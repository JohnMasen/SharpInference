namespace SharpInference.Graphs;

/// <summary>Represents an executable backend plan for an execution graph.</summary>
public interface IBackendExecutablePlan
{
    /// <summary>Gets the graph represented by this model or plan.</summary>
    ExecutionGraph Graph { get; }
}

/// <summary>Identifies why a backend could not prepare an execution node.</summary>
public enum BackendPreparationFailureReason
{
    /// <summary>The backend does not support the operation.</summary>
    UnsupportedOperation,
    /// <summary>No implementation matches the input and output element types.</summary>
    SignatureMismatch,
    /// <summary>No matching implementation meets the required precision.</summary>
    PrecisionRequirementNotMet,
    /// <summary>The backend does not support the tensor shape.</summary>
    ShapeNotSupported,
    /// <summary>The required hardware capability is unavailable.</summary>
    HardwareCapabilityMissing,
    /// <summary>Runtime compilation of an implementation failed.</summary>
    DynamicCompilationFailed,
    /// <summary>The preparation exceeded a backend resource limit.</summary>
    ResourceLimitExceeded,
}

/// <summary>Describes one backend preparation failure and the implementations considered.</summary>
/// <param name="NodeId">The failing execution node identifier, or null for a graph-level failure.</param>
/// <param name="Operation">The operation contract identifier.</param>
/// <param name="SourceNodes">The logical nodes associated with the preparation failure.</param>
/// <param name="Reason">The category of backend preparation failure.</param>
/// <param name="Message">The human-readable failure description.</param>
/// <param name="RequiredPrecision">The required operator precision, or null for a graph-level failure.</param>
/// <param name="AvailableImplementations">The backend implementations considered for the operation.</param>
public sealed record BackendPreparationDiagnostic(
    ExecutionNodeId? NodeId,
    GraphOperationId? Operation,
    IReadOnlyList<LogicalNodeId> SourceNodes,
    BackendPreparationFailureReason Reason,
    string Message,
    PrecisionRequirement? RequiredPrecision,
    IReadOnlyList<OperatorImplementationDescription> AvailableImplementations);

/// <summary>Represents either a prepared backend plan or preparation diagnostics.</summary>
public abstract record BackendPreparationResult
{
    private BackendPreparationResult()
    {
    }

    /// <summary>Contains an executable plan produced by successful backend preparation.</summary>
    public sealed record Success : BackendPreparationResult
    {
        /// <summary>Creates a successful result containing an executable plan.</summary>
        /// <param name="plan">The prepared executable plan.</param>
        public Success(IBackendExecutablePlan plan)
        {
            Plan = plan ?? throw new ArgumentNullException(nameof(plan));
        }

        /// <summary>Gets the successfully prepared backend executable plan.</summary>
        public IBackendExecutablePlan Plan { get; }
    }

    /// <summary>Contains diagnostics produced by unsuccessful backend preparation.</summary>
    public sealed record Failure : BackendPreparationResult
    {
        /// <summary>Creates a failed result containing one or more diagnostics.</summary>
        /// <param name="diagnostics">The diagnostics explaining preparation failures.</param>
        public Failure(IEnumerable<BackendPreparationDiagnostic> diagnostics)
        {
            Diagnostics = diagnostics?.ToArray() ?? throw new ArgumentNullException(nameof(diagnostics));
            if (Diagnostics.Count == 0)
            {
                throw new ArgumentException("A failed backend preparation requires at least one diagnostic.", nameof(diagnostics));
            }
        }

        /// <summary>Gets the diagnostics describing backend preparation failures.</summary>
        public IReadOnlyList<BackendPreparationDiagnostic> Diagnostics { get; }
    }

    /// <summary>Returns the prepared plan or throws an exception containing failure diagnostics.</summary>
    /// <param name="backendId">The identifier of the backend that performed preparation.</param>
    /// <returns>The prepared executable plan.</returns>
    public IBackendExecutablePlan GetPlanOrThrow(string backendId) => this switch
    {
        Success success => success.Plan,
        Failure failure => throw new BackendPreparationException(backendId, failure.Diagnostics),
        _ => throw new InvalidOperationException("Unknown backend preparation result."),
    };
}

/// <summary>Reports that a backend could not prepare one or more execution nodes.</summary>
public sealed class BackendPreparationException : InvalidOperationException
{
    /// <summary>Creates an exception identifying the backend and its preparation failures.</summary>
    /// <param name="backendId">The backend identifier.</param>
    /// <param name="diagnostics">The diagnostics explaining backend preparation failures.</param>
    public BackendPreparationException(
        string backendId,
        IEnumerable<BackendPreparationDiagnostic> diagnostics)
        : base(CreateMessage(backendId, diagnostics, out var values))
    {
        BackendId = backendId;
        Diagnostics = values;
    }

    /// <summary>Gets the backend identifier.</summary>
    public string BackendId { get; }
    /// <summary>Gets the diagnostics describing backend preparation failures.</summary>
    public IReadOnlyList<BackendPreparationDiagnostic> Diagnostics { get; }

    private static string CreateMessage(
        string backendId,
        IEnumerable<BackendPreparationDiagnostic> diagnostics,
        out IReadOnlyList<BackendPreparationDiagnostic> values)
    {
        if (string.IsNullOrWhiteSpace(backendId))
        {
            throw new ArgumentException("A backend identifier is required.", nameof(backendId));
        }

        values = diagnostics?.ToArray() ?? throw new ArgumentNullException(nameof(diagnostics));
        if (values.Count == 0)
        {
            throw new ArgumentException("At least one backend preparation diagnostic is required.", nameof(diagnostics));
        }

        return $"Backend '{backendId}' cannot prepare {values.Count} execution node(s): " +
               string.Join("; ", values.Select(value =>
                   $"{value.NodeId?.Value ?? "<graph>"} {value.Operation?.ToString() ?? "<resource>"}: {value.Message}"));
    }
}

/// <summary>Contains implementation selections and diagnostics from backend preflight.</summary>
public sealed class BackendPreflightResult
{
    internal BackendPreflightResult(
        IReadOnlyDictionary<ExecutionNodeId, OperatorImplementationDescription> selections,
        IReadOnlyList<BackendPreparationDiagnostic> diagnostics)
    {
        Selections = selections;
        Diagnostics = diagnostics;
    }

    /// <summary>Gets the compatible implementation selected for each execution node.</summary>
    public IReadOnlyDictionary<ExecutionNodeId, OperatorImplementationDescription> Selections { get; }
    /// <summary>Gets the diagnostics describing backend preparation failures.</summary>
    public IReadOnlyList<BackendPreparationDiagnostic> Diagnostics { get; }
    /// <summary>Gets whether preflight completed without preparation diagnostics.</summary>
    public bool Succeeded => Diagnostics.Count == 0;
}

/// <summary>Selects compatible backend implementations before execution.</summary>
public static class BackendPreparation
{
    /// <summary>Preflights each graph node against implementations supplied by the resolver.</summary>
    /// <param name="graph">The execution graph to inspect.</param>
    /// <param name="implementationResolver">Resolves candidate implementations for each node.</param>
    /// <returns>Selections for compatible nodes and diagnostics for nodes without a selection.</returns>
    public static BackendPreflightResult Preflight(
        ExecutionGraph graph,
        Func<ExecutionNode, IReadOnlyList<OperatorImplementationDescription>> implementationResolver)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(implementationResolver);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var selections = new Dictionary<ExecutionNodeId, OperatorImplementationDescription>();
        var diagnostics = new List<BackendPreparationDiagnostic>();

        foreach (var node in graph.Nodes)
        {
            var implementations = implementationResolver(node)?.ToArray()
                ?? throw new InvalidOperationException($"The implementation resolver returned null for node '{node.Id}'.");
            var signature = CreateSignature(node, resources);
            var selected = OperatorImplementationSelector.Select(
                node.Operation,
                signature,
                node.Requirements,
                implementations);
            if (selected is not null)
            {
                selections.Add(node.Id, selected);
                continue;
            }

            var operationCandidates = implementations
                .Where(candidate => candidate.Operation == node.Operation)
                .ToArray();
            var signatureCandidates = operationCandidates
                .Where(candidate => candidate.Signature.Matches(signature))
                .ToArray();
            var reason = operationCandidates.Length == 0
                ? BackendPreparationFailureReason.UnsupportedOperation
                : signatureCandidates.Length == 0
                    ? BackendPreparationFailureReason.SignatureMismatch
                    : BackendPreparationFailureReason.PrecisionRequirementNotMet;
            var message = reason switch
            {
                BackendPreparationFailureReason.UnsupportedOperation =>
                    $"Operation '{node.Operation}' is not supported.",
                BackendPreparationFailureReason.SignatureMismatch =>
                    $"No implementation matches inputs [{string.Join(", ", signature.InputTypes)}] and outputs [{string.Join(", ", signature.OutputTypes)}].",
                _ =>
                    $"No matching implementation satisfies arithmetic >= {node.Requirements.MinimumArithmeticType} and accumulator >= {node.Requirements.MinimumAccumulatorType}.",
            };
            diagnostics.Add(new BackendPreparationDiagnostic(
                node.Id,
                node.Operation,
                node.Source.LogicalNodes,
                reason,
                message,
                node.Requirements,
                operationCandidates));
        }

        return new BackendPreflightResult(selections, diagnostics);
    }

    /// <summary>Builds an ordered input and output element-type signature from an execution node&apos;s resource accesses.</summary>
    /// <param name="node">The operation node to inspect.</param>
    /// <param name="resources">The graph resource descriptors.</param>
    public static OperatorSignature CreateSignature(
        ExecutionNode node,
        IReadOnlyDictionary<ResourceId, GraphResource> resources)
    {
        ArgumentNullException.ThrowIfNull(node);
        ArgumentNullException.ThrowIfNull(resources);
        var inputs = node.Resources
            .Where(binding => binding.Access != GraphResourceAccess.Write)
            .Select(binding => resources[binding.Resource].Tensor.ElementType);
        var outputs = node.Resources
            .Where(binding => binding.Access != GraphResourceAccess.Read)
            .Select(binding => resources[binding.Resource].Tensor.ElementType);
        return new OperatorSignature(inputs, outputs);
    }
}

/// <summary>Prepares execution graphs and exposes primitive operators and kernel capabilities.</summary>
public interface IExecutionGraphBackend : IProcessorSessionBackend
{
    /// <summary>Gets the primitive operation descriptions supported by this backend.</summary>
    IPrimitiveOperatorBackend PrimitiveOperators { get; }
    /// <summary>Gets the backend&apos;s operation capabilities and node definitions.</summary>
    IExecutionKernelCatalog KernelCatalog { get; }
    /// <summary>Returns candidate backend implementations for an execution node.</summary>
    /// <param name="graph">The graph supplying operations and resource binding metadata.</param>
    /// <param name="node">The operation node to inspect.</param>
    IReadOnlyList<OperatorImplementationDescription> GetOperatorImplementations(
        ExecutionGraph graph,
        ExecutionNode node);
    /// <summary>Prepares an executable graph plan or returns diagnostics describing preparation failures.</summary>
    /// <param name="graph">The graph supplying operations and resource binding metadata.</param>
    BackendPreparationResult Prepare(ExecutionGraph graph);
}

/// <summary>Prepares model weights for a graph backend and declares CPU-copy requirements.</summary>
public interface IGraphModelWeightBackend : IExecutionGraphBackend
{
    /// <summary>Gets whether model weights must remain available in CPU memory.</summary>
    bool RequiresCpuWeightCopy { get; }
    /// <summary>Prepares the model&apos;s weight resources for execution by this backend.</summary>
    /// <param name="model">The model instance or graph model signature.</param>
    void PrepareModelWeights(PortableGraphModel model);
}

/// <summary>Optional session backend accepting a separately prepared prefill plan.</summary>
public interface IProcessorPrefillBackend : IExecutionGraphBackend
{
    /// <summary>Determines whether the backend can prepare a separate prefill plan for the inference graph.</summary>
    /// <param name="inferenceGraph">The execution graph used for single-token inference.</param>
    bool CanPreparePrefill(ExecutionGraph inferenceGraph) => true;

    /// <summary>Creates a processor executor using the supplied model, state, and prepared plans.</summary>
    /// <param name="model">The model executed by the processor session.</param>
    /// <param name="state">The model session state.</param>
    /// <param name="inferencePlan">The prepared single-token inference plan.</param>
    /// <param name="prefillPlan">The separately prepared multi-token prefill plan.</param>
    IProcessorSessionExecutor CreateSessionExecutor(
        IModel model,
        IModelState state,
        IBackendExecutablePlan inferencePlan,
        IBackendExecutablePlan prefillPlan);
}
