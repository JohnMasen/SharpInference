namespace SharpInference.Graphs;

/// <summary>Represents an executable backend plan for an execution graph.</summary>
public interface IBackendExecutablePlan
{
    ExecutionGraph Graph { get; }
}

/// <summary>Identifies why a backend could not prepare an execution node.</summary>
public enum BackendPreparationFailureReason
{
    UnsupportedOperation,
    SignatureMismatch,
    PrecisionRequirementNotMet,
    ShapeNotSupported,
    HardwareCapabilityMissing,
    DynamicCompilationFailed,
    ResourceLimitExceeded,
}

/// <summary>Describes one backend preparation failure and the implementations considered.</summary>
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

    public sealed record Success : BackendPreparationResult
    {
        /// <summary>Creates a successful result containing an executable plan.</summary>
        /// <param name="plan">The prepared executable plan.</param>
        public Success(IBackendExecutablePlan plan)
        {
            Plan = plan ?? throw new ArgumentNullException(nameof(plan));
        }

        public IBackendExecutablePlan Plan { get; }
    }

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
    public BackendPreparationException(
        string backendId,
        IEnumerable<BackendPreparationDiagnostic> diagnostics)
        : base(CreateMessage(backendId, diagnostics, out var values))
    {
        BackendId = backendId;
        Diagnostics = values;
    }

    public string BackendId { get; }
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

    public IReadOnlyDictionary<ExecutionNodeId, OperatorImplementationDescription> Selections { get; }
    public IReadOnlyList<BackendPreparationDiagnostic> Diagnostics { get; }
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

public interface IExecutionGraphBackend : IProcessorSessionBackend
{
    IPrimitiveOperatorBackend PrimitiveOperators { get; }
    IExecutionKernelCatalog KernelCatalog { get; }
    IReadOnlyList<OperatorImplementationDescription> GetOperatorImplementations(
        ExecutionGraph graph,
        ExecutionNode node);
    BackendPreparationResult Prepare(ExecutionGraph graph);
}

public interface IGraphModelWeightBackend : IExecutionGraphBackend
{
    bool RequiresCpuWeightCopy { get; }
    void PrepareModelWeights(PortableGraphModel model);
}

/// <summary>Optional session backend accepting a separately prepared prefill plan.</summary>
public interface IProcessorPrefillBackend : IExecutionGraphBackend
{
    bool CanPreparePrefill(ExecutionGraph inferenceGraph) => true;

    IProcessorSessionExecutor CreateSessionExecutor(
        IRwkvModel model,
        IRwkvState state,
        IBackendExecutablePlan inferencePlan,
        IBackendExecutablePlan prefillPlan);
}
