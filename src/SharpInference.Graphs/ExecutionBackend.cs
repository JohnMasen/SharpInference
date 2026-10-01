namespace SharpInference.Graphs;

public interface IBackendExecutablePlan
{
    ExecutionGraph Graph { get; }
}

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

public sealed record BackendPreparationDiagnostic(
    ExecutionNodeId? NodeId,
    GraphOperationId? Operation,
    IReadOnlyList<LogicalNodeId> SourceNodes,
    BackendPreparationFailureReason Reason,
    string Message,
    PrecisionRequirement? RequiredPrecision,
    IReadOnlyList<OperatorImplementationDescription> AvailableImplementations);

public abstract record BackendPreparationResult
{
    private BackendPreparationResult()
    {
    }

    public sealed record Success : BackendPreparationResult
    {
        public Success(IBackendExecutablePlan plan)
        {
            Plan = plan ?? throw new ArgumentNullException(nameof(plan));
        }

        public IBackendExecutablePlan Plan { get; }
    }

    public sealed record Failure : BackendPreparationResult
    {
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

    public IBackendExecutablePlan GetPlanOrThrow(string backendId) => this switch
    {
        Success success => success.Plan,
        Failure failure => throw new BackendPreparationException(backendId, failure.Diagnostics),
        _ => throw new InvalidOperationException("Unknown backend preparation result."),
    };
}

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

public static class BackendPreparation
{
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
