using System.Runtime.CompilerServices;
using SharpInference.Graphs;

namespace SharpInference.Backends.Cpu;

/// <summary>Processor adapter for primitive-only graphs with portable model and state types.</summary>
public sealed class CpuPrimitiveGraphBackend : IGraphModelWeightBackend
{
    private readonly ConditionalWeakTable<PortableGraphModel, CpuPrimitiveGraphExecutor> models = new();

    public static CpuPrimitiveGraphBackend Instance { get; } = new();

    public IPrimitiveOperatorBackend PrimitiveOperators => CpuPrimitiveOperatorBackend.Instance;
    public IExecutionKernelCatalog KernelCatalog { get; } = new CpuExpressionKernelCatalog();

    private static readonly IReadOnlyList<OperatorImplementationDescription> TensorImplementations =
        PortableTensorOperationContracts.Contracts.Select(contract =>
            new OperatorImplementationDescription(
                $"cpu.{contract.Operation.Name}.float32", contract.Operation, contract.Signature,
                new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32)))
            .ToArray();

    // The Processor must own the catalog after its reader is released. The executor retains
    // tensor references, not a second copy of each weight.
    public bool RequiresCpuWeightCopy => true;

    public IReadOnlyList<OperatorImplementationDescription> GetOperatorImplementations(
        ExecutionGraph graph, ExecutionNode node)
    {
        if (node.Operation == FusedElementwiseExpressionContract.Operation)
        {
            var tensors = graph.Resources.ToDictionary(resource => resource.Id, resource => resource.Tensor);
            var output = node.Resources.Single(binding => binding.Access == GraphResourceAccess.Write);
            return ((CpuExpressionKernelCatalog)KernelCatalog).GetExpressionImplementations(
                new OperatorSignature(node.Resources
                    .Where(binding => binding.Access == GraphResourceAccess.Read)
                    .Select(binding => tensors[binding.Resource].ElementType),
                    [tensors[output.Resource].ElementType]), tensors[output.Resource]);
        }
        return PrimitiveOperators.OperatorImplementations.Concat(TensorImplementations)
            .Where(implementation => implementation.Operation == node.Operation).ToArray();
    }

    public BackendPreparationResult Prepare(ExecutionGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        try
        {
            CpuPrimitiveGraphExecutor.Validate(graph, PrimitiveOperators);
            return new BackendPreparationResult.Success(new CpuPrimitiveGraphPlan(graph));
        }
        catch (Exception error) when (error is InvalidDataException or NotSupportedException or OverflowException)
        {
            var node = graph.Nodes.FirstOrDefault(candidate =>
                error.Message.Contains($"'{candidate.Id}'", StringComparison.Ordinal));
            return new BackendPreparationResult.Failure(
            [
                new BackendPreparationDiagnostic(node?.Id, node?.Operation,
                    node?.Source.LogicalNodes ?? [],
                    error.Message.Contains("Unsupported primitive", StringComparison.Ordinal) ||
                    error.Message.Contains("Unsupported tensor operation", StringComparison.Ordinal)
                        ? BackendPreparationFailureReason.UnsupportedOperation
                        : error.Message.Contains("unsupported tensor types or precision", StringComparison.Ordinal)
                            ? BackendPreparationFailureReason.SignatureMismatch
                            : BackendPreparationFailureReason.ShapeNotSupported,
                    error.Message, node?.Requirements,
                    node is null ? [] : GetOperatorImplementations(graph, node)),
            ]);
        }
    }

    public void PrepareModelWeights(PortableGraphModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        models.GetValue(model, bound => new CpuPrimitiveGraphExecutor(bound, PrimitiveOperators));
    }

    public IProcessorSessionExecutor CreateSessionExecutor(
        IModel model, IModelState state, IBackendExecutablePlan plan)
    {
        if (model is not PortableGraphModel portable ||
            state is not PortableGraphState portableState ||
            plan is not CpuPrimitiveGraphPlan cpuPlan ||
            !ReferenceEquals(portable.Graph, cpuPlan.Graph))
            throw new ArgumentException("The portable model, state, and CPU plan do not match.");
        var graph = cpuPlan.Graph;
        if (portableState.ArchitectureId != graph.Identity.ArchitectureId)
            throw new ArgumentException("The state architecture does not match the execution graph.", nameof(state));
        if (graph.Inputs.Count != 1 || graph.Outputs.Count != 1)
            throw new NotSupportedException("Processor token sessions require one token input and one logits output.");
        var input = graph.Resources.Single(resource => resource.Id == graph.Inputs[0]);
        var output = graph.Resources.Single(resource => resource.Id == graph.Outputs[0]);
        if (input.Tensor.ElementType != GraphElementType.Int32 ||
            !input.Tensor.Dimensions.SequenceEqual([1]) ||
            output.Tensor.ElementType != GraphElementType.Float32 ||
            output.Tensor.Dimensions.Count != 1)
            throw new NotSupportedException("Processor token sessions require int32[1] input and float32[vocabulary] output.");
        GraphSessionStateAccess.Read(graph.GraphState, graph.Resources, portableState);
        var executor = models.GetValue(portable, bound =>
            new CpuPrimitiveGraphExecutor(bound, PrimitiveOperators));
        return new Session(executor, portableState, graph, graph.Inputs[0], graph.Outputs[0],
            output.Tensor.Dimensions[0]);
    }

    private sealed class Session(CpuPrimitiveGraphExecutor executor, PortableGraphState state,
        ExecutionGraph graph, ResourceId input, ResourceId output, int vocabularySize)
        : IProcessorSessionExecutor, IProcessorStateExecutor
    {
        private bool disposed;

        public void ForwardToken(int token, Span<float> logits)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (logits.Length != vocabularySize)
                throw new ArgumentException("The logits buffer does not match the graph vocabulary.", nameof(logits));
            executor.ExecuteInto(
                new Dictionary<ResourceId, Array> { [input] = new[] { token } },
                state, output, logits);
        }

        public IReadOnlyList<GraphStateValue> ReadState(GraphState schema) =>
            GraphSessionStateAccess.Read(schema, graph.Resources, state);

        public void WriteState(GraphState schema, IReadOnlyList<GraphStateValue> values) =>
            GraphSessionStateAccess.Write(schema, graph.Resources, state, values);

        public void Dispose()
        {
            disposed = true;
        }
    }
}

public sealed record CpuPrimitiveGraphPlan(ExecutionGraph Graph) : IBackendExecutablePlan;

internal sealed class CpuExpressionKernelCatalog : IExecutionKernelCatalog,
    IFusedElementwiseExpressionProvider
{
    private readonly ExecutionKernelCatalog inner = new("cpu",
        CpuPrimitiveOperatorBackend.Instance.PrimitiveOperators
            .Select(description => description.Operation)
            .Concat(PortableTensorOperationContracts.Contracts.Select(contract => contract.Operation))
            .Append(FusedElementwiseExpressionContract.Operation));

    public string BackendId => inner.BackendId;
    public IReadOnlySet<GraphOperationId> SupportedOperations => inner.SupportedOperations;
    public IReadOnlyList<GraphNodeDefinition> NodeDefinitions => inner.NodeDefinitions;
    public bool Supports(GraphOperationId operation) => inner.Supports(operation);

    public IReadOnlyList<OperatorImplementationDescription> GetExpressionImplementations(
        OperatorSignature signature, TensorDescriptor tensor) =>
        tensor.ElementType == GraphElementType.Float32 && tensor.Layout == "dense" &&
        signature.InputTypes.Count > 0 &&
        signature.InputTypes.All(type => type == GraphElementType.Float32) &&
        signature.OutputTypes.SequenceEqual([GraphElementType.Float32])
            ? [new OperatorImplementationDescription("cpu.fused-elementwise-expression.float32",
                FusedElementwiseExpressionContract.Operation, signature,
                new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32))]
            : [];
}
