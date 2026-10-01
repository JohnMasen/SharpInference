using SharpInference.Graphs;
using Vortice.Direct3D12;

namespace SharpInference.Backends.Vortice;

/// <summary>
/// Opt-in graph backend for dense FP32 primitive graphs. The caller owns the D3D12 device;
/// dispose sessions before disposing this backend. Does not retain model tensor catalogs.
/// </summary>
public sealed class VorticePrimitiveGraphBackend : IGraphModelWeightBackend, IDisposable
{
    private static readonly GraphOperationId[] Operations =
    [
        PrimitiveGraphOperations.Copy, PrimitiveGraphOperations.Add,
        PrimitiveGraphOperations.Subtract, PrimitiveGraphOperations.Multiply,
        PrimitiveGraphOperations.Divide, PrimitiveGraphOperations.Maximum,
        PrimitiveGraphOperations.Relu, PrimitiveGraphOperations.Sigmoid,
        PrimitiveGraphOperations.Exp, PrimitiveGraphOperations.Tanh,
        PrimitiveGraphOperations.ReciprocalSquareRoot, PrimitiveGraphOperations.Square,
        PrimitiveGraphOperations.ReduceSum, PrimitiveGraphOperations.ReduceMean,
        PrimitiveGraphOperations.MatVec, PrimitiveGraphOperations.GatherRow,
        PortableTensorOperationContracts.Fill, PortableTensorOperationContracts.CastFp16ToFp32,
        PortableTensorOperationContracts.Reshape,
        PortableTensorOperationContracts.Slice, PortableTensorOperationContracts.Broadcast,
        PortableTensorOperationContracts.BatchedMatVec,
        PortableTensorOperationContracts.ReduceLastSum, PortableTensorOperationContracts.ReduceLastMean,
        PortableTensorOperationContracts.HeadOuter,
        FusedElementwiseExpressionContract.Operation,
    ];
    private readonly ID3D12Device device;
    private readonly bool ownsDevice;
    private VorticePrimitiveOperatorBackend? primitives;
    private VorticePrimitiveGraphExecutor? executor;
    private bool profileCommandRecording;
    private bool enableCommandReplay;
    private bool disposed;

    /// <summary>The caller retains ownership of the supplied device and must dispose it after this backend.</summary>
    public VorticePrimitiveGraphBackend(ID3D12Device device)
        : this(device, ownsDevice: false, deviceName: null) { }

    private VorticePrimitiveGraphBackend(ID3D12Device device, bool ownsDevice, string? deviceName)
    {
        this.device = device ?? throw new ArgumentNullException(nameof(device));
        this.ownsDevice = ownsDevice;
        DeviceName = deviceName;
    }

    /// <summary>
    /// Selects a supported hardware adapter in high-performance order. The returned backend owns
    /// and disposes its device. EnableCommandReplay opts into per-session closed-list replay;
    /// The other config option selects the hardware adapter.
    /// </summary>
    public static VorticePrimitiveGraphBackend FromConfig(VorticeRuntimeConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        var (device, name) = VorticePrimitiveGraphDeviceFactory.Create(config.AdapterIndex);
        return new VorticePrimitiveGraphBackend(device, ownsDevice: true, name)
        {
            EnableCommandReplay = config.EnableCommandReplay,
        };
    }

    public string? DeviceName { get; }
    public bool RequiresCpuWeightCopy => false;
    public VorticePrimitiveGraphMetrics? Metrics => executor?.Metrics;
    public VorticePrimitiveGraphMemoryMetrics? Memory => executor?.Memory;
    public int? DispatchesPerToken => executor?.DispatchesPerToken;
    public VorticePrimitiveGraphExecutionProfile? LastExecutionProfile => executor?.LastExecutionProfile;
    /// <summary>Opt-in per-session closed-list replay for scalar-token graphs; unsupported inputs fail preparation.</summary>
    public bool EnableCommandReplay
    {
        get => enableCommandReplay;
        set
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (executor is not null) executor.EnableCommandReplay = value;
            enableCommandReplay = value;
        }
    }
    public bool ProfileCommandRecording
    {
        get => profileCommandRecording;
        set
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            profileCommandRecording = value;
            if (executor is not null) executor.ProfileCommandRecording = value;
        }
    }
    public IExecutionKernelCatalog KernelCatalog { get; } =
        new VorticeExpressionKernelCatalog(Operations);
    public IPrimitiveOperatorBackend PrimitiveOperators
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return primitives ??= new VorticePrimitiveOperatorBackend(device, native16BitShaderOpsSupported: false);
        }
    }

    public IReadOnlyList<OperatorImplementationDescription> GetOperatorImplementations(
        ExecutionGraph graph, ExecutionNode node)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(node);
        if (!KernelCatalog.Supports(node.Operation)) return [];
        if (node.Operation == FusedElementwiseExpressionContract.Operation)
        {
            var tensors = graph.Resources.ToDictionary(resource => resource.Id, resource => resource.Tensor);
            var output = node.Resources.Single(binding => binding.Access == GraphResourceAccess.Write);
            return ((VorticeExpressionKernelCatalog)KernelCatalog).GetExpressionImplementations(
                new OperatorSignature(node.Resources
                    .Where(binding => binding.Access == GraphResourceAccess.Read)
                    .Select(binding => tensors[binding.Resource].ElementType),
                    [tensors[output.Resource].ElementType]), tensors[output.Resource]);
        }
        return
        [
            new OperatorImplementationDescription(
                $"vortice.primitive.{node.Operation.Name}.fp32", node.Operation,
                BackendPreparation.CreateSignature(node, graph.Resources.ToDictionary(resource => resource.Id)),
                new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32)),
        ];
    }

    public BackendPreparationResult Prepare(ExecutionGraph graph)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(graph);
        try
        {
            var prepared = VorticePrimitiveGraphPlan.Compile(graph);
            if (enableCommandReplay) VorticePrimitiveGraphExecutor.ValidateReplay(prepared);
            return new BackendPreparationResult.Success(prepared);
        }
        catch (Exception error) when (error is NotSupportedException or InvalidDataException or OverflowException)
        {
            var node = graph.Nodes.FirstOrDefault(candidate =>
                error.Message.Contains($"node '{candidate.Id}'", StringComparison.Ordinal));
            return new BackendPreparationResult.Failure(
            [
                new BackendPreparationDiagnostic(node?.Id, node?.Operation,
                    node?.Source.LogicalNodes ?? [],
                    node is not null && !KernelCatalog.Supports(node.Operation)
                        ? BackendPreparationFailureReason.UnsupportedOperation
                        : BackendPreparationFailureReason.ShapeNotSupported,
                    error.Message, node?.Requirements,
                    node is null ? [] : GetOperatorImplementations(graph, node)),
            ]);
        }
    }

    internal sealed class VorticeExpressionKernelCatalog(GraphOperationId[] operations)
        : IExecutionKernelCatalog, IFusedElementwiseExpressionProvider
    {
        private readonly ExecutionKernelCatalog inner = new("vortice.primitive.graph", operations);

        public string BackendId => inner.BackendId;
        public IReadOnlySet<GraphOperationId> SupportedOperations => inner.SupportedOperations;
        public IReadOnlyList<GraphNodeDefinition> NodeDefinitions => inner.NodeDefinitions;
        public bool Supports(GraphOperationId operation) => inner.Supports(operation);

        public IReadOnlyList<OperatorImplementationDescription> GetExpressionImplementations(
            OperatorSignature signature, TensorDescriptor tensor) =>
            tensor.ElementType == GraphElementType.Float32 && tensor.Layout == "dense" &&
            signature.InputTypes.Count is > 0 and <= VorticeFusedExpressionKernel.MaximumInputs &&
            signature.InputTypes.All(type => type == GraphElementType.Float32) &&
            signature.OutputTypes.SequenceEqual([GraphElementType.Float32])
                ? [new OperatorImplementationDescription("vortice.fused-elementwise-expression.float32",
                    FusedElementwiseExpressionContract.Operation, signature,
                    new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32))]
                : [];
    }

    /// <summary>Called during architecture binding while the source catalog is still alive.</summary>
    public void PrepareModelWeights(PortableGraphModel model)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(model);
        if (executor is not null)
            throw new InvalidOperationException(
                "This graph backend already owns resident model weights; use a separate backend for another model.");
        var prepared = new VorticePrimitiveGraphExecutor(device, model.Graph);
        try
        {
            prepared.ProfileCommandRecording = profileCommandRecording;
            prepared.EnableCommandReplay = enableCommandReplay;
            prepared.PrepareModelWeights(model);
            executor = prepared;
        }
        catch
        {
            prepared.Dispose();
            throw;
        }
    }

    public IProcessorSessionExecutor CreateSessionExecutor(
        IRwkvModel model, IRwkvState state, IBackendExecutablePlan plan)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (executor is null)
            throw new InvalidOperationException("Graph weights must be prepared before creating a GPU session.");
        if (model is not PortableGraphModel portable ||
            state is not PortableGraphState portableState ||
            plan is not VorticePrimitiveGraphPlan prepared ||
            !ReferenceEquals(portable.Graph, executor.Plan.Graph) ||
            !ReferenceEquals(prepared.Graph, executor.Plan.Graph))
            throw new ArgumentException("The model, state, and prepared plan must belong to this GPU graph.");
        var graph = prepared.Graph;
        var outputResource = graph.Outputs.Count == 1
            ? graph.Resources.Single(resource => resource.Id == graph.Outputs[0])
            : null;
        if (graph.Inputs.Count != 1 ||
            graph.Resources.Single(resource => resource.Id == graph.Inputs[0]).Tensor.ElementType != GraphElementType.Int32 ||
            !graph.Resources.Single(resource => resource.Id == graph.Inputs[0]).Tensor.Dimensions.SequenceEqual([1]) ||
            outputResource?.Tensor.ElementType != GraphElementType.Float32 ||
            !outputResource.Tensor.Dimensions.SequenceEqual([portable.Metadata.VocabularySize]))
            throw new NotSupportedException(
                "The token session adapter requires an Int32[1] input and FP32[vocabulary] output; use CreateSession/ExecuteToken for other graphs.");
        var session = executor.CreateSession();
        try
        {
            var bridge = new VorticePrimitiveGraphStateBridge(session, portableState, graph);
            return new TokenSession(session, bridge, graph.Inputs[0], graph.Outputs[0],
                portable.Metadata.VocabularySize);
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        try
        {
            executor?.Dispose();
            primitives?.Dispose();
        }
        finally
        {
            if (ownsDevice) device.Dispose();
        }
    }

    private sealed class TokenSession(
        VorticePrimitiveGraphSession session,
        VorticePrimitiveGraphStateBridge bridge,
        ResourceId tokenInput,
        ResourceId output,
        int outputCount) : IProcessorSessionExecutor, IProcessorStateExecutor
    {
        private bool disposed;

        public void ForwardToken(int token, Span<float> logits)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (logits.Length != outputCount)
                throw new ArgumentException($"The graph output requires {outputCount} FP32 values.", nameof(logits));
            bridge.BeforeToken();
            var result = session.ExecuteToken(
                new Dictionary<ResourceId, float[]>(),
                new Dictionary<ResourceId, int> { [tokenInput] = token });
            result[output].CopyTo(logits);
            bridge.AfterToken();
        }

        public IReadOnlyList<GraphStateValue> ReadState(GraphState graph) => bridge.ReadState(graph);
        public void WriteState(GraphState graph, IReadOnlyList<GraphStateValue> values) =>
            bridge.WriteState(graph, values);

        public void Dispose()
        {
            if (disposed) return;
            bridge.Dispose();
            session.Dispose();
            disposed = true;
        }
    }
}
