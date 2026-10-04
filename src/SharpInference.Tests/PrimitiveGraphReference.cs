using System.Runtime.InteropServices;
using SharpInference.Gguf;
using SharpInference.Graphs;
using SharpInference.Runtime;

namespace SharpInference.Tests;

// Standalone primitive-backend fixtures retain reference kernel coverage outside the VM runtime.
internal sealed class PrimitiveGraphReference : IDisposable
{
    private readonly GgmlModelFile catalog;
    private readonly IExecutionGraphBackend backend;
    private readonly IBackendExecutablePlan plan;
    private readonly PortableGraphModel model;

    private PrimitiveGraphReference(GgmlModelFile catalog, IExecutionGraphBackend backend,
        IBackendExecutablePlan plan, PortableGraphModel model)
    {
        this.catalog = catalog;
        this.backend = backend;
        this.plan = plan;
        this.model = model;
    }

    public static PrimitiveGraphReference LoadGraph(string path, LogicalGraph logical,
        IExecutionGraphBackend backend, GraphOptimizationOptions options)
    {
        var catalog = GgmlModelFile.Open(path);
        try
        {
            var graph = new GraphOptimizer().Optimize(logical, options, backend.KernelCatalog);
            var plan = backend.Prepare(graph).GetPlanOrThrow(backend.KernelCatalog.BackendId);
            var model = new PortableGraphModel(new GraphArchitectureMetadataReader(logical).Read(catalog),
                catalog, graph);
            if (backend is IGraphModelWeightBackend weights) weights.PrepareModelWeights(model);
            return new PrimitiveGraphReference(catalog, backend, plan, model);
        }
        catch (Exception error)
        {
            try { catalog.Dispose(); }
            catch (Exception cleanup) { throw new AggregateException(error, cleanup); }
            throw;
        }
    }

    public PrimitiveGraphReferenceSession CreateSession() =>
        CreateSession(new PortableGraphState(model.Graph));

    internal PrimitiveGraphReferenceSession CreateSession(PortableGraphState state) =>
        new(this, state, backend.CreateSessionExecutor(model, state, plan), model.Graph,
            model.Metadata.VocabularySize);

    public void Dispose() => catalog.Dispose();
}

internal sealed class PrimitiveGraphReferenceSession(
    PrimitiveGraphReference owner, PortableGraphState state,
    IProcessorSessionExecutor executor, ExecutionGraph graph, int vocabularySize) : IDisposable
{
    private readonly float[] logits = new float[vocabularySize];
    private readonly IProcessorStateExecutor stateAccess = Assert.IsAssignableFrom<IProcessorStateExecutor>(executor);

    public ReadOnlyMemory<float> ForwardToken(int token)
    {
        executor.ForwardToken(token, logits);
        return logits;
    }

    public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens)
    {
        executor.ForwardTokens(tokens, logits);
        return logits;
    }

    public PrimitiveGraphReferenceSession Fork() =>
        owner.CreateSession(Assert.IsType<PortableGraphState>(state.Clone()));

    public void Reset() => state.Reset();

    public GgufState Capture() => new(graph.GraphState.Schema.Name,
        stateAccess.ReadState(graph.GraphState).Select(value =>
            new GgufStateTensor(value.Name, GgufTensorType.Float32,
                value.Dimensions.Select(size => (ulong)size),
                MemoryMarshal.AsBytes(value.Values.AsSpan()).ToArray())));

    public void SaveState(Stream stream) => GgufStateFile.Write(stream, Capture());

    public void LoadState(Stream stream)
    {
        var saved = GgufStateFile.Read(stream);
        Assert.Equal(graph.GraphState.Schema.Name, saved.SchemaName);
        Assert.All(saved.Tensors, tensor => Assert.Equal(GgufTensorType.Float32, tensor.Type));
        stateAccess.WriteState(graph.GraphState, saved.Tensors.Select(tensor =>
            new GraphStateValue(tensor.Name, tensor.Dimensions.Select(size => checked((int)size)).ToArray(),
                MemoryMarshal.Cast<byte, float>(tensor.Data.Span).ToArray())).ToArray());
    }

    public void Dispose() => executor.Dispose();
}
