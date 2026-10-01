using System.Text;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Backends.Cpu;
using SharpInference.Graphs;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class ProcessorPipelineTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GenericPrefillOptimizer_PreparesAndRoutesOnlyMatchingGraphs(bool matches)
    {
        var backend = new PrefillBackend();
        using var processor = PrefillBuilder(backend, new TestPrefillOptimizer(matches)).Build();

        Assert.Equal(matches ? 2 : 1, backend.Prepared.Count);
        Assert.Equal(matches, processor.PrefillPreparedPlan is not null);
        using var session = processor.CreateSession();
        Assert.Equal(matches, backend.DualSessionCreated);
        Assert.Equal(2, session.ForwardToken(2).Span[0]);
    }

    [Theory]
    [InlineData("schema")]
    [InlineData("tensor")]
    [InlineData("output")]
    public void GenericPrefillOptimizer_RejectsIncompatibleGraphBeforePrefillPreparation(string mismatch)
    {
        var backend = new PrefillBackend();
        Assert.Throws<InvalidDataException>(() =>
            PrefillBuilder(backend, new TestPrefillOptimizer(true, mismatch)).Build());
        Assert.Single(backend.Prepared);
    }

    [Fact]
    public void GenericPrefillOptimizer_RejectsAmbiguousMatches()
    {
        var backend = new PrefillBackend();
        Assert.Throws<InvalidOperationException>(() =>
            PrefillBuilder(backend, new TestPrefillOptimizer(true), new TestPrefillOptimizer(true)).Build());
        Assert.Single(backend.Prepared);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphExports_WriteUtf8AtCurrentPositionAndLeaveStreamsOpen(bool rwkv7)
    {
        using var processor = LoadPortable(rwkv7);
        Assert.NotNull(processor.LogicalGraph);
        Assert.Same(processor.InferenceExecutionGraph, processor.PreparedPlan!.Graph);

        Check(processor.ExportLogicalGraph, GraphJson.Serialize(processor.LogicalGraph));
        Check(stream => processor.ExportExecutionGraph(stream),
            GraphJson.Serialize(processor.InferenceExecutionGraph!));
        Check(stream => processor.ExportExecutionGraph(stream, ProcessorExecutionGraphKind.Inference),
            GraphJson.Serialize(processor.InferenceExecutionGraph!));

        using var invalid = new MemoryStream();
        Assert.Throws<InvalidOperationException>(() =>
            processor.ExportExecutionGraph(invalid, ProcessorExecutionGraphKind.Prefill));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            processor.ExportExecutionGraph(invalid, (ProcessorExecutionGraphKind)999));
        processor.Dispose();
        Assert.Throws<ObjectDisposedException>(() => processor.ExportLogicalGraph(invalid));
        Assert.Throws<ObjectDisposedException>(() => processor.ExportExecutionGraph(invalid));

        static void Check(Action<Stream> export, string expected)
        {
            using var stream = new MemoryStream();
            stream.WriteByte(0x7F);
            export(stream);
            Assert.True(stream.CanWrite);
            Assert.Equal(0x7F, stream.GetBuffer()[0]);
            Assert.Equal(stream.Length, stream.Position);
            Assert.Equal(Encoding.UTF8.GetBytes(expected), stream.ToArray().AsSpan(1).ToArray());
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ProcessorLoad_AndSuppliedPortableGraphAgreeAcrossPrefillForkAndReset(bool rwkv7)
    {
        var model = rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6;
        var path = TestModelLoader.GetPath(model);
        using var automatic = Processor.Load(path);
        using var supplied = LoadPortable(rwkv7);
        using var first = automatic.CreateSession();
        using var second = supplied.CreateSession();
        Assert.Equal(rwkv7 ? "rwkv-7" : "rwkv-6", automatic.Metadata.ArchitectureId);
        Assert.Same(automatic.InferenceExecutionGraph, automatic.PreparedPlan!.Graph);
        Assert.Equal(first.Prefill([0, 1, 2]).ToArray(), second.Prefill([0, 1, 2]).ToArray());
        StateSnapshotAssertions.Equal(StateSnapshotAssertions.Capture(first),
            StateSnapshotAssertions.Capture(second));

        using var fork = second.Fork();
        Assert.Equal(first.ForwardToken(3).ToArray(), second.ForwardToken(3).ToArray());
        Assert.Equal(second.ForwardToken(4).ToArray(), fork.Prefill([3, 4]).ToArray());
        second.Reset();
        using var fresh = supplied.CreateSession();
        Assert.Equal(fresh.ForwardToken(0).ToArray(), second.ForwardToken(0).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void XmlExecutionSource_RoundTripsWithoutLogicalSourceAndRunsSameGraph(bool rwkv7)
    {
        var path = TestModelLoader.GetPath(rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6);
        using var baseline = LoadPortable(rwkv7);
        var original = baseline.InferenceExecutionGraph!;
        var serialized = GraphXml.Serialize(new ExecutionGraph(original.Identity, original.Model,
            original.Resources, original.Regions,
            original.Nodes.Select(node => node with { Source = ExecutionSourceMap.XmlOnly }),
            original.Inputs, original.Outputs, original.GraphState));
        using var restored = new ProcessorPipelineBuilder(path)
            .UseReader(new GgmlModelReader(), new GraphArchitectureMetadataReader(original))
            .UseExecutionXml(serialized)
            .UseBackend(CpuPrimitiveGraphBackend.Instance)
            .UsePortableGraphArchitecture()
            .Build();

        Assert.Null(restored.LogicalGraph);
        Assert.All(restored.InferenceExecutionGraph!.Nodes, node => Assert.Empty(node.Source.LogicalNodes));
        using var expected = baseline.CreateSession();
        using var actual = restored.CreateSession();
        foreach (var token in new[] { 0, 1 })
            Assert.Equal(expected.ForwardToken(token).ToArray(), actual.ForwardToken(token).ToArray());
        StateSnapshotAssertions.Equal(StateSnapshotAssertions.Capture(expected),
            StateSnapshotAssertions.Capture(actual));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphReader_RejectsIncompatibleWeightBeforeBackendPreparation(bool rwkv7)
    {
        var model = rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6;
        var path = TestModelLoader.GetPath(model);
        using var catalog = TestModelLoader.OpenCatalog(model);
        ILogicalGraphProvider provider = rwkv7
            ? new PortableRwkv7GraphProvider() : new PortableRwkv6GraphProvider();
        var graph = provider.Build(catalog);
        var head = graph.Resources.Single(resource =>
            resource.Kind == GraphResourceKind.Weight && resource.BindingKey == "head.weight");
        var corrupted = new LogicalGraph(graph.Identity, graph.Model,
            graph.Resources.Select(resource => resource.Id == head.Id
                ? resource with { BindingKey = "nonexistent.weight" }
                : resource), graph.Regions, graph.Nodes,
            graph.Inputs, graph.Outputs, graph.GraphState);

        Assert.Throws<InvalidDataException>(() =>
            Processor.LoadGraph(path, corrupted, CpuPrimitiveGraphBackend.Instance));
    }

    private static Processor LoadPortable(bool rwkv7) =>
        Processor.LoadGraph(TestModelLoader.GetPath(rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6),
            rwkv7 ? new PortableRwkv7GraphProvider() : new PortableRwkv6GraphProvider(),
            CpuPrimitiveGraphBackend.Instance);

    private static ProcessorPipelineBuilder PrefillBuilder(
        PrefillBackend backend, params IPrefillGraphOptimizer[] optimizers)
    {
        var logical = new LogicalGraphBuilder(
                new GraphIdentity("rwkv-6", 1, "test-prefill"),
                new GraphModelSignature(8, 4, 1, 2, 2, "rwkv-6.state.fp32@1"))
            .SetStateSchema(new StateSchema("Test_State"))
            .AddRegion("root", GraphRegionTypes.Graph, "Root")
            .AddResource("token", "Token", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true)
            .AddResource("logits", "Logits", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [8]), graphOutput: true)
            .AddResource("state", "State", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                new TensorDescriptor(GraphElementType.Float32, [8]))
            .AddStateSlot("memory", "state")
            .AddNode("copy", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "state"), GraphBindings.Write("output", "logits")])
            .Build();
        return new ProcessorPipelineBuilder("prefill-model.ggml")
            .UseReader(new PrefillReader(), new GraphArchitectureMetadataReader(logical))
            .UseProvider(new SuppliedLogicalGraphProvider(logical))
            .UsePrefillOptimizers(optimizers)
            .UseBackend(backend)
            .UsePortableGraphArchitecture();
    }

    private sealed class TestPrefillOptimizer(bool matches, string? mismatch = null) : IPrefillGraphOptimizer
    {
        public bool CanOptimize(GraphPrefillOptimizationContext context) => matches;

        public ExecutionGraph Optimize(GraphPrefillOptimizationContext context)
        {
            var graph = context.InferenceGraph;
            var resources = graph.Resources.Select(resource =>
                mismatch == "tensor" && resource.Id.Value == "state"
                    ? resource with { Tensor = new TensorDescriptor(GraphElementType.Float32, [4]) }
                    : mismatch == "output" && resource.Id.Value == "logits"
                        ? resource with { Tensor = new TensorDescriptor(GraphElementType.Float32, [4]) }
                        : resource);
            GraphState entries = mismatch == "schema"
                ? new GraphState(new StateSchema("Incompatible_State"), graph.GraphState.Entries)
                : graph.GraphState;
            return new ExecutionGraph(graph.Identity with { Name = "test-prefill-alternate" }, graph.Model,
                resources, graph.Regions, graph.Nodes, graph.Inputs, graph.Outputs, entries);
        }
    }

    private sealed class PrefillBackend : IProcessorPrefillBackend
    {
        public List<ExecutionGraph> Prepared { get; } = [];
        public bool DualSessionCreated { get; private set; }
        public IExecutionKernelCatalog KernelCatalog { get; } =
            new ExecutionKernelCatalog("test-prefill", [PrimitiveGraphOperations.Copy]);
        public IPrimitiveOperatorBackend PrimitiveOperators => CpuPrimitiveOperatorBackend.Instance;
        public IReadOnlyList<OperatorImplementationDescription> GetOperatorImplementations(
            ExecutionGraph graph, ExecutionNode node) => [];

        public BackendPreparationResult Prepare(ExecutionGraph graph)
        {
            Prepared.Add(graph);
            return new BackendPreparationResult.Success(new PrefillPlan(graph));
        }

        public IProcessorSessionExecutor CreateSessionExecutor(
            IRwkvModel model, IRwkvState state, IBackendExecutablePlan plan) => new PrefillExecutor();

        public IProcessorSessionExecutor CreateSessionExecutor(
            IRwkvModel model, IRwkvState state, IBackendExecutablePlan inferencePlan,
            IBackendExecutablePlan prefillPlan)
        {
            DualSessionCreated = true;
            return new PrefillExecutor();
        }
    }

    private sealed record PrefillPlan(ExecutionGraph Graph) : IBackendExecutablePlan;

    private sealed class PrefillExecutor : IProcessorSessionExecutor
    {
        public void ForwardToken(int token, Span<float> logits) => logits.Fill(token);
        public void Dispose() { }
    }

    private sealed class PrefillReader : IModelReader
    {
        public IModelFile Open(string path) => new PrefillFile(path);
    }

    private sealed class PrefillFile(string path) : IModelFile
    {
        public int VocabularySize => 8;
        public int EmbeddingSize => 4;
        public int LayerCount => 1;
        public string Path => path;
        public IReadOnlyCollection<string> Names => [];
        public bool TryGet(string name, out IModelTensor tensor)
        {
            tensor = null!;
            return false;
        }
        public IModelTensor GetRequired(string name) => throw new KeyNotFoundException(name);
        public void Dispose() { }
    }
}
