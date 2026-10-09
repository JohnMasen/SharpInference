using SharpInference.Backends.Cpu;
using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class CpuPrimitiveGraphExecutorTests
{
    private static readonly GraphModelSignature Model = TestGraphSignatures.Create(3, 2, 1, 1, 2, "test.state");

    [Fact]
    public void LogicalGraph_GathersProjectsAddsStateAndReturnsIndependentSnapshots()
    {
        var graph = new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "primitive"), Model)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("index", "index", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true)
            .AddResource("embedding", "embedding", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float32, [3, 2]), bindingKey: "lookup")
            .AddResource("projection", "projection", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float32, [2, 2]), bindingKey: "linear")
            .AddResource("row", "row", GraphResourceKind.TokenTransient, GraphResourceLifetime.Token,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("projected", "projected", GraphResourceKind.TokenTransient, GraphResourceLifetime.Token,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("memory", "memory", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddStateSlot("memory.slot", "memory")
            .AddNode("gather", PrimitiveGraphOperations.GatherRow, "root",
                [GraphBindings.Read("table", "embedding"), GraphBindings.Read("index", "index"),
                    GraphBindings.Write("output", "row")])
            .AddNode("project", PrimitiveGraphOperations.MatVec, "root",
                [GraphBindings.Read("matrix", "projection"), GraphBindings.Read("input", "row"),
                    GraphBindings.Write("output", "projected")], ["gather"])
            .AddNode("add", PrimitiveGraphOperations.Add, "root",
                [GraphBindings.Read("left", "projected"), GraphBindings.Read("right", "memory"),
                    GraphBindings.Write("output", "result")], ["project"])
            .AddNode("update", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "result"), GraphBindings.Write("output", "memory")], ["add"])
            .Build();
        var matrix = new float[] { 2, 0, 0, 3 };
        var catalog = new Catalog(new Tensor("lookup", [3, 2], new float[] { 1, 2, 3, 4, 5, 6 }),
            new Tensor("linear", [2, 2], matrix));
        var executor = new CpuPrimitiveGraphExecutor(graph, catalog);
        matrix[0] = 100;
        var portable = new PortableGraphModel(
            new RwkvModelMetadata(3, 2, 1, 1, 2, "synthetic"), catalog,
            new GraphOptimizer().Optimize(graph));
        var borrowed = new CpuPrimitiveGraphExecutor(portable);
        Assert.Equal([310f, 32f], Assert.IsType<float[]>(borrowed.Execute(
            new Dictionary<ResourceId, Array> { [new("index")] = new[] { 1 } },
            [new GraphStateValue("memory.slot", [2], [10, 20])]).Outputs[new("result")]));
        Assert.Equal([16f, 32f], Assert.IsType<float[]>(executor.Execute(
            new Dictionary<ResourceId, Array> { [new("index")] = new[] { 1 } },
            [new GraphStateValue("memory.slot", [2], [10, 20])]).Outputs[new("result")]));
        var initial = new[] { 10f, 20f };
        var state = new[] { new GraphStateValue("memory.slot", [2], initial) };
        var first = executor.Execute(new Dictionary<ResourceId, Array> { [new("index")] = new[] { 1 } }, state);
        Assert.Equal([16f, 32f], Assert.IsType<float[]>(first.Outputs[new("result")]));
        Assert.Equal([16f, 32f], first.State[0].Values);
        Assert.Equal([10f, 20f], initial);
        var second = executor.Execute(new Dictionary<ResourceId, Array> { [new("index")] = new[] { 2 } },
            first.State);
        Assert.Equal([26f, 50f], Assert.IsType<float[]>(second.Outputs[new("result")]));
        Assert.Equal([16f, 32f], first.State[0].Values);
        Assert.Equal([16f, 32f], Assert.IsType<float[]>(first.Outputs[new("result")]));

        Assert.Throws<InvalidDataException>(() => executor.Execute(
            new Dictionary<ResourceId, Array> { [new("index")] = new[] { 0 } },
            [new GraphStateValue("wrong", [2], [0, 0])]));
        Assert.Throws<InvalidDataException>(() => executor.Execute(
            new Dictionary<ResourceId, Array> { [new("index")] = new float[] { 0 } }, state));
        Assert.Throws<ArgumentOutOfRangeException>(() => executor.Execute(
            new Dictionary<ResourceId, Array> { [new("index")] = new[] { 3 } }, state));
    }

    [Theory]
    [InlineData("core.copy", 2f, 4f)]
    [InlineData("core.exp", 7.389056f, 54.59815f)]
    [InlineData("core.tanh", 0.9640276f, 0.9993293f)]
    [InlineData("core.sigmoid", 0.880797f, 0.9820138f)]
    [InlineData("core.rsqrt", 0.70710677f, 0.5f)]
    [InlineData("core.square", 4f, 16f)]
    [InlineData("core.relu", 2f, 4f)]
    [InlineData("core.reduce-sum", 6f, 6f)]
    [InlineData("core.reduce-mean", 3f, 3f)]
    [InlineData("core.add", 5f, 9f)]
    [InlineData("core.subtract", -1f, -1f)]
    [InlineData("core.multiply", 6f, 20f)]
    [InlineData("core.divide", 0.6666667f, 0.8f)]
    [InlineData("core.maximum", 3f, 5f)]
    public void DispatchesStandardElementwiseAndReductionOperations(string name, float first, float second)
    {
        var operation = new GraphOperationId(name);
        var binary = name is "core.add" or "core.subtract" or "core.multiply" or "core.divide" or "core.maximum";
        var reduction = name is "core.reduce-sum" or "core.reduce-mean";
        var builder = new LogicalGraphBuilder(new GraphIdentity("independent", 1, name), Model)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("x", "x", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("y", "y", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, reduction ? [1] : [2]), graphOutput: true);
        if (binary)
            builder.AddResource("rhs", "rhs", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true);
        builder.AddNode("operation", operation, "root",
            binary
                ? [GraphBindings.Read("left", "x"), GraphBindings.Read("right", "rhs"), GraphBindings.Write("output", "y")]
                : [GraphBindings.Read("input", "x"), GraphBindings.Write("output", "y")]);
        var executor = new CpuPrimitiveGraphExecutor(builder.Build(), new Catalog());
        var input = new Dictionary<ResourceId, Array> { [new("x")] = new float[] { 2, 4 } };
        if (binary) input[new("rhs")] = new float[] { 3, 5 };
        var actual = Assert.IsType<float[]>(executor.Execute(input).Outputs[new("y")]);
        Assert.Equal(first, actual[0], 0.0001f);
        if (!reduction) Assert.Equal(second, actual[1], 0.0001f);
    }

    [Fact]
    public void ExecutionGraph_UsesDependencyOrderAndFp16Weights()
    {
        var logical = new LogicalGraphBuilder(new GraphIdentity("unrelated", 1, "half"), Model)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("input", "input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float16, [2]), graphInput: true)
            .AddResource("weight", "weight", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float16, [2, 2]), bindingKey: "half.weight")
            .AddResource("intermediate", "intermediate", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float16, [2]))
            .AddResource("output", "output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float16, [2]), graphOutput: true)
            .AddNode("matvec", PrimitiveGraphOperations.MatVec, "root",
                [GraphBindings.Read("matrix", "weight"), GraphBindings.Read("input", "input"),
                    GraphBindings.Write("output", "intermediate")])
            .AddNode("square", PrimitiveGraphOperations.Square, "root",
                [GraphBindings.Read("input", "intermediate"), GraphBindings.Write("output", "output")],
                ["matvec"])
            .Build();
        var execution = new GraphOptimizer().Optimize(logical);
        var reversed = new ExecutionGraph(execution.Identity, execution.Model, execution.Resources,
            execution.Regions, execution.Nodes.Reverse(), execution.Inputs, execution.Outputs);
        var executor = new CpuPrimitiveGraphExecutor(reversed,
            new Catalog(new Tensor("half.weight", [2, 2], [(Half)1, (Half)2, (Half)3, (Half)4])));
        var result = executor.Execute(new Dictionary<ResourceId, Array>
            { [new("input")] = new Half[] { (Half)2, (Half)3 } });
        Assert.Equal(new Half[] { (Half)64, (Half)324 },
            Assert.IsType<Half[]>(result.Outputs[new("output")]));
        Assert.Empty(result.State);
    }

    [Fact]
    public void RejectsUnknownOperationsAndMismatchedWeightsBeforeExecution()
    {
        var graph = new LogicalGraphBuilder(new GraphIdentity("test", 1, "reject"), Model)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("input", "input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("weight", "weight", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float32, [2]), bindingKey: "weight")
            .AddResource("output", "output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddNode("add", PrimitiveGraphOperations.Add, "root",
                [GraphBindings.Read("left", "input"), GraphBindings.Read("right", "weight"),
                    GraphBindings.Write("output", "output")])
            .Build();
        Assert.Throws<InvalidDataException>(() => new CpuPrimitiveGraphExecutor(graph,
            new Catalog(new Tensor("weight", [1, 2], [1f, 2f]))));
        var fused = new ExecutionGraph(graph.Identity, graph.Model, graph.Resources, graph.Regions,
            [new ExecutionNode(new ExecutionNodeId("fused"), new GraphOperationId("custom.fused"),
                new RegionId("root"), graph.Nodes[0].Resources, [],
                graph.Nodes[0].Attributes, graph.Nodes[0].Requirements, ExecutionSourceMap.XmlOnly)],
            graph.Inputs, graph.Outputs);
        Assert.Throws<NotSupportedException>(() => new CpuPrimitiveGraphExecutor(fused,
            new Catalog(new Tensor("weight", [2], [1f, 2f]))));
    }

    [Fact]
    public void PortableModelAndState_ImplementRuntimeContractsWithoutArchitectureForward()
    {
        var logical = new LogicalGraphBuilder(new GraphIdentity("portable-test", 1, "state"), Model)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("input", "input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("accumulator", "accumulator", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("output", "output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddStateSlot("memory", "accumulator")
            .AddNode("add", PrimitiveGraphOperations.Add, "root",
                [GraphBindings.Read("left", "input"), GraphBindings.Read("right", "accumulator"),
                    GraphBindings.Write("output", "output")])
            .AddNode("update", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "output"), GraphBindings.Write("output", "accumulator")],
                ["add"])
            .Build();
        var graph = new GraphOptimizer().Optimize(logical);
        IModel runtimeModel = new PortableGraphModel(
            new RwkvModelMetadata(3, 2, 1, 1, 2, "portable-test"), new Catalog(), graph);
        var model = Assert.IsType<PortableGraphModel>(runtimeModel);
        var executor = new CpuPrimitiveGraphExecutor(model);
        Assert.Same(graph, executor.Graph);
        IModelState runtimeState = new PortableGraphState(graph);
        var state = Assert.IsType<PortableGraphState>(runtimeState);
        IReadOnlyList<GraphStateValue> Read(PortableGraphState current) =>
            GraphSessionStateAccess.Read(graph.GraphState, graph.Resources, current);
        Assert.Equal("portable-test", state.ArchitectureId);
        Assert.Equal(2, state.ElementCount);
        Assert.Equal("memory", Assert.Single(((INamedFloat32ModelState)state).Views).Name);
        state.Restore([4, 5]);
        Assert.Equal([4f, 5f], Assert.Single(Read(state)).Values);
        var inputs = new Dictionary<ResourceId, Array> { [new("input")] = new float[] { 2, 3 } };
        var first = executor.Execute(inputs, state);
        Assert.Equal([6f, 8f], Assert.IsType<float[]>(first.Outputs[new("output")]));
        Assert.Equal([6f, 8f], Read(state)[0].Values);
        state.Views[0].Values[0] = 7f;
        state.CommitViews();
        var clone = Assert.IsType<PortableGraphState>(state.Clone());
        Assert.Equal([9f, 11f], Assert.IsType<float[]>(executor.Execute(inputs, clone).Outputs[new("output")]));
        Assert.Equal([7f, 8f], Read(state)[0].Values);
        state.Reset();
        Assert.Equal([0f, 0f], Read(state)[0].Values);
        GraphSessionStateAccess.Write(model.Graph.GraphState, model.Graph.Resources, state,
            [new GraphStateValue("memory", [2], [1, 2])]);
        Assert.Equal([1f, 2f], Read(state)[0].Values);
        state.Reset();
        Assert.Throws<ArgumentException>(() => state.Restore([1]));
        Assert.Throws<InvalidDataException>(() => GraphSessionStateAccess.Write(
            model.Graph.GraphState, model.Graph.Resources, state,
            [new GraphStateValue("wrong", [2], [1, 2])]));
        Assert.Equal([0f, 0f], Read(state)[0].Values);
    }

    [Theory]
    [InlineData(new int[] { 1 }, "incompatible elementwise dimensions")]
    [InlineData(new int[] { 1, 2 }, "incompatible elementwise dimensions")]
    public void BinaryOperationsRejectBroadcastAndReshapeWithNodeDiagnostic(
        int[] rightShape, string diagnostic)
    {
        var graph = new LogicalGraphBuilder(new GraphIdentity("test", 1, "shape"), Model)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("left", "left", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("right", "right", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, rightShape), graphInput: true)
            .AddResource("output", "output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddNode("not-broadcastable", PrimitiveGraphOperations.Add, "root",
                [GraphBindings.Read("left", "left"), GraphBindings.Read("right", "right"),
                    GraphBindings.Write("output", "output")])
            .Build();
        var error = Assert.Throws<InvalidDataException>(() =>
            new CpuPrimitiveGraphExecutor(graph, new Catalog()));
        Assert.Contains(diagnostic, error.Message, StringComparison.Ordinal);
        Assert.Contains("not-broadcastable", error.Message, StringComparison.Ordinal);
        var execution = new GraphOptimizer().Optimize(graph);
        var failure = Assert.IsType<BackendPreparationResult.Failure>(
            CpuPrimitiveGraphBackend.Instance.Prepare(execution));
        Assert.Equal(BackendPreparationFailureReason.ShapeNotSupported,
            Assert.Single(failure.Diagnostics).Reason);
        Assert.Contains("broadcasting is not supported",
            failure.Diagnostics[0].Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MatVecAndUnsupportedNormalizationFailBeforeRunning()
    {
        LogicalGraph Graph(GraphOperationId operation) =>
            new LogicalGraphBuilder(new GraphIdentity("test", 1, "invalid"), Model)
                .AddRegion("root", GraphRegionTypes.Graph, "root")
                .AddResource("matrix", "matrix", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                    new TensorDescriptor(GraphElementType.Float32, [2, 3]), bindingKey: "matrix")
                .AddResource("input", "input", GraphResourceKind.Input, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
                .AddResource("output", "output", GraphResourceKind.Output, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
                .AddNode("projection", operation, "root",
                    [GraphBindings.Read("matrix", "matrix"), GraphBindings.Read("input", "input"),
                        GraphBindings.Write("output", "output")])
                .Build();
        var catalog = new Catalog(new Tensor("matrix", [2, 3], new float[6]));
        var mismatch = Assert.Throws<InvalidDataException>(() =>
            new CpuPrimitiveGraphExecutor(Graph(PrimitiveGraphOperations.MatVec), catalog));
        Assert.Contains("projection", mismatch.Message, StringComparison.Ordinal);
        Assert.Contains("matvec dimensions", mismatch.Message, StringComparison.Ordinal);
        var unsupported = Assert.Throws<NotSupportedException>(() =>
            new CpuPrimitiveGraphExecutor(Graph(new GraphOperationId("core.layer-normalize")), catalog));
        Assert.Contains("core.layer-normalize", unsupported.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Backend_UsesPortableModelAndStateForProcessorTokenSession()
    {
        var signature = TestGraphSignatures.Create(2, 2, 1, 1, 2, "portable.state");
        var logical = new LogicalGraphBuilder(new GraphIdentity("portable", 1, "token"), signature)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("token", "token", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true)
            .AddResource("table", "table", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float32, [2, 2]), bindingKey: "table")
            .AddResource("row", "row", GraphResourceKind.TokenTransient, GraphResourceLifetime.Token,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("memory", "memory", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("logits", "logits", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddStateSlot("memory", "memory")
            .AddNode("lookup", PrimitiveGraphOperations.GatherRow, "root",
                [GraphBindings.Read("table", "table"), GraphBindings.Read("index", "token"),
                    GraphBindings.Write("output", "row")])
            .AddNode("accumulate", PrimitiveGraphOperations.Add, "root",
                [GraphBindings.Read("left", "row"), GraphBindings.Read("right", "memory"),
                    GraphBindings.Write("output", "logits")], ["lookup"])
            .AddNode("store", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "logits"), GraphBindings.Write("output", "memory")],
                ["accumulate"])
            .Build();
        var graph = new GraphOptimizer().Optimize(logical);
        var table = new float[] { 1, 2, 3, 4 };
        var model = new PortableGraphModel(
            new RwkvModelMetadata(2, 2, 1, 1, 2, "portable"),
            new Catalog(new Tensor("table", [2, 2], table)), graph);
        IGraphModelWeightBackend backend = CpuPrimitiveGraphBackend.Instance;
        Assert.True(backend.RequiresCpuWeightCopy);
        var plan = backend.Prepare(graph).GetPlanOrThrow("cpu");
        backend.PrepareModelWeights(model);
        var state = new PortableGraphState(graph);
        state.Restore([10, 20]);
        using var session = backend.CreateSessionExecutor(model, state, plan);
        var logits = new float[2];
        session.ForwardToken(1, logits);
        Assert.Equal([13f, 24f], logits);
        table[2] = 5;
        session.ForwardToken(1, logits);
        Assert.Equal([18f, 28f], logits);
        Assert.Equal([18f, 28f], Assert.Single(GraphSessionStateAccess.Read(
            graph.GraphState, graph.Resources, state)).Values);
        Assert.Throws<ArgumentException>(() => session.ForwardToken(0, new float[1]));
        Assert.Same(graph, Assert.IsType<CpuPrimitiveGraphPlan>(plan).Graph);
        session.Dispose();
        Assert.Throws<ObjectDisposedException>(() => session.ForwardToken(0, logits));
    }

    private sealed class Tensor : IModelTensor
    {
        private readonly float[] floats;
        private readonly Half[] halves;
        public Tensor(string name, int[] dimensions, float[] values)
        {
            Name = name;
            Dimensions = dimensions;
            floats = values;
            halves = [];
            DataType = TensorDataType.Float32;
        }
        public Tensor(string name, int[] dimensions, Half[] values)
        {
            Name = name;
            Dimensions = dimensions;
            halves = values;
            floats = [];
            DataType = TensorDataType.Float16;
        }
        public string Name { get; }
        public IReadOnlyList<int> Dimensions { get; }
        public TensorDataType DataType { get; }
        public ReadOnlySpan<float> FloatValues => floats;
        public ReadOnlySpan<Half> HalfValues => halves;
    }

    private sealed class Catalog(params Tensor[] tensors) : IModelTensorCatalog
    {
        private readonly Dictionary<string, IModelTensor> values = tensors.ToDictionary(tensor => tensor.Name,
            tensor => (IModelTensor)tensor);
        public int VocabularySize => 3;
        public int EmbeddingSize => 2;
        public int LayerCount => 1;
        public IReadOnlyCollection<string> Names => values.Keys;
        public bool TryGet(string name, out IModelTensor tensor) => values.TryGetValue(name, out tensor!);
        public IModelTensor GetRequired(string name) => values[name];
    }
}
