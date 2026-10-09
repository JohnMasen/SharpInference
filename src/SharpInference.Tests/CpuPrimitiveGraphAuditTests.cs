using SharpInference.Backends.Cpu;
using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class CpuPrimitiveGraphAuditTests
{
    private static readonly GraphModelSignature Model = TestGraphSignatures.Create(2, 2, 1, 1, 2, "audit.state");

    [Theory]
    [InlineData("core.copy", 2f, 4f)]
    [InlineData("core.add", 5f, 9f)]
    [InlineData("core.subtract", -1f, -1f)]
    [InlineData("core.multiply", 6f, 20f)]
    [InlineData("core.divide", 0.6666667f, 0.8f)]
    [InlineData("core.maximum", 3f, 5f)]
    [InlineData("core.exp", 7.389056f, 54.59815f)]
    [InlineData("core.tanh", 0.9640276f, 0.9993293f)]
    [InlineData("core.sigmoid", 0.880797f, 0.9820138f)]
    [InlineData("core.rsqrt", 0.70710677f, 0.5f)]
    [InlineData("core.square", 4f, 16f)]
    [InlineData("core.relu", 2f, 4f)]
    [InlineData("core.reduce-sum", 6f, 6f)]
    [InlineData("core.reduce-mean", 3f, 3f)]
    public void EveryHalfElementwiseAndReductionPrimitiveExecutesItsAdvertisedSignature(
        string name, float first, float second)
    {
        var operation = new GraphOperationId(name);
        var binary = operation is var op && (op == PrimitiveGraphOperations.Add ||
            op == PrimitiveGraphOperations.Subtract || op == PrimitiveGraphOperations.Multiply ||
            op == PrimitiveGraphOperations.Divide || op == PrimitiveGraphOperations.Maximum);
        var reduction = operation == PrimitiveGraphOperations.ReduceSum ||
            operation == PrimitiveGraphOperations.ReduceMean;
        var builder = new LogicalGraphBuilder(new GraphIdentity("audit", 1, name), Model)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("x", "x", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float16, [2]), graphInput: true)
            .AddResource("y", "y", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float16, reduction ? [1] : [2]), graphOutput: true);
        if (binary)
            builder.AddResource("rhs", "rhs", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float16, [2]), graphInput: true);
        builder.AddNode("operation", operation, "root",
            binary
                ? [GraphBindings.Read("left", "x"), GraphBindings.Read("right", "rhs"),
                    GraphBindings.Write("output", "y")]
                : [GraphBindings.Read("input", "x"), GraphBindings.Write("output", "y")]);
        var graph = new GraphOptimizer().Optimize(builder.Build());
        var backend = CpuPrimitiveGraphBackend.Instance;
        Assert.IsType<CpuPrimitiveGraphPlan>(backend.Prepare(graph).GetPlanOrThrow("cpu"));
        Assert.Contains(backend.GetOperatorImplementations(graph, graph.Nodes[0]),
            description => description.Signature.OutputTypes.SequenceEqual([GraphElementType.Float16]));
        var inputs = new Dictionary<ResourceId, Array>
            { [new("x")] = new Half[] { (Half)2, (Half)4 } };
        if (binary) inputs[new("rhs")] = new Half[] { (Half)3, (Half)5 };
        var actual = Assert.IsType<Half[]>(new CpuPrimitiveGraphExecutor(graph, new EmptyCatalog())
            .Execute(inputs).Outputs[new("y")]);
        Assert.InRange((float)actual[0], first - 0.04f, first + 0.04f);
        if (!reduction) Assert.InRange((float)actual[1], second - 0.04f, second + 0.04f);
    }

    [Fact]
    public void StandardKernelCatalogListsExactlyAllPrimitiveOperations()
    {
        var actual = CpuPrimitiveGraphBackend.Instance.KernelCatalog.SupportedOperations
            .Where(operation => !operation.Name.StartsWith("core.tensor.", StringComparison.Ordinal) &&
                operation != FusedElementwiseExpressionContract.Operation)
            .ToHashSet();
        var expected = PrimitiveGraphOperations.CreateStandardDescriptions()
            .Select(description => description.Operation).ToHashSet();
        Assert.Equal(16, expected.Count);
        Assert.True(expected.SetEquals(actual));
    }

    [Fact]
    public void HalfGatherFeedsHalfMatVecThroughGraphResources()
    {
        var logical = new LogicalGraphBuilder(new GraphIdentity("audit", 1, "half-projection"), Model)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("index", "index", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true)
            .AddResource("table", "table", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float16, [2, 2]), bindingKey: "table")
            .AddResource("matrix", "matrix", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float16, [2, 2]), bindingKey: "matrix")
            .AddResource("row", "row", GraphResourceKind.TokenTransient, GraphResourceLifetime.Token,
                new TensorDescriptor(GraphElementType.Float16, [2]))
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float16, [2]), graphOutput: true)
            .AddNode("gather", PrimitiveGraphOperations.GatherRow, "root",
                [GraphBindings.Read("table", "table"), GraphBindings.Read("index", "index"),
                    GraphBindings.Write("output", "row")])
            .AddNode("matvec", PrimitiveGraphOperations.MatVec, "root",
                [GraphBindings.Read("matrix", "matrix"), GraphBindings.Read("input", "row"),
                    GraphBindings.Write("output", "result")], ["gather"])
            .Build();
        var graph = new GraphOptimizer().Optimize(logical);
        var backend = CpuPrimitiveGraphBackend.Instance;
        backend.Prepare(graph).GetPlanOrThrow("cpu");
        var catalog = new HalfCatalog(
            ("table", new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 }),
            ("matrix", new Half[] { (Half)2, (Half)0, (Half)0, (Half)3 }));
        var result = new CpuPrimitiveGraphExecutor(graph, catalog).Execute(
            new Dictionary<ResourceId, Array> { [new("index")] = new[] { 1 } });
        Assert.Equal(new Half[] { (Half)6, (Half)12 },
            Assert.IsType<Half[]>(result.Outputs[new("result")]));
    }

    [Fact]
    public void GatherIndexAndReductionOutputMustBeRankOneSingletons()
    {
        var reduction = Graph(PrimitiveGraphOperations.ReduceSum, [2], [1, 1]);
        var diagnostic = Assert.Single(Assert.IsType<BackendPreparationResult.Failure>(
            CpuPrimitiveGraphBackend.Instance.Prepare(reduction)).Diagnostics);
        Assert.Contains("scalar output [1]", diagnostic.Message, StringComparison.Ordinal);

        var logical = new LogicalGraphBuilder(new GraphIdentity("audit", 1, "gather"), Model)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("table", "table", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float32, [2, 2]), bindingKey: "table")
            .AddResource("index", "index", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1, 1]), graphInput: true)
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddNode("gather", PrimitiveGraphOperations.GatherRow, "root",
                [GraphBindings.Read("table", "table"), GraphBindings.Read("index", "index"),
                    GraphBindings.Write("output", "result")])
            .Build();
        var failure = Assert.IsType<BackendPreparationResult.Failure>(
            CpuPrimitiveGraphBackend.Instance.Prepare(new GraphOptimizer().Optimize(logical)));
        Assert.Contains("invalid gather dimensions", failure.Diagnostics[0].Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RejectsUnknownVersionAndAttributesInsteadOfSilentlyDispatching()
    {
        var unknown = Graph(new GraphOperationId(PrimitiveGraphOperations.Tanh.Name, 2), [2], [2]);
        var failure = Assert.IsType<BackendPreparationResult.Failure>(
            CpuPrimitiveGraphBackend.Instance.Prepare(unknown));
        Assert.Equal(BackendPreparationFailureReason.UnsupportedOperation,
            Assert.Single(failure.Diagnostics).Reason);
        var graph = Graph(PrimitiveGraphOperations.Tanh, [2], [2]);
        var node = graph.Nodes[0] with
        {
            Attributes = new Dictionary<string, string> { ["unrecognized"] = "ignored" },
        };
        var attributed = new ExecutionGraph(graph.Identity, graph.Model, graph.Resources, graph.Regions,
            [node], graph.Inputs, graph.Outputs);
        failure = Assert.IsType<BackendPreparationResult.Failure>(
            CpuPrimitiveGraphBackend.Instance.Prepare(attributed));
        Assert.Contains("unsupported attributes", failure.Diagnostics[0].Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SharedWorkspaceKeepsParallelSessionStateAndResultsIsolated()
    {
        var graph = new GraphOptimizer().Optimize(
            new LogicalGraphBuilder(new GraphIdentity("audit", 1, "sessions"), Model)
                .AddRegion("root", GraphRegionTypes.Graph, "root")
                .AddResource("input", "input", GraphResourceKind.Input, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
                .AddResource("state", "state", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                    new TensorDescriptor(GraphElementType.Float32, [2]))
                .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
                .AddStateSlot("memory", "state")
                .AddNode("add", PrimitiveGraphOperations.Add, "root",
                    [GraphBindings.Read("left", "state"), GraphBindings.Read("right", "input"),
                        GraphBindings.Write("output", "result")])
                .AddNode("store", PrimitiveGraphOperations.Copy, "root",
                    [GraphBindings.Read("input", "result"), GraphBindings.Write("output", "state")],
                    ["add"])
                .Build());
        var executor = new CpuPrimitiveGraphExecutor(graph, new EmptyCatalog());
        Parallel.For(0, 12, _ =>
        {
            var state = new PortableGraphState(graph);
            var input = new Dictionary<ResourceId, Array> { [new("input")] = new float[] { 1, 2 } };
            var first = executor.Execute(input, state);
            for (var step = 1; step < 10; step++)
                executor.Execute(input, state);
            Assert.Equal([1f, 2f], Assert.IsType<float[]>(first.Outputs[new("result")]));
            Assert.Equal([10f, 20f], Assert.Single(GraphSessionStateAccess.Read(
                graph.GraphState, graph.Resources, state)).Values);
        });
    }

    [Fact]
    public void WarmGraphDoesNotAllocateOneBufferPerNodePerToken()
    {
        var builder = new LogicalGraphBuilder(new GraphIdentity("audit", 1, "reuse"), Model)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("input", "input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [4]), graphInput: true)
            .AddResource("output", "output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [4]), graphOutput: true);
        var previous = "input";
        for (var index = 0; index < 128; index++)
        {
            var next = index == 127 ? "output" : $"scratch.{index}";
            if (index != 127)
                builder.AddResource(next, next, GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                    new TensorDescriptor(GraphElementType.Float32, [4]));
            builder.AddNode($"copy.{index}", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", previous), GraphBindings.Write("output", next)],
                index == 0 ? null : [$"copy.{index - 1}"]);
            previous = next;
        }
        var executor = new CpuPrimitiveGraphExecutor(builder.Build(), new EmptyCatalog());
        var inputs = new Dictionary<ResourceId, Array> { [new("input")] = new float[] { 1, 2, 3, 4 } };
        for (var index = 0; index < 8; index++) executor.Execute(inputs);
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var index = 0; index < 32; index++) executor.Execute(inputs);
        var bytesPerToken = (GC.GetAllocatedBytesForCurrentThread() - before) / 32;
        Assert.True(bytesPerToken < 8_000,
            $"Warm 128-node graph allocated {bytesPerToken} bytes per token.");
    }

    private static ExecutionGraph Graph(GraphOperationId operation, int[] input, int[] output)
    {
        var logical = new LogicalGraphBuilder(new GraphIdentity("audit", 1, "primitive"), Model)
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("input", "input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, input), graphInput: true)
            .AddResource("output", "output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, output), graphOutput: true)
            .AddNode("operation", operation, "root",
                [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "output")])
            .Build();
        return new GraphOptimizer().Optimize(logical);
    }

    private sealed class EmptyCatalog : IModelTensorCatalog
    {
        public int VocabularySize => 2;
        public int EmbeddingSize => 2;
        public int LayerCount => 1;
        public IReadOnlyCollection<string> Names => [];
        public bool TryGet(string name, out IModelTensor tensor)
        {
            tensor = null!;
            return false;
        }
        public IModelTensor GetRequired(string name) => throw new InvalidDataException(name);
    }

    private sealed class HalfCatalog(params (string Name, Half[] Values)[] tensors) : IModelTensorCatalog
    {
        private readonly IReadOnlyDictionary<string, IModelTensor> values = tensors
            .ToDictionary(item => item.Name,
                item => (IModelTensor)new HalfTensor(item.Name, item.Values));
        public int VocabularySize => 2;
        public int EmbeddingSize => 2;
        public int LayerCount => 1;
        public IReadOnlyCollection<string> Names => values.Keys.ToArray();
        public bool TryGet(string name, out IModelTensor tensor) => values.TryGetValue(name, out tensor!);
        public IModelTensor GetRequired(string name) => values[name];

        private sealed class HalfTensor(string name, Half[] values) : IModelTensor
        {
            public string Name => name;
            public TensorDataType DataType => TensorDataType.Float16;
            public IReadOnlyList<int> Dimensions => [2, 2];
            public ReadOnlySpan<float> FloatValues => [];
            public ReadOnlySpan<Half> HalfValues => values;
        }
    }
}
