using SharpInference.Backends.Cpu;
using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class CpuPortableTensorOperationsTests
{
    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(2, 3, 7)]
    [InlineData(2, 4, 257)]
    public void HeadOuterMatchesScalarReference(int heads, int rows, int columns)
    {
        var left = Enumerable.Range(0, heads * rows).Select(i => (i - 3) / 8f).ToArray();
        var right = Enumerable.Range(0, heads * columns).Select(i => (i % 19 - 9) / 16f).ToArray();
        var expected = new float[heads * rows * columns];
        for (var head = 0; head < heads; head++)
            for (var row = 0; row < rows; row++)
                for (var column = 0; column < columns; column++)
                    expected[(head * rows + row) * columns + column] =
                        left[head * rows + row] * right[head * columns + column];

        Assert.Equal(expected, Run(PortableTensorOperationContracts.HeadOuter,
            [heads, rows, columns], null,
            ("left", new[] { heads, rows }, left), ("right", new[] { heads, columns }, right)));
    }

    [Fact]
    public void BroadcastHandlesMultipleSingletonAxesAndRankExpansion()
    {
        var source = Enumerable.Range(1, 6).Select(i => (float)i).ToArray();
        var expected = new float[2 * 3 * 4 * 2];
        for (var outer = 0; outer < 2; outer++)
            for (var row = 0; row < 3; row++)
                for (var repeat = 0; repeat < 4; repeat++)
                    for (var column = 0; column < 2; column++)
                        expected[((outer * 3 + row) * 4 + repeat) * 2 + column] =
                            source[row * 2 + column];
        Assert.Equal(expected, Run(PortableTensorOperationContracts.Broadcast, [2, 3, 4, 2], null,
            ("input", new[] { 3, 1, 2 }, source)));
        Assert.Equal(Enumerable.Repeat(3f, 257),
            Run(PortableTensorOperationContracts.Broadcast, [257], null,
                ("input", new[] { 1 }, new[] { 3f })));
    }

    [Fact]
    public void ExecutesFillCastReshapeSliceAndBroadcast()
    {
        Assert.Equal([2.5f, 2.5f, 2.5f, 2.5f],
            Run(PortableTensorOperationContracts.Fill, [2, 2],
                new TensorFillValue(2.5f).ToAttributes()));
        Assert.Equal([1f, -2f, 3f, 4f],
            Run(PortableTensorOperationContracts.CastFp16ToFp32, [2, 2], null,
                ("input", new[] { 2, 2 }, new Half[] { (Half)1, (Half)(-2), (Half)3, (Half)4 })));
        Assert.Equal([1f, 2f, 3f, 4f],
            Run(PortableTensorOperationContracts.Reshape, [2, 2], null,
                ("input", new[] { 4 }, new float[] { 1, 2, 3, 4 })));
        Assert.Equal([2f, 3f, 4f, 5f, 8f, 9f, 10f, 11f],
            Run(PortableTensorOperationContracts.Slice, [2, 2, 2],
                new TensorSlice(1, 1, 2).ToAttributes(),
                ("input", new[] { 2, 3, 2 }, Enumerable.Range(0, 12).Select(i => (float)i).ToArray())));
        Assert.Equal([1f, 1f, 1f, 2f, 2f, 2f],
            Run(PortableTensorOperationContracts.Broadcast, [2, 3], null,
                ("input", new[] { 2, 1 }, new float[] { 1, 2 })));
        Assert.Equal([3f, 4f, 3f, 4f, 3f, 4f, 3f, 4f, 3f, 4f, 3f, 4f],
            Run(PortableTensorOperationContracts.Broadcast, [2, 3, 2], null,
                ("input", new[] { 2 }, new float[] { 3, 4 })));
    }

    [Fact]
    public void ExecutesBatchedMatVecLastAxisReductionsAndHeadOuter()
    {
        Assert.Equal([8f, 18f, 17f, 23f],
            Run(PortableTensorOperationContracts.BatchedMatVec, [2, 2], null,
                ("matrix", new[] { 2, 2, 2 }, new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }),
                ("vector", new[] { 2, 2 }, new float[] { 2, 3, 1, 2 })));
        Assert.Equal([3f, 7f, 11f, 15f],
            Run(PortableTensorOperationContracts.ReduceLastSum, [2, 2], null,
                ("input", new[] { 2, 2, 2 }, Enumerable.Range(1, 8).Select(i => (float)i).ToArray())));
        Assert.Equal([1.5f, 3.5f, 5.5f, 7.5f],
            Run(PortableTensorOperationContracts.ReduceLastMean, [2, 2], null,
                ("input", new[] { 2, 2, 2 }, Enumerable.Range(1, 8).Select(i => (float)i).ToArray())));
        Assert.Equal([5f, 6f, 7f, 10f, 12f, 14f, 24f, 27f, 30f, 32f, 36f, 40f],
            Run(PortableTensorOperationContracts.HeadOuter, [2, 2, 3], null,
                ("left", new[] { 2, 2 }, new float[] { 1, 2, 3, 4 }),
                ("right", new[] { 2, 3 }, new float[] { 5, 6, 7, 8, 9, 10 })));
    }

    [Fact]
    public void CastReadsBorrowedFp16ModelWeightWithoutCopying()
    {
        var original = Create(PortableTensorOperationContracts.CastFp16ToFp32, [2], null,
            ("input", new[] { 2 }, new Half[2]));
        var graph = new ExecutionGraph(original.Identity, original.Model,
            original.Resources.Select(resource => resource.Id == new ResourceId("input")
                ? resource with
                {
                    Kind = GraphResourceKind.Weight,
                    Lifetime = GraphResourceLifetime.Model,
                    BindingKey = "weight",
                }
                : resource),
            original.Regions, original.Nodes, [], original.Outputs);
        var values = new Half[] { (Half)1, (Half)2 };
        var model = new PortableGraphModel(
            new RwkvModelMetadata(2, 2, 1, 1, 2, "tensor-test"), new HalfCatalog(values), graph);
        var executor = new CpuPrimitiveGraphExecutor(model);
        values[0] = (Half)3;
        Assert.Equal([3f, 2f], Assert.IsType<float[]>(executor.Execute(
            new Dictionary<ResourceId, Array>()).Outputs[new ResourceId("output")]));
    }

    [Fact]
    public void InvalidShapeAttributePortAndPrecisionReturnPreparationDiagnostics()
    {
        var wrongShape = Create(PortableTensorOperationContracts.BatchedMatVec, [2, 2], null,
            ("matrix", new[] { 2, 2, 3 }, new float[12]),
            ("vector", new[] { 2, 2 }, new float[4]));
        Failure(wrongShape, "incompatible dimensions");

        var badAttribute = Create(PortableTensorOperationContracts.Slice, [2, 1], new Dictionary<string, string>
            { ["axis"] = "0", ["start"] = "0", ["length"] = "-1" },
            ("input", new[] { 2, 2 }, new float[4]));
        Failure(badAttribute, "non-negative integer");
        var nonFinite = Create(PortableTensorOperationContracts.Fill, [1],
            new Dictionary<string, string> { ["value"] = "NaN" });
        Failure(nonFinite, "finite invariant-culture");
        var extraAttribute = Create(PortableTensorOperationContracts.Reshape, [2], new Dictionary<string, string>
            { ["unexpected"] = "1" }, ("input", new[] { 2 }, new float[2]));
        Failure(extraAttribute, "invalid attributes");

        var invalidNode = wrongShape.Nodes.Single() with
        {
            Resources = [GraphBindings.Read("incorrect", "matrix"),
                GraphBindings.Read("vector", "vector"), GraphBindings.Write("output", "output")],
        };
        Failure(WithNode(wrongShape, invalidNode), "invalid ports or access");
        var lowPrecision = wrongShape.Nodes.Single() with
        {
            Requirements = new PrecisionRequirement(GraphElementType.Float16, GraphElementType.Float16),
        };
        Failure(WithNode(wrongShape, lowPrecision), "requires FP32 arithmetic");
    }

    [Theory]
    [InlineData("cast")]
    [InlineData("reshape")]
    [InlineData("slice")]
    [InlineData("broadcast")]
    [InlineData("batched")]
    [InlineData("sum")]
    [InlineData("mean")]
    [InlineData("outer")]
    public void RejectsEachMalformedTensorShapeDuringPreparation(string kind)
    {
        var graph = kind switch
        {
            "cast" => Create(PortableTensorOperationContracts.CastFp16ToFp32, [1], null,
                ("input", new[] { 2 }, new Half[2])),
            "reshape" => Create(PortableTensorOperationContracts.Reshape, [3], null,
                ("input", new[] { 2 }, new float[2])),
            "slice" => Create(PortableTensorOperationContracts.Slice, [2, 3],
                new TensorSlice(1, 1, 2).ToAttributes(),
                ("input", new[] { 2, 3 }, new float[6])),
            "broadcast" => Create(PortableTensorOperationContracts.Broadcast, [2, 3], null,
                ("input", new[] { 2, 2 }, new float[4])),
            "batched" => Create(PortableTensorOperationContracts.BatchedMatVec, [2, 2], null,
                ("matrix", new[] { 2, 2, 3 }, new float[12]),
                ("vector", new[] { 2, 2 }, new float[4])),
            "sum" => Create(PortableTensorOperationContracts.ReduceLastSum, [2, 2], null,
                ("input", new[] { 2, 3 }, new float[6])),
            "mean" => Create(PortableTensorOperationContracts.ReduceLastMean, [2, 2], null,
                ("input", new[] { 2, 3 }, new float[6])),
            "outer" => Create(PortableTensorOperationContracts.HeadOuter, [2, 2, 2], null,
                ("left", new[] { 2, 2 }, new float[4]),
                ("right", new[] { 2, 3 }, new float[6])),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
        Failure(graph, "incompatible dimensions");
    }

    private static float[] Run(GraphOperationId operation, int[] shape,
        IReadOnlyDictionary<string, string>? attributes,
        params (string Port, int[] Shape, Array Data)[] sources)
    {
        var graph = Create(operation, shape, attributes, sources);
        var backend = CpuPrimitiveGraphBackend.Instance;
        Assert.Contains(operation, backend.KernelCatalog.SupportedOperations);
        Assert.Single(backend.GetOperatorImplementations(graph, graph.Nodes[0]));
        Assert.IsType<CpuPrimitiveGraphPlan>(backend.Prepare(graph).GetPlanOrThrow("cpu"));
        var inputs = sources.ToDictionary(source => new ResourceId(source.Port),
            source => source.Data);
        return Assert.IsType<float[]>(new CpuPrimitiveGraphExecutor(graph, new EmptyCatalog())
            .Execute(inputs).Outputs[new ResourceId("output")]);
    }

    private static ExecutionGraph Create(GraphOperationId operation, int[] shape,
        IReadOnlyDictionary<string, string>? attributes,
        params (string Port, int[] Shape, Array Data)[] sources)
    {
        var builder = new LogicalGraphBuilder(
            new GraphIdentity("tensor-test", 1, "portable-tensor"),
            new GraphModelSignature(2, 2, 1, 1, 2, "tensor.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root");
        foreach (var source in sources)
            builder.AddResource(source.Port, source.Port, GraphResourceKind.Input,
                GraphResourceLifetime.External,
                new TensorDescriptor(source.Data is Half[] ? GraphElementType.Float16 : GraphElementType.Float32,
                    source.Shape), graphInput: true);
        builder.AddResource("output", "output", GraphResourceKind.Output,
            GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Float32, shape),
            graphOutput: true);
        builder.AddNode("operation", operation, "root",
            sources.Select(source => GraphBindings.Read(source.Port, source.Port))
                .Append(GraphBindings.Write("output", "output")), attributes: attributes,
            requirements: new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32));
        return new GraphOptimizer().Optimize(builder.Build());
    }

    private static ExecutionGraph WithNode(ExecutionGraph graph, ExecutionNode node) =>
        new(graph.Identity, graph.Model, graph.Resources, graph.Regions,
            [node], graph.Inputs, graph.Outputs, graph.GraphState);

    private static void Failure(ExecutionGraph graph, string expected)
    {
        var result = Assert.IsType<BackendPreparationResult.Failure>(
            CpuPrimitiveGraphBackend.Instance.Prepare(graph));
        var diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(graph.Nodes[0].Id, diagnostic.NodeId);
        Assert.Contains(expected, diagnostic.Message, StringComparison.Ordinal);
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

    private sealed class HalfCatalog(Half[] values) : IModelTensorCatalog, IModelTensor
    {
        public int VocabularySize => 2;
        public int EmbeddingSize => 2;
        public int LayerCount => 1;
        public IReadOnlyCollection<string> Names => ["weight"];
        public bool TryGet(string name, out IModelTensor tensor)
        {
            tensor = name == "weight" ? this : null!;
            return tensor is not null;
        }
        public IModelTensor GetRequired(string name) =>
            name == "weight" ? this : throw new InvalidDataException(name);
        public string Name => "weight";
        public RwkvTensorDataType DataType => RwkvTensorDataType.Float16;
        public IReadOnlyList<int> Dimensions => [2];
        public ReadOnlySpan<float> FloatValues => [];
        public ReadOnlySpan<Half> HalfValues => values;
    }
}
