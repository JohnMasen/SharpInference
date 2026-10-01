using SharpInference.Backends.Cpu;
using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class CpuWeightCastResidencyTests
{
    [Fact]
    public void HalfMatrixAndReadOnlyReshapeRunWithoutResidentFp32Copy()
    {
        const int width = 256;
        var source = Enumerable.Repeat((Half)1, width * width).ToArray();
        var tensor = new CountingTensor(source, [width, width]);
        var graph = new GraphOptimizer().Optimize(
            new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "weight-cache"),
                new GraphModelSignature(2, width, 1, 1, width, "synthetic.state"))
                .AddRegion("root", GraphRegionTypes.Graph, "root")
                .AddResource("weight", "weight", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                    new TensorDescriptor(GraphElementType.Float16, [width, width]), bindingKey: "weight")
                .AddResource("converted", "converted", GraphResourceKind.Temporary,
                    GraphResourceLifetime.Invocation,
                    new TensorDescriptor(GraphElementType.Float32, [width, width]))
                .AddResource("reshaped", "reshaped", GraphResourceKind.Temporary,
                    GraphResourceLifetime.Invocation,
                    new TensorDescriptor(GraphElementType.Float32, [width, width]))
                .AddResource("vector", "vector", GraphResourceKind.Input, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [width]), graphInput: true)
                .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [width]), graphOutput: true)
                .AddNode("cast", PortableTensorOperationContracts.CastFp16ToFp32, "root",
                    [GraphBindings.Read("input", "weight"), GraphBindings.Write("output", "converted")],
                    requirements: FloatPrecision)
                .AddNode("reshape", PortableTensorOperationContracts.Reshape, "root",
                    [GraphBindings.Read("input", "converted"), GraphBindings.Write("output", "reshaped")],
                    ["cast"], requirements: FloatPrecision)
                .AddNode("matvec", PrimitiveGraphOperations.MatVec, "root",
                    [GraphBindings.Read("matrix", "reshaped"), GraphBindings.Read("input", "vector"),
                        GraphBindings.Write("output", "result")], ["reshape"])
                .Build());
        var model = new PortableGraphModel(
            new RwkvModelMetadata(2, width, 1, 1, width, "synthetic"),
            new OneTensorCatalog(tensor), graph);
        var inputs = new Dictionary<ResourceId, Array>
            { [new("vector")] = Enumerable.Repeat(1f, width).ToArray() };
        var beforeConstruction = GC.GetAllocatedBytesForCurrentThread();
        var executor = new CpuPrimitiveGraphExecutor(model);
        var constructionBytes = GC.GetAllocatedBytesForCurrentThread() - beforeConstruction;
        Assert.True(constructionBytes < width * width * sizeof(float) / 2,
            $"Binding eagerly allocated {constructionBytes} bytes for an unexecuted weight cast.");

        var beforeFirst = GC.GetAllocatedBytesForCurrentThread();
        var first = executor.Execute(inputs);
        var firstBytes = GC.GetAllocatedBytesForCurrentThread() - beforeFirst;
        Assert.True(firstBytes < width * width * sizeof(float) / 2,
            $"Executing the half matrix allocated {firstBytes} bytes.");
        Assert.Equal(width, Assert.IsType<float[]>(first.Outputs[new("result")])[0]);
        var readsAfterFirst = tensor.HalfReadCount;
        source[0] = (Half)10;
        var second = executor.Execute(inputs);
        Assert.Equal(width + 9, Assert.IsType<float[]>(second.Outputs[new("result")])[0]);
        Assert.True(tensor.HalfReadCount > readsAfterFirst);
        Assert.Equal(0, tensor.FloatReadCount);

        var copiedExecutor = new CpuPrimitiveGraphExecutor(graph, new OneTensorCatalog(tensor));
        source[0] = (Half)1;
        var beforeCopiedExecution = GC.GetAllocatedBytesForCurrentThread();
        var copiedResult = copiedExecutor.Execute(inputs);
        Assert.True(GC.GetAllocatedBytesForCurrentThread() - beforeCopiedExecution <
            width * width * sizeof(float) / 2,
            "The direct graph/catalog path must not materialize a full FP32 matrix on execution.");
        Assert.Equal(width + 9, Assert.IsType<float[]>(copiedResult.Outputs[new("result")])[0]);
    }

    [Fact]
    public void CastOfTokenInputStillRunsEveryTokenAndKeepsPreviousResultsIndependent()
    {
        var graph = new LogicalGraphBuilder(
                new GraphIdentity("synthetic", 1, "dynamic-cast"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("input", "input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float16, [2]), graphInput: true)
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddNode("cast", PortableTensorOperationContracts.CastFp16ToFp32, "root",
                [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "result")],
                requirements: FloatPrecision)
            .Build();
        var executor = new CpuPrimitiveGraphExecutor(graph, new OneTensorCatalog());
        var first = executor.Execute(new Dictionary<ResourceId, Array>
            { [new("input")] = new Half[] { (Half)1, (Half)2 } });
        var second = executor.Execute(new Dictionary<ResourceId, Array>
            { [new("input")] = new Half[] { (Half)3, (Half)4 } });
        Assert.Equal([1f, 2f], Assert.IsType<float[]>(first.Outputs[new("result")]));
        Assert.Equal([3f, 4f], Assert.IsType<float[]>(second.Outputs[new("result")]));
    }

    [Fact]
    public void HalfEmbeddingGatherDoesNotCreateFullFp32Table()
    {
        const int width = 256;
        var tensor = new CountingTensor(
            Enumerable.Repeat((Half)2, width * width).ToArray(), [width, width]);
        var graph = new LogicalGraphBuilder(
                new GraphIdentity("synthetic", 1, "half-gather"),
                new GraphModelSignature(width, width, 1, 1, width, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("weight", "weight", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float16, [width, width]), bindingKey: "weight")
            .AddResource("cast", "cast", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [width, width]))
            .AddResource("table", "table", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [width, width]))
            .AddResource("index", "index", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true)
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphOutput: true)
            .AddNode("cast", PortableTensorOperationContracts.CastFp16ToFp32, "root",
                [GraphBindings.Read("input", "weight"), GraphBindings.Write("output", "cast")],
                requirements: FloatPrecision)
            .AddNode("reshape", PortableTensorOperationContracts.Reshape, "root",
                [GraphBindings.Read("input", "cast"), GraphBindings.Write("output", "table")],
                ["cast"], requirements: FloatPrecision)
            .AddNode("gather", PrimitiveGraphOperations.GatherRow, "root",
                [GraphBindings.Read("table", "table"), GraphBindings.Read("index", "index"),
                    GraphBindings.Write("output", "result")], ["reshape"])
            .Build();
        var model = new PortableGraphModel(
            new RwkvModelMetadata(width, width, 1, 1, width, "synthetic"),
            new OneTensorCatalog(tensor), new GraphOptimizer().Optimize(graph));
        var executor = new CpuPrimitiveGraphExecutor(model);
        var before = GC.GetAllocatedBytesForCurrentThread();
        var result = executor.Execute(new Dictionary<ResourceId, Array> { [new("index")] = new[] { 1 } });
        Assert.True(GC.GetAllocatedBytesForCurrentThread() - before < width * width * sizeof(float) / 2);
        Assert.All(Assert.IsType<float[]>(result.Outputs[new("result")]), value => Assert.Equal(2f, value));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            executor.Execute(new Dictionary<ResourceId, Array> { [new("index")] = new[] { width } }));
        Assert.Equal(0, tensor.FloatReadCount);
    }

    [Fact]
    public void VectorizedHalfMatVecMatchesScalarFp32AccumulationTolerance()
    {
        const int rows = 128;
        const int columns = 512;
        var matrix = Enumerable.Range(0, rows * columns)
            .Select(index => (Half)((index % 19 - 9) / 16f)).ToArray();
        var vector = Enumerable.Range(0, columns)
            .Select(index => (index % 23 - 11) / 10f).ToArray();
        var graph = new GraphOptimizer().Optimize(new LogicalGraphBuilder(
                new GraphIdentity("synthetic", 1, "vectorized-half"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("weight", "weight", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float16, [rows, columns]), bindingKey: "weight")
            .AddResource("converted", "converted", GraphResourceKind.Temporary,
                GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [rows, columns]))
            .AddResource("vector", "vector", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [columns]), graphInput: true)
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [rows]), graphOutput: true)
            .AddNode("cast", PortableTensorOperationContracts.CastFp16ToFp32, "root",
                [GraphBindings.Read("input", "weight"), GraphBindings.Write("output", "converted")],
                requirements: FloatPrecision)
            .AddNode("matvec", PrimitiveGraphOperations.MatVec, "root",
                [GraphBindings.Read("matrix", "converted"), GraphBindings.Read("input", "vector"),
                    GraphBindings.Write("output", "result")], ["cast"])
            .Build());
        var tensor = new CountingTensor(matrix, [rows, columns]);
        var model = new PortableGraphModel(
            new RwkvModelMetadata(2, 2, 1, 1, 2, "synthetic"),
            new OneTensorCatalog(tensor), graph);
        var result = Assert.IsType<float[]>(new CpuPrimitiveGraphExecutor(model).Execute(
            new Dictionary<ResourceId, Array> { [new("vector")] = vector })
            .Outputs[new ResourceId("result")]);
        for (var row = 0; row < rows; row++)
        {
            float expected = 0;
            for (var column = 0; column < columns; column++)
                expected += (float)matrix[row * columns + column] * vector[column];
            Assert.InRange(MathF.Abs(result[row] - expected), 0,
                3e-4f * MathF.Max(1, MathF.Abs(expected)));
        }
        Assert.Equal(0, tensor.FloatReadCount);
    }

    [Fact]
    public void WeightCastWithElementwiseConsumerStillCachesConvertedValues()
    {
        var tensor = new CountingTensor([(Half)1, (Half)2], [2]);
        var graph = new LogicalGraphBuilder(
                new GraphIdentity("synthetic", 1, "elementwise-cast"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("weight", "weight", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float16, [2]), bindingKey: "weight")
            .AddResource("cast", "cast", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("input", "input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddNode("cast", PortableTensorOperationContracts.CastFp16ToFp32, "root",
                [GraphBindings.Read("input", "weight"), GraphBindings.Write("output", "cast")],
                requirements: FloatPrecision)
            .AddNode("add", PrimitiveGraphOperations.Add, "root",
                [GraphBindings.Read("left", "cast"), GraphBindings.Read("right", "input"),
                    GraphBindings.Write("output", "result")], ["cast"])
            .Build();
        var executor = new CpuPrimitiveGraphExecutor(new PortableGraphModel(
            new RwkvModelMetadata(2, 2, 1, 1, 2, "synthetic"),
            new OneTensorCatalog(tensor), new GraphOptimizer().Optimize(graph)));
        var inputs = new Dictionary<ResourceId, Array> { [new("input")] = new[] { 10f, 20f } };
        Assert.Equal([11f, 22f], Assert.IsType<float[]>(executor.Execute(inputs).Outputs[new("result")]));
        var reads = tensor.HalfReadCount;
        Assert.Equal([11f, 22f], Assert.IsType<float[]>(executor.Execute(inputs).Outputs[new("result")]));
        Assert.Equal(reads, tensor.HalfReadCount);
    }

    [Fact]
    public void NativeHalfWeightSurvivesIndependentProcessorSessions()
    {
        var graph = new GraphOptimizer().Optimize(
            new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "sessions"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
                .AddRegion("root", GraphRegionTypes.Graph, "root")
                .AddResource("token", "token", GraphResourceKind.Input, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true)
                .AddResource("weight", "weight", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                    new TensorDescriptor(GraphElementType.Float16, [2, 2]), bindingKey: "weight")
                .AddResource("converted", "converted", GraphResourceKind.Temporary,
                    GraphResourceLifetime.Invocation,
                    new TensorDescriptor(GraphElementType.Float32, [2, 2]))
                .AddResource("reshaped", "reshaped", GraphResourceKind.Temporary,
                    GraphResourceLifetime.Invocation,
                    new TensorDescriptor(GraphElementType.Float32, [2, 2]))
                .AddResource("logits", "logits", GraphResourceKind.Output, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
                .AddNode("cast", PortableTensorOperationContracts.CastFp16ToFp32, "root",
                    [GraphBindings.Read("input", "weight"), GraphBindings.Write("output", "converted")],
                    requirements: FloatPrecision)
                .AddNode("reshape", PortableTensorOperationContracts.Reshape, "root",
                    [GraphBindings.Read("input", "converted"), GraphBindings.Write("output", "reshaped")],
                    ["cast"], requirements: FloatPrecision)
                .AddNode("gather", PrimitiveGraphOperations.GatherRow, "root",
                    [GraphBindings.Read("table", "reshaped"), GraphBindings.Read("index", "token"),
                        GraphBindings.Write("output", "logits")], ["reshape"])
                .Build());
        var values = new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 };
        var tensor = new CountingTensor(values, [2, 2]);
        var model = new PortableGraphModel(
            new RwkvModelMetadata(2, 2, 1, 1, 2, "synthetic"), new OneTensorCatalog(tensor), graph);
        var backend = CpuPrimitiveGraphBackend.Instance;
        var plan = backend.Prepare(graph).GetPlanOrThrow("cpu");
        backend.PrepareModelWeights(model);
        var logits = new float[2];
        using (var first = backend.CreateSessionExecutor(model, new PortableGraphState(graph), plan))
            first.ForwardToken(1, logits);
        Assert.Equal([3f, 4f], logits);
        var reads = tensor.HalfReadCount;
        values[2] = (Half)99;
        using (var second = backend.CreateSessionExecutor(model, new PortableGraphState(graph), plan))
            second.ForwardToken(1, logits);
        Assert.Equal([99f, 4f], logits);
        Assert.True(tensor.HalfReadCount > reads);
    }

    [Fact]
    public void ProcessorForwardDoesNotAllocateFullStateSnapshotPerToken()
    {
        const int stateLength = 8192;
        var graph = new GraphOptimizer().Optimize(
            new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "large-state"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
                .AddRegion("root", GraphRegionTypes.Graph, "root")
                .AddResource("token", "token", GraphResourceKind.Input, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true)
                .AddResource("memory", "memory", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                    new TensorDescriptor(GraphElementType.Float32, [stateLength]))
                .AddResource("logits", "logits", GraphResourceKind.Output, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
                .AddStateSlot("memory", "memory")
                .AddNode("store", PrimitiveGraphOperations.Copy, "root",
                    [GraphBindings.Read("input", "memory"), GraphBindings.Write("output", "memory")])
                .AddNode("fill", PortableTensorOperationContracts.Fill, "root",
                    [GraphBindings.Write("output", "logits")], ["store"],
                    new TensorFillValue(2).ToAttributes(), FloatPrecision)
                .Build());
        var model = new PortableGraphModel(new RwkvModelMetadata(2, 2, 1, 1, 2, "synthetic"),
            new OneTensorCatalog(), graph);
        var backend = CpuPrimitiveGraphBackend.Instance;
        var plan = backend.Prepare(graph).GetPlanOrThrow("cpu");
        backend.PrepareModelWeights(model);
        var state = new PortableGraphState(graph);
        state.Restore(Enumerable.Repeat(1f, stateLength).ToArray());
        using var session = backend.CreateSessionExecutor(model, state, plan);
        var logits = new float[2];
        for (var index = 0; index < 4; index++) session.ForwardToken(0, logits);
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var index = 0; index < 16; index++) session.ForwardToken(0, logits);
        var allocatedPerToken = (GC.GetAllocatedBytesForCurrentThread() - before) / 16;
        Assert.True(allocatedPerToken < stateLength * sizeof(float) / 2,
            $"ForwardToken allocated {allocatedPerToken} bytes per token for a {stateLength}-float state.");
        Assert.Equal([2f, 2f], logits);
        Assert.All(state.Views[0].Values, value => Assert.Equal(1f, value));
    }

    [Fact]
    public void LargeDynamicWorkspaceIsDeferredUntilFirstExecution()
    {
        const int width = 65536;
        var graph = new GraphOptimizer().Optimize(
            new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "lazy-scratch"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
                .AddRegion("root", GraphRegionTypes.Graph, "root")
                .AddResource("logits", "logits", GraphResourceKind.Output, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [width]), graphOutput: true)
                .AddNode("fill", PortableTensorOperationContracts.Fill, "root",
                    [GraphBindings.Write("output", "logits")],
                    attributes: new TensorFillValue(1).ToAttributes(), requirements: FloatPrecision)
                .Build());
        var before = GC.GetAllocatedBytesForCurrentThread();
        var executor = new CpuPrimitiveGraphExecutor(graph, new OneTensorCatalog());
        var constructionBytes = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(constructionBytes < width * sizeof(float) / 2,
            $"Binding eagerly allocated {constructionBytes} bytes for an unexecuted temporary.");
        Assert.All(Assert.IsType<float[]>(executor.Execute(new Dictionary<ResourceId, Array>())
            .Outputs[new ResourceId("logits")]), value => Assert.Equal(1f, value));
    }

    private static PrecisionRequirement FloatPrecision =>
        new(GraphElementType.Float32, GraphElementType.Float32);

    private sealed class CountingTensor(Half[] values, int[] dimensions) : IModelTensor
    {
        public int HalfReadCount { get; private set; }
        public int FloatReadCount { get; private set; }
        public string Name => "weight";
        public RwkvTensorDataType DataType => RwkvTensorDataType.Float16;
        public IReadOnlyList<int> Dimensions => dimensions;
        public ReadOnlySpan<float> FloatValues
        {
            get
            {
                FloatReadCount++;
                throw new InvalidOperationException("A half weight must not request eager FP32 catalog conversion.");
            }
        }
        public ReadOnlySpan<Half> HalfValues
        {
            get
            {
                HalfReadCount++;
                return values;
            }
        }
    }

    private sealed class OneTensorCatalog(CountingTensor? tensor = null) : IModelTensorCatalog
    {
        public int VocabularySize => 2;
        public int EmbeddingSize => 2;
        public int LayerCount => 1;
        public IReadOnlyCollection<string> Names => tensor is null ? [] : ["weight"];
        public bool TryGet(string name, out IModelTensor value)
        {
            value = name == "weight" ? tensor! : null!;
            return value is not null;
        }
        public IModelTensor GetRequired(string name) =>
            tensor is not null && name == "weight" ? tensor : throw new InvalidDataException(name);
    }
}
