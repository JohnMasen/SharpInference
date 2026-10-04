using System.Diagnostics;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Backends.Cpu;
using SharpInference.Graphs;
using SharpInference.Runtime;
using Xunit.Abstractions;

namespace SharpInference.Tests;

public sealed class CpuFusedElementwiseExpressionTests(ITestOutputHelper output)
{
    [Fact]
    public void OptimizerOnFusesTypedExpressionAndMatchesOffWithWarmTiming()
    {
        const int width = 32768;
        var logical = BuildChain(width);
        var backend = CpuPrimitiveGraphBackend.Instance;
        var optimizer = new GraphOptimizer();
        var off = optimizer.Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), backend.KernelCatalog);
        var on = optimizer.Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), backend.KernelCatalog);
        Assert.Equal(4, off.Nodes.Count);
        Assert.Single(on.Nodes);
        Assert.Equal(FusedElementwiseExpressionContract.Operation, on.Nodes[0].Operation);
        Assert.Equal(4, FusedElementwiseExpressionContract.Read(
            on.Nodes[0], on.Resources.ToDictionary(resource => resource.Id)).Steps.Count);
        Assert.Single(backend.GetOperatorImplementations(on, on.Nodes[0]));
        Assert.IsType<CpuPrimitiveGraphPlan>(backend.Prepare(on).GetPlanOrThrow("cpu"));

        var noExpressionProvider = new ExecutionKernelCatalog("cpu-no-fusion",
            backend.KernelCatalog.SupportedOperations);
        var fallback = optimizer.Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), noExpressionProvider);
        Assert.Equal(off.Nodes.Count, fallback.Nodes.Count);

        var inputs = new Dictionary<ResourceId, Array>
        {
            [new("a")] = Enumerable.Range(0, width).Select(i => i % 13 / 10f).ToArray(),
            [new("b")] = Enumerable.Range(0, width).Select(i => (i % 7 + 1) / 13f).ToArray(),
            [new("c")] = Enumerable.Range(0, width).Select(i => (i % 11 + 1) / 6f).ToArray(),
        };
        var unfused = new CpuPrimitiveGraphExecutor(off, new EmptyCatalog());
        var fused = new CpuPrimitiveGraphExecutor(on, new EmptyCatalog());
        var expected = Assert.IsType<float[]>(unfused.Execute(inputs).Outputs[new("result")]);
        Assert.Equal(expected, Assert.IsType<float[]>(fused.Execute(inputs).Outputs[new("result")]));
        for (var warm = 0; warm < 8; warm++)
        {
            unfused.Execute(inputs);
            fused.Execute(inputs);
        }

        const int iterations = 32;
        var stopwatch = Stopwatch.StartNew();
        for (var iteration = 0; iteration < iterations; iteration++) unfused.Execute(inputs);
        stopwatch.Stop();
        var offMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
        stopwatch.Restart();
        for (var iteration = 0; iteration < iterations; iteration++) fused.Execute(inputs);
        stopwatch.Stop();
        var onMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
        Assert.Equal(expected, Assert.IsType<float[]>(fused.Execute(inputs).Outputs[new("result")]));
        output.WriteLine(
            $"FP32 elementwise {width} elements x {iterations}: Off={offMilliseconds:F3}ms " +
            $"({off.Nodes.Count} nodes), On={onMilliseconds:F3}ms ({on.Nodes.Count} node); exact parity");
    }

    [Fact]
    public void InvalidFusedExpressionReportsPreparationFailure()
    {
        var backend = CpuPrimitiveGraphBackend.Instance;
        var on = new GraphOptimizer().Optimize(BuildChain(2),
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), backend.KernelCatalog);
        var invalid = new ExecutionGraph(on.Identity, on.Model, on.Resources, on.Regions,
            [on.Nodes[0] with
            {
                Attributes = new Dictionary<string, string>
                    { [FusedElementwiseExpressionContract.Attribute] = "{\"InputCount\":0}" },
            }], on.Inputs, on.Outputs, on.GraphState);
        var failure = Assert.IsType<BackendPreparationResult.Failure>(backend.Prepare(invalid));
        Assert.Contains(failure.Diagnostics, diagnostic =>
            diagnostic.Message.Contains("fused expression", StringComparison.OrdinalIgnoreCase));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(255)]
    [InlineData(256)]
    [InlineData(257)]
    [InlineData(1025)]
    public void BranchedStepReferencesRetainEarlierFp32Results(int width)
    {
        var expression = new FusedElementwiseExpression(2,
        [
            new ElementwiseInstruction(PrimitiveGraphOperations.Add,
                [new ElementwiseOperand(InputIndex: 0), new ElementwiseOperand(InputIndex: 1)]),
            new ElementwiseInstruction(PrimitiveGraphOperations.Square,
                [new ElementwiseOperand(StepIndex: 0)]),
            new ElementwiseInstruction(PrimitiveGraphOperations.Add,
                [new ElementwiseOperand(StepIndex: 1), new ElementwiseOperand(StepIndex: 0)]),
        ]);
        var logical = new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "branched"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("a", "a", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphInput: true)
            .AddResource("b", "b", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphInput: true)
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphOutput: true)
            .AddNode("expression", FusedElementwiseExpressionContract.Operation, "root",
                [GraphBindings.Read("input0", "a"), GraphBindings.Read("input1", "b"),
                    GraphBindings.Write("output", "result")],
                attributes: FusedElementwiseExpressionContract.ToAttributes(expression))
            .Build();
        var graph = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded),
            CpuPrimitiveGraphBackend.Instance.KernelCatalog);
        Assert.IsType<CpuPrimitiveGraphPlan>(
            CpuPrimitiveGraphBackend.Instance.Prepare(graph).GetPlanOrThrow("cpu"));
        var result = new CpuPrimitiveGraphExecutor(graph, new EmptyCatalog()).Execute(
            new Dictionary<ResourceId, Array>
            {
                [new("a")] = Enumerable.Range(0, width).Select(i => i % 2 == 0 ? 2f : 3f).ToArray(),
                [new("b")] = Enumerable.Range(0, width).Select(i => i % 2 == 0 ? 4f : 5f).ToArray(),
            });
        Assert.Equal(Enumerable.Range(0, width).Select(i => i % 2 == 0 ? 42f : 72f),
            Assert.IsType<float[]>(result.Outputs[new("result")]));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(257)]
    [InlineData(4097)]
    public void BranchedTranscendentalExpressionMatchesPrimitiveBackendWithPooledScratch(int width)
    {
        var expression = new FusedElementwiseExpression(2,
        [
            new(PrimitiveGraphOperations.Add, [new(InputIndex: 0), new(InputIndex: 1)]),
            new(PrimitiveGraphOperations.Square, [new(StepIndex: 0)]),
            new(PrimitiveGraphOperations.Tanh, [new(StepIndex: 1)]),
            new(PrimitiveGraphOperations.Multiply, [new(StepIndex: 1), new(StepIndex: 2)]),
            new(PrimitiveGraphOperations.Add, [new(StepIndex: 3), new(StepIndex: 0)]),
        ]);
        var logical = new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "branched-tanh"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("a", "a", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphInput: true)
            .AddResource("b", "b", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphInput: true)
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphOutput: true)
            .AddNode("expression", FusedElementwiseExpressionContract.Operation, "root",
                [GraphBindings.Read("input0", "a"), GraphBindings.Read("input1", "b"),
                    GraphBindings.Write("output", "result")],
                attributes: FusedElementwiseExpressionContract.ToAttributes(expression))
            .Build();
        var a = Enumerable.Range(0, width).Select(i => (i % 19 - 9) / 16f).ToArray();
        var b = Enumerable.Range(0, width).Select(i => (i % 11 - 5) / 8f).ToArray();
        var sum = new float[width];
        var square = new float[width];
        var tanh = new float[width];
        var expected = new float[width];
        var backend = CpuPrimitiveOperatorBackend.Instance;
        backend.Add(a, b, sum);
        backend.Square(sum, square);
        backend.Tanh(square, tanh);
        backend.Multiply(square, tanh, expected);
        backend.Add(expected, sum, expected);
        var executor = new CpuPrimitiveGraphExecutor(logical, new EmptyCatalog());
        var inputs = new Dictionary<ResourceId, Array> { [new("a")] = a, [new("b")] = b };
        for (var iteration = 0; iteration < 3; iteration++)
        {
            var actual = Assert.IsType<float[]>(executor.Execute(inputs).Outputs[new("result")]);
            for (var index = 0; index < width; index++)
                Assert.InRange(MathF.Abs(actual[index] - expected[index]), 0,
                    2e-6f * MathF.Max(1, MathF.Abs(expected[index])));
        }
    }

    [Fact]
    public void FusedExpressionRejectsMismatchedInputShape()
    {
        var backend = CpuPrimitiveGraphBackend.Instance;
        var on = new GraphOptimizer().Optimize(BuildChain(2),
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), backend.KernelCatalog);
        var resources = on.Resources.Select(resource => resource.Id == new ResourceId("b")
            ? resource with { Tensor = new TensorDescriptor(GraphElementType.Float32, [3]) }
            : resource).ToArray();
        var invalid = new ExecutionGraph(on.Identity, on.Model, resources, on.Regions,
            on.Nodes, on.Inputs, on.Outputs, on.GraphState);
        var failure = Assert.IsType<BackendPreparationResult.Failure>(backend.Prepare(invalid));
        Assert.Contains(failure.Diagnostics, diagnostic =>
            diagnostic.Message.Contains("identical dense FP32", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("core.copy")]
    [InlineData("core.add")]
    [InlineData("core.subtract")]
    [InlineData("core.multiply")]
    [InlineData("core.divide")]
    [InlineData("core.maximum")]
    [InlineData("core.exp")]
    [InlineData("core.tanh")]
    [InlineData("core.sigmoid")]
    [InlineData("core.rsqrt")]
    [InlineData("core.square")]
    [InlineData("core.relu")]
    public void EveryTypedStepMatchesUnfusedPrimitiveExactly(string name)
    {
        var operation = PrimitiveGraphOperations.CreateStandardDescriptions()
            .Single(description => description.Operation.Name == name).Operation;
        var binary = FusedElementwiseExpressionContract.Arity(operation) == 2;
        var builder = new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, name),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("a", "a", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("b", "b", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("temporary", "temporary", GraphResourceKind.Temporary,
                GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddNode("copy", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "a"), GraphBindings.Write("output", "temporary")])
            .AddNode("step", operation, "root",
                binary
                    ? [GraphBindings.Read("left", "temporary"), GraphBindings.Read("right", "b"),
                        GraphBindings.Write("output", "result")]
                    : [GraphBindings.Read("input", "temporary"), GraphBindings.Write("output", "result")],
                ["copy"]);
        var backend = CpuPrimitiveGraphBackend.Instance;
        var logical = builder.Build();
        var optimizer = new GraphOptimizer();
        var off = optimizer.Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), backend.KernelCatalog);
        var on = optimizer.Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), backend.KernelCatalog);
        Assert.Contains(on.Nodes, node => node.Operation == FusedElementwiseExpressionContract.Operation);
        var inputs = new Dictionary<ResourceId, Array>
        {
            [new("a")] = new[] { 2f, 4f },
            [new("b")] = new[] { 3f, 5f },
        };
        Assert.Equal(
            Assert.IsType<float[]>(new CpuPrimitiveGraphExecutor(off, new EmptyCatalog())
                .Execute(inputs).Outputs[new("result")]),
            Assert.IsType<float[]>(new CpuPrimitiveGraphExecutor(on, new EmptyCatalog())
                .Execute(inputs).Outputs[new("result")]));
    }

    [Fact]
    public void CompiledProcessorPipelineAcceptsElementwiseChainWithOptimizationOnAndOff()
    {
        var graph = new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "processor-expression"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state.fp32@1"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("token", "token", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true)
            .AddResource("weight", "weight", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float32, [2, 2]), bindingKey: "table")
            .AddResource("bias", "bias", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float32, [2]), bindingKey: "bias")
            .AddResource("row", "row", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("sum", "sum", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("logits", "logits", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddNode("gather", PrimitiveGraphOperations.GatherRow, "root",
                [GraphBindings.Read("table", "weight"), GraphBindings.Read("index", "token"),
                    GraphBindings.Write("output", "row")])
            .AddNode("sum", PrimitiveGraphOperations.Add, "root",
                [GraphBindings.Read("left", "row"), GraphBindings.Read("right", "bias"),
                    GraphBindings.Write("output", "sum")], ["gather"])
            .AddNode("relu", PrimitiveGraphOperations.Relu, "root",
                [GraphBindings.Read("input", "sum"), GraphBindings.Write("output", "logits")], ["sum"])
            .Build();
        var backend = CpuPrimitiveGraphBackend.Instance;
        var enabled = new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        var disabled = new GraphOptimizationOptions(OptimizationBoundary.Off,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        Assert.Contains(new GraphOptimizer().Optimize(graph, enabled, backend.KernelCatalog).Nodes,
            node => node.Operation == FusedElementwiseExpressionContract.Operation);

        float[] Forward(GraphOptimizationOptions options)
        {
            using var processor = new ProcessorPipelineBuilder("virtual-model")
                .UseReader(new VirtualReader(), new GraphArchitectureMetadataReader(graph))
                .UseProvider(new SuppliedLogicalGraphProvider(graph))
                .UseBackend(VmBackendFactory.CreateCpu(), options)
                .UsePortableGraphArchitecture()
                .Build();
            using var session = processor.CreateSession();
            Assert.Equal([11f, 22f], session.ForwardToken(0).ToArray());
            return session.ForwardToken(1).ToArray();
        }
        Assert.Equal([13f, 24f], Forward(disabled));
        Assert.Equal(Forward(disabled), Forward(enabled));
    }

    [Theory]
    [Trait("Category", "ExternalModel")]
    [InlineData(false)]
    [InlineData(true)]
    public void RealTinyPortableGraphMatchesWithExpressionFusionOnAndOff(bool rwkv7)
    {
        var path = TestModelLoader.GetPath(rwkv7 ? TestModel.Rwkv7Fp16 : TestModel.Rwkv6);
        ILogicalGraphProvider provider = rwkv7
            ? new PortableRwkv7GraphProvider() : new PortableRwkv6GraphProvider();
        var backend = CpuPrimitiveGraphBackend.Instance;
        var enabled = new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        var disabled = new GraphOptimizationOptions(OptimizationBoundary.Off,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        using (var catalog = GgmlModelFile.Open(path))
        {
            var graph = provider.Build(catalog);
            var on = new GraphOptimizer().Optimize(graph, enabled, backend.KernelCatalog);
            Assert.Contains(on.Nodes, node =>
                node.Operation == FusedElementwiseExpressionContract.Operation);
        }

        using var off = Processor.LoadGraph(path, provider, VmBackendFactory.CreateCpu(), disabled);
        using var onProcessor = Processor.LoadGraph(path, provider, VmBackendFactory.CreateCpu(), enabled);
        using var offSession = off.CreateSession();
        using var onSession = onProcessor.CreateSession();
        foreach (var token in new[] { 2, 4, 1 })
        {
            Assert.Equal(offSession.ForwardToken(token).ToArray(),
                onSession.ForwardToken(token).ToArray());
            Assert.Equal(
                StateSnapshotAssertions.Values(StateSnapshotAssertions.Capture(offSession)),
                StateSnapshotAssertions.Values(StateSnapshotAssertions.Capture(onSession)));
        }

        static double Measure(Processor processor)
        {
            using var session = processor.CreateSession();
            foreach (var token in new[] { 2, 4, 1 }) session.ForwardToken(token);
            var timer = Stopwatch.StartNew();
            foreach (var token in new[] { 2, 4, 1, 2, 4, 1, 2, 4 })
                session.ForwardToken(token);
            return timer.Elapsed.TotalMilliseconds;
        }
        output.WriteLine(
            $"{(rwkv7 ? "RWKV7 FP16" : "RWKV6 FP32")} tiny CPU 8 warmed tokens: " +
            $"Off={Measure(off):F3}ms, On={Measure(onProcessor):F3}ms; exact logits/state parity");
    }

    private static LogicalGraph BuildChain(int width) =>
        new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "fused-expression"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("a", "a", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphInput: true)
            .AddResource("b", "b", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphInput: true)
            .AddResource("c", "c", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphInput: true)
            .AddResource("sum", "sum", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [width]))
            .AddResource("product", "product", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [width]))
            .AddResource("activation", "activation", GraphResourceKind.Temporary,
                GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float32, [width]))
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphOutput: true)
            .AddNode("sum", PrimitiveGraphOperations.Add, "root",
                [GraphBindings.Read("left", "a"), GraphBindings.Read("right", "b"),
                    GraphBindings.Write("output", "sum")])
            .AddNode("product", PrimitiveGraphOperations.Multiply, "root",
                [GraphBindings.Read("left", "sum"), GraphBindings.Read("right", "c"),
                    GraphBindings.Write("output", "product")], ["sum"])
            .AddNode("activation", PrimitiveGraphOperations.Tanh, "root",
                [GraphBindings.Read("input", "product"), GraphBindings.Write("output", "activation")],
                ["product"])
            .AddNode("maximum", PrimitiveGraphOperations.Maximum, "root",
                [GraphBindings.Read("left", "activation"), GraphBindings.Read("right", "b"),
                    GraphBindings.Write("output", "result")], ["activation"])
            .Build();

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

    private sealed class VirtualReader : IModelReader
    {
        public IModelFile Open(string path) => new VirtualFile();
    }

    private sealed class VirtualFile : IModelFile
    {
        public string Path => "virtual-model";
        public int VocabularySize => 2;
        public int EmbeddingSize => 2;
        public int LayerCount => 1;
        public IReadOnlyCollection<string> Names => ["table", "bias"];
        public bool TryGet(string name, out IModelTensor tensor)
        {
            tensor = name is "table" or "bias" ? new VirtualTensor(name) : null!;
            return name is "table" or "bias";
        }
        public IModelTensor GetRequired(string name) =>
            TryGet(name, out var tensor) ? tensor : throw new InvalidDataException(name);
        public void Dispose() { }
    }

    private sealed class VirtualTensor(string name) : IModelTensor
    {
        public string Name => name;
        public RwkvTensorDataType DataType => RwkvTensorDataType.Float32;
        public IReadOnlyList<int> Dimensions => name == "table" ? [2, 2] : [2];
        public ReadOnlySpan<float> FloatValues => name == "table"
            ? [1f, 2f, 3f, 4f] : [10f, 20f];
        public ReadOnlySpan<Half> HalfValues => [];
    }
}
