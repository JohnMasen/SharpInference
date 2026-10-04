using System.Diagnostics;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Backends.Cpu;
using SharpInference.Backends.Vortice;
using SharpInference.Graphs;
using SharpInference.Runtime;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;
using Xunit.Abstractions;

namespace SharpInference.Tests;

public sealed class VorticePrimitiveGraphExecutorTests(ITestOutputHelper output)
{
    private static ResourceId Id(string name) => new(name);

    [Fact]
    public void PlanUsesDescriptorsAndRejectsUnsupportedNodes()
    {
        var graph = Graph();
        var plan = VorticePrimitiveGraphPlan.Compile(graph);
        Assert.Equal(["GatherRow", "Add", "MatVec", "Copy"], plan.Steps.Select(step => step.Kernel));
        Assert.Equal(new ResourceId("table"), plan.Steps[0].Input0);
        Assert.Equal(new ResourceId("token"), plan.Steps[0].Index);
        Assert.Equal((uint)2, plan.Steps[2].Rows);
        Assert.Equal((uint)3, plan.Steps[2].Columns);

        var invalid = Graph(PrimitiveGraphOperations.ReduceSum);
        var error = Assert.Throws<NotSupportedException>(() => VorticePrimitiveGraphPlan.Compile(invalid));
        Assert.Contains("gather", error.Message);
        Assert.Contains("core.reduce-sum", error.Message);

        var mismatched = new ExecutionGraph(graph.Identity, graph.Model,
            graph.Resources.Select(resource => resource.Id == Id("matrix")
                ? resource with { Tensor = new TensorDescriptor(GraphElementType.Float32, [3, 2]) }
                : resource),
            graph.Regions, graph.Nodes, graph.Inputs, graph.Outputs, graph.GraphState);
        Assert.Contains("matvec", Assert.Throws<NotSupportedException>(
            () => VorticePrimitiveGraphPlan.Compile(mismatched)).Message);
    }

    [Fact]
    public void ConfigFactorySelectsHardwareAndOwnsItsDevice()
    {
        Assert.False(new VorticeRuntimeConfig().EnableCommandReplay);
        var runtimeConfiguration = new ConfigurationBuilder().AddInMemoryCollection(
            new Dictionary<string, string?>
            {
                ["Rwkv:Runtime:Vortice:EnableCommandReplay"] = "true",
            }).Build();
        var enabled = Assert.IsType<VorticeRuntimeConfig>(runtimeConfiguration
            .GetSection("Rwkv:Runtime:Vortice").Get<VorticeRuntimeConfig>());
        Assert.True(enabled.EnableCommandReplay);
        using var probe = TryCreateDevice();
        if (probe is null) return;
        using var backend = VorticePrimitiveGraphBackend.FromConfig(
            new VorticeRuntimeConfig { AdapterIndex = 0 });
        using var replayBackend = VorticePrimitiveGraphBackend.FromConfig(enabled);
        Assert.False(backend.EnableCommandReplay);
        Assert.True(replayBackend.EnableCommandReplay);
        var unsupportedGraph = new GraphOptimizer().Optimize(FusedExpressionGraph(),
            new GraphOptimizationOptions(OptimizationBoundary.Off),
            replayBackend.KernelCatalog);
        var failure = Assert.IsType<BackendPreparationResult.Failure>(
            replayBackend.Prepare(unsupportedGraph));
        Assert.Contains("Int32[1]", failure.Diagnostics.Single().Message);
        Assert.False(string.IsNullOrWhiteSpace(backend.DeviceName));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            VorticePrimitiveGraphBackend.FromConfig(new VorticeRuntimeConfig
            {
                AdapterIndex = int.MaxValue,
            }));
        backend.Dispose();
        Assert.Throws<ObjectDisposedException>(() => backend.Prepare(Graph()));
    }

    [Fact]
    public void MultipleNodesExecutePerTokenWithResidentWeightsAndIsolatedState()
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        using var executor = new VorticePrimitiveGraphExecutor(device, Graph());
        Assert.False(executor.RequiresCpuWeightCopy);
        using var first = executor.CreateSession();
        using var second = executor.CreateSession();
        Assert.Null(first.LastTokenMetrics);
        Assert.Equal((ulong)0, executor.Metrics.TokenCommandSubmissions);
        first.UploadState(new ResourceId("state"), [0f, 0f, 0f]);
        second.UploadState(new ResourceId("state"), [10f, 0f, 0f]);
        Assert.Throws<InvalidOperationException>(() => first.ExecuteToken(
            new Dictionary<ResourceId, float[]>(), new Dictionary<ResourceId, int> { [Id("token")] = 0 }));
        var table = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var matrix = new[] { 1f, 1f, 1f, 2f, -1f, 0.5f };
        var model = new PortableGraphModel(
            new RwkvModelMetadata(2, 3, 1, 1, 3, "test-primitive"),
            new Catalog(new Tensor("embedding", [2, 3], table),
                new Tensor("projection", [2, 3], matrix)), executor.Plan.Graph);
        executor.PrepareModelWeights(model);
        Array.Clear(table);
        Array.Clear(matrix);

        var token0 = new Dictionary<ResourceId, int> { [Id("token")] = 0 };
        var token1 = new Dictionary<ResourceId, int> { [Id("token")] = 1 };
        Assert.Equal([6f, 1.5f], first.ExecuteToken(new Dictionary<ResourceId, float[]>(), token0)[Id("result")]);
        Assert.Equal([21f, 7.5f], first.ExecuteToken(new Dictionary<ResourceId, float[]>(), token1)[Id("result")]);
        Assert.Equal([16f, 21.5f], second.ExecuteToken(new Dictionary<ResourceId, float[]>(), token0)[Id("result")]);
        Assert.Equal(new VorticePrimitiveGraphTokenMetrics(1, 4, 0, 8), first.LastTokenMetrics);
        Assert.Equal((ulong)3, executor.Metrics.TokenCount);
        Assert.Equal((ulong)3, executor.Metrics.TokenCommandSubmissions);
        Assert.Equal((ulong)2, executor.Metrics.WeightCommandSubmissions);
        Assert.Equal((ulong)48, executor.Metrics.WeightUploadBytes);
        Assert.Equal((ulong)24, executor.Metrics.StateUploadBytes);
        Assert.Equal((ulong)0, executor.Metrics.ActivationUploadBytes);
        Assert.Equal((ulong)24, executor.Metrics.OutputReadbackBytes);
        Assert.Equal((ulong)0, executor.Metrics.StateReadbackBytes);
        Assert.Equal([5f, 7f, 9f], first.ReadState(Id("state")));
        Assert.Equal([11f, 2f, 3f], second.ReadState(Id("state")));
        Assert.Equal((ulong)24, executor.Metrics.StateReadbackBytes);
        Assert.Equal((ulong)2, executor.Metrics.StateReadbackCommandSubmissions);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            first.ExecuteToken(new Dictionary<ResourceId, float[]>(),
                new Dictionary<ResourceId, int> { [Id("token")] = 2 }));
    }

    [Fact]
    public void ReplayReusesPerSessionClosedListsAndObservesTokenAndStateUploads()
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        using var baseline = new VorticePrimitiveGraphExecutor(device, Graph());
        using var replay = new VorticePrimitiveGraphExecutor(device, Graph())
        {
            EnableCommandReplay = true,
        };
        using var unsupportedBackend = new VorticePrimitiveGraphBackend(device)
        {
            EnableCommandReplay = true,
        };
        var unsupportedGraph = new GraphOptimizer().Optimize(FusedExpressionGraph(),
            new GraphOptimizationOptions(OptimizationBoundary.Off),
            unsupportedBackend.KernelCatalog);
        var failure = Assert.IsType<BackendPreparationResult.Failure>(
            unsupportedBackend.Prepare(unsupportedGraph));
        Assert.Contains("Int32[1]", failure.Diagnostics.Single().Message);
        foreach (var executor in new[] { baseline, replay })
        {
            executor.UploadWeight(Id("table"), [1f, 2f, 3f, 4f, 5f, 6f]);
            executor.UploadWeight(Id("matrix"), [1f, 1f, 1f, 2f, -1f, 0.5f]);
        }
        using var expected = baseline.CreateSession();
        using var actual = replay.CreateSession();
        using var second = replay.CreateSession();
        expected.UploadState(Id("state"), [0f, 0f, 0f]);
        actual.UploadState(Id("state"), [0f, 0f, 0f]);
        second.UploadState(Id("state"), [10f, 0f, 0f]);
        foreach (var token in new[] { 0, 1, 0 })
        {
            var scalar = new Dictionary<ResourceId, int> { [Id("token")] = token };
            Assert.Equal(expected.ExecuteToken(
                    new Dictionary<ResourceId, float[]>(), scalar)[Id("result")],
                actual.ExecuteToken(new Dictionary<ResourceId, float[]>(), scalar)[Id("result")]);
            Assert.Equal(expected.ReadState(Id("state")), actual.ReadState(Id("state")));
        }
        Assert.Equal([16f, 21.5f], second.ExecuteToken(
            new Dictionary<ResourceId, float[]>(),
            new Dictionary<ResourceId, int> { [Id("token")] = 0 })[Id("result")]);
        expected.UploadState(Id("state"), [10f, 0f, 0f]);
        actual.UploadState(Id("state"), [10f, 0f, 0f]);
        var next = new Dictionary<ResourceId, int> { [Id("token")] = 1 };
        Assert.Equal(expected.ExecuteToken(new Dictionary<ResourceId, float[]>(), next)[Id("result")],
            actual.ExecuteToken(new Dictionary<ResourceId, float[]>(), next)[Id("result")]);
        Assert.Equal(expected.ReadState(Id("state")), actual.ReadState(Id("state")));
        Assert.Equal((ulong)2, replay.Metrics.ReplayListRecordings);
        Assert.Equal((ulong)5, replay.Metrics.ReplayedTokens);
        Assert.Equal((ulong)5, replay.Metrics.TokenCommandSubmissions);
        Assert.Equal((ulong)20, replay.Metrics.ScalarUploadBytes);
        Assert.Equal((ulong)0, replay.Metrics.ActivationUploadBytes);
    }

    [Fact]
    public void FloatInputsAreUploadedOncePerTokenAndOutputsReadBack()
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        var source = Graph();
        var graph = new ExecutionGraph(source.Identity, source.Model,
            [
                new GraphResource(Id("input"), "input", GraphResourceKind.Input, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [3])),
                new GraphResource(Id("bias"), "bias", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                    new TensorDescriptor(GraphElementType.Float32, [3]), "bias"),
                new GraphResource(Id("result"), "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [3])),
            ],
            source.Regions,
            [new ExecutionNode(new ExecutionNodeId("add"), PrimitiveGraphOperations.Add, new RegionId("root"),
                [GraphBindings.Read("left", "input"), GraphBindings.Read("right", "bias"),
                 GraphBindings.Write("output", "result")],
                [], new Dictionary<string, string>(),
                new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32),
                ExecutionSourceMap.XmlOnly)],
            [Id("input")], [Id("result")]);
        using var executor = new VorticePrimitiveGraphExecutor(device, graph);
        executor.UploadWeight(Id("bias"), [10f, 20f, 30f]);
        using var session = executor.CreateSession();
        Assert.Equal([11f, 22f, 33f], session.ExecuteToken(
            new Dictionary<ResourceId, float[]> { [Id("input")] = [1f, 2f, 3f] })[Id("result")]);
        Assert.Equal([9f, 18f, 27f], session.ExecuteToken(
            new Dictionary<ResourceId, float[]> { [Id("input")] = [-1f, -2f, -3f] })[Id("result")]);
        Assert.Equal(new VorticePrimitiveGraphTokenMetrics(1, 1, 12, 12), session.LastTokenMetrics);
        Assert.Equal((ulong)24, executor.Metrics.ActivationUploadBytes);
        Assert.Equal((ulong)24, executor.Metrics.OutputReadbackBytes);
        Assert.Equal((ulong)12, executor.Metrics.WeightUploadBytes);
        Assert.Throws<ArgumentException>(() => session.ExecuteToken(
            new Dictionary<ResourceId, float[]> { [Id("input")] = [1f] }));
    }

    [Fact]
    public void StateBridgeFlushesOnCloneAndImportsHostEditsBeforeNextToken()
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        var graph = Graph();
        using var executor = new VorticePrimitiveGraphExecutor(device, graph);
        executor.UploadWeight(Id("table"), [1f, 2f, 3f, 4f, 5f, 6f]);
        executor.UploadWeight(Id("matrix"), [1f, 1f, 1f, 2f, -1f, 0.5f]);
        using var session = executor.CreateSession();
        var state = new PortableGraphState(graph);
        using var bridge = new VorticePrimitiveGraphStateBridge(session, state, graph);
        bridge.BeforeToken();
        session.ExecuteToken(new Dictionary<ResourceId, float[]>(),
            new Dictionary<ResourceId, int> { [Id("token")] = 0 });
        bridge.AfterToken();
        Assert.Equal((ulong)0, executor.Metrics.StateReadbackBytes);
        Assert.Equal((ulong)12, executor.Metrics.StateUploadBytes);
        var fork = Assert.IsType<PortableGraphState>(state.Clone());
        Assert.Equal((ulong)12, executor.Metrics.StateReadbackBytes);
        Assert.Equal([1f, 2f, 3f], fork.Views.Single().Values);

        state.Reset();
        bridge.BeforeToken();
        Assert.Equal((ulong)24, executor.Metrics.StateUploadBytes);
        session.ExecuteToken(new Dictionary<ResourceId, float[]>(),
            new Dictionary<ResourceId, int> { [Id("token")] = 1 });
        bridge.AfterToken();
        Assert.Equal((ulong)12, executor.Metrics.StateReadbackBytes);
        Assert.Equal([4f, 5f, 6f], bridge.ReadState(graph.GraphState).Single().Values);

        bridge.WriteState(graph.GraphState, [new GraphStateValue("state", [3], [10f, 20f, 30f])]);
        bridge.BeforeToken();
        session.ExecuteToken(new Dictionary<ResourceId, float[]>(),
            new Dictionary<ResourceId, int> { [Id("token")] = 0 });
        bridge.AfterToken();
        Assert.Equal([11f, 22f, 33f], bridge.ReadState(graph.GraphState).Single().Values);
        state.Views.Single().Values[0] = 100f;
        state.CommitViews();
        bridge.BeforeToken();
        Assert.Equal([161f, 196f], session.ExecuteToken(
            new Dictionary<ResourceId, float[]>(),
            new Dictionary<ResourceId, int> { [Id("token")] = 0 })[Id("result")]);
        bridge.AfterToken();
        Assert.Equal([1f, 2f, 3f], fork.Views.Single().Values);
    }

    [Fact]
    public void PrimitiveBackendBindsResidentWeightsBeforeSourceRelease()
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        using var backend = new VorticePrimitiveGraphBackend(device);
        var graph = Graph();
        var plan = backend.Prepare(graph).GetPlanOrThrow(backend.KernelCatalog.BackendId);
        Assert.False(backend.RequiresCpuWeightCopy);
        Assert.IsAssignableFrom<IGraphModelWeightBackend>(backend);
        var table = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var matrix = new[] { 1f, 1f, 1f, 2f, -1f, 0.5f };
        var signature = graph.Model;
        var model = new PortableGraphModel(new RwkvModelMetadata(
            signature.VocabularySize, signature.EmbeddingSize, signature.LayerCount,
            signature.HeadCount, signature.HeadSize, graph.Identity.ArchitectureId), new Catalog(
            new Tensor("embedding", [2, 3], table),
            new Tensor("projection", [2, 3], matrix)), graph);
        backend.PrepareModelWeights(model);
        Array.Clear(table);
        Array.Clear(matrix);
        var state = new PortableGraphState(graph);
        using var session = backend.CreateSessionExecutor(model, state, plan);
        var logits = new float[2];
        session.ForwardToken(0, logits);
        Assert.Equal([6f, 1.5f], logits);
        var fork = state.Clone();
        Assert.Equal([1f, 2f, 3f], Assert.IsType<PortableGraphState>(fork).Views.Single().Values);
        var stateAccess = Assert.IsAssignableFrom<IProcessorStateExecutor>(session);
        Assert.Equal([1f, 2f, 3f], stateAccess.ReadState(graph.GraphState).Single().Values);
        stateAccess.WriteState(graph.GraphState,
            [new GraphStateValue("state", [3], [10f, 20f, 30f])]);
        session.ForwardToken(1, logits);
        Assert.Equal([75f, 21f], logits);
    }

    [Fact]
    public void PortableTensorFp32KernelsExecuteWithValidatedShapesOnGpu()
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        Run(PortableTensorOperationContracts.Fill, [], [2, 2], [], [1.25f, 1.25f, 1.25f, 1.25f],
            new TensorFillValue(1.25f).ToAttributes());
        Run(PortableTensorOperationContracts.Reshape, [[2, 3]], [3, 2],
            [[1f, 2f, 3f, 4f, 5f, 6f]], [1f, 2f, 3f, 4f, 5f, 6f]);
        Run(PortableTensorOperationContracts.Slice, [[3, 2]], [2, 2],
            [[1f, 2f, 3f, 4f, 5f, 6f]], [3f, 4f, 5f, 6f],
            new TensorSlice(0, 1, 2).ToAttributes());
        Run(PortableTensorOperationContracts.Slice, [[2, 3]], [2, 2],
            [[1f, 2f, 3f, 4f, 5f, 6f]], [2f, 3f, 5f, 6f],
            new TensorSlice(1, 1, 2).ToAttributes());
        Run(PortableTensorOperationContracts.Slice, [[2, 2, 3]], [2, 1, 3],
            [[1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f]],
            [4f, 5f, 6f, 10f, 11f, 12f], new TensorSlice(1, 1, 1).ToAttributes());
        Run(PortableTensorOperationContracts.Slice, [[2, 2, 2, 2]], [2, 2, 1, 2],
            [Enumerable.Range(1, 16).Select(value => (float)value).ToArray()],
            [3f, 4f, 7f, 8f, 11f, 12f, 15f, 16f], new TensorSlice(2, 1, 1).ToAttributes());
        Run(PortableTensorOperationContracts.Broadcast, [[1, 3]], [2, 3],
            [[1f, 2f, 3f]], [1f, 2f, 3f, 1f, 2f, 3f]);
        Run(PortableTensorOperationContracts.Broadcast, [[3]], [2, 3],
            [[1f, 2f, 3f]], [1f, 2f, 3f, 1f, 2f, 3f]);
        Run(PortableTensorOperationContracts.Broadcast, [[2, 1, 1]], [2, 3, 2],
            [[10f, 20f]], [10f, 10f, 10f, 10f, 10f, 10f,
                20f, 20f, 20f, 20f, 20f, 20f]);
        Run(PortableTensorOperationContracts.Broadcast, [[1, 2, 1, 2]], [2, 2, 3, 2],
            [[1f, 2f, 3f, 4f]],
            [1f, 2f, 1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f, 3f, 4f,
                1f, 2f, 1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f, 3f, 4f]);
        Run(PortableTensorOperationContracts.BatchedMatVec, [[2, 2, 3], [2, 3]], [2, 2],
            [[1f, 2f, 3f, 4f, 5f, 6f, 2f, 0f, 1f, 0f, 1f, 2f], [1f, 0f, 1f, 3f, 4f, 5f]],
            [4f, 10f, 11f, 14f]);
        Run(PortableTensorOperationContracts.ReduceLastSum, [[2, 3]], [2],
            [[1f, 2f, 3f, 4f, 5f, 6f]], [6f, 15f]);
        Run(PortableTensorOperationContracts.ReduceLastMean, [[2, 3]], [2],
            [[1f, 2f, 3f, 4f, 5f, 6f]], [2f, 5f]);
        Run(PortableTensorOperationContracts.HeadOuter, [[2, 2], [2, 3]], [2, 2, 3],
            [[1f, 2f, 3f, 4f], [1f, 2f, 3f, 4f, 5f, 6f]],
            [1f, 2f, 3f, 2f, 4f, 6f, 12f, 15f, 18f, 16f, 20f, 24f]);

        void Run(GraphOperationId op, int[][] inputDims, int[] outputDims, float[][] inputs,
            float[] expected, IReadOnlyDictionary<string, string>? attributes = null)
        {
            var graph = SingleNodeGraph(op, inputDims, outputDims, attributes);
            using var executor = new VorticePrimitiveGraphExecutor(device, graph);
            using var session = executor.CreateSession();
            var values = inputDims.Select((_, index) =>
                new KeyValuePair<ResourceId, float[]>(Id($"input{index}"), inputs[index]))
                .ToDictionary(pair => pair.Key, pair => pair.Value);
            var actual = session.ExecuteToken(values)[Id("output")];
            Assert.Equal(expected.Length, actual.Length);
            for (var index = 0; index < actual.Length; index++)
                Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0f, 0.0001f);
        }
    }

    [Fact]
    public void AdditionalPrimitiveKernelsRunAndRankFiveIsExplicitlyUnsupported()
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        foreach (var (operation, expected) in new (GraphOperationId, float[])[]
        {
            (PrimitiveGraphOperations.Exp, [1f, MathF.E]),
            (PrimitiveGraphOperations.Tanh, [0f, MathF.Tanh(1f)]),
            (PrimitiveGraphOperations.ReciprocalSquareRoot, [1f, 0.5f]),
            (PrimitiveGraphOperations.Square, [1f, 16f]),
            (PrimitiveGraphOperations.ReduceSum, [5f]),
            (PrimitiveGraphOperations.ReduceMean, [2.5f]),
        })
        {
            var reduction = operation is var op && (op == PrimitiveGraphOperations.ReduceSum ||
                op == PrimitiveGraphOperations.ReduceMean);
            var source = reduction ? new[] { 1f, 4f } :
                operation == PrimitiveGraphOperations.ReciprocalSquareRoot ? [1f, 4f] :
                operation == PrimitiveGraphOperations.Square ? [1f, 4f] : [0f, 1f];
            using var executor = new VorticePrimitiveGraphExecutor(device,
                SingleNodeGraph(operation, [[2]], [reduction ? 1 : 2]));
            using var session = executor.CreateSession();
            var actual = session.ExecuteToken(new Dictionary<ResourceId, float[]>
                { [Id("input0")] = source })[Id("output")];
            Assert.Equal(expected.Length, actual.Length);
            for (var index = 0; index < expected.Length; index++)
                Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0f, 0.0001f);
        }
        var unsupportedShape = SingleNodeGraph(PortableTensorOperationContracts.Slice,
            [[1, 1, 1, 1, 2]], [1, 1, 1, 1, 1], new TensorSlice(4, 0, 1).ToAttributes());
        Assert.Contains("ranks 1 through 4", Assert.Throws<NotSupportedException>(() =>
            VorticePrimitiveGraphPlan.Compile(unsupportedShape)).Message);
    }

    [Fact]
    public void Fp16WeightStaysResidentAndCastConvertsOddElementCountOnGpu()
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        var original = SingleNodeGraph(PortableTensorOperationContracts.CastFp16ToFp32,
            [[5]], [5]);
        var graph = new ExecutionGraph(original.Identity, original.Model,
            original.Resources.Select(resource => resource.Id == Id("input0")
                ? resource with
                {
                    Kind = GraphResourceKind.Weight,
                    Lifetime = GraphResourceLifetime.Model,
                    Tensor = new TensorDescriptor(GraphElementType.Float16, [5]),
                    BindingKey = "fp16.tensor",
                }
                : resource), original.Regions, original.Nodes, [], original.Outputs);
        using var executor = new VorticePrimitiveGraphExecutor(device, graph);
        var halves = new Half[] { (Half)1f, (Half)(-2f), (Half)0.5f, (Half)65504f, (Half)(1f / 3f) };
        var expected = halves.Select(value => (float)value).ToArray();
        executor.PrepareModelWeights(new PortableGraphModel(
            new RwkvModelMetadata(2, 2, 1, 1, 2, "tensor-test"),
            new Catalog(new Tensor("fp16.tensor", [5], halves)), graph));
        Array.Clear(halves);
        Assert.Equal((ulong)12, executor.Memory.ModelBufferBytes);
        Assert.Equal((ulong)20, executor.Memory.LogicalCastOutputBytes);
        using var session = executor.CreateSession();
        Assert.Equal(session.AllocatedBufferBytes, executor.Memory.ActiveSessionBufferBytes);
        Assert.Equal(expected, session.ExecuteToken(new Dictionary<ResourceId, float[]>())[Id("output")]);
        Assert.Equal(checked(executor.Memory.ModelBufferBytes + session.AllocatedBufferBytes),
            executor.Memory.PeakBufferBytes);
        Assert.Equal((ulong)10, executor.Metrics.WeightUploadBytes);
        Assert.Equal((ulong)0, executor.Metrics.ActivationUploadBytes);
        Assert.Equal(new VorticePrimitiveGraphTokenMetrics(1, 1, 0, 20), session.LastTokenMetrics);
    }

    [Theory]
    [InlineData("FP32")]
    [InlineData("FP16")]
    public void TinyPortableRwkv6ProcessorMatchesCpuAcrossTokens(string precision)
    {
        var directory = Environment.GetEnvironmentVariable("RWKV_TEST_MODEL_DIRECTORY");
        if (string.IsNullOrWhiteSpace(directory)) return;
        var path = Path.Combine(directory, $"tiny-rwkv-6v0-3m-{precision}.bin");
        if (!File.Exists(path)) return;
        using var device = TryCreateDevice();
        if (device is null) return;
        LogicalGraph graph;
        using (var catalog = GgmlModelFile.Open(path))
            graph = new PortableRwkv6GraphProvider().Build(catalog);
        var options = new GraphOptimizationOptions(OptimizationBoundary.Off,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        using var cpu = PrimitiveGraphReference.LoadGraph(path, graph, CpuPrimitiveGraphBackend.Instance, options);
        using var backend = new VorticePrimitiveGraphBackend(device);
        using var gpu = PrimitiveGraphReference.LoadGraph(path, graph, backend, options);
        using var cpuSession = cpu.CreateSession();
        using var gpuSession = gpu.CreateSession();
        foreach (var token in new[] { 2, 4 })
        {
            var expected = cpuSession.ForwardToken(token).ToArray();
            var actual = gpuSession.ForwardToken(token).ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (var index = 0; index < actual.Length; index++)
                Assert.True(float.IsFinite(actual[index]) &&
                    MathF.Abs(actual[index] - expected[index]) <= 0.005f * MathF.Max(1f, MathF.Abs(expected[index])),
                    $"token {token} logits[{index}]: expected {expected[index]}, actual {actual[index]}");
            StateSnapshotAssertions.Near(StateSnapshotAssertions.Capture(cpuSession),
                StateSnapshotAssertions.Capture(gpuSession), 0.005f, 0.005f,
                $"token {token}", scaleAtLeastOne: true);
        }
        Assert.Equal((ulong)2, backend.Metrics!.TokenCommandSubmissions);
        Assert.Equal((ulong)0, backend.Metrics.ActivationUploadBytes);
        Assert.True(backend.Metrics.WeightUploadBytes > 0);
    }

    [Fact]
    public void OptionalLargeRwkv6Fp16ModelUploadsResidentWeights()
    {
        var path = Environment.GetEnvironmentVariable("RWKV_TEST_RWKV6_LARGE_FP16_MODEL");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return;
        using var device = TryCreateDevice();
        if (device is null) return;
        LogicalGraph graph;
        using (var catalog = GgmlModelFile.Open(path))
            graph = new PortableRwkv6GraphProvider().Build(catalog);
        using var backend = new VorticePrimitiveGraphBackend(device);
        using var processor = PrimitiveGraphReference.LoadGraph(path, graph, backend,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded));
        Assert.True(backend.Metrics!.WeightUploadBytes > 10UL * 1024 * 1024 * 1024,
            $"Expected >10 GiB of resident FP16 weights, uploaded {backend.Metrics.WeightUploadBytes} bytes.");
        output.WriteLine($"RWKV6 7B model buffers: {backend.Memory!.ModelBufferBytes:N0} bytes; " +
            $"logical cast outputs: {backend.Memory.LogicalCastOutputBytes:N0} bytes; " +
            $"pooled local buffers/session: {backend.Memory.PooledLocalBufferBytesPerSession:N0} bytes.");
        Assert.True(backend.Memory!.LogicalCastOutputBytes > backend.Memory.PooledLocalBufferBytesPerSession,
            $"Expected cast scratch to be reused: logical {backend.Memory.LogicalCastOutputBytes}, " +
            $"pooled {backend.Memory.PooledLocalBufferBytesPerSession} bytes.");
        if (Environment.GetEnvironmentVariable("RWKV_TEST_RWKV6_LARGE_FORWARD") == "1")
        {
            using var session = processor.CreateSession();
            var logits = session.ForwardToken(1).ToArray();
            Assert.Equal(graph.Model.VocabularySize, logits.Length);
            Assert.All(logits, value => Assert.True(float.IsFinite(value)));
            Assert.Equal((ulong)1, backend.Metrics.TokenCommandSubmissions);
            Assert.True(backend.Memory.PeakBufferBytes >= backend.Memory.ModelBufferBytes);
            output.WriteLine($"RWKV6 7B peak committed model+session buffers: " +
                $"{backend.Memory.PeakBufferBytes:N0} bytes.");
        }
    }

    [Fact]
    public void OptionalLargeRwkv7Fp16ModelUsesPooledCastScratch()
    {
        var path = Environment.GetEnvironmentVariable("RWKV_TEST_RWKV7_LARGE_FP16_MODEL");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return;
        using var device = TryCreateDevice();
        if (device is null) return;
        LogicalGraph graph;
        using (var catalog = GgmlModelFile.Open(path))
            graph = new PortableRwkv7GraphProvider().Build(catalog);
        using var backend = new VorticePrimitiveGraphBackend(device);
        using var processor = PrimitiveGraphReference.LoadGraph(path, graph, backend,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded));
        Assert.True(backend.Metrics!.WeightUploadBytes > 10UL * 1024 * 1024 * 1024);
        output.WriteLine($"RWKV7 7.2B model buffers: {backend.Memory!.ModelBufferBytes:N0} bytes; " +
            $"logical cast outputs: {backend.Memory.LogicalCastOutputBytes:N0} bytes; " +
            $"pooled local buffers/session: {backend.Memory.PooledLocalBufferBytesPerSession:N0} bytes.");
        Assert.True(backend.Memory!.LogicalCastOutputBytes > backend.Memory.PooledLocalBufferBytesPerSession,
            $"Expected cast scratch to be reused: logical {backend.Memory.LogicalCastOutputBytes}, " +
            $"pooled {backend.Memory.PooledLocalBufferBytesPerSession} bytes.");
        if (Environment.GetEnvironmentVariable("RWKV_TEST_RWKV7_LARGE_FORWARD") == "1")
        {
            using var session = processor.CreateSession();
            var logits = session.ForwardToken(1).ToArray();
            Assert.Equal(graph.Model.VocabularySize, logits.Length);
            Assert.All(logits, value => Assert.True(float.IsFinite(value)));
            Assert.Equal((ulong)1, backend.Metrics.TokenCommandSubmissions);
            output.WriteLine($"RWKV7 7.2B peak committed model+session buffers: " +
                $"{backend.Memory.PeakBufferBytes:N0} bytes.");
        }
    }

    [Fact]
    public void OptionalLargeRwkv7GpuDecodeForkSnapshotAndWarmThroughput()
    {
        var path = Environment.GetEnvironmentVariable("RWKV_TEST_RWKV7_LARGE_FP16_MODEL");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path) ||
            Environment.GetEnvironmentVariable("RWKV_TEST_RWKV7_LARGE_SEQUENCE") != "1")
            return;
        var modelName = Path.GetFileName(path);
        using var device = TryCreateDevice();
        if (device is null) return;
        LogicalGraph graph;
        using (var catalog = GgmlModelFile.Open(path))
            graph = new PortableRwkv7GraphProvider().Build(catalog);
        var options = new GraphOptimizationOptions(OptimizationBoundary.Off,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        (float[][] Logits, SharpInference.Gguf.GgufState[] States)? reference =
            Environment.GetEnvironmentVariable("RWKV_TEST_RWKV7_LARGE_COMPARE_CPU") == "1"
            ? CpuReference() : null;
        using var backend = new VorticePrimitiveGraphBackend(device);
        using var gpu = PrimitiveGraphReference.LoadGraph(path, graph, backend, options);
        using var original = gpu.CreateSession();
        var first = original.ForwardToken(2).ToArray();
        Assert.All(first, value => Assert.True(float.IsFinite(value)));
        var firstState = StateSnapshotAssertions.Capture(original);
        if (reference is not null)
            CompareCpu(first, firstState, reference.Value.Logits[0], reference.Value.States[0], 2);

        using var snapshot = new MemoryStream();
        original.SaveState(snapshot);
        float[][] continuation = new float[2][];
        var continuationStates = new SharpInference.Gguf.GgufState[2];
        using (var fork = original.Fork())
        {
            StateSnapshotAssertions.Near(firstState, StateSnapshotAssertions.Capture(fork),
                0f, 0f, "fork after first token");
            var timer = new Stopwatch();
            for (var index = 0; index < 2; index++)
            {
                timer.Start();
                continuation[index] = original.ForwardToken(index == 0 ? 4 : 1).ToArray();
                timer.Stop();
                continuationStates[index] = StateSnapshotAssertions.Capture(original);
            }
            for (var index = 0; index < 2; index++)
            {
                var token = index == 0 ? 4 : 1;
                var forkLogits = fork.ForwardToken(token).ToArray();
                AssertLogitsNear(continuation[index], forkLogits, 1e-5f, token, "fork");
                StateSnapshotAssertions.Near(continuationStates[index],
                    StateSnapshotAssertions.Capture(fork), 1e-5f, 1e-5f,
                    $"fork after token {token}", scaleAtLeastOne: true);
                if (reference is not null)
                    CompareCpu(continuation[index], continuationStates[index],
                        reference.Value.Logits[index + 1], reference.Value.States[index + 1], token);
            }
            output.WriteLine($"{modelName} warm two-token decode: {timer.Elapsed.TotalSeconds:F3} s, " +
                $"{2d / timer.Elapsed.TotalSeconds:F3} token/s (excludes model load and warmup).");
        }

        using (var restored = gpu.CreateSession())
        {
            snapshot.Position = 0;
            restored.LoadState(snapshot);
            StateSnapshotAssertions.Near(firstState, StateSnapshotAssertions.Capture(restored),
                0f, 0f, "snapshot restore");
            for (var index = 0; index < 2; index++)
            {
                var token = index == 0 ? 4 : 1;
                AssertLogitsNear(continuation[index], restored.ForwardToken(token).ToArray(),
                    1e-5f, token, "snapshot");
            }
            StateSnapshotAssertions.Near(StateSnapshotAssertions.Capture(original),
                StateSnapshotAssertions.Capture(restored), 1e-5f, 1e-5f,
                "snapshot continuation", scaleAtLeastOne: true);
        }
        Assert.Equal((ulong)7, backend.Metrics!.TokenCommandSubmissions);
        Assert.Equal((ulong)0, backend.Metrics.ActivationUploadBytes);
        Assert.True(backend.Metrics.StateReadbackBytes > 0);
        output.WriteLine($"{modelName} submissions: tokens {backend.Metrics.TokenCount}, " +
            $"token commands {backend.Metrics.TokenCommandSubmissions}, " +
            $"weight commands {backend.Metrics.WeightCommandSubmissions}, " +
            $"state upload commands {backend.Metrics.StateUploadCommandSubmissions}, " +
            $"state readback commands {backend.Metrics.StateReadbackCommandSubmissions}.");
        output.WriteLine($"{modelName} memory: model {backend.Memory!.ModelBufferBytes:N0}, " +
            $"pooled scratch/session {backend.Memory.PooledLocalBufferBytesPerSession:N0}, " +
            $"active session {backend.Memory.ActiveSessionBufferBytes:N0}, " +
            $"logical cast outputs {backend.Memory.LogicalCastOutputBytes:N0}, " +
            $"peak model+sessions {backend.Memory.PeakBufferBytes:N0} bytes; " +
            $"weight uploads {backend.Metrics.WeightUploadBytes:N0}, " +
            $"activation uploads {backend.Metrics.ActivationUploadBytes:N0}, " +
            $"state uploads {backend.Metrics.StateUploadBytes:N0}, " +
            $"output readbacks {backend.Metrics.OutputReadbackBytes:N0}, " +
            $"state readbacks {backend.Metrics.StateReadbackBytes:N0} bytes.");
        if (Environment.GetEnvironmentVariable("RWKV_TEST_RWKV7_LARGE_PROMPT") == "1")
        {
            using var promptSession = gpu.CreateSession();
            var tokenizer = RwkvWorldTokenizer.LoadBundled();
            var prompt = tokenizer.Encode(
                "System: Answer with only the number.\n\nUser: What is 2 + 2?\n\nAssistant:");
            var clock = Stopwatch.StartNew();
            var logits = promptSession.Prefill(prompt.ToArray()).ToArray();
            var prefillSeconds = clock.Elapsed.TotalSeconds;
            var tokens = new List<int>();
            const int maxGenerated = 128;
            clock.Restart();
            while (tokens.Count < maxGenerated)
            {
                var next = tokenizer.TokenIds.Where(id => (uint)id < (uint)logits.Length)
                    .MaxBy(id => logits[id]);
                Assert.True(float.IsFinite(logits[next]));
                tokens.Add(next);
                var text = tokenizer.Decode(tokens);
                if (text.Contains("\n\nUser:", StringComparison.Ordinal) ||
                    text.Contains("\n\nSystem:", StringComparison.Ordinal))
                    break;
                logits = promptSession.ForwardToken(next).ToArray();
            }
            clock.Stop();
            output.WriteLine($"{modelName} prompt tokens {prompt.Count}, prefill {prefillSeconds:F3} s; " +
                $"generated {tokens.Count} tokens in {clock.Elapsed.TotalSeconds:F3} s: " +
                $"{tokenizer.Decode(tokens)}");
            Assert.NotEmpty(tokens);
        }

        (float[][] Logits, SharpInference.Gguf.GgufState[] States) CpuReference()
        {
            var logits = new float[3][];
            var states = new SharpInference.Gguf.GgufState[3];
            using (var cpu = PrimitiveGraphReference.LoadGraph(path, graph, CpuPrimitiveGraphBackend.Instance, options))
            using (var session = cpu.CreateSession())
            {
                for (var index = 0; index < 3; index++)
                {
                    logits[index] = session.ForwardToken(index switch { 0 => 2, 1 => 4, _ => 1 }).ToArray();
                    states[index] = StateSnapshotAssertions.Capture(session);
                }
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            output.WriteLine($"{modelName} CPU reference released before GPU model load.");
            return (logits, states);
        }
    }

    private static void AssertLogitsNear(
        float[] expected, float[] actual, float relative, int token, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < actual.Length; index++)
            Assert.True(float.IsFinite(actual[index]) &&
                MathF.Abs(actual[index] - expected[index]) <= relative * MathF.Max(1f, MathF.Abs(expected[index])),
                $"{label} token {token} logits[{index}]: expected {expected[index]}, actual {actual[index]}");
    }

    private static void CompareCpu(
        float[] gpuLogits, SharpInference.Gguf.GgufState gpuState,
        float[] cpuLogits, SharpInference.Gguf.GgufState cpuState, int token)
    {
        AssertLogitsNear(cpuLogits, gpuLogits, 0.005f, token, "CPU/GPU");
        StateSnapshotAssertions.Near(cpuState, gpuState, 0.005f, 0.005f,
            $"CPU/GPU token {token}", scaleAtLeastOne: true);
    }

    [Fact]
    public void FusedFp32ExpressionPreservesOrderedOperandsAndExceptionalValuesInOneDispatch()
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        var logical = FusedExpressionGraph();
        using var backend = new VorticePrimitiveGraphBackend(device);
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
        Assert.Equal(3, FusedElementwiseExpressionContract.Read(on.Nodes[0]).InputCount);
        Assert.Single(backend.GetOperatorImplementations(on, on.Nodes[0]));
        using var unfused = new VorticePrimitiveGraphExecutor(device, off);
        using var fused = new VorticePrimitiveGraphExecutor(device, on);
        using var offSession = unfused.CreateSession();
        using var onSession = fused.CreateSession();
        var inputs = new Dictionary<ResourceId, float[]>
        {
            [Id("a")] = [7f, float.NaN, float.PositiveInfinity, -0f, 2f],
            [Id("b")] = [2f, 1f, 3f, +0f, 3f],
            [Id("c")] = [2f, 1f, 2f, 1f, 0f],
        };
        var expected = offSession.ExecuteToken(inputs)[Id("result")];
        var actual = onSession.ExecuteToken(inputs)[Id("result")];
        for (var index = 0; index < expected.Length; index++)
            Assert.Equal(BitConverter.SingleToUInt32Bits(expected[index]),
                BitConverter.SingleToUInt32Bits(actual[index]));
        Assert.Equal(4, offSession.LastTokenMetrics!.NodeDispatches);
        Assert.Equal(1, onSession.LastTokenMetrics!.NodeDispatches);
        Assert.Equal(1, onSession.LastTokenMetrics.CommandSubmissions);
        Assert.Empty(((IFusedElementwiseExpressionProvider)backend.KernelCatalog)
            .GetExpressionImplementations(
                new OperatorSignature(Enumerable.Repeat(GraphElementType.Float32, 9),
                    [GraphElementType.Float32]),
                new TensorDescriptor(GraphElementType.Float32, [5])));
        var invalid = new ExecutionGraph(on.Identity, on.Model,
            on.Resources.Select(resource => resource.Id == Id("c")
                ? resource with { Tensor = new TensorDescriptor(GraphElementType.Float32, [4]) }
                : resource), on.Regions, on.Nodes, on.Inputs, on.Outputs, on.GraphState);
        var failure = Assert.IsType<BackendPreparationResult.Failure>(backend.Prepare(invalid));
        Assert.Contains(failure.Diagnostics, diagnostic =>
            diagnostic.Message.Contains("identical dense FP32", StringComparison.Ordinal));

        var oversized = new FusedElementwiseExpression(3,
            Enumerable.Range(0, 33).Select(index => new ElementwiseInstruction(
                PrimitiveGraphOperations.Copy,
                [index == 0 ? new ElementwiseOperand(InputIndex: 0) :
                    new ElementwiseOperand(StepIndex: index - 1)])).ToArray());
        var tooLarge = new ExecutionGraph(on.Identity, on.Model, on.Resources, on.Regions,
            [on.Nodes[0] with { Attributes = FusedElementwiseExpressionContract.ToAttributes(oversized) }],
            on.Inputs, on.Outputs, on.GraphState);
        Assert.Contains("at most", Assert.IsType<BackendPreparationResult.Failure>(
            backend.Prepare(tooLarge)).Diagnostics.Single().Message);
        var malformed = new ExecutionGraph(on.Identity, on.Model, on.Resources, on.Regions,
            [on.Nodes[0] with
            {
                Attributes = new Dictionary<string, string>
                    { [FusedElementwiseExpressionContract.Attribute] = "{\"InputCount\":0}" },
            }], on.Inputs, on.Outputs, on.GraphState);
        Assert.Contains("fused expression", Assert.IsType<BackendPreparationResult.Failure>(
            backend.Prepare(malformed)).Diagnostics.Single().Message,
            StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("Copy")]
    [InlineData("Add")]
    [InlineData("Subtract")]
    [InlineData("Multiply")]
    [InlineData("Divide")]
    [InlineData("Maximum")]
    [InlineData("Exp")]
    [InlineData("Tanh")]
    [InlineData("Sigmoid")]
    [InlineData("ReciprocalSquareRoot")]
    [InlineData("Square")]
    [InlineData("Relu")]
    public void FusedFp32StepMatchesUnfusedForNaNAndOrderedOperands(string operation)
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        var op = operation switch
        {
            "Copy" => PrimitiveGraphOperations.Copy,
            "Add" => PrimitiveGraphOperations.Add,
            "Subtract" => PrimitiveGraphOperations.Subtract,
            "Multiply" => PrimitiveGraphOperations.Multiply,
            "Divide" => PrimitiveGraphOperations.Divide,
            "Maximum" => PrimitiveGraphOperations.Maximum,
            "Exp" => PrimitiveGraphOperations.Exp,
            "Tanh" => PrimitiveGraphOperations.Tanh,
            "Sigmoid" => PrimitiveGraphOperations.Sigmoid,
            "ReciprocalSquareRoot" => PrimitiveGraphOperations.ReciprocalSquareRoot,
            "Square" => PrimitiveGraphOperations.Square,
            "Relu" => PrimitiveGraphOperations.Relu,
            _ => throw new ArgumentOutOfRangeException(nameof(operation)),
        };
        var logical = FusedOperationGraph(op);
        using var backend = new VorticePrimitiveGraphBackend(device);
        var optimizer = new GraphOptimizer();
        var off = optimizer.Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), backend.KernelCatalog);
        var on = optimizer.Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), backend.KernelCatalog);
        Assert.Equal(2, off.Nodes.Count);
        Assert.Single(on.Nodes);
        using var unfused = new VorticePrimitiveGraphExecutor(device, off);
        using var fused = new VorticePrimitiveGraphExecutor(device, on);
        using var offSession = unfused.CreateSession();
        using var onSession = fused.CreateSession();
        var inputs = new Dictionary<ResourceId, float[]>
        {
            [Id("a")] = [float.NaN, float.PositiveInfinity, -0f, -1f, 4f],
            [Id("b")] = [1f, -2f, +0f, 3f, float.NaN],
        };
        var expected = offSession.ExecuteToken(inputs)[Id("result")];
        var actual = onSession.ExecuteToken(inputs)[Id("result")];
        for (var index = 0; index < expected.Length; index++)
            Assert.Equal(BitConverter.SingleToUInt32Bits(expected[index]),
                BitConverter.SingleToUInt32Bits(actual[index]));
        Assert.Equal(2, offSession.LastTokenMetrics!.NodeDispatches);
        Assert.Equal(1, onSession.LastTokenMetrics!.NodeDispatches);
    }

    [Fact]
    public void FusedFp32BranchedStepReferencesKeepEarlierResults()
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        var expression = new FusedElementwiseExpression(2,
        [
            new ElementwiseInstruction(PrimitiveGraphOperations.Add,
                [new ElementwiseOperand(InputIndex: 0), new ElementwiseOperand(InputIndex: 1)]),
            new ElementwiseInstruction(PrimitiveGraphOperations.Square,
                [new ElementwiseOperand(StepIndex: 0)]),
            new ElementwiseInstruction(PrimitiveGraphOperations.Add,
                [new ElementwiseOperand(StepIndex: 1), new ElementwiseOperand(StepIndex: 0)]),
        ]);
        var graph = new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "branched-gpu"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("a", "a", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("b", "b", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddNode("expression", FusedElementwiseExpressionContract.Operation, "root",
                [GraphBindings.Read("input0", "a"), GraphBindings.Read("input1", "b"),
                    GraphBindings.Write("output", "result")],
                attributes: FusedElementwiseExpressionContract.ToAttributes(expression))
            .Build();
        using var backend = new VorticePrimitiveGraphBackend(device);
        var execution = new GraphOptimizer().Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), backend.KernelCatalog);
        using var executor = new VorticePrimitiveGraphExecutor(device, execution);
        using var session = executor.CreateSession();
        Assert.Equal([42f, 72f], session.ExecuteToken(
            new Dictionary<ResourceId, float[]>
            {
                [Id("a")] = [2f, 3f],
                [Id("b")] = [4f, 5f],
            })[Id("result")]);
        Assert.Equal(1, session.LastTokenMetrics!.NodeDispatches);
    }

    [Theory]
    [InlineData(false, 2)]
    [InlineData(true, 4)]
    public void ReshapeAliasesOnlyDeadLocalSource(bool readSourceAfterReshape, int dispatches)
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        var builder = new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "reshape-view"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("input", "input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2, 3]), graphInput: true)
            .AddResource("temp", "temp", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [2, 3]))
            .AddResource("flat", "flat", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [3, 2]))
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [3, 2]), graphOutput: true)
            .AddNode("copy", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "temp")])
            .AddNode("reshape", PortableTensorOperationContracts.Reshape, "root",
                [GraphBindings.Read("input", "temp"), GraphBindings.Write("output", "flat")],
                ["copy"])
            .AddNode("result", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "flat"), GraphBindings.Write("output", "result")],
                ["reshape"]);
        if (readSourceAfterReshape)
            builder.AddResource("other", "other", GraphResourceKind.Output,
                    GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Float32, [2, 3]),
                    graphOutput: true)
                .AddNode("other", PrimitiveGraphOperations.Copy, "root",
                    [GraphBindings.Read("input", "temp"), GraphBindings.Write("output", "other")],
                    ["result"]);
        using var backend = new VorticePrimitiveGraphBackend(device);
        var execution = new GraphOptimizer().Optimize(builder.Build(),
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), backend.KernelCatalog);
        using var executor = new VorticePrimitiveGraphExecutor(device, execution);
        Assert.Equal(dispatches, executor.DispatchesPerToken);
        using var session = executor.CreateSession();
        float[] values = [1f, 2f, 3f, 4f, 5f, 6f];
        var outputs = session.ExecuteToken(new Dictionary<ResourceId, float[]> { [Id("input")] = values });
        Assert.Equal(values, outputs[Id("result")]);
        if (readSourceAfterReshape) Assert.Equal(values, outputs[Id("other")]);
        Assert.Equal(dispatches, session.LastTokenMetrics!.NodeDispatches);
    }

    private static LogicalGraph FusedOperationGraph(GraphOperationId operation)
    {
        var binary = FusedElementwiseExpressionContract.Arity(operation) == 2;
        return new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "gpu-step"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("a", "a", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [5]), graphInput: true)
            .AddResource("b", "b", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [5]), graphInput: true)
            .AddResource("temp", "temp", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [5]))
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [5]), graphOutput: true)
            .AddNode("copy", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "a"), GraphBindings.Write("output", "temp")])
            .AddNode("step", operation, "root",
                binary
                    ? [GraphBindings.Read("left", "temp"), GraphBindings.Read("right", "b"),
                        GraphBindings.Write("output", "result")]
                    : [GraphBindings.Read("input", "temp"), GraphBindings.Write("output", "result")],
                ["copy"])
            .Build();
    }

    [Theory]
    [InlineData(false, "FP32")]
    [InlineData(false, "FP16")]
    [InlineData(true, "FP32")]
    [InlineData(true, "FP16")]
    public void TinyPortableGpuExpressionOnMatchesOffAndReducesDispatches(bool rwkv7, string precision)
    {
        var directory = Environment.GetEnvironmentVariable("RWKV_TEST_MODEL_DIRECTORY");
        if (string.IsNullOrWhiteSpace(directory)) return;
        var path = Path.Combine(directory, rwkv7
            ? $"tiny-rwkv-7v0-834K-{precision}.bin"
            : $"tiny-rwkv-6v0-3m-{precision}.bin");
        if (!File.Exists(path)) return;
        using var device = TryCreateDevice();
        if (device is null) return;
        ILogicalGraphProvider provider = rwkv7
            ? new PortableRwkv7GraphProvider() : new PortableRwkv6GraphProvider();
        LogicalGraph graph;
        using (var catalog = GgmlModelFile.Open(path))
            graph = provider.Build(catalog);
        var disabled = new GraphOptimizationOptions(OptimizationBoundary.Off,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        var enabled = new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        using var offBackend = new VorticePrimitiveGraphBackend(device);
        using var onBackend = new VorticePrimitiveGraphBackend(device);
        var optimizer = new GraphOptimizer();
        var offGraph = optimizer.Optimize(graph, disabled, offBackend.KernelCatalog);
        var onGraph = optimizer.Optimize(graph, enabled, onBackend.KernelCatalog);
        var offDispatches = VorticePrimitiveGraphPlan.Compile(offGraph).Steps.Count;
        var onDispatches = VorticePrimitiveGraphPlan.Compile(onGraph).Steps.Count;
        Assert.True(onDispatches < offDispatches,
            $"Expected expression fusion to reduce dispatches: Off={offDispatches}, On={onDispatches}.");
        Assert.Contains(onGraph.Nodes, node =>
            node.Operation == FusedElementwiseExpressionContract.Operation);
        using var offProcessor = PrimitiveGraphReference.LoadGraph(path, graph, offBackend, disabled);
        using var onProcessor = PrimitiveGraphReference.LoadGraph(path, graph, onBackend, enabled);
        using var off = offProcessor.CreateSession();
        using var on = onProcessor.CreateSession();
        foreach (var token in new[] { 2, 4, 1 })
        {
            AssertLogitsNear(off.ForwardToken(token).ToArray(),
                on.ForwardToken(token).ToArray(), 0.0001f, token, "GPU On/Off");
            StateSnapshotAssertions.Near(StateSnapshotAssertions.Capture(off),
                StateSnapshotAssertions.Capture(on), 0.0001f, 0.0001f,
                $"GPU On/Off token {token}", scaleAtLeastOne: true);
        }
        static (double Milliseconds, float[] LastLogits) Measure(
            PrimitiveGraphReferenceSession session, IReadOnlyList<int> tokens)
        {
            var clock = Stopwatch.StartNew();
            float[] logits = [];
            foreach (var token in tokens)
                logits = session.ForwardToken(token).ToArray();
            clock.Stop();
            return (clock.Elapsed.TotalMilliseconds, logits);
        }
        int[] warmTokens = [2, 4, 1, 2, 4, 1, 2, 4];
        var offTime = Measure(off, warmTokens);
        var onTime = Measure(on, warmTokens);
        AssertLogitsNear(offTime.LastLogits, onTime.LastLogits, 0.0001f,
            warmTokens[^1], "GPU warmed On/Off");
        StateSnapshotAssertions.Near(StateSnapshotAssertions.Capture(off),
            StateSnapshotAssertions.Capture(on), 0.0001f, 0.0001f,
            "GPU warmed On/Off", scaleAtLeastOne: true);
        var offRecorded = offBackend.DispatchesPerToken!.Value;
        var onRecorded = onBackend.DispatchesPerToken!.Value;
        Assert.True(onRecorded < offRecorded);
        output.WriteLine($"{(rwkv7 ? "RWKV7" : "RWKV6")} {precision} GPU dispatches/token: " +
            $"Off={offRecorded}, On={onRecorded}; reduction " +
            $"{100d * (offRecorded - onRecorded) / offRecorded:F1}%; " +
            $"8 warmed tokens: Off={offTime.Milliseconds:F3}ms, On={onTime.Milliseconds:F3}ms.");
        Assert.Equal((ulong)11, offBackend.Metrics!.TokenCommandSubmissions);
        Assert.Equal((ulong)11, onBackend.Metrics!.TokenCommandSubmissions);
    }

    [Theory]
    [InlineData(false, "FP32")]
    [InlineData(false, "FP16")]
    [InlineData(true, "FP32")]
    [InlineData(true, "FP16")]
    public void TinyPortableGpuReplayMatchesRecordedAfterForkResetAndRestore(
        bool rwkv7, string precision)
    {
        var directory = Environment.GetEnvironmentVariable("RWKV_TEST_MODEL_DIRECTORY");
        if (string.IsNullOrWhiteSpace(directory)) return;
        var path = Path.Combine(directory, rwkv7
            ? $"tiny-rwkv-7v0-834K-{precision}.bin"
            : $"tiny-rwkv-6v0-3m-{precision}.bin");
        if (!File.Exists(path)) return;
        using var device = TryCreateDevice();
        if (device is null) return;
        ILogicalGraphProvider provider = rwkv7
            ? new PortableRwkv7GraphProvider() : new PortableRwkv6GraphProvider();
        LogicalGraph graph;
        using (var catalog = GgmlModelFile.Open(path))
            graph = provider.Build(catalog);
        var options = new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        using var recordedBackend = new VorticePrimitiveGraphBackend(device);
        using var replayBackend = new VorticePrimitiveGraphBackend(device)
        {
            EnableCommandReplay = true,
        };
        using var recordedProcessor = PrimitiveGraphReference.LoadGraph(path, graph, recordedBackend, options);
        using var replayProcessor = PrimitiveGraphReference.LoadGraph(path, graph, replayBackend, options);
        using var recorded = recordedProcessor.CreateSession();
        using var replay = replayProcessor.CreateSession();

        void Compare(PrimitiveGraphReferenceSession expected, PrimitiveGraphReferenceSession actual, int token)
        {
            AssertLogitsNear(expected.ForwardToken(token).ToArray(),
                actual.ForwardToken(token).ToArray(), 0.0001f, token, "recorded/replayed");
            StateSnapshotAssertions.Near(StateSnapshotAssertions.Capture(expected),
                StateSnapshotAssertions.Capture(actual), 0.0001f, 0.0001f,
                $"recorded/replayed token {token}", scaleAtLeastOne: true);
        }

        foreach (var token in new[] { 2, 4, 1 }) Compare(recorded, replay, token);
        using (var recordedFork = recorded.Fork())
        using (var replayFork = replay.Fork())
        {
            Compare(recorded, replay, 2);
            Compare(recordedFork, replayFork, 2);
        }
        using var recordedSnapshot = new MemoryStream();
        using var replaySnapshot = new MemoryStream();
        recorded.SaveState(recordedSnapshot);
        replay.SaveState(replaySnapshot);
        recorded.Reset();
        replay.Reset();
        Compare(recorded, replay, 4);
        recordedSnapshot.Position = 0;
        replaySnapshot.Position = 0;
        recorded.LoadState(recordedSnapshot);
        replay.LoadState(replaySnapshot);
        Compare(recorded, replay, 1);
        Assert.Equal((ulong)2, replayBackend.Metrics!.ReplayListRecordings);
        Assert.Equal(replayBackend.Metrics.ReplayedTokens,
            replayBackend.Metrics.TokenCommandSubmissions);
        Assert.Equal(replayBackend.Metrics.ReplayedTokens * sizeof(uint),
            replayBackend.Metrics.ScalarUploadBytes);
        Assert.Equal((ulong)0, replayBackend.Metrics.ActivationUploadBytes);
        output.WriteLine($"{(rwkv7 ? "RWKV7" : "RWKV6")} {precision} replay: " +
            $"{replayBackend.Metrics.ReplayedTokens} tokens, " +
            $"{replayBackend.Metrics.ReplayListRecordings} per-session list recordings, " +
            $"{replayBackend.DispatchesPerToken} GPU dispatches/token.");
    }

    [Fact]
    public void TinyRwkv6Fp32GpuOptimizationMatchesCpuAndUnfusedGpu()
    {
        var directory = Environment.GetEnvironmentVariable("RWKV_TEST_MODEL_DIRECTORY");
        if (string.IsNullOrWhiteSpace(directory)) return;
        var path = Path.Combine(directory, "tiny-rwkv-6v0-3m-FP32.bin");
        if (!File.Exists(path)) return;
        using var device = TryCreateDevice();
        if (device is null) return;
        var config = new VorticeRuntimeConfig
        {
            AdapterIndex = 0,
        };
        LogicalGraph graph;
        using (var catalog = GgmlModelFile.Open(path))
            graph = new PortableRwkv6GraphProvider().Build(catalog);
        var offOptions = new GraphOptimizationOptions(OptimizationBoundary.Off,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        var onOptions = new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        using var offBackend = VorticePrimitiveGraphBackend.FromConfig(config);
        using var onBackend = VorticePrimitiveGraphBackend.FromConfig(config);
        using var offProcessor = PrimitiveGraphReference.LoadGraph(path, graph, offBackend, offOptions);
        using var onProcessor = PrimitiveGraphReference.LoadGraph(path, graph, onBackend, onOptions);
        using var cpuProcessor = PrimitiveGraphReference.LoadGraph(path, graph, CpuPrimitiveGraphBackend.Instance, onOptions);
        using var off = offProcessor.CreateSession();
        using var on = onProcessor.CreateSession();
        using var cpu = cpuProcessor.CreateSession();
        foreach (var token in new[] { 2, 4, 1 })
        {
            var reference = cpu.ForwardToken(token).ToArray();
            AssertLogitsNear(reference, off.ForwardToken(token).ToArray(), 0.005f,
                token, "CPU/GPU Off");
            AssertLogitsNear(reference, on.ForwardToken(token).ToArray(), 0.005f,
                token, "CPU/GPU On");
            var cpuState = StateSnapshotAssertions.Capture(cpu);
            StateSnapshotAssertions.Near(cpuState, StateSnapshotAssertions.Capture(off),
                0.005f, 0.005f, $"CPU/GPU Off token {token}", scaleAtLeastOne: true);
            StateSnapshotAssertions.Near(cpuState, StateSnapshotAssertions.Capture(on),
                0.005f, 0.005f, $"CPU/GPU On token {token}", scaleAtLeastOne: true);
        }
        int[] warmTokens = [2, 4, 1, 2, 4, 1, 2, 4];
        static (double Milliseconds, float[] LastLogits) Measure(PrimitiveGraphReferenceSession session, int[] tokens)
        {
            var clock = Stopwatch.StartNew();
            float[] last = [];
            foreach (var token in tokens)
                last = session.ForwardToken(token).ToArray();
            clock.Stop();
            return (clock.Elapsed.TotalMilliseconds, last);
        }
        var cpuTime = Measure(cpu, warmTokens);
        var offTime = Measure(off, warmTokens);
        var onTime = Measure(on, warmTokens);
        AssertLogitsNear(cpuTime.LastLogits, offTime.LastLogits,
            0.005f, warmTokens[^1], "warmed CPU/GPU Off");
        AssertLogitsNear(cpuTime.LastLogits, onTime.LastLogits,
            0.005f, warmTokens[^1], "warmed CPU/GPU On");
        var cpuStateAfter = StateSnapshotAssertions.Capture(cpu);
        StateSnapshotAssertions.Near(cpuStateAfter, StateSnapshotAssertions.Capture(off),
            0.005f, 0.005f, "warmed CPU/GPU Off", scaleAtLeastOne: true);
        StateSnapshotAssertions.Near(cpuStateAfter, StateSnapshotAssertions.Capture(on),
            0.005f, 0.005f, "warmed CPU/GPU On", scaleAtLeastOne: true);
        var optimizer = new GraphOptimizer();
        var offDispatches = VorticePrimitiveGraphPlan.Compile(
            optimizer.Optimize(graph, offOptions, offBackend.KernelCatalog)).Steps.Count;
        var onPlan = VorticePrimitiveGraphPlan.Compile(
            optimizer.Optimize(graph, onOptions, onBackend.KernelCatalog));
        var onDispatches = onPlan.Steps.Count;
        Assert.True(onDispatches < offDispatches);
        Assert.Equal((ulong)11, onBackend.Metrics!.TokenCommandSubmissions);
        Assert.Equal((ulong)11, offBackend.Metrics!.TokenCommandSubmissions);
        output.WriteLine($"RWKV6 tiny FP32 ({onBackend.DeviceName}) 8 warmed tokens: " +
            $"CPU {cpuTime.Milliseconds:F3}ms " +
            $"({8000d / cpuTime.Milliseconds:F2} tokens/s), " +
            $"portable Off {offTime.Milliseconds:F3}ms ({8000d / offTime.Milliseconds:F2} tokens/s), " +
            $"portable fused On {onTime.Milliseconds:F3}ms ({8000d / onTime.Milliseconds:F2} tokens/s); " +
            $"portable dispatches/token Off={offBackend.DispatchesPerToken}, " +
            $"On={onBackend.DispatchesPerToken} " +
            $"(skipped reshapes: Off={offDispatches - offBackend.DispatchesPerToken}, " +
            $"On={onDispatches - onBackend.DispatchesPerToken}).");
        output.WriteLine("Portable On kernels/token: " +
            string.Join(", ", onPlan.Steps.GroupBy(step => step.Kernel)
                .OrderByDescending(group => group.Count())
                .Select(group => $"{group.Key}={group.Count()}")));
        var descriptors = onPlan.Graph.Resources.ToDictionary(item => item.Id);
        output.WriteLine("Portable reshape source/output kinds: " +
            string.Join(", ", onPlan.Steps.Where(step => step.Kernel == "PortableCopy")
                .GroupBy(step => $"{descriptors[step.Input0].Kind}/{descriptors[step.Input0].Scope}" +
                    $" → {descriptors[step.Output].Kind}/{descriptors[step.Output].Scope}")
                .OrderByDescending(group => group.Count())
                .Select(group => $"{group.Key}={group.Count()}")));
        onBackend.ProfileCommandRecording = true;
        _ = on.ForwardToken(2);
        var profile = Assert.IsType<VorticePrimitiveGraphExecutionProfile>(
            onBackend.LastExecutionProfile);
        Assert.Equal(onBackend.DispatchesPerToken,
            profile.Kernels.Sum(kernel => kernel.Dispatches));
        output.WriteLine($"Portable On CPU command recording {profile.CpuRecordingMilliseconds:F3}ms, " +
            $"submission + fence wait {profile.SubmissionAndFenceMilliseconds:F3}ms; " +
            "top kernel CPU recording (not GPU timestamps): " +
            string.Join(", ", profile.Kernels.Take(10)
                .Select(kernel => $"{kernel.Kernel}={kernel.Dispatches}/" +
                    $"{kernel.CpuRecordingMilliseconds:F3}ms")));
    }

    [Fact]
    public void TinyRwkv6Fp32ReplayThreeWarmWindowsMatchesRecordedAndCpu()
    {
        var directory = Environment.GetEnvironmentVariable("RWKV_TEST_MODEL_DIRECTORY");
        if (string.IsNullOrWhiteSpace(directory)) return;
        var path = Path.Combine(directory, "tiny-rwkv-6v0-3m-FP32.bin");
        if (!File.Exists(path)) return;
        using var device = TryCreateDevice();
        if (device is null) return;
        var config = new VorticeRuntimeConfig
        {
            AdapterIndex = 0,
        };
        var replayConfig = new VorticeRuntimeConfig
        {
            AdapterIndex = config.AdapterIndex,
            EnableCommandReplay = true,
        };
        LogicalGraph graph;
        using (var catalog = GgmlModelFile.Open(path))
            graph = new PortableRwkv6GraphProvider().Build(catalog);
        var options = new GraphOptimizationOptions(OptimizationBoundary.Unrestricted,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        using var recordedBackend = VorticePrimitiveGraphBackend.FromConfig(config);
        using var replayBackend = VorticePrimitiveGraphBackend.FromConfig(replayConfig);
        Assert.True(replayBackend.EnableCommandReplay);
        using var recordedProcessor = PrimitiveGraphReference.LoadGraph(path, graph, recordedBackend, options);
        using var replayProcessor = PrimitiveGraphReference.LoadGraph(path, graph, replayBackend, options);
        using var cpuProcessor = PrimitiveGraphReference.LoadGraph(path, graph, CpuPrimitiveGraphBackend.Instance, options);
        using var recorded = recordedProcessor.CreateSession();
        using var replay = replayProcessor.CreateSession();
        using var cpu = cpuProcessor.CreateSession();

        void AssertParity(float[] expected, float[] actual, PrimitiveGraphReferenceSession session, string label)
        {
            AssertLogitsNear(expected, actual, 0.005f, 4, label);
            StateSnapshotAssertions.Near(StateSnapshotAssertions.Capture(cpu),
                StateSnapshotAssertions.Capture(session), 0.005f, 0.005f,
                label, scaleAtLeastOne: true);
        }

        foreach (var token in new[] { 2, 4, 1 })
        {
            var expected = cpu.ForwardToken(token).ToArray();
            AssertParity(expected, recorded.ForwardToken(token).ToArray(), recorded, "recorded warmup");
            AssertParity(expected, replay.ForwardToken(token).ToArray(), replay, "replay warmup");
        }
        int[] tokens = [2, 4, 1, 2, 4, 1, 2, 4];
        static (double Milliseconds, float[] LastLogits) Measure(PrimitiveGraphReferenceSession session, int[] sequence)
        {
            var timer = Stopwatch.StartNew();
            float[] logits = [];
            foreach (var token in sequence)
                logits = session.ForwardToken(token).ToArray();
            timer.Stop();
            return (timer.Elapsed.TotalMilliseconds, logits);
        }
        var cpuTimes = new double[3];
        var recordedTimes = new double[3];
        var replayTimes = new double[3];
        for (var round = 0; round < 3; round++)
        {
            var baselineCpu = Measure(cpu, tokens);
            var baseline = Measure(recorded, tokens);
            var cached = Measure(replay, tokens);
            cpuTimes[round] = baselineCpu.Milliseconds;
            recordedTimes[round] = baseline.Milliseconds;
            replayTimes[round] = cached.Milliseconds;
            AssertParity(baselineCpu.LastLogits, baseline.LastLogits, recorded,
                $"recorded window {round}");
            AssertParity(baselineCpu.LastLogits, cached.LastLogits, replay,
                $"replay window {round}");
        }
        recordedBackend.ProfileCommandRecording = true;
        replayBackend.ProfileCommandRecording = true;
        var finalLogits = cpu.ForwardToken(2).ToArray();
        AssertParity(finalLogits, recorded.ForwardToken(2).ToArray(), recorded, "recorded profile");
        AssertParity(finalLogits, replay.ForwardToken(2).ToArray(), replay, "replay profile");
        var recordedProfile = Assert.IsType<VorticePrimitiveGraphExecutionProfile>(
            recordedBackend.LastExecutionProfile);
        var replayProfile = Assert.IsType<VorticePrimitiveGraphExecutionProfile>(
            replayBackend.LastExecutionProfile);
        Assert.Equal(0d, replayProfile.CpuRecordingMilliseconds);
        Assert.Equal((ulong)1, replayBackend.Metrics!.ReplayListRecordings);
        Assert.Equal((ulong)28, replayBackend.Metrics.ReplayedTokens);
        Assert.Equal((ulong)28, replayBackend.Metrics.TokenCommandSubmissions);
        Assert.Equal((ulong)28, recordedBackend.Metrics!.TokenCommandSubmissions);
        Assert.Equal((ulong)112, replayBackend.Metrics.ScalarUploadBytes);
        Assert.Equal((ulong)0, replayBackend.Metrics.ActivationUploadBytes);
        static double Median(double[] samples) => samples.Order().ElementAt(1);
        output.WriteLine($"RWKV6 tiny FP32 ({recordedBackend.DeviceName}) 3x8 warmed tokens: " +
            $"CPU={string.Join("/", cpuTimes.Select(value => value.ToString("F3")))}ms " +
            $"(median {8000d / Median(cpuTimes):F2} tokens/s), " +
            $"portable recorded={string.Join("/", recordedTimes.Select(value => value.ToString("F3")))}ms " +
            $"(median {8000d / Median(recordedTimes):F2} tokens/s), " +
            $"portable replay={string.Join("/", replayTimes.Select(value => value.ToString("F3")))}ms " +
            $"(median {8000d / Median(replayTimes):F2} tokens/s); " +
            $"dispatches/token {replayBackend.DispatchesPerToken}, " +
            $"token submissions {replayBackend.Metrics.TokenCommandSubmissions}, " +
            $"list recordings {replayBackend.Metrics.ReplayListRecordings}, " +
            $"scalar upload {replayBackend.Metrics.ScalarUploadBytes} bytes.");
        output.WriteLine($"CPU record/fence sample (ms/token): recorded " +
            $"{recordedProfile.CpuRecordingMilliseconds:F3}/" +
            $"{recordedProfile.SubmissionAndFenceMilliseconds:F3}, " +
            $"replay {replayProfile.CpuRecordingMilliseconds:F3}/" +
            $"{replayProfile.SubmissionAndFenceMilliseconds:F3}.");
    }

    private static LogicalGraph FusedExpressionGraph()
    {
        var builder = new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "gpu-expression"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root");
        foreach (var name in new[] { "a", "b", "c" })
            builder.AddResource(name, name, GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [5]), graphInput: true);
        foreach (var name in new[] { "sub", "div", "max" })
            builder.AddResource(name, name, GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [5]));
        return builder
            .AddResource("result", "result", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [5]), graphOutput: true)
            .AddNode("sub", PrimitiveGraphOperations.Subtract, "root",
                [GraphBindings.Read("left", "a"), GraphBindings.Read("right", "b"),
                    GraphBindings.Write("output", "sub")])
            .AddNode("div", PrimitiveGraphOperations.Divide, "root",
                [GraphBindings.Read("left", "sub"), GraphBindings.Read("right", "c"),
                    GraphBindings.Write("output", "div")], ["sub"])
            .AddNode("max", PrimitiveGraphOperations.Maximum, "root",
                [GraphBindings.Read("left", "div"), GraphBindings.Read("right", "b"),
                    GraphBindings.Write("output", "max")], ["div"])
            .AddNode("square", PrimitiveGraphOperations.Square, "root",
                [GraphBindings.Read("input", "max"), GraphBindings.Write("output", "result")], ["max"])
            .Build();
    }

    [Theory]
    [InlineData("FP32")]
    [InlineData("FP16")]
    public void TinyPortableRwkv7ProcessorMatchesCpuAcrossTokens(string precision)
    {
        var directory = Environment.GetEnvironmentVariable("RWKV_TEST_MODEL_DIRECTORY");
        if (string.IsNullOrWhiteSpace(directory)) return;
        var path = Path.Combine(directory, $"tiny-rwkv-7v0-834K-{precision}.bin");
        if (!File.Exists(path)) return;
        using var device = TryCreateDevice();
        if (device is null) return;
        LogicalGraph graph;
        using (var catalog = GgmlModelFile.Open(path))
            graph = new PortableRwkv7GraphProvider().Build(catalog);
        var options = new GraphOptimizationOptions(OptimizationBoundary.Off,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded);
        using var cpu = PrimitiveGraphReference.LoadGraph(path, graph, CpuPrimitiveGraphBackend.Instance, options);
        using var backend = new VorticePrimitiveGraphBackend(device);
        using var gpu = PrimitiveGraphReference.LoadGraph(path, graph, backend, options);
        using var cpuSession = cpu.CreateSession();
        using var gpuSession = gpu.CreateSession();
        foreach (var token in new[] { 2, 4 })
        {
            var expected = cpuSession.ForwardToken(token).ToArray();
            var actual = gpuSession.ForwardToken(token).ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (var index = 0; index < actual.Length; index++)
                Assert.True(float.IsFinite(actual[index]) &&
                    MathF.Abs(actual[index] - expected[index]) <= 0.005f * MathF.Max(1f, MathF.Abs(expected[index])),
                    $"token {token} logits[{index}]: expected {expected[index]}, actual {actual[index]}");
            StateSnapshotAssertions.Near(StateSnapshotAssertions.Capture(cpuSession),
                StateSnapshotAssertions.Capture(gpuSession), 0.005f, 0.005f,
                $"token {token}", scaleAtLeastOne: true);
        }
        Assert.Equal((ulong)2, backend.Metrics!.TokenCommandSubmissions);
    }

    [Fact]
    public void HeadOuterUsesResidentGpuWeightAndSessionState()
    {
        using var device = TryCreateDevice();
        if (device is null) return;
        var original = SingleNodeGraph(PortableTensorOperationContracts.HeadOuter,
            [[2, 2], [2, 3]], [2, 2, 3]);
        var graph = new ExecutionGraph(original.Identity, original.Model,
            original.Resources.Select(resource => resource.Id.Value switch
            {
                "input0" => resource with
                {
                    Kind = GraphResourceKind.SessionState,
                    Lifetime = GraphResourceLifetime.Session,
                },
                "input1" => resource with
                {
                    Kind = GraphResourceKind.Weight,
                    Lifetime = GraphResourceLifetime.Model,
                    BindingKey = "head.weight",
                },
                _ => resource,
            }),
            original.Regions, original.Nodes, [], original.Outputs,
            [new GraphStateEntry("head", Id("input0"))]);
        using var executor = new VorticePrimitiveGraphExecutor(device, graph);
        var weights = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        executor.UploadWeight(Id("input1"), weights);
        Array.Clear(weights);
        using var session = executor.CreateSession();
        session.UploadState(Id("input0"), [1f, 2f, 3f, 4f]);
        Assert.Equal([1f, 2f, 3f, 2f, 4f, 6f, 12f, 15f, 18f, 16f, 20f, 24f],
            session.ExecuteToken(new Dictionary<ResourceId, float[]>())[Id("output")]);
        Assert.Equal([1f, 2f, 3f, 4f], session.ReadState(Id("input0")));
    }

    private static ExecutionGraph SingleNodeGraph(GraphOperationId operation,
        int[][] inputDims, int[] outputDims, IReadOnlyDictionary<string, string>? attributes = null)
    {
        var contract = PortableTensorOperationContracts.Contracts.SingleOrDefault(item => item.Operation == operation);
        var ports = contract?.InputPorts ??
            Enumerable.Repeat("input", inputDims.Length).ToArray();
        var resources = inputDims.Select((dims, index) =>
            new GraphResource(Id($"input{index}"), $"input{index}", GraphResourceKind.Input,
                GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Float32, dims))).ToList();
        resources.Add(new GraphResource(Id("output"), "output", GraphResourceKind.Output,
            GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Float32, outputDims)));
        var bindings = ports.Select((port, index) => GraphBindings.Read(port, $"input{index}"))
            .Append(GraphBindings.Write("output", "output")).ToArray();
        return new ExecutionGraph(new GraphIdentity("tensor-test", 1, "tensor"),
            new GraphModelSignature(2, 2, 1, 1, 2, "state"),
            resources, [new GraphRegion(new RegionId("root"), null, GraphRegionTypes.Graph, null,
                "root", new Dictionary<string, string>())],
            [new ExecutionNode(new ExecutionNodeId("tensor"), operation, new RegionId("root"),
                bindings, [], attributes ?? new Dictionary<string, string>(),
                new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32),
                ExecutionSourceMap.XmlOnly)],
            resources.Where(item => item.Kind == GraphResourceKind.Input).Select(item => item.Id),
            [Id("output")]);
    }

    private static ExecutionGraph Graph(GraphOperationId? gatherOperation = null)
    {
        static GraphResource Resource(string id, GraphResourceKind kind, GraphResourceLifetime lifetime,
            GraphElementType type, int[] dimensions, string? binding = null) =>
            new(new ResourceId(id), id, kind, lifetime, new TensorDescriptor(type, dimensions), binding);

        var resources = new[]
        {
            Resource("token", GraphResourceKind.Input, GraphResourceLifetime.External, GraphElementType.Int32, [1]),
            Resource("table", GraphResourceKind.Weight, GraphResourceLifetime.Model, GraphElementType.Float32, [2, 3], "embedding"),
            Resource("matrix", GraphResourceKind.Weight, GraphResourceLifetime.Model, GraphElementType.Float32, [2, 3], "projection"),
            Resource("state", GraphResourceKind.SessionState, GraphResourceLifetime.Session, GraphElementType.Float32, [3]),
            Resource("row", GraphResourceKind.TokenTransient, GraphResourceLifetime.Token, GraphElementType.Float32, [3]),
            Resource("sum", GraphResourceKind.TokenTransient, GraphResourceLifetime.Token, GraphElementType.Float32, [3]),
            Resource("result", GraphResourceKind.Output, GraphResourceLifetime.External, GraphElementType.Float32, [2]),
        };
        static ExecutionNode Node(string id, GraphOperationId op, NodeResourceBinding[] bindings,
            params string[] dependencies) =>
            new(new ExecutionNodeId(id), op, new RegionId("root"), bindings,
                dependencies.Select(value => new ExecutionNodeId(value)).ToArray(),
                new Dictionary<string, string>(),
                new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32),
                ExecutionSourceMap.XmlOnly);
        var nodes = new[]
        {
            Node("gather", gatherOperation ?? PrimitiveGraphOperations.GatherRow,
                [GraphBindings.Read("table", "table"), GraphBindings.Read("index", "token"), GraphBindings.Write("output", "row")]),
            Node("add", PrimitiveGraphOperations.Add,
                [GraphBindings.Read("left", "row"), GraphBindings.Read("right", "state"), GraphBindings.Write("output", "sum")], "gather"),
            Node("matvec", PrimitiveGraphOperations.MatVec,
                [GraphBindings.Read("weight", "matrix"), GraphBindings.Read("input", "sum"), GraphBindings.Write("output", "result")], "add"),
            Node("store", PrimitiveGraphOperations.Copy,
                [GraphBindings.Read("input", "sum"), GraphBindings.Write("output", "state")], "matvec"),
        };
        return new ExecutionGraph(new GraphIdentity("test-primitive", 1, "resident"),
            new GraphModelSignature(2, 3, 1, 1, 3, "test-state"),
            resources, [new GraphRegion(new RegionId("root"), null, GraphRegionTypes.Graph, null, "root", new Dictionary<string, string>())],
            nodes, [new ResourceId("token")], [new ResourceId("result")],
            [new GraphStateEntry("state", new ResourceId("state"))]);
    }

    private static ID3D12Device? TryCreateDevice()
    {
        try
        {
            using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory6>();
            for (uint index = 0; factory.EnumAdapterByGpuPreference(index, GpuPreference.HighPerformance, out IDXGIAdapter1? adapter).Success; index++)
            {
                using (adapter)
                {
                    if ((adapter!.Description1.Flags & AdapterFlags.Software) != 0) continue;
                    return D3D12CreateDevice<ID3D12Device>(adapter, FeatureLevel.Level_11_0);
                }
            }
        }
        catch (Exception error) when (error is PlatformNotSupportedException or InvalidOperationException or NotSupportedException)
        {
            return null;
        }
        return null;
    }

    private sealed class Tensor(string name, int[] dimensions, float[] values) : IModelTensor
    {
        private readonly Half[] halves = [];
        public Tensor(string name, int[] dimensions, Half[] halfValues)
            : this(name, dimensions, Array.Empty<float>()) => halves = halfValues;
        public string Name => name;
        public RwkvTensorDataType DataType => halves.Length == 0
            ? RwkvTensorDataType.Float32 : RwkvTensorDataType.Float16;
        public IReadOnlyList<int> Dimensions => dimensions;
        public ReadOnlySpan<float> FloatValues => values;
        public ReadOnlySpan<Half> HalfValues => halves;
    }

    private sealed class Catalog(params Tensor[] tensors) : IModelTensorCatalog
    {
        private readonly Dictionary<string, IModelTensor> values =
            tensors.ToDictionary(tensor => tensor.Name, tensor => (IModelTensor)tensor);
        public int VocabularySize => 2;
        public int EmbeddingSize => 3;
        public int LayerCount => 1;
        public IReadOnlyCollection<string> Names => values.Keys;
        public bool TryGet(string name, out IModelTensor tensor) => values.TryGetValue(name, out tensor!);
        public IModelTensor GetRequired(string name) => values[name];
    }
}
