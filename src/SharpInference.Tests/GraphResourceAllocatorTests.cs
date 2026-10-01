using SharpInference.Architectures.Rwkv6;
using SharpInference.Backends.Cpu;
using SharpInference.Graphs;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class GraphResourceAllocatorTests
{
    [Fact]
    public void CompatibleOrderedPrivateResourcesShareOneInvocationSlot()
    {
        var graph = CreateGraph(
            [Scratch("a"), Scratch("b"), Scratch("c")],
            [Node("first", ["a"]), Node("second", ["b"], ["first"]), Node("third", ["c"], ["second"])]);

        var plan = GraphResourceAllocator.Plan(graph);

        Assert.Single(plan.Slots);
        Assert.Equal(3, plan.SlotByResource.Count);
        Assert.Equal([new ResourceId("a"), new ResourceId("b"), new ResourceId("c")],
            plan.SlotByResource.Keys);
        Assert.All(plan.SlotByResource.Values, slot => Assert.Equal(0, slot));
        Assert.Equal(GraphResourceLifetime.Invocation, plan.Slots[0].Lifetime);
        Assert.Equal(GraphElementType.Float32, plan.Slots[0].Tensor.ElementType);
        Assert.Equal([2, 4], plan.Slots[0].Tensor.Dimensions);
        Assert.Equal("dense", plan.Slots[0].Tensor.Layout);

        var calls = 0;
        var invocation = plan.AllocateInvocation(slot =>
        {
            Assert.Equal(0, slot.Id);
            calls++;
            return new TestBuffer();
        });
        Assert.Equal(1, calls);
        Assert.Same(invocation.GetBuffer(new("a")), invocation.GetBuffer(new("c")));
        var buffer = invocation.GetBuffer(new("a"));
        invocation.Dispose();
        Assert.Equal(1, buffer.DisposeCount);
    }

    [Fact]
    public void ParallelBranchesDoNotReuseEvenWhenListedSequentially()
    {
        var graph = CreateGraph(
            [Scratch("a"), Scratch("b"), Scratch("c")],
            [Node("first", ["a"]), Node("branch", ["b"]), Node("join", ["c"], ["first", "branch"])]);

        var plan = GraphResourceAllocator.Plan(graph);

        Assert.NotEqual(plan.SlotByResource[new("a")], plan.SlotByResource[new("b")]);
        Assert.Equal(plan.SlotByResource[new("a")], plan.SlotByResource[new("c")]);
        Assert.Equal(2, plan.Slots.Count);
    }

    [Theory]
    [InlineData(GraphElementType.Float16, "dense", GraphResourceLifetime.Invocation, 2, 4)]
    [InlineData(GraphElementType.Float32, "packed", GraphResourceLifetime.Invocation, 2, 4)]
    [InlineData(GraphElementType.Float32, "dense", GraphResourceLifetime.Invocation, 4, 2)]
    [InlineData(GraphElementType.Float32, "dense", GraphResourceLifetime.Token, 2, 4)]
    public void IncompatibleDescriptorsOrLifetimesNeverShare(
        GraphElementType type, string layout, GraphResourceLifetime lifetime, int first, int second)
    {
        var graph = CreateGraph(
            [Scratch("a"), Scratch("b", type, layout, lifetime, first, second)],
            [Node("first", ["a"]), Node("second", ["b"], ["first"])]);

        var plan = GraphResourceAllocator.Plan(graph);

        Assert.Equal(2, plan.Slots.Count);
        Assert.NotEqual(plan.SlotByResource[new("a")], plan.SlotByResource[new("b")]);
    }

    [Fact]
    public void PublicLocalLivenessIncludesLastConsumerNotJustProducer()
    {
        var graph = CreateGraph([Scratch("a"), Scratch("b"), Scratch("c")],
        [
            Node("produceA", []) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("produceB", [], ["produceA"]) with { Resources = [GraphBindings.Write("out", "b")] },
            Node("consumeA", [], ["produceB"]) with { Resources = [GraphBindings.Read("in", "a")] },
            Node("produceC", [], ["consumeA"]) with { Resources = [GraphBindings.Write("out", "c")] },
        ]);
        var plan = GraphResourceAllocator.PlanLocal(graph);

        Assert.NotEqual(plan.SlotByResource[new("a")], plan.SlotByResource[new("b")]);
        Assert.Equal(plan.SlotByResource[new("a")], plan.SlotByResource[new("c")]);
    }

    [Fact]
    public void RepeatedSequentialPublicEpochsReuseOneSlot()
    {
        var graph = CreateGraph([Scratch("a"), Scratch("b")],
        [
            Node("a1", []) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("readA1", [], ["a1"]) with { Resources = [GraphBindings.Read("in", "a")] },
            Node("b1", [], ["readA1"]) with { Resources = [GraphBindings.Write("out", "b")] },
            Node("readB1", [], ["b1"]) with { Resources = [GraphBindings.Read("in", "b")] },
            Node("a2", [], ["readB1"]) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("readA2", [], ["a2"]) with { Resources = [GraphBindings.Read("in", "a")] },
            Node("b2", [], ["readA2"]) with { Resources = [GraphBindings.Write("out", "b")] },
            Node("readB2", [], ["b2"]) with { Resources = [GraphBindings.Read("in", "b")] },
        ]);

        var plan = GraphResourceAllocator.PlanLocal(graph);

        Assert.Single(plan.Slots);
        Assert.Equal(plan.SlotByResource[new("a")], plan.SlotByResource[new("b")]);
    }

    [Fact]
    public void ActualPortableRwkv6LayerGraphReusesLocalSlots()
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv6);
        var graph = new GraphOptimizer().Optimize(
            new PortableRwkv6GraphProvider().Build(catalog),
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            CpuPrimitiveGraphBackend.Instance.KernelCatalog);

        var plan = GraphResourceAllocator.PlanLocal(graph);

        Assert.True(plan.Slots.Count < plan.SlotByResource.Count);
        Assert.Contains(plan.SlotByResource.GroupBy(pair => pair.Value), group => group.Count() > 1);
    }

    [Fact]
    public void OverlappingEpochsDoNotReuseEvenWhenLaterEpochsAreSequential()
    {
        var graph = CreateGraph([Scratch("a"), Scratch("b")],
        [
            Node("a1", []) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("b1", [], ["a1"]) with { Resources = [GraphBindings.Write("out", "b")] },
            Node("readA1", [], ["b1"]) with { Resources = [GraphBindings.Read("in", "a")] },
            Node("readB1", [], ["readA1"]) with { Resources = [GraphBindings.Read("in", "b")] },
            Node("a2", [], ["readB1"]) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("b2", [], ["a2"]) with { Resources = [GraphBindings.Write("out", "b")] },
        ]);

        var plan = GraphResourceAllocator.PlanLocal(graph);

        Assert.NotEqual(plan.SlotByResource[new("a")], plan.SlotByResource[new("b")]);
    }

    [Fact]
    public void UnorderedEpochsOnParallelBranchesDoNotReuse()
    {
        var graph = CreateGraph([Scratch("a"), Scratch("b")],
        [
            Node("a1", []) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("readA1", [], ["a1"]) with { Resources = [GraphBindings.Read("in", "a")] },
            Node("b1", [], ["readA1"]) with { Resources = [GraphBindings.Write("out", "b")] },
            Node("readB1", [], ["b1"]) with { Resources = [GraphBindings.Read("in", "b")] },
            Node("a2", [], ["readA1"]) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("readA2", [], ["a2"]) with { Resources = [GraphBindings.Read("in", "a")] },
        ]);

        var plan = GraphResourceAllocator.PlanLocal(graph);

        Assert.NotEqual(plan.SlotByResource[new("a")], plan.SlotByResource[new("b")]);
    }

    [Fact]
    public void UnprovedReadWriteDoesNotStartNewEpoch()
    {
        var graph = CreateGraph([Scratch("a"), Scratch("b")],
        [
            Node("a1", []) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("b1", [], ["a1"]) with { Resources = [GraphBindings.Write("out", "b")] },
            Node("updateA", [], ["b1"]) with { Resources = [GraphBindings.ReadWrite("value", "a")] },
            Node("readB1", [], ["updateA"]) with { Resources = [GraphBindings.Read("in", "b")] },
            Node("a2", [], ["readB1"]) with { Resources = [GraphBindings.Write("out", "a")] },
        ]);

        var plan = GraphResourceAllocator.PlanLocal(graph);
        Assert.NotEqual(plan.SlotByResource[new("a")], plan.SlotByResource[new("b")]);
    }

    [Fact]
    public void ProvenReadWriteStartsNewEpoch()
    {
        var graph = CreateGraph([Scratch("a"), Scratch("b")],
        [
            Node("a1", []) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("b1", [], ["a1"]) with { Resources = [GraphBindings.Write("out", "b")] },
            Node("readB1", [], ["b1"]) with { Resources = [GraphBindings.Read("in", "b")] },
            Node("replaceA", [], ["readB1"]) with
            {
                Resources = [GraphBindings.ReadWrite("value", "a")],
                FirstWriteResources = [new ResourceId("a")],
            },
            Node("readA2", [], ["replaceA"]) with { Resources = [GraphBindings.Read("in", "a")] },
        ]);

        Assert.Single(GraphResourceAllocator.PlanLocal(graph).Slots);
        var unproved = graph.Nodes.Select(node => node.Id == new ExecutionNodeId("replaceA")
            ? node with { FirstWriteResources = [] }
            : node).ToArray();
        var withoutProof = CreateGraph([Scratch("a"), Scratch("b")], unproved);
        var unprovedPlan = GraphResourceAllocator.PlanLocal(withoutProof);
        Assert.NotEqual(unprovedPlan.SlotByResource[new("a")], unprovedPlan.SlotByResource[new("b")]);
        Assert.Single(GraphResourceAllocator.PlanLocal(withoutProof,
            (node, id) => node.Id == new ExecutionNodeId("replaceA") && id == new ResourceId("a")).Slots);
    }

    [Fact]
    public void SameNodeReadAndWriteDoNotSplitAnExistingEpochWithoutProof()
    {
        var graph = CreateGraph([Scratch("a"), Scratch("b")],
        [
            Node("a1", []) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("b1", [], ["a1"]) with { Resources = [GraphBindings.Write("out", "b")] },
            Node("readB1", [], ["b1"]) with { Resources = [GraphBindings.Read("in", "b")] },
            Node("updateA", [], ["readB1"]) with
            {
                Resources = [GraphBindings.Read("in", "a"), GraphBindings.Write("out", "a")],
            },
        ]);

        var plan = GraphResourceAllocator.PlanLocal(graph);
        Assert.NotEqual(plan.SlotByResource[new("a")], plan.SlotByResource[new("b")]);
    }

    [Fact]
    public void SameNodeReadAndWriteCannotEstablishFirstWriteWithoutProof()
    {
        var graph = CreateGraph([Scratch("a")],
        [
            Node("mixed", []) with
            {
                Resources = [GraphBindings.Read("in", "a"), GraphBindings.Write("out", "a")],
            },
        ]);

        Assert.Throws<InvalidDataException>(() => GraphResourceAllocator.PlanLocal(graph));
    }

    [Fact]
    public void OrderedReadBeforeFirstWriteIsRejected()
    {
        var graph = CreateGraph([Scratch("a"), Scratch("b")],
        [
            Node("readA", []) with { Resources = [GraphBindings.Read("in", "a")] },
            Node("writeA", [], ["readA"]) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("writeB", [], ["writeA"]) with { Resources = [GraphBindings.Write("out", "b")] },
        ]);

        Assert.Throws<InvalidDataException>(() => GraphResourceAllocator.PlanLocal(graph));
    }

    [Fact]
    public void ParallelConsumerAndProducerDoNotShareBacking()
    {
        var graph = CreateGraph([Scratch("a"), Scratch("b")],
        [
            Node("produceA", []) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("consumeA", [], ["produceA"]) with { Resources = [GraphBindings.Read("in", "a")] },
            Node("produceB", [], ["produceA"]) with { Resources = [GraphBindings.Write("out", "b")] },
        ]);
        var plan = GraphResourceAllocator.PlanLocal(graph);
        Assert.NotEqual(plan.SlotByResource[new("a")], plan.SlotByResource[new("b")]);
    }

    [Fact]
    public void PublicLocalReadMustHaveOrderedFirstWrite()
    {
        var graph = CreateGraph([Scratch("a")],
        [
            Node("write", []) with { Resources = [GraphBindings.Write("out", "a")] },
            Node("read", []) with { Resources = [GraphBindings.Read("in", "a")] },
        ]);
        Assert.Throws<InvalidDataException>(() => GraphResourceAllocator.PlanLocal(graph));
    }

    [Fact]
    public void PublicReadWriteRequiresBackendFirstWriteProof()
    {
        var graph = CreateGraph([Scratch("a")],
        [
            Node("fused", []) with { Resources = [GraphBindings.ReadWrite("value", "a")] },
            Node("consumer", [], ["fused"]) with { Resources = [GraphBindings.Read("value", "a")] },
        ]);
        Assert.Throws<InvalidDataException>(() => GraphResourceAllocator.PlanLocal(graph));
        Assert.Equal([new ResourceId("a")],
            GraphResourceAllocator.PlanLocal(graph, id => id == new ResourceId("a")).SlotByResource.Keys);
        Assert.Throws<InvalidDataException>(() =>
            GraphResourceAllocator.PlanLocal(graph, (node, id) =>
                node.Id == new ExecutionNodeId("consumer") && id == new ResourceId("a")));
        Assert.Equal([new ResourceId("a")],
            GraphResourceAllocator.PlanLocal(graph, (node, id) =>
                node.Id == new ExecutionNodeId("fused") && id == new ResourceId("a")).SlotByResource.Keys);
    }

    [Fact]
    public void OptimizerPublicFirstWriteProofSurvivesFusionAndExecutionXml()
    {
        var builder = new LogicalGraphBuilder(new("test", 1, "public proof"), new(1, 1, 1, 1, 1, "test.state"));
        builder.AddRegion("root", GraphRegionTypes.Graph, "Root")
            .AddResource("value", "Value", GraphResourceKind.Temporary,
                GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float32, [2, 4]))
            .AddResource("result", "Result", GraphResourceKind.Output,
                GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Float32, [2, 4]),
                graphOutput: true)
            .AddNode("first", new("test.first"), "root", [GraphBindings.Write("out", "value")])
            .AddNode("second", new("test.second"), "root", [GraphBindings.Read("in", "value")], ["first"])
            .AddNode("third", new("test.third"), "root",
                [GraphBindings.Read("in", "value"), GraphBindings.Write("out", "result")], ["second"]);
        var graph = new GraphOptimizer().Optimize(builder.Build(),
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted),
            [new SequenceFusionRule("pair", [new("test.first"), new("test.second")], new("test.pair"))]);
        var fused = graph.Nodes[0];
        Assert.Contains(new ResourceId("value"), fused.FirstWriteResources);
        Assert.DoesNotContain(graph.Nodes[1].FirstWriteResources,
            id => id == new ResourceId("result"));
        var restored = GraphXml.DeserializeExecution(GraphXml.Serialize(graph));
        Assert.Equal(fused.FirstWriteResources, restored.Nodes[0].FirstWriteResources);
        Assert.Contains(new ResourceId("value"), GraphResourceAllocator.PlanLocal(restored).SlotByResource.Keys);
    }

    [Fact]
    public void UnorderedWritesToSameLocalResourceAreRejected()
    {
        var graph = CreateGraph([Scratch("a")],
        [
            Node("left", []) with { Resources = [GraphBindings.Write("value", "a")] },
            Node("right", []) with { Resources = [GraphBindings.Write("value", "a")] },
        ]);
        Assert.Throws<InvalidDataException>(() => GraphResourceAllocator.PlanLocal(graph));
    }

    [Fact]
    public void DeviceAndGraphEndpointPreventUnsafeReuse()
    {
        var a = Scratch("a") with { DeviceId = "gpu0" };
        var b = Scratch("b") with { DeviceId = "gpu1" };
        var output = Scratch("output") with { DeviceId = "gpu0" };
        var graph = CreateGraph([a, b, output],
        [
            Node("first", ["a"]),
            Node("second", ["b"], ["first"]),
            Node("last", [], ["second"]) with { Resources = [GraphBindings.Write("out", "output")] },
        ]);
        var plan = GraphResourceAllocator.PlanLocal(new ExecutionGraph(graph.Identity, graph.Model,
            graph.Resources, graph.Regions, graph.Nodes, [], [new ResourceId("output")]));
        Assert.Equal(2, plan.Slots.Count);
        Assert.NotEqual(plan.SlotByResource[a.Id], plan.SlotByResource[b.Id]);
        Assert.Equal("gpu0", plan.Slots[plan.SlotByResource[a.Id]].DeviceId);
        Assert.False(plan.SlotByResource.ContainsKey(output.Id));
    }

    [Fact]
    public void EarlyGraphOutputStaysExternallyOwnedThroughReturn()
    {
        var output = new GraphResource(new("logits"), "Logits", GraphResourceKind.Output,
            GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Float32, [2, 4]));
        var scratch = Scratch("later");
        var graph = CreateGraph([output, scratch],
        [
            Node("output", []) with { Resources = [GraphBindings.Write("logits", "logits")] },
            Node("later", [], ["output"]) with { Resources = [GraphBindings.Write("scratch", "later")] },
        ]);
        var bound = new ExecutionGraph(graph.Identity, graph.Model, graph.Resources, graph.Regions,
            graph.Nodes, [], [output.Id]);

        Assert.Equal(GraphResourceScope.Local, output.Scope);
        var plan = GraphResourceAllocator.PlanLocal(bound);
        Assert.Equal([scratch.Id], plan.SlotByResource.Keys);
    }

    [Fact]
    public void TypedResourceMapValidatesMetadataAndDoesNotCopyValuesOnRead()
    {
        var descriptor = Scratch("a") with { DeviceId = "gpu0" };
        IResourceMapBase<TestBuffer> map = new GraphResourceMap<TestBuffer>(
            [descriptor], (metadata, _) => metadata.DeviceId == "gpu0");
        using var value = new TestBuffer();
        Assert.Equal(descriptor, Assert.Single(map.Resources));
        Assert.True(map.TryGetMetadata(descriptor.Id, out var metadata));
        Assert.Same(descriptor, metadata);
        map.SetResource(descriptor.Id, value);
        Assert.True(map.TryGetResource(descriptor.Id, out var found));
        Assert.Same(value, found);
        Assert.False(map.TryGetMetadata(new("unknown"), out _));
        Assert.False(map.TryGetResource(new("unknown"), out _));
        Assert.Throws<InvalidOperationException>(() => map.SetResource(descriptor.Id, value));
        Assert.Throws<KeyNotFoundException>(() => map.SetResource(new("unknown"), value));
        Assert.Throws<ArgumentNullException>(() => map.SetResource(descriptor.Id, null!));
        var rejectingMap = new GraphResourceMap<TestBuffer>([descriptor], (_, _) => false);
        Assert.Throws<ArgumentException>(() => rejectingMap.SetResource(descriptor.Id, value));
    }

    [Fact]
    public void GraphResourceMapEnforcesPrivateOwnerAndSealedBindPhase()
    {
        var graph = CreateGraph([Scratch("private")], [Node("owner", ["private"])]);
        var map = new GraphResourceMap<TestBuffer>(graph, (_, _) => true);
        using var buffer = new TestBuffer();
        Assert.Throws<InvalidOperationException>(() => map.SetResource(new("private"), buffer));
        Assert.Throws<InvalidOperationException>(() =>
            map.SetPrivateResource(new("other"), new("private"), buffer));
        map.SetPrivateResource(new("owner"), new("private"), buffer);
        map.SealBindings();
        Assert.True(map.TryGetResource(new("private"), out var bound));
        Assert.Same(buffer, bound);
        Assert.Throws<InvalidOperationException>(() =>
            map.SetPrivateResource(new("owner"), new("private"), new TestBuffer()));
    }

    [Fact]
    public void GlobalAndSessionResourcesNeverEnterLocalSlots()
    {
        var global = new GraphResource(new("global"), "Global", GraphResourceKind.Constant,
            GraphResourceLifetime.Model, new TensorDescriptor(GraphElementType.Float32, [2, 4]));
        var session = new GraphResource(new("session"), "Session", GraphResourceKind.SessionState,
            GraphResourceLifetime.Session, new TensorDescriptor(GraphElementType.Float32, [2, 4]));
        var local = Scratch("local");
        var graph = CreateGraph([global, session, local],
        [
            Node("first", []) with
            {
                Resources = [GraphBindings.Read("global", "global"), GraphBindings.Read("session", "session"),
                    GraphBindings.Write("out", "local")],
            },
        ]);
        var plan = GraphResourceAllocator.PlanLocal(graph);

        Assert.Equal(GraphResourceScope.Global, global.Scope);
        Assert.Equal(GraphResourceScope.Session, session.Scope);
        Assert.Equal(GraphResourceScope.Local, local.Scope);
        Assert.Equal([local.Id], plan.SlotByResource.Keys);
    }

    [Fact]
    public void OneLeaseMayServeRepeatedSequentialReplays()
    {
        var plan = GraphResourceAllocator.Plan(CreateGraph([Scratch("a")], [Node("first", ["a"])]));
        var allocations = 0;
        using var lease = plan.AllocateInvocation(_ =>
        {
            allocations++;
            return new TestBuffer();
        });
        for (var token = 0; token < 128; token++)
            Assert.Same(lease.GetBuffer(new("a")), lease.GetBuffer(new("a")));
        Assert.Equal(1, allocations);
    }

    [Fact]
    public void AllUsedLocalResourcesArePlannedWithoutIncludingUnusedResources()
    {
        var publicResource = Scratch("public");
        var node = Node("first", ["private"]) with
        {
            Resources = [GraphBindings.Write("output", "public")],
        };
        var graph = CreateGraph([Scratch("private"), publicResource, Scratch("unused")], [node]);

        var plan = GraphResourceAllocator.PlanLocal(graph);

        Assert.Equal([new ResourceId("private"), new ResourceId("public")],
            plan.SlotByResource.Keys.OrderBy(id => id.Value));
        Assert.Equal(2, plan.SlotByResource.Count);
        Assert.NotEqual(plan.SlotByResource[new("private")], plan.SlotByResource[new("public")]);
        Assert.False(plan.SlotByResource.ContainsKey(new("unused")));
    }

    [Fact]
    public void ReadBeforeInitializationIsRejectedRatherThanRelyingOnZeroedBuffers()
    {
        var read = CreateGraph([Scratch("a")],
            [Node("first", ["a"], access: GraphResourceAccess.Read)]);
        var readWrite = CreateGraph([Scratch("a")],
            [Node("first", ["a"], access: GraphResourceAccess.ReadWrite)]);
        var proved = CreateGraph([Scratch("a")],
            [Node("first", ["a"], access: GraphResourceAccess.ReadWrite,
                initializedBeforeRead: true)]);

        Assert.Throws<InvalidDataException>(() => GraphResourceAllocator.Plan(read));
        Assert.Throws<InvalidDataException>(() => GraphResourceAllocator.Plan(read, _ => true));
        Assert.Throws<InvalidDataException>(() => GraphResourceAllocator.Plan(readWrite));
        Assert.Single(GraphResourceAllocator.Plan(proved).Slots);
        Assert.Single(GraphResourceAllocator.Plan(readWrite, id => id == new ResourceId("a")).Slots);
    }

    [Fact]
    public void OptimizerFirstWriteProofSurvivesNestedFusionAndEnablesPlanning()
    {
        var builder = new LogicalGraphBuilder(
            new GraphIdentity("test", 1, "nested scratch"),
            new GraphModelSignature(1, 1, 1, 1, 1, "test.state"));
        builder.AddRegion("root", GraphRegionTypes.Graph, "Root")
            .AddResource("a", "A", GraphResourceKind.Temporary,
                GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float32, [2, 4]))
            .AddResource("b", "B", GraphResourceKind.Temporary,
                GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float32, [2, 4]))
            .AddResource("output", "Output", GraphResourceKind.Output,
                GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Float32, [2, 4]),
                graphOutput: true)
            .AddNode("first", new("test.first"), "root", [GraphBindings.Write("result", "a")])
            .AddNode("second", new("test.second"), "root",
                [GraphBindings.Read("input", "a"), GraphBindings.Write("result", "b")],
                ["first"])
            .AddNode("third", new("test.third"), "root",
                [GraphBindings.Read("input", "b"), GraphBindings.Write("result", "output")],
                ["second"]);

        var execution = new GraphOptimizer().Optimize(builder.Build(),
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted),
            [
                new SequenceFusionRule("first-second", [new("test.first"), new("test.second")],
                    new("test.first-second")),
                new SequenceFusionRule("all", [new("test.first-second"), new("test.third")],
                    new("test.all")),
            ]);

        var fused = Assert.Single(execution.Nodes);
        Assert.Equal([new ResourceId("a"), new ResourceId("b")],
            fused.InternalResources.Select(binding => binding.Resource));
        Assert.All(fused.InternalResources, binding =>
        {
            Assert.Equal(GraphResourceAccess.ReadWrite, binding.Access);
            Assert.True(binding.InitializedBeforeRead);
        });
        var plan = GraphResourceAllocator.Plan(execution);
        Assert.Equal(2, plan.SlotByResource.Count);
    }

    [Fact]
    public void ConcurrentInvocationsCannotShareBackingObjects()
    {
        var graph = CreateGraph([Scratch("a")], [Node("first", ["a"])]);
        var plan = GraphResourceAllocator.Plan(graph);
        var buffer = new TestBuffer();
        using var first = plan.AllocateInvocation(_ => buffer);

        Assert.Throws<InvalidOperationException>(() => plan.AllocateInvocation(_ => buffer));
        Assert.Throws<InvalidOperationException>(() =>
            GraphResourceAllocator.Plan(graph).AllocateInvocation(_ => buffer));
        Assert.Equal(0, buffer.DisposeCount);
        TestBuffer secondBuffer;
        using (var second = plan.AllocateInvocation(_ => new TestBuffer()))
        {
            secondBuffer = second.GetBuffer(new("a"));
            Assert.NotSame(first.GetBuffer(new("a")), secondBuffer);
        }
        Assert.Equal(1, secondBuffer.DisposeCount);
        Assert.Throws<KeyNotFoundException>(() => first.GetBuffer(new("missing")));
        first.Dispose();
        first.Dispose();
        Assert.Equal(1, buffer.DisposeCount);
        Assert.Throws<ObjectDisposedException>(() => first.GetBuffer(new("a")));
    }

    [Fact]
    public void LaterFactoryFailureDisposesAllEarlierSlotsExactlyOnce()
    {
        var plan = GraphResourceAllocator.Plan(CreateGraph(
            [Scratch("a"), Scratch("b"), Scratch("c")],
            [Node("first", ["a", "b", "c"])]));
        var allocated = new List<TestBuffer>();

        var failure = Assert.Throws<InvalidOperationException>(() => plan.AllocateInvocation(slot =>
        {
            if (slot.Id == 2) throw new InvalidOperationException("allocation failed");
            var buffer = new TestBuffer();
            allocated.Add(buffer);
            return buffer;
        }));

        Assert.Equal("allocation failed", failure.Message);
        Assert.Equal(2, allocated.Count);
        Assert.All(allocated, buffer => Assert.Equal(1, buffer.DisposeCount));
        using var retry = plan.AllocateInvocation(_ => new TestBuffer());
        Assert.Equal(3, plan.Slots.Count);
    }

    [Fact]
    public void DuplicateInLaterSlotDoesNotDisposeAlreadyOwnedBufferTwice()
    {
        var plan = GraphResourceAllocator.Plan(CreateGraph(
            [Scratch("a"), Scratch("b")],
            [Node("first", ["a", "b"])]));
        var buffer = new TestBuffer();

        Assert.Throws<InvalidOperationException>(() => plan.AllocateInvocation(_ => buffer));

        Assert.Equal(1, buffer.DisposeCount);
    }

    [Fact]
    public void FailedAllocationReleasesOnlyItsOwnBuffersNotAnotherInvocationsBuffer()
    {
        var owner = GraphResourceAllocator.Plan(CreateGraph([Scratch("a")], [Node("first", ["a"])]));
        var requester = GraphResourceAllocator.Plan(CreateGraph(
            [Scratch("b"), Scratch("c")], [Node("second", ["b", "c"])]));
        var shared = new TestBuffer();
        var partial = new TestBuffer();
        using var active = owner.AllocateInvocation(_ => shared);

        Assert.Throws<InvalidOperationException>(() =>
            requester.AllocateInvocation(slot => slot.Id == 0 ? partial : shared));

        Assert.Equal(1, partial.DisposeCount);
        Assert.Equal(0, shared.DisposeCount);
        Assert.Same(shared, active.GetBuffer(new("a")));
    }

    [Fact]
    public void CallbackReleasesOpaqueBuffersOnlyWhenLeaseEnds()
    {
        var plan = GraphResourceAllocator.Plan(CreateGraph([Scratch("a")], [Node("first", ["a"])]));
        var buffer = new object();
        var releases = 0;
        var lease = plan.AllocateInvocation(_ => buffer, _ => releases++);

        Assert.Equal(0, releases);
        lease.Dispose();
        lease.Dispose();
        Assert.Equal(1, releases);
    }

    [Fact]
    public void ReleaseFailureDoesNotLeakOtherBuffersOrActiveRegistrations()
    {
        var plan = GraphResourceAllocator.Plan(CreateGraph(
            [Scratch("a"), Scratch("b")], [Node("first", ["a", "b"])]));
        var allocated = new List<TestBuffer>();
        var lease = plan.AllocateInvocation(_ =>
        {
            var buffer = new TestBuffer();
            allocated.Add(buffer);
            return buffer;
        }, buffer =>
        {
            buffer.Dispose();
            if (ReferenceEquals(buffer, allocated[0])) throw new InvalidOperationException("release failed");
        });

        Assert.Throws<AggregateException>(() => lease.Dispose());
        Assert.All(allocated, buffer => Assert.Equal(1, buffer.DisposeCount));
        using var retry = plan.AllocateInvocation(_ => new TestBuffer());
    }

    [Fact]
    public void FactoryFailureAndCleanupFailureBothRemainObservable()
    {
        var plan = GraphResourceAllocator.Plan(CreateGraph(
            [Scratch("a"), Scratch("b")], [Node("first", ["a", "b"])]));
        var buffer = new TestBuffer();

        var error = Assert.Throws<AggregateException>(() => plan.AllocateInvocation(
            slot => slot.Id == 0 ? buffer : throw new InvalidOperationException("factory failed"),
            value =>
            {
                value.Dispose();
                throw new InvalidOperationException("release failed");
            }));

        Assert.Equal(["factory failed", "release failed"],
            error.InnerExceptions.Select(exception => exception.Message));
        Assert.Equal(1, buffer.DisposeCount);
    }

    private sealed class TestBuffer : IDisposable
    {
        public int DisposeCount { get; private set; }
        public void Dispose() => DisposeCount++;
    }

    private static GraphResource Scratch(
        string id,
        GraphElementType type = GraphElementType.Float32,
        string layout = "dense",
        GraphResourceLifetime lifetime = GraphResourceLifetime.Invocation,
        params int[] dimensions) =>
        new(new(id), id,
            lifetime == GraphResourceLifetime.Token ? GraphResourceKind.TokenTransient : GraphResourceKind.Temporary,
            lifetime, new TensorDescriptor(type, dimensions.Length == 0 ? [2, 4] : dimensions, layout));

    private static ExecutionNode Node(
        string id,
        string[] privateResources,
        string[]? dependencies = null,
        GraphResourceAccess access = GraphResourceAccess.Write,
        bool initializedBeforeRead = false) =>
        new(new(id), new("test.scratch"), new("root"), [],
            dependencies?.Select(dependency => new ExecutionNodeId(dependency)).ToArray() ?? [],
            new Dictionary<string, string>(),
            new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32),
            ExecutionSourceMap.XmlOnly)
        {
            InternalResources = privateResources
                .Select(resource => new NodeResourceBinding(resource, new ResourceId(resource), access)
                {
                    InitializedBeforeRead = initializedBeforeRead,
                })
                .ToArray(),
        };

    private static ExecutionGraph CreateGraph(GraphResource[] resources, ExecutionNode[] nodes) =>
        new(new GraphIdentity("test", 1, "scratch"),
            new GraphModelSignature(1, 1, 1, 1, 1, "test.state"),
            resources,
            [new GraphRegion(new("root"), null, GraphRegionTypes.Graph, null, "Root",
                new Dictionary<string, string>())],
            nodes, [], []);
}
