using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Backends.Cpu;
using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class GraphTests
{
    [Fact]
    public void Optimizer_WithinStageDoesNotCrossSiblingStages()
    {
        var graph = CreateBoundaryGraph();
        var rules = new[]
        {
            new SequenceFusionRule(
                "fuse-a-b",
                [new GraphOperationId("test.a"), new GraphOperationId("test.b")],
                new GraphOperationId("test.ab")),
            new SequenceFusionRule(
                "fuse-ab-c",
                [new GraphOperationId("test.ab"), new GraphOperationId("test.c")],
                new GraphOperationId("test.abc")),
        };

        var execution = new GraphOptimizer().Optimize(
            graph,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            rules);

        Assert.Equal(["test.ab@1", "test.c@1"], execution.Nodes.Select(node => node.Operation.ToString()));
        Assert.Equal("stage.a", execution.Nodes[0].Region.Value);
        Assert.Equal(["a", "b"], execution.Nodes[0].Source.LogicalNodes.Select(node => node.Value));
    }

    [Fact]
    public void Optimizer_WithinLayerCanCrossSiblingStages()
    {
        var graph = CreateBoundaryGraph();
        var rules = new[]
        {
            new SequenceFusionRule(
                "fuse-a-b",
                [new GraphOperationId("test.a"), new GraphOperationId("test.b")],
                new GraphOperationId("test.ab")),
            new SequenceFusionRule(
                "fuse-ab-c",
                [new GraphOperationId("test.ab"), new GraphOperationId("test.c")],
                new GraphOperationId("test.abc")),
        };

        var execution = new GraphOptimizer().Optimize(
            graph,
            new GraphOptimizationOptions(OptimizationBoundary.WithinLayer),
            rules);

        var node = Assert.Single(execution.Nodes);
        Assert.Equal("test.abc@1", node.Operation.ToString());
        Assert.Equal("layer.0", node.Region.Value);
        Assert.Equal(["a", "b", "c"], node.Source.LogicalNodes.Select(source => source.Value));
    }

    [Fact]
    public void Optimizer_ExposesNamedArgumentsAndKeepsScratchPrivateAcrossFusionPasses()
    {
        var graph = CreateBoundaryGraph();
        var execution = new GraphOptimizer().Optimize(
            graph,
            new GraphOptimizationOptions(OptimizationBoundary.WithinLayer),
            [
                new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab"),
                    Arguments:
                    [
                        new("source", 0, "input"),
                        new("result", 1, "output"),
                    ]),
                new SequenceFusionRule("abc", [new("test.ab"), new("test.c")], new("test.abc"),
                    Arguments:
                    [
                        new("input", 0, "source"),
                        new("output", 1, "output"),
                    ]),
            ]);

        var fused = Assert.Single(execution.Nodes);
        Assert.Equal(["input", "output"], fused.Resources.Select(binding => binding.Port));
        Assert.Equal(["x", "output"], fused.Resources.Select(binding => binding.Resource.Value));
        Assert.Equal(["y", "z"], fused.InternalResources.Select(binding => binding.Resource.Value));
        Assert.All(fused.InternalResources, binding => Assert.True(binding.InitializedBeforeRead));
        Assert.Equal(2, GraphResourceAllocator.Plan(execution).Slots.Count);
        Assert.Equal(graph.Resources.Select(resource => resource.Id), execution.Resources.Select(resource => resource.Id));
        Assert.Equal(["a", "b", "c"], fused.Source.LogicalNodes.Select(id => id.Value));
    }

    [Fact]
    public void BackendPreflight_ExcludesPrivateScratchButStillChecksFusedPrecision()
    {
        var logical = CreateBoundaryGraph(GraphElementType.Float16,
            new PrecisionRequirement(GraphElementType.Float16, GraphElementType.Float32));
        var execution = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            [new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab"))]);
        var fused = execution.Nodes[0];
        Assert.Single(fused.InternalResources);
        Assert.Equal(["node0.input", "node1.output"], fused.Resources.Select(binding => binding.Port));
        var signature = BackendPreparation.CreateSignature(fused, execution.Resources.ToDictionary(resource => resource.Id));
        Assert.Equal([GraphElementType.Float16], signature.InputTypes);
        Assert.Equal([GraphElementType.Float16], signature.OutputTypes);

        var insufficient = BackendPreparation.Preflight(execution, node =>
            node == fused
                ? [new OperatorImplementationDescription("fp16", fused.Operation, signature,
                    new KernelPrecisionProfile(GraphElementType.Float16, GraphElementType.Float16))]
                : []);
        Assert.Contains(insufficient.Diagnostics,
            diagnostic => diagnostic.NodeId == fused.Id &&
                diagnostic.Reason == BackendPreparationFailureReason.PrecisionRequirementNotMet);
    }

    [Fact]
    public void Optimizer_EmitsDistinctNamedInputAndOutputForAliasedResource()
    {
        var original = CreateBoundaryGraph();
        var a = original.Nodes[0] with
        {
            Resources = [GraphBindings.Read("input", "x"), GraphBindings.Write("output", "x")],
        };
        var b = original.Nodes[1] with
        {
            Resources = [GraphBindings.Read("input", "x"), GraphBindings.Write("output", "z")],
        };
        var logical = new LogicalGraph(original.Identity, original.Model, original.Resources,
            original.Regions, [a, b, original.Nodes[2]], original.Inputs, original.Outputs);
        var execution = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            [
                new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab"),
                    Arguments:
                    [
                        new("source", 0, "input"),
                        new("updated", 0, "output"),
                        new("result", 1, "output"),
                    ]),
            ]);
        var fused = execution.Nodes[0];

        Assert.Equal(["source", "updated", "result"], fused.Resources.Select(binding => binding.Port));
        Assert.Equal(["x", "x", "z"], fused.Resources.Select(binding => binding.Resource.Value));
        Assert.Equal([GraphResourceAccess.Read, GraphResourceAccess.Write, GraphResourceAccess.Write],
            fused.Resources.Select(binding => binding.Access));
        var signature = BackendPreparation.CreateSignature(fused, execution.Resources.ToDictionary(resource => resource.Id));
        Assert.Single(signature.InputTypes);
        Assert.Equal(2, signature.OutputTypes.Count);

        var fallback = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            [new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab"))]);
        var aliased = Assert.Single(fallback.Nodes[0].Resources,
            binding => binding.Resource == new ResourceId("x"));
        Assert.Equal(GraphResourceAccess.ReadWrite, aliased.Access);
    }

    [Fact]
    public void Optimizer_KeepsExternallyConsumedIntermediatePublic()
    {
        var original = CreateBoundaryGraph();
        var observer = new LogicalNode(new("observer"), new("test.observe"), new("stage.b"),
            [GraphBindings.Read("observed", "y")], [], new Dictionary<string, string>(),
            new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32));
        var graph = new LogicalGraph(original.Identity, original.Model, original.Resources,
            original.Regions, [.. original.Nodes, observer], original.Inputs, original.Outputs);
        var execution = new GraphOptimizer().Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            [new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab"))]);

        Assert.Contains(execution.Nodes[0].Resources, binding => binding.Resource == new ResourceId("y"));
        Assert.DoesNotContain(execution.Nodes[0].InternalResources, binding => binding.Resource == new ResourceId("y"));
    }

    [Fact]
    public void Optimizer_DefaultArgumentNamesFollowPatternPortsAndRejectInternalAliases()
    {
        var graph = CreateBoundaryGraph();
        var optimizer = new GraphOptimizer();
        var options = new GraphOptimizationOptions(OptimizationBoundary.WithinStage);
        var fused = optimizer.Optimize(graph, options,
            [new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab"))]);
        Assert.Equal(["node0.input", "node1.output"], fused.Nodes[0].Resources.Select(binding => binding.Port));
        Assert.Throws<InvalidDataException>(() => optimizer.Optimize(graph, options,
            [new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab"),
                Arguments: [new("bad", 0, "output")])]));
    }

    [Fact]
    public void Optimizer_DoesNotInternalizeScratchWithoutFirstWrite()
    {
        var original = CreateBoundaryGraph();
        var a = original.Nodes[0] with
        {
            Resources = [GraphBindings.Read("input", "x"), GraphBindings.ReadWrite("output", "y")],
        };
        var logical = new LogicalGraph(original.Identity, original.Model, original.Resources,
            original.Regions, [a, .. original.Nodes.Skip(1)], original.Inputs, original.Outputs);
        var execution = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            [new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab"))]);

        Assert.Contains(execution.Nodes[0].Resources, binding => binding.Resource == new ResourceId("y"));
        Assert.DoesNotContain(new ResourceId("y"), execution.InternalResourceOwners.Keys);
    }

    [Fact]
    public void Optimizer_DoesNotTreatPortOrderWithinOneNodeAsFirstWriteProof()
    {
        var original = CreateBoundaryGraph();
        var a = original.Nodes[0] with
        {
            Resources =
            [
                GraphBindings.Read("input", "x"),
                GraphBindings.Write("produce", "y"),
                GraphBindings.Read("consume", "y"),
            ],
        };
        var logical = new LogicalGraph(original.Identity, original.Model, original.Resources,
            original.Regions, [a, .. original.Nodes.Skip(1)], original.Inputs, original.Outputs);
        var execution = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            [new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab"))]);

        Assert.Contains(execution.Nodes[0].Resources, binding => binding.Resource == new ResourceId("y"));
        Assert.DoesNotContain(new ResourceId("y"), execution.InternalResourceOwners.Keys);
    }

    [Fact]
    public void ExecutionGraph_RejectsConsumptionOfInternalizedScratch()
    {
        var logical = CreateBoundaryGraph();
        var execution = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            [new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab"))]);
        var consumer = execution.Nodes[1] with
        {
            Resources = [.. execution.Nodes[1].Resources, GraphBindings.Read("leak", "y")],
        };

        var error = Assert.Throws<InvalidDataException>(() => new ExecutionGraph(
            execution.Identity, execution.Model, execution.Resources, execution.Regions,
            [execution.Nodes[0], consumer], execution.Inputs, execution.Outputs));
        Assert.Contains("Internal resource", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ExecutionGraph_TracksPrivateOwnersAndConservativeBackingCompatibility()
    {
        var original = CreateBoundaryGraph();
        var w = new GraphResource(new("w"), "W", GraphResourceKind.Temporary,
            GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float32, [1]));
        var c = original.Nodes[2] with { Resources = [GraphBindings.Read("input", "z"), GraphBindings.Write("output", "w")] };
        var d = new LogicalNode(new("d"), new("test.d"), new("stage.b"),
            [GraphBindings.Read("input", "w"), GraphBindings.Write("output", "output")],
            [new("c")], new Dictionary<string, string>(),
            new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32));
        var logical = new LogicalGraph(original.Identity, original.Model, [.. original.Resources, w],
            original.Regions, [original.Nodes[0], original.Nodes[1], c, d], original.Inputs, original.Outputs);
        var execution = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.WithinLayer),
            [
                new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab")),
                new SequenceFusionRule("cd", [new("test.c"), new("test.d")], new("test.cd")),
            ]);

        Assert.Equal(2, execution.Nodes.Count);
        Assert.NotEqual(new ResourceId("y"), new ResourceId("w"));
        Assert.Equal(execution.Nodes[0].Id, execution.InternalResourceOwners[new ResourceId("y")]);
        Assert.Equal(execution.Nodes[1].Id, execution.InternalResourceOwners[new ResourceId("w")]);
        Assert.True(execution.CanReuseInternalBacking(new("y"), new("w")));
        Assert.False(execution.CanReuseInternalBacking(new("y"), new("y")));
        Assert.False(execution.CanReuseInternalBacking(new("y"), new("z")));

        var independent = new ExecutionGraph(execution.Identity, execution.Model, execution.Resources,
            execution.Regions, [execution.Nodes[0], execution.Nodes[1] with { Dependencies = [] }],
            execution.Inputs, execution.Outputs);
        Assert.False(independent.CanReuseInternalBacking(new("y"), new("w")));
    }

    [Fact]
    public void ExecutionGraph_AllowsEmptySourceMappingWithoutLogicalGraph()
    {
        var execution = new GraphOptimizer().Optimize(CreateBoundaryGraph(),
            new GraphOptimizationOptions(OptimizationBoundary.Off));
        ExecutionGraph Rebuild(ExecutionSourceMap source) => new(
            execution.Identity, execution.Model, execution.Resources, execution.Regions,
            [execution.Nodes[0] with { Source = source }, .. execution.Nodes.Skip(1)],
            execution.Inputs, execution.Outputs);

        Assert.Empty(Rebuild(ExecutionSourceMap.XmlOnly).Nodes[0].Source.LogicalNodes);
        Assert.Equal("test.rule",
            Rebuild(new ExecutionSourceMap([], "test.rule")).Nodes[0].Source.AppliedRuleId);
    }

    [Fact]
    public void ExecutionGraph_RejectsInvalidInitializationMetadata()
    {
        var original = CreateBoundaryGraph();
        var execution = new GraphOptimizer().Optimize(original,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            [new SequenceFusionRule("ab", [new("test.a"), new("test.b")], new("test.ab"))]);

        ExecutionGraph Rebuild(ExecutionNode modified) => new(
            execution.Identity, execution.Model, execution.Resources, execution.Regions,
            [modified, .. execution.Nodes.Skip(1)], execution.Inputs, execution.Outputs);

        var fused = execution.Nodes[0];
        Assert.Throws<InvalidDataException>(() => Rebuild(fused with
        {
            Resources = [fused.Resources[0] with { InitializedBeforeRead = true }, .. fused.Resources.Skip(1)],
        }));
        Assert.Throws<InvalidDataException>(() => Rebuild(fused with
        {
            InternalResources = [fused.InternalResources[0] with
            {
                Access = GraphResourceAccess.Read,
                InitializedBeforeRead = true,
            }],
        }));
    }

    [Fact]
    public void Optimizer_OffPreservesLogicalNodesAndSourceMapping()
    {
        var graph = CreateBoundaryGraph();

        var execution = new GraphOptimizer().Optimize(
            graph,
            new GraphOptimizationOptions(OptimizationBoundary.Off),
            [
                new SequenceFusionRule(
                    "unused",
                    [new GraphOperationId("test.a"), new GraphOperationId("test.b")],
                    new GraphOperationId("test.ab")),
            ]);

        Assert.Equal(3, execution.Nodes.Count);
        Assert.All(execution.Nodes, node =>
        {
            Assert.Single(node.Source.LogicalNodes);
            Assert.Null(node.Source.AppliedRuleId);
        });
    }

    [Fact]
    public void Optimizer_EliminatesDeadPortableAndPrimitiveTemporariesAndRewiresDependencies()
    {
        var graph = CreateDeadTemporaryGraph();
        var catalog = new ExecutionKernelCatalog("test.backend",
            [new("test.a"), new("test.b"), new("test.c")]);
        PortableTensorOperationContracts.ValidateGraph(graph);

        var execution = new GraphOptimizer().Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage), catalog);

        Assert.Equal(["a", "b", "c"], execution.Nodes.SelectMany(node =>
            node.Source.LogicalNodes.Select(source => source.Value)));
        Assert.Equal(["e0000", "e0001"], execution.Nodes[2].Dependencies.Select(id => id.Value));
        Assert.Equal(graph.Resources.Select(resource => resource.Id), execution.Resources.Select(resource => resource.Id));
        Assert.Equal(graph.Outputs, execution.Outputs);
    }

    [Fact]
    public void Optimizer_DeadNodeEliminationPreservesExecutedOutputs()
    {
        var logical = new LogicalGraphBuilder(new GraphIdentity("test", 1, "dead-path"),
                new GraphModelSignature(1, 2, 1, 1, 2, "test.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "Root")
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("dead-first", "Dead first", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("dead-second", "Dead second", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddNode("fill", PortableTensorOperationContracts.Fill, "root",
                [GraphBindings.Write("output", "dead-first")],
                attributes: new TensorFillValue(5).ToAttributes())
            .AddNode("unused-copy", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "dead-first"), GraphBindings.Write("output", "dead-second")],
                ["fill"])
            .AddNode("result", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "output")],
                ["unused-copy"])
            .Build();
        var optimizer = new GraphOptimizer();
        var original = optimizer.Optimize(logical, new GraphOptimizationOptions(OptimizationBoundary.Off));
        var optimized = optimizer.Optimize(logical);
        var inputs = new Dictionary<ResourceId, Array> { [new("input")] = new float[] { 4, 7 } };

        Assert.Single(optimized.Nodes);
        Assert.Empty(optimized.Nodes[0].Dependencies);
        Assert.Equal(Assert.IsType<float[]>(new CpuPrimitiveGraphExecutor(original, new EmptyTensorCatalog())
                .Execute(inputs).Outputs[new("output")]),
            Assert.IsType<float[]>(new CpuPrimitiveGraphExecutor(optimized, new EmptyTensorCatalog())
                .Execute(inputs).Outputs[new("output")]));
    }

    [Fact]
    public void Optimizer_OffDoesNotEliminateDeadPureNodes()
    {
        var graph = CreateDeadTemporaryGraph();

        var execution = new GraphOptimizer().Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.Off));

        Assert.Equal(graph.Nodes.Count, execution.Nodes.Count);
        Assert.Equal(["dead-fill", "dead-copy"],
            execution.Nodes.Skip(2).Take(2).SelectMany(node =>
                node.Source.LogicalNodes.Select(source => source.Value)));

        var catalog = new ExecutionKernelCatalog("test.backend",
            [new("test.ab"), new("test.c"), PortableTensorOperationContracts.Fill,
                PrimitiveGraphOperations.Copy],
            [new GraphNodeDefinition("ab", new("test.ab"), [new("test.a"), new("test.b")])]);
        var canonicalized = new GraphOptimizer().Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.Off), catalog);

        Assert.Equal(4, canonicalized.Nodes.Count);
        Assert.Equal(["dead-fill", "dead-copy"],
            canonicalized.Nodes.Skip(1).Take(2).SelectMany(node =>
                node.Source.LogicalNodes.Select(source => source.Value)));
    }

    [Fact]
    public void Optimizer_PreservesAliasedReadWriteAndUnknownOperationVersions()
    {
        var original = CreateBoundaryGraph();
        var temp = new GraphResource(new("scratch"), "Scratch", GraphResourceKind.Temporary,
            GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float32, [1]));
        var alias = original.Nodes[0] with
        {
            Id = new("alias"),
            Operation = PrimitiveGraphOperations.Copy,
            Resources = [GraphBindings.Read("input", "scratch"), GraphBindings.Write("output", "scratch")],
            Dependencies = [],
        };
        var readWrite = alias with
        {
            Id = new("read-write"),
            Resources = [GraphBindings.ReadWrite("output", "scratch")],
        };
        var unknownVersion = alias with
        {
            Id = new("unknown-version"),
            Operation = new GraphOperationId(PrimitiveGraphOperations.Copy.Name, 2),
            Resources = [GraphBindings.Read("input", "x"), GraphBindings.Write("output", "scratch")],
        };
        var unknownSemantics = unknownVersion with
        {
            Id = new("unknown-operation"),
            Operation = new("custom.copy"),
        };
        foreach (var candidate in new[] { alias, readWrite, unknownVersion, unknownSemantics })
        {
            var graph = new LogicalGraph(original.Identity, original.Model, [.. original.Resources, temp],
                original.Regions, [.. original.Nodes, candidate], original.Inputs, original.Outputs);
            var execution = new GraphOptimizer().Optimize(graph);
            Assert.Equal(graph.Nodes.Count, execution.Nodes.Count);
        }

        var competingWriters = new LogicalGraph(original.Identity, original.Model,
            [.. original.Resources, temp], original.Regions,
            [.. original.Nodes, unknownVersion with { Operation = PrimitiveGraphOperations.Copy },
                unknownVersion with { Id = new("other-writer"), Operation = PrimitiveGraphOperations.Copy }],
            original.Inputs, original.Outputs);
        Assert.Equal(competingWriters.Nodes.Count,
            new GraphOptimizer().Optimize(competingWriters).Nodes.Count);
    }

    [Fact]
    public void Optimizer_KeepsPureWritesToGraphOutputsAndGraphState()
    {
        var original = CreateBoundaryGraph();
        var state = new GraphResource(new("state"), "State", GraphResourceKind.SessionState,
            GraphResourceLifetime.Session, new TensorDescriptor(GraphElementType.Float32, [1]));
        var output = original.Nodes[2] with
        {
            Operation = PrimitiveGraphOperations.Copy,
        };
        var stateWrite = output with
        {
            Id = new("state-write"),
            Resources = [GraphBindings.Read("input", "x"), GraphBindings.Write("output", "state")],
        };
        var graph = new LogicalGraph(original.Identity, original.Model, [.. original.Resources, state],
            original.Regions, [original.Nodes[0], original.Nodes[1], output, stateWrite],
            original.Inputs, original.Outputs,
            [new GraphStateSlot("state", new("state"))]);

        var execution = new GraphOptimizer().Optimize(graph);

        Assert.Equal(graph.Nodes.Count, execution.Nodes.Count);
        Assert.Equal(new ResourceId("state"), Assert.Single(execution.GraphState).Resource);
    }

    [Fact]
    public void Optimizer_DoesNotDiscardFusedNodeWithFirstWriteMetadata()
    {
        var original = CreateBoundaryGraph();
        var observer = new LogicalNode(new("observer"), new("test.observe"), new("stage.b"),
            [GraphBindings.Read("input", "y")], [new("b")], new Dictionary<string, string>(),
            new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32));
        var graph = new LogicalGraph(original.Identity, original.Model, original.Resources,
            original.Regions, [.. original.Nodes, observer], original.Inputs, original.Outputs);

        var execution = new GraphOptimizer().Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            [new SequenceFusionRule("ab", [new("test.a"), new("test.b")], PrimitiveGraphOperations.Copy)]);

        var fused = Assert.Single(execution.Nodes, node => node.Source.AppliedRuleId == "ab");
        Assert.Contains(new ResourceId("y"), fused.FirstWriteResources);
        Assert.Equal(["a", "b"], fused.Source.LogicalNodes.Select(id => id.Value));
    }

    [Fact]
    public void Optimizer_FusesTypedElementwiseExpressionOnlyWithMatchingImplementation()
    {
        var logical = CreateElementwiseChain();
        var optimizer = new GraphOptimizer();
        var off = optimizer.Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded),
            new ExpressionCatalog(true, true));
        var noImplementation = optimizer.Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            new ExpressionCatalog(true, false));
        var noCapability = optimizer.Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            new ExpressionCatalog(false, true));
        var on = optimizer.Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage),
            new ExpressionCatalog(true, true));

        Assert.Equal(3, off.Nodes.Count);
        Assert.Equal(3, noImplementation.Nodes.Count);
        Assert.Equal(3, noCapability.Nodes.Count);
        var fused = Assert.Single(on.Nodes);
        Assert.Equal(FusedElementwiseExpressionContract.Operation, fused.Operation);
        Assert.Equal(["input0", "input1", "input2", "output"],
            fused.Resources.Select(binding => binding.Port));
        Assert.Equal(["a", "b", "c"], fused.Source.LogicalNodes.Select(id => id.Value));
        var expression = FusedElementwiseExpressionContract.Read(
            fused, on.Resources.ToDictionary(resource => resource.Id));
        Assert.Equal([PrimitiveGraphOperations.Add, PrimitiveGraphOperations.Multiply, PrimitiveGraphOperations.Relu],
            expression.Steps.Select(step => step.Operation));
        var restored = GraphXml.DeserializeExecution(GraphXml.Serialize(on));
        var roundTripped = FusedElementwiseExpressionContract.Read(Assert.Single(restored.Nodes));
        Assert.Equal(expression.InputCount, roundTripped.InputCount);
        Assert.Equal(fused.Attributes[FusedElementwiseExpressionContract.Attribute],
            restored.Nodes[0].Attributes[FusedElementwiseExpressionContract.Attribute]);
        Assert.Throws<InvalidDataException>(() => FusedElementwiseExpressionContract.ToAttributes(
            expression with
            {
                Steps = [expression.Steps[0] with
                    { Operation = new GraphOperationId(PrimitiveGraphOperations.Add.Name, 2) },
                    .. expression.Steps.Skip(1)],
            }));

        var inputs = new Dictionary<ResourceId, Array>
        {
            [new("x")] = new float[] { 2, -4 },
            [new("y")] = new float[] { 1, 2 },
            [new("z")] = new float[] { 3, 2 },
        };
        var expected = Assert.IsType<float[]>(new CpuPrimitiveGraphExecutor(off, new EmptyTensorCatalog())
            .Execute(inputs).Outputs[new("output")]);
        var actual = EvaluateExpression(fused, expression, inputs);
        Assert.Equal(expected, actual);
        Assert.Equal([9f, 0f], actual);
    }

    [Fact]
    public void Optimizer_RejectsShapeMismatchAndSharedOrAliasedIntermediates()
    {
        var original = CreateElementwiseChain();
        var catalog = new ExpressionCatalog(true, true);
        ExecutionGraph Optimize(LogicalGraph graph) => new GraphOptimizer().Optimize(
            graph, new GraphOptimizationOptions(OptimizationBoundary.WithinStage), catalog);

        var mismatched = new LogicalGraph(original.Identity, original.Model,
            original.Resources.Select(resource => resource.Id.Value == "tmp1"
                ? resource with { Tensor = new TensorDescriptor(GraphElementType.Float32, [3]) }
                : resource),
            original.Regions, original.Nodes, original.Inputs, original.Outputs);
        Assert.Equal(3, Optimize(mismatched).Nodes.Count);

        var observer = original.Nodes[0] with
        {
            Id = new("observer"),
            Operation = new("test.observe"),
            Resources = [GraphBindings.Read("first", "tmp1"), GraphBindings.Read("second", "tmp2")],
            Dependencies = [new("c")],
        };
        var shared = new LogicalGraph(original.Identity, original.Model, original.Resources,
            original.Regions, [.. original.Nodes, observer], original.Inputs, original.Outputs);
        Assert.Equal(4, Optimize(shared).Nodes.Count);

        var alias = original.Nodes[1] with
        {
            Resources = [GraphBindings.Read("left", "tmp1"), GraphBindings.Read("right", "z"),
                GraphBindings.Write("output", "x")],
        };
        var afterAlias = original.Nodes[2] with
        {
            Resources = [GraphBindings.Read("input", "x"), GraphBindings.Write("output", "output")],
        };
        var aliased = new LogicalGraph(original.Identity, original.Model, original.Resources,
            original.Regions, [original.Nodes[0], alias, afterAlias], original.Inputs, original.Outputs);
        Assert.Equal(3, Optimize(aliased).Nodes.Count);

        var side = original.Nodes[0] with
        {
            Id = new("side"),
            Operation = new("test.observe"),
            Resources = [GraphBindings.Read("input", "x")],
            Dependencies = [new("a")],
        };
        var throughSide = original.Nodes[1] with { Dependencies = [new("a"), new("side")] };
        var indirectDependency = new LogicalGraph(original.Identity, original.Model, original.Resources,
            original.Regions, [original.Nodes[0], throughSide, side, original.Nodes[2]],
            original.Inputs, original.Outputs);
        Assert.Equal(4, Optimize(indirectDependency).Nodes.Count);
    }

    [Fact]
    public void Optimizer_PortableRwkv6HistogramShowsNoSizableViewOnlyOpportunity()
    {
        var graph = new PortableRwkv6GraphProvider().Build(new Rwkv6ShapeCatalog(2));
        var histogram = graph.Nodes.GroupBy(node => node.Operation)
            .ToDictionary(group => group.Key, group => group.Count());
        var writers = graph.Nodes.ToDictionary(
            node => node.Resources.Single(binding => binding.Port == "output").Resource);
        int Pairs(GraphOperationId first, GraphOperationId second) => graph.Nodes.Count(node =>
            node.Operation == second &&
            node.Resources.Any(binding => binding.Port == "input" &&
                writers.TryGetValue(binding.Resource, out var producer) && producer.Operation == first));
        var catalog = new ExecutionKernelCatalog("histogram",
            graph.Nodes.Select(node => node.Operation).Distinct());
        var optimizer = new GraphOptimizer();
        var off = optimizer.Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded), catalog);
        var optimized = optimizer.Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted), catalog);

        Assert.Equal(299, graph.Nodes.Count);
        Assert.Equal(299, off.Nodes.Count);
        Assert.Equal(58, histogram[PortableTensorOperationContracts.Reshape]);
        Assert.Equal(54, histogram[PrimitiveGraphOperations.Add]);
        Assert.Equal(44, histogram[PrimitiveGraphOperations.Multiply]);
        Assert.Equal(23, histogram[PrimitiveGraphOperations.MatVec]);
        Assert.Equal(22, histogram[PortableTensorOperationContracts.Broadcast]);
        Assert.Equal(7, histogram[PrimitiveGraphOperations.Copy]);
        Assert.Equal(10, Pairs(PortableTensorOperationContracts.Reshape,
            PortableTensorOperationContracts.Reshape));
        Assert.Equal(10, Pairs(PortableTensorOperationContracts.Reshape,
            PortableTensorOperationContracts.Broadcast));
        Assert.Equal(0, Pairs(PrimitiveGraphOperations.Copy,
            PortableTensorOperationContracts.Reshape));
        Assert.Equal(0, Pairs(PrimitiveGraphOperations.Copy,
            PrimitiveGraphOperations.Copy));
        Assert.Equal(0, Pairs(PortableTensorOperationContracts.Broadcast,
            PortableTensorOperationContracts.Broadcast));
        Assert.Equal(299, optimized.Nodes.Count);
        Assert.Equal(58, optimized.Nodes.Count(node =>
            node.Operation == PortableTensorOperationContracts.Reshape));
        Assert.Equal(off.Nodes.Select(node => node.Operation),
            optimized.Nodes.Select(node => node.Operation));
        Assert.Equal(graph.Outputs, optimized.Outputs);
        Assert.Equal(graph.GraphState.Slots, optimized.GraphState.Slots);
    }

    [Fact]
    public void Optimizer_DoesNotCollapseShapeCompatibleButValueChangingReshapeBroadcast()
    {
        var graph = new LogicalGraphBuilder(new GraphIdentity("test", 1, "broadcast"),
                new GraphModelSignature(1, 4, 1, 1, 4, "test.state"))
            .AddRegion("graph", GraphRegionTypes.Graph, "Graph")
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2, 2]), graphInput: true)
            .AddResource("reshaped", "Reshaped", GraphResourceKind.Temporary,
                GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float32, [2, 2, 1]))
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2, 2, 2]), graphOutput: true)
            .AddNode("reshape", PortableTensorOperationContracts.Reshape, "graph",
                [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "reshaped")])
            .AddNode("broadcast", PortableTensorOperationContracts.Broadcast, "graph",
                [GraphBindings.Read("input", "reshaped"), GraphBindings.Write("output", "output")],
                ["reshape"])
            .Build();
        var catalog = new ExecutionKernelCatalog("views",
            [PortableTensorOperationContracts.Reshape, PortableTensorOperationContracts.Broadcast]);
        var optimized = new GraphOptimizer().Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted), catalog);
        var inputs = new Dictionary<ResourceId, Array> { [new("input")] = new float[] { 1, 2, 3, 4 } };
        var expected = Assert.IsType<float[]>(
            new CpuPrimitiveGraphExecutor(optimized, new EmptyTensorCatalog())
                .Execute(inputs).Outputs[new("output")]);

        Assert.Equal(2, optimized.Nodes.Count);
        Assert.Equal([1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f], expected);
        var direct = new LogicalGraphBuilder(new GraphIdentity("test", 1, "direct-broadcast"), graph.Model)
            .AddRegion("graph", GraphRegionTypes.Graph, "Graph")
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2, 2]), graphInput: true)
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2, 2, 2]), graphOutput: true)
            .AddNode("broadcast", PortableTensorOperationContracts.Broadcast, "graph",
                [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "output")])
            .Build();
        PortableTensorOperationContracts.ValidateGraph(direct);
        var wrong = Assert.IsType<float[]>(
            new CpuPrimitiveGraphExecutor(direct, new EmptyTensorCatalog())
                .Execute(inputs).Outputs[new("output")]);
        Assert.False(expected.SequenceEqual(wrong));
    }

    [Fact]
    public void Optimizer_ChecksCapabilitiesAfterAllFusionPasses()
    {
        var graph = CreateBoundaryGraph();
        var execution = new GraphOptimizer().Optimize(
            graph,
            new GraphOptimizationOptions(OptimizationBoundary.WithinLayer),
            [
                new SequenceFusionRule(
                    "fuse-a-b",
                    [new GraphOperationId("test.a"), new GraphOperationId("test.b")],
                    new GraphOperationId("test.ab")),
                new SequenceFusionRule(
                    "fuse-ab-c",
                    [new GraphOperationId("test.ab"), new GraphOperationId("test.c")],
                    new GraphOperationId("test.abc")),
            ],
            new SingleOperationCapabilities(new GraphOperationId("test.abc")));

        Assert.Equal(new GraphOperationId("test.abc"), Assert.Single(execution.Nodes).Operation);
    }

    [Fact]
    public void Optimizer_ReportsEveryUnsupportedCatalogOperationWithSources()
    {
        var graph = CreateBoundaryGraph();
        var catalog = new ExecutionKernelCatalog("test.backend", [new GraphOperationId("test.b")]);

        var exception = Assert.Throws<BackendPreparationException>(() =>
            new GraphOptimizer().Optimize(graph,
                new GraphOptimizationOptions(OptimizationBoundary.Off,
                    DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded),
                catalog));

        Assert.Equal("test.backend", exception.BackendId);
        Assert.Equal(["test.a", "test.c"], exception.Diagnostics.Select(diagnostic => diagnostic.Operation?.Name));
        Assert.Equal(["a", "c"], exception.Diagnostics.SelectMany(diagnostic =>
            diagnostic.SourceNodes.Select(source => source.Value)));
        Assert.All(exception.Diagnostics, diagnostic =>
            Assert.Equal(BackendPreparationFailureReason.UnsupportedOperation, diagnostic.Reason));
    }

    [Fact]
    public void Optimizer_ReportsUnsupportedOperationForNonCatalogCapabilities()
    {
        var exception = Assert.Throws<BackendPreparationException>(() =>
            new GraphOptimizer().Optimize(CreateBoundaryGraph(),
                new GraphOptimizationOptions(OptimizationBoundary.Off),
                capabilities: new SingleOperationCapabilities(new("test.b"))));

        Assert.Equal(2, exception.Diagnostics.Count);
        Assert.Contains("test.a@1", exception.Message, StringComparison.Ordinal);
        Assert.All(exception.Diagnostics, diagnostic => Assert.NotNull(diagnostic.NodeId));
    }

    [Fact]
    public void Optimizer_SkipsUnsupportedTerminalCatalogFusion()
    {
        var graph = CreateBoundaryGraph();
        var catalog = new ExecutionKernelCatalog("test.backend",
            [new("test.a"), new("test.b"), new("test.c")],
            fusedOperators:
            [
                new FusedOperatorDescription("unsupported",
                    new SequenceFusionRule("unsupported", [new("test.a"), new("test.b")], new("test.ab"))),
            ]);

        var execution = new GraphOptimizer().Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.WithinStage), catalog);

        Assert.Equal(graph.Nodes.Select(node => node.Operation), execution.Nodes.Select(node => node.Operation));
        Assert.All(execution.Nodes, node => Assert.Single(node.Source.LogicalNodes));
    }

    [Fact]
    public void Optimizer_OffCanonicalizeReportsUnmatchedUnsupportedLogicalNodes()
    {
        var graph = CreateBoundaryGraph();
        var catalog = new ExecutionKernelCatalog("test.backend",
            [new("test.c")],
            [new GraphNodeDefinition("ab", new("test.ab"), [new("test.a"), new("test.b")])]);

        var exception = Assert.Throws<BackendPreparationException>(() =>
            new GraphOptimizer().Optimize(graph,
                new GraphOptimizationOptions(OptimizationBoundary.Off), catalog));

        Assert.Equal(2, exception.Diagnostics.Count);
        Assert.Equal(["a", "b"], exception.Diagnostics.SelectMany(diagnostic =>
            diagnostic.SourceNodes.Select(source => source.Value)));
        Assert.All(exception.Diagnostics, diagnostic =>
            Assert.Equal(BackendPreparationFailureReason.UnsupportedOperation, diagnostic.Reason));
    }

    [Fact]
    public void Optimizer_OffCanonicalizeCanReplaceUnsupportedLogicalNodes()
    {
        var graph = CreateBoundaryGraph();
        var catalog = new ExecutionKernelCatalog("test.backend",
            [new("test.ab"), new("test.c")],
            [new GraphNodeDefinition("ab", new("test.ab"), [new("test.a"), new("test.b")])]);

        var execution = new GraphOptimizer().Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.Off), catalog);

        Assert.Equal(["test.ab", "test.c"], execution.Nodes.Select(node => node.Operation.Name));
        Assert.Equal(["a", "b"], execution.Nodes[0].Source.LogicalNodes.Select(source => source.Value));
    }

    [Fact]
    public void Optimizer_GreedilyCanonicalizesDefinitionsBeforeBackendFusion()
    {
        var graph = CreateBoundaryGraph();
        var abc = new GraphOperationId("test.abc");
        var catalog = new ExecutionKernelCatalog(
            "test",
            [abc],
            [
                new GraphNodeDefinition(
                    "test.definition.ab",
                    new GraphOperationId("test.ab"),
                    [new GraphOperationId("test.a"), new GraphOperationId("test.b")]),
            ],
            [
                new FusedOperatorDescription(
                    "test.fusion.abc",
                    new SequenceFusionRule(
                        "test.fusion.abc",
                        [new GraphOperationId("test.ab"), new GraphOperationId("test.c")],
                        abc,
                        OptimizationBoundary.WithinLayer)),
            ]);

        var execution = new GraphOptimizer().Optimize(
            graph,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted),
            catalog);

        var node = Assert.Single(execution.Nodes);
        Assert.Equal(abc, node.Operation);
        Assert.Equal(["a", "b", "c"], node.Source.LogicalNodes.Select(source => source.Value));
    }

    [Fact]
    public void Optimizer_RespectsKernelPatternScopeUnderUnrestrictedPolicy()
    {
        var graph = CreateBoundaryGraph();
        var ab = new GraphOperationId("test.ab");
        var c = new GraphOperationId("test.c");
        var abc = new GraphOperationId("test.abc");
        var catalog = new ExecutionKernelCatalog(
            "test",
            [ab, c, abc],
            [
                new GraphNodeDefinition(
                    "test.definition.ab",
                    ab,
                    [new GraphOperationId("test.a"), new GraphOperationId("test.b")]),
            ],
            [
                new FusedOperatorDescription(
                    "test.fusion.abc",
                    new SequenceFusionRule(
                        "test.fusion.abc",
                        [ab, c],
                        abc,
                        OptimizationBoundary.WithinStage)),
            ]);

        var execution = new GraphOptimizer().Optimize(
            graph,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted),
            catalog);

        Assert.Equal([ab, c], execution.Nodes.Select(node => node.Operation));
    }

    [Fact]
    public void Optimizer_DefinitionPolicyIsIndependentFromFusionBoundary()
    {
        var graph = CreateBoundaryGraph();
        var a = new GraphOperationId("test.a");
        var b = new GraphOperationId("test.b");
        var c = new GraphOperationId("test.c");
        var ab = new GraphOperationId("test.ab");
        var catalog = new ExecutionKernelCatalog(
            "test",
            [a, b, c, ab],
            [new GraphNodeDefinition("test.definition.ab", ab, [a, b])]);

        var canonical = new GraphOptimizer().Optimize(
            graph,
            new GraphOptimizationOptions(OptimizationBoundary.Off),
            catalog);
        var expanded = new GraphOptimizer().Optimize(
            graph,
            new GraphOptimizationOptions(
                OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded),
            catalog);

        Assert.Equal([ab, c], canonical.Nodes.Select(node => node.Operation));
        Assert.Equal([a, b, c], expanded.Nodes.Select(node => node.Operation));
    }

    [Fact]
    public void Optimizer_FusionPropagatesHighestSourcePrecisionRequirement()
    {
        var graph = CreateBoundaryGraph(
            GraphElementType.Float16,
            new PrecisionRequirement(GraphElementType.Float16, GraphElementType.Float32));
        var ab = new GraphOperationId("test.ab");
        var abc = new GraphOperationId("test.abc");
        var catalog = new ExecutionKernelCatalog(
            "test",
            [abc],
            [new GraphNodeDefinition("test.definition.ab", ab, [new GraphOperationId("test.a"), new GraphOperationId("test.b")])],
            [
                new FusedOperatorDescription(
                    "test.fusion.abc",
                    new SequenceFusionRule(
                        "test.fusion.abc",
                        [ab, new GraphOperationId("test.c")],
                        abc,
                        OptimizationBoundary.WithinLayer)),
            ]);

        var execution = new GraphOptimizer().Optimize(
            graph,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted),
            catalog);

        var fused = Assert.Single(execution.Nodes);
        Assert.Equal(abc, fused.Operation);
        Assert.Equal(GraphElementType.Float16, fused.Requirements.MinimumArithmeticType);
        Assert.Equal(GraphElementType.Float32, fused.Requirements.MinimumAccumulatorType);
    }

    [Fact]
    public void ImplementationSelector_ChoosesLowestQualifyingPrecisionBeforePreference()
    {
        var operation = new GraphOperationId("test.mat-vec");
        var signature = new OperatorSignature(
            [GraphElementType.Float16, GraphElementType.Float16],
            [GraphElementType.Float16]);
        var selected = OperatorImplementationSelector.Select(
            operation,
            signature,
            new PrecisionRequirement(GraphElementType.Float16, GraphElementType.Float32),
            [
                new("native", operation, signature, new(GraphElementType.Float16, GraphElementType.Float16)),
                new("promoted-preferred", operation, signature, new(GraphElementType.Float32, GraphElementType.Float32), 0),
                new("minimum", operation, signature, new(GraphElementType.Float16, GraphElementType.Float32), 100),
            ]);

        Assert.Equal("minimum", Assert.IsType<OperatorImplementationDescription>(selected).ImplementationId);
    }

    [Fact]
    public void ImplementationSelector_UsesPreferenceForIncomparableMinimumProfiles()
    {
        var operation = new GraphOperationId("test.reduce");
        var signature = new OperatorSignature([GraphElementType.Float16], [GraphElementType.Float16]);
        var selected = OperatorImplementationSelector.Select(
            operation,
            signature,
            new PrecisionRequirement(GraphElementType.Float16, GraphElementType.Float16),
            [
                new("fp16-fp32", operation, signature, new(GraphElementType.Float16, GraphElementType.Float32), 10),
                new("fp32-fp16", operation, signature, new(GraphElementType.Float32, GraphElementType.Float16), 0),
            ]);

        Assert.Equal("fp32-fp16", Assert.IsType<OperatorImplementationDescription>(selected).ImplementationId);
    }

    [Fact]
    public void BackendPreflight_ReturnsAllUnsupportedSignatureAndPrecisionFailures()
    {
        var logical = CreateBoundaryGraph(
            GraphElementType.Float16,
            new PrecisionRequirement(GraphElementType.Float16, GraphElementType.Float32));
        var execution = new GraphOptimizer().Optimize(
            logical,
            new GraphOptimizationOptions(OptimizationBoundary.Off));
        var fp16Signature = new OperatorSignature([GraphElementType.Float16], [GraphElementType.Float16]);
        var fp32Signature = new OperatorSignature([GraphElementType.Float32], [GraphElementType.Float32]);

        var result = BackendPreparation.Preflight(
            execution,
            node => node.Operation.Name switch
            {
                "test.a" => [],
                "test.b" =>
                [
                    new("b-native", node.Operation, fp16Signature, new(GraphElementType.Float16, GraphElementType.Float16)),
                ],
                _ =>
                [
                    new("c-wrong-signature", node.Operation, fp32Signature, new(GraphElementType.Float32, GraphElementType.Float32)),
                ],
            });

        Assert.False(result.Succeeded);
        Assert.Equal(3, result.Diagnostics.Count);
        Assert.Equal(
            [
                BackendPreparationFailureReason.UnsupportedOperation,
                BackendPreparationFailureReason.PrecisionRequirementNotMet,
                BackendPreparationFailureReason.SignatureMismatch,
            ],
            result.Diagnostics.Select(diagnostic => diagnostic.Reason));

        var failure = new BackendPreparationResult.Failure(result.Diagnostics);
        var exception = Assert.Throws<BackendPreparationException>(() =>
            failure.GetPlanOrThrow("test"));
        Assert.Equal(3, exception.Diagnostics.Count);
        Assert.Contains("e0000", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PortableRwkv6Provider_ProducesValidCpuGraphWithStateAndWeightBindings()
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv6);
        var logical = new PortableRwkv6GraphProvider().Build(catalog);
        Assert.Equal("RWKV6_State", logical.GraphState.Schema.Name);
        var backend = CpuPrimitiveGraphBackend.Instance;
        var execution = new GraphOptimizer().Optimize(
            logical,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted),
            backend.KernelCatalog);

        GraphValidator.Validate(execution);
        Assert.Equal(catalog.LayerCount * 3, logical.Resources.Count(resource => resource.Kind == GraphResourceKind.SessionState));
        Assert.Single(logical.Regions, region => region.Type == GraphRegionTypes.Graph);
        Assert.Equal(logical.Nodes.Count, execution.Nodes.Sum(node => node.Source.LogicalNodes.Count));
        Assert.Contains(execution.Resources, resource => resource.Kind == GraphResourceKind.Weight);
        Assert.NotEmpty(backend.Prepare(execution).GetPlanOrThrow("cpu").Graph.Nodes);
        Assert.Contains("\"Kind\": \"LogicalGraph\"", GraphJson.Serialize(logical), StringComparison.Ordinal);
        Assert.Contains("\"Kind\": \"ExecutionGraph\"", GraphJson.Serialize(execution), StringComparison.Ordinal);
    }

    [Fact]
    public void PortableRwkv7Provider_ModelsFirstValueAndProducesCpuPlan()
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv7Fp32);
        var logical = new PortableRwkv7GraphProvider().Build(catalog);
        Assert.Equal("RWKV7_State", logical.GraphState.Schema.Name);
        var backend = CpuPrimitiveGraphBackend.Instance;
        var execution = new GraphOptimizer().Optimize(
            logical,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted),
            backend.KernelCatalog);

        GraphValidator.Validate(execution);
        Assert.NotEmpty(logical.Nodes);
        Assert.Equal(catalog.LayerCount * 3, logical.Resources.Count(resource => resource.Kind == GraphResourceKind.SessionState));
        Assert.Equal(logical.Nodes.Count, execution.Nodes.Sum(node => node.Source.LogicalNodes.Count));
        Assert.NotEmpty(backend.Prepare(execution).GetPlanOrThrow("cpu").Graph.Nodes);
    }

    private static LogicalGraph CreateElementwiseChain() =>
        new LogicalGraphBuilder(new GraphIdentity("test", 1, "elementwise"),
                new GraphModelSignature(1, 2, 1, 1, 2, "test.state"))
            .AddRegion("graph", GraphRegionTypes.Graph, "Graph")
            .AddRegion("stage", GraphRegionTypes.Stage, "Stage", "graph")
            .AddResource("x", "X", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("y", "Y", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("z", "Z", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("tmp1", "Tmp1", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("tmp2", "Tmp2", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
                new TensorDescriptor(GraphElementType.Float32, [2]))
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddNode("a", PrimitiveGraphOperations.Add, "stage",
                [GraphBindings.Read("left", "x"), GraphBindings.Read("right", "y"),
                    GraphBindings.Write("output", "tmp1")])
            .AddNode("b", PrimitiveGraphOperations.Multiply, "stage",
                [GraphBindings.Read("left", "tmp1"), GraphBindings.Read("right", "z"),
                    GraphBindings.Write("output", "tmp2")], ["a"])
            .AddNode("c", PrimitiveGraphOperations.Relu, "stage",
                [GraphBindings.Read("input", "tmp2"), GraphBindings.Write("output", "output")], ["b"])
            .Build();

    private static float[] EvaluateExpression(
        ExecutionNode node,
        FusedElementwiseExpression expression,
        IReadOnlyDictionary<ResourceId, Array> inputs)
    {
        var values = new List<float[]>();
        foreach (var step in expression.Steps)
        {
            float[] Resolve(ElementwiseOperand operand) =>
                operand.StepIndex is int stepIndex
                    ? values[stepIndex]
                    : (float[])inputs[node.Resources[operand.InputIndex!.Value].Resource];
            var first = Resolve(step.Operands[0]);
            var second = step.Operands.Count == 2 ? Resolve(step.Operands[1]) : null;
            values.Add(Enumerable.Range(0, first.Length).Select(index => step.Operation switch
            {
                var operation when operation == PrimitiveGraphOperations.Add => first[index] + second![index],
                var operation when operation == PrimitiveGraphOperations.Multiply => first[index] * second![index],
                var operation when operation == PrimitiveGraphOperations.Relu => MathF.Max(0, first[index]),
                _ => throw new InvalidOperationException("Unexpected test expression step."),
            }).ToArray());
        }
        return values[^1];
    }

    private sealed class ExpressionCatalog(bool supportsExpression, bool hasImplementation)
        : IExecutionKernelCatalog, IFusedElementwiseExpressionProvider
    {
        public string BackendId => "test.expression";
        public IReadOnlySet<GraphOperationId> SupportedOperations { get; } =
            new[]
            {
                PrimitiveGraphOperations.Add, PrimitiveGraphOperations.Multiply,
                PrimitiveGraphOperations.Relu, new GraphOperationId("test.observe"),
            }.Concat(supportsExpression
                ? [FusedElementwiseExpressionContract.Operation]
                : Array.Empty<GraphOperationId>()).ToHashSet();
        public IReadOnlyList<GraphNodeDefinition> NodeDefinitions => [];
        public bool Supports(GraphOperationId operation) => SupportedOperations.Contains(operation);
        public IReadOnlyList<OperatorImplementationDescription> GetExpressionImplementations(
            OperatorSignature signature, TensorDescriptor tensor) =>
            hasImplementation && tensor.Dimensions.SequenceEqual([2]) &&
            signature.InputTypes.Count == 3
                ? [new OperatorImplementationDescription("test.expression.fp32",
                    FusedElementwiseExpressionContract.Operation, signature,
                    new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32))]
                : [];
    }

    private static LogicalGraph CreateDeadTemporaryGraph()
    {
        var original = CreateBoundaryGraph();
        var first = new GraphResource(new("dead-first"), "Dead first", GraphResourceKind.Temporary,
            GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float32, [1]));
        var second = new GraphResource(new("dead-second"), "Dead second", GraphResourceKind.Temporary,
            GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float32, [1]));
        var requirement = new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32);
        var fill = new LogicalNode(new("dead-fill"), PortableTensorOperationContracts.Fill, new("stage.a"),
            [GraphBindings.Write("output", "dead-first")], [new("a")],
            new Dictionary<string, string> { ["value"] = "1" }, requirement);
        var copy = new LogicalNode(new("dead-copy"), PrimitiveGraphOperations.Copy, new("stage.a"),
            [GraphBindings.Read("input", "dead-first"), GraphBindings.Write("output", "dead-second")],
            [new("dead-fill")], new Dictionary<string, string>(), requirement);
        var consumer = original.Nodes[2] with { Dependencies = [new("b"), new("dead-copy")] };
        return new LogicalGraph(original.Identity, original.Model,
            [.. original.Resources, first, second], original.Regions,
            [original.Nodes[0], original.Nodes[1], fill, copy, consumer],
            original.Inputs, original.Outputs);
    }

    private static LogicalGraph CreateBoundaryGraph(
        GraphElementType elementType = GraphElementType.Float32,
        PrecisionRequirement? middleRequirement = null)
    {
        var builder = new LogicalGraphBuilder(
            new GraphIdentity("test", 1, "boundary"),
            new GraphModelSignature(1, 1, 1, 1, 1, "test.state@1"));
        builder
            .AddRegion("graph", GraphRegionTypes.Graph, "Graph")
            .AddRegion("layer.0", GraphRegionTypes.Layer, "Layer", "graph")
            .AddRegion("stage.a", GraphRegionTypes.Stage, "Stage A", "layer.0")
            .AddRegion("stage.b", GraphRegionTypes.Stage, "Stage B", "layer.0")
            .AddResource("x", "X", GraphResourceKind.Input, GraphResourceLifetime.External, new TensorDescriptor(elementType, [1]), graphInput: true)
            .AddResource("y", "Y", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, new TensorDescriptor(elementType, [1]))
            .AddResource("z", "Z", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, new TensorDescriptor(elementType, [1]))
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External, new TensorDescriptor(elementType, [1]), graphOutput: true)
            .AddNode("a", new GraphOperationId("test.a"), "stage.a", [GraphBindings.Read("input", "x"), GraphBindings.Write("output", "y")])
            .AddNode(
                "b",
                new GraphOperationId("test.b"),
                "stage.a",
                [GraphBindings.Read("input", "y"), GraphBindings.Write("output", "z")],
                ["a"],
                requirements: middleRequirement)
            .AddNode("c", new GraphOperationId("test.c"), "stage.b", [GraphBindings.Read("input", "z"), GraphBindings.Write("output", "output")], ["b"]);
        return builder.Build();
    }

    private sealed class SingleOperationCapabilities(GraphOperationId operation) : IExecutionCapabilityProvider
    {
        public bool Supports(GraphOperationId candidate) => candidate == operation;
    }

    private sealed class EmptyTensorCatalog : IModelTensorCatalog
    {
        public int VocabularySize => 1;
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

    private sealed class Rwkv6ShapeCatalog : IModelTensorCatalog
    {
        private readonly Dictionary<string, IModelTensor> tensors = new(StringComparer.Ordinal);

        public Rwkv6ShapeCatalog(int layers)
        {
            LayerCount = layers;
            void Add(string name, params int[] shape) => tensors.Add(name, new ShapeTensor(name, shape));
            Add("emb.weight", 4, 5);
            Add("head.weight", 4, 5);
            foreach (var name in new[] { "ln_out.weight", "ln_out.bias",
                "blocks.0.ln0.weight", "blocks.0.ln0.bias" }) Add(name, 4);
            for (var layer = 0; layer < layers; layer++)
            {
                var prefix = $"blocks.{layer}.";
                foreach (var name in new[]
                {
                    "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
                    "att.time_maa_x", "att.time_maa_w", "att.time_maa_k",
                    "att.time_maa_v", "att.time_maa_r", "att.time_maa_g",
                    "att.time_decay", "att.ln_x.weight", "att.ln_x.bias",
                    "ffn.time_maa_k", "ffn.time_maa_r",
                }) Add(prefix + name, 4);
                Add(prefix + "att.time_faaaa", 1, 2, 2);
                Add(prefix + "att.time_maa_w1", 4, 10);
                Add(prefix + "att.time_maa_w2", 2, 4, 5);
                Add(prefix + "att.time_decay_w1", 4, 3);
                Add(prefix + "att.time_decay_w2", 3, 4);
                foreach (var name in new[]
                {
                    "att.receptance.weight", "att.key.weight", "att.value.weight",
                    "att.gate.weight", "att.output.weight", "ffn.receptance.weight",
                }) Add(prefix + name, 4, 4);
                Add(prefix + "ffn.key.weight", 4, 6);
                Add(prefix + "ffn.value.weight", 6, 4);
            }
        }

        public int VocabularySize => 5;
        public int EmbeddingSize => 4;
        public int LayerCount { get; }
        public IReadOnlyCollection<string> Names => tensors.Keys;
        public bool TryGet(string name, out IModelTensor tensor) => tensors.TryGetValue(name, out tensor!);
        public IModelTensor GetRequired(string name) => tensors[name];

        private sealed class ShapeTensor(string name, int[] shape) : IModelTensor
        {
            public string Name => name;
            public IReadOnlyList<int> Dimensions => shape;
            public RwkvTensorDataType DataType => RwkvTensorDataType.Float32;
            public ReadOnlySpan<float> FloatValues => [];
            public ReadOnlySpan<Half> HalfValues => [];
        }
    }
}
