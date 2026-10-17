using SharpInference.Instructions;
using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;
using SharpInference.Graphs.Logical;

namespace SharpInference.Tests;

public sealed class ExecutionGraphPostprocessorTests
{
    private static readonly TensorDescriptor F = new(ElementType.Float32, [4]);
    private static readonly ExecutionGraphPostprocessContext Context = new(
        LogicalGraphDefinition.Define("logical", _ => { }),
        new("arbitrary-model", "arbitrary-backend"), 0, "test");

    [Fact]
    public void DependencyReductionPreservesEveryReachabilityAndDataEdge()
    {
        var builder = Builder();
        Add(builder, "a", "a");
        Add(builder, "b", "b", dependencies: ["a"]);
        Add(builder, "c", "left-output", dependencies: ["a", "b"]);
        Add(builder, "d", "right-output", dependencies: ["a", "b", "c"]);
        var before = builder.Build();
        var after = Process(new DependencyReductionPostprocessor(), before).Graphs[0];
        Assert.Equal(3, after.Structure.Edges.OfType<ExecutionDependencyEdge>().Count());
        foreach (var first in before.Nodes)
        {
            foreach (var second in before.Nodes)
            {
                Assert.Equal(
                    before.Structure.DependencyView().IsReachable(first.GraphId, second.GraphId),
                    after.Structure.DependencyView().IsReachable(first.GraphId, second.GraphId));
            }
        }

        Assert.Equal(before.Structure.Edges.Where(edge => edge is not ExecutionDependencyEdge),
            after.Structure.Edges.Where(edge => edge is not ExecutionDependencyEdge));
        AssertIdempotent(new DependencyReductionPostprocessor(), new([before]));
    }

    [Fact]
    public void DependencyReductionRetainsOneParallelEdge()
    {
        var builder = Builder();
        Add(builder, "a", "a");
        Add(builder, "b", "b", dependencies: ["a"]);
        var graph = builder.Build();
        var dependency = graph.Structure.Edges.OfType<ExecutionDependencyEdge>().Single();
        graph = WithEdges(graph, graph.Structure.Edges.Append(dependency with { Id = new("parallel") }));
        var after = Process(new DependencyReductionPostprocessor(), graph).Graphs[0];
        Assert.Single(after.Structure.Edges.OfType<ExecutionDependencyEdge>());
    }

    [Fact]
    public void SerialResourcesReuseWithoutChangingOperationsOrDependencies()
    {
        var builder = Builder();
        Add(builder, "a", "a");
        Add(builder, "read", "output", "a", ["a"]);
        Add(builder, "b", "b", dependencies: ["read"]);
        var before = builder.Build();
        var after = Process(new TemporaryResourceReusePostprocessor(), before).Graphs[0];
        Assert.DoesNotContain(after.Resources, resource => resource.Id.Value == "b");
        Assert.Equal("a", after.Bindings(after.Nodes.Single(node => node.Id.Value == "b"))
            .Single().Resource.Resource.Value);
        Assert.Equal(before.Nodes, after.Nodes);
        Assert.Equal(before.Structure.Edges.OfType<ExecutionDependencyEdge>(),
            after.Structure.Edges.OfType<ExecutionDependencyEdge>());
        AssertIdempotent(new TemporaryResourceReusePostprocessor(), new([before]));
    }

    [Fact]
    public void IndependentBranchesCannotReuseEvenWhenTopologicalIntervalsDoNotOverlap()
    {
        var builder = Builder();
        Add(builder, "a", "a");
        Add(builder, "read-a", "output", "a", ["a"]);
        Add(builder, "b", "b");
        var before = builder.Build();
        Assert.Equal(before.Resources.Count,
            Process(new TemporaryResourceReusePostprocessor(), before).Graphs[0].Resources.Count);
    }

    [Fact]
    public void DiamondMustWaitForEveryLastReader()
    {
        var builder = Builder();
        Add(builder, "a", "a");
        Add(builder, "left", "left-output", "a", ["a"]);
        Add(builder, "right", "right-output", "a", ["a"]);
        Add(builder, "b", "b", dependencies: ["left"]);
        var before = builder.Build();
        var after = Process(new TemporaryResourceReusePostprocessor(), before).Graphs[0];
        Assert.Contains(after.Resources, resource => resource.Id.Value == "b");
        Assert.Equal("b", after.Bindings(after.Nodes.Single(node => node.Id.Value == "b"))
            .Single().Resource.Resource.Value);
    }

    [Fact]
    public void DiamondJoinAllowsReuseAfterAllReaders()
    {
        var builder = Builder();
        Add(builder, "a", "a");
        Add(builder, "left", "left-output", "a", ["a"]);
        Add(builder, "right", "right-output", "a", ["a"]);
        Add(builder, "b", "b", dependencies: ["left", "right"]);
        Assert.DoesNotContain(Process(new TemporaryResourceReusePostprocessor(), builder.Build()).Graphs[0].Resources,
            resource => resource.Id.Value == "b");
    }

    [Fact]
    public void ResourceSafetyDoesNotDependOnDeclarationOrderOrNames()
    {
        var builder = Builder();
        Add(builder, "z-last-name", "a");
        Add(builder, "a-first-name", "output", "a", ["z-last-name"]);
        Add(builder, "middle-name", "b", dependencies: ["a-first-name"]);
        var graph = builder.Build();
        var reversed = new ExecutionGraph(graph.Id,
            new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(
                graph.Structure.Nodes.Reverse(), graph.Structure.Edges.Reverse()));
        var pass = new TemporaryResourceReusePostprocessor();
        foreach (var candidate in new[] { graph, reversed })
        {
            var after = Process(pass, candidate).Graphs[0];
            Assert.DoesNotContain(after.Resources, resource => resource.Id.Value == "b");
            Assert.Equal("a", after.Bindings(after.Nodes.Single(node => node.Id.Value == "middle-name"))
                .Single().Resource.Resource.Value);
        }
    }

    [Theory]
    [InlineData("initial-read")]
    [InlineData("read-write")]
    [InlineData("unknown-write")]
    [InlineData("unknown-version")]
    public void UnprovenInitializationCannotReuse(string scenario)
    {
        var builder = Builder();
        Add(builder, "a", "a");
        if (scenario == "initial-read")
        {
            Add(builder, "b", "output", "b", ["a"]);
        }
        else if (scenario == "read-write")
        {
            builder.AddOperation(Operation("b", "core.copy"),
                [Binding("output", "b", ResourceAccess.ReadWrite, "b")], [new("a")]);
        }
        else
        {
            builder.AddOperation(Operation("b", scenario == "unknown-write" ? "custom.partial" : "core.copy",
                scenario == "unknown-version" ? 2 : 1),
                [Binding("output", "b", ResourceAccess.Write, "b")], [new("a")]);
        }

        Assert.Contains(Process(new TemporaryResourceReusePostprocessor(), builder.Build()).Graphs[0].Resources,
            resource => resource.Id.Value == "b");
    }

    [Theory]
    [InlineData(GraphDefinitionResourceKind.State)]
    [InlineData(GraphDefinitionResourceKind.Output)]
    [InlineData(GraphDefinitionResourceKind.InOut)]
    [InlineData(GraphDefinitionResourceKind.Weight)]
    [InlineData(GraphDefinitionResourceKind.Input)]
    public void NonTemporaryResourcesAreProtected(GraphDefinitionResourceKind kind)
    {
        var builder = Builder(secondKind: kind);
        Add(builder, "a", "a");
        if (kind == GraphDefinitionResourceKind.Weight)
        {
            Add(builder, "b", "output", "b", ["a"]);
        }
        else
        {
            Add(builder, "b", "b", dependencies: ["a"]);
        }
        Assert.Contains(Process(new TemporaryResourceReusePostprocessor(), builder.Build()).Graphs[0].Resources,
            resource => resource.Id.Value == "b");
    }

    [Fact]
    public void ExecutionGraphRejectsWeightWrites()
    {
        var builder = Builder(secondKind: GraphDefinitionResourceKind.Weight);
        Add(builder, "write-weight", "b");

        var exception = Assert.Throws<InvalidDataException>(() => builder.Build());

        Assert.Contains("EG0004", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BindingKeysAndIncompatibleDescriptorsAreProtected()
    {
        foreach (var tensor in new[]
        {
            new TensorDescriptor(ElementType.Float32, [2, 2]),
            new TensorDescriptor(ElementType.Float16, [4]),
            new TensorDescriptor(ElementType.Float32, [4], "other-layout"),
        })
        {
            var builder = Builder(secondTensor: tensor);
            Add(builder, "a", "a");
            Add(builder, "b", "b", dependencies: ["a"]);
            Assert.Equal(6, Process(new TemporaryResourceReusePostprocessor(), builder.Build()).Graphs[0].Resources.Count);
        }
        var bound = Builder(bindingKey: "externally-visible");
        Add(bound, "a", "a");
        Add(bound, "b", "b", dependencies: ["a"]);
        Assert.Equal(6, Process(new TemporaryResourceReusePostprocessor(), bound.Build()).Graphs[0].Resources.Count);
    }

    [Fact]
    public void SameOperationInputsAndOutputsNeverCoalesce()
    {
        var builder = Builder();
        Add(builder, "a", "a");
        Add(builder, "b", "b", "a", ["a"]);
        Assert.Equal(6, Process(new TemporaryResourceReusePostprocessor(), builder.Build()).Graphs[0].Resources.Count);
    }

    [Fact]
    public void MultipleWritesRemainInOneLifetime()
    {
        var builder = Builder();
        Add(builder, "first", "a");
        Add(builder, "b", "b", dependencies: ["first"]);
        Add(builder, "later", "a", dependencies: ["b"]);
        var result = Process(new TemporaryResourceReusePostprocessor(), builder.Build()).Graphs[0];
        Assert.Contains(result.Resources, resource => resource.Id.Value == "b");
    }

    [Fact]
    public void TemplateSharingPreservesCallsPortsAndSourceMaps()
    {
        var builder = Builder();
        Add(builder, "a", "a", "input");
        Add(builder, "b", "b", "input", ["a"]);
        var before = builder.Build();
        var after = Process(new OperationTemplateSharingPostprocessor(), before);
        Assert.Equal(2, after.Graphs.Count);
        var calls = after.Graphs[0].Nodes.OfType<ExecutionGraphCallNode>().ToArray();
        Assert.Equal(2, calls.Length);
        Assert.Equal(calls[0].Target, calls[1].Target);
        Assert.Equal(before.Structure.Edges, after.Graphs[0].Structure.Edges);
        Assert.Equal(before.Nodes[0].Source.LogicalNodes, calls[0].Source.LogicalNodes);
        Assert.Single(after.Graphs[1].Nodes);
        Assert.All(after.Graphs[1].Resources, resource => Assert.Null(resource.BindingKey));
        AssertIdempotent(new OperationTemplateSharingPostprocessor(), new([before]));
    }

    [Theory]
    [InlineData("attributes")]
    [InlineData("precision")]
    [InlineData("version")]
    [InlineData("shape")]
    [InlineData("alias")]
    public void DifferentOrAliasedContractsAreNotShared(string difference)
    {
        var builder = Builder(secondTensor: difference == "shape"
            ? new(ElementType.Float32, [2]) : null);
        Add(builder, "a", "a", "input");
        var op = Operation("b", "core.copy", difference == "version" ? 2 : 1) with
        {
            Attributes = difference == "attributes"
                ? new Dictionary<string, string> { ["mode"] = "other" } : new Dictionary<string, string>(),
            Requirements = difference == "precision"
                ? new(ElementType.Float16, ElementType.Float16)
                : new(ElementType.Float32, ElementType.Float32),
        };
        builder.AddOperation(op,
            [Binding("input", difference == "alias" ? "b" : "input", ResourceAccess.Read, "b"),
             Binding("output", "b", ResourceAccess.Write, "b")], [new("a")]);
        Assert.Single(Process(new OperationTemplateSharingPostprocessor(), builder.Build()).Graphs);
    }

    [Fact]
    public void CallerBoundTemporaryAndCallAccessesAreProtected()
    {
        var child = new ExecutionGraphBuilder(new("child"));
        child.AddResource(new(new("a"), "a", GraphDefinitionResourceKind.Output, F));
        child.AddResource(new(new("b"), "b", GraphDefinitionResourceKind.Output, F));
        Add(child, "a", "a");
        Add(child, "b", "b", dependencies: ["a"]);
        var root = Builder();
        root.AddCall(new(new("call"), new("child"), GraphCompileBoundary.Atomic, new([])),
            [Binding("a", "a", ResourceAccess.Write, "call"),
             Binding("b", "b", ResourceAccess.Write, "call")]);
        var before = new ExecutionGraphProgram([root.Build(), child.Build()]);
        var after = new TemporaryResourceReusePostprocessor().Process(before, Context);
        Assert.Equal(6, after.Graphs.Single(graph => graph.Id.Value == "root").Resources.Count);
        Assert.Equal(2, after.Graphs.Single(graph => graph.Id.Value == "child").Resources.Count);
        var rootAfter = after.Graphs.Single(graph => graph.Id.Value == "root");
        Assert.Equal(["a", "b"],
            rootAfter.Bindings(Assert.Single(rootAfter.Nodes)).Select(binding => binding.Resource.Resource.Value));
    }

    [Fact]
    public void ReadWriteTemplatePortsPreserveAccess()
    {
        var builder = Builder();
        foreach (var name in new[] { "a", "b" })
        {
            builder.AddOperation(Operation(name, "custom.update"),
                [Binding("state", name, ResourceAccess.ReadWrite, name)]);
        }

        var after = Process(new OperationTemplateSharingPostprocessor(), builder.Build());
        Assert.Equal(GraphDefinitionResourceKind.InOut, Assert.Single(after.Graphs[1].Resources).Kind);
        Assert.Equal(ResourceAccess.ReadWrite,
            Assert.Single(after.Graphs[1].Bindings(Assert.Single(after.Graphs[1].Nodes))).Access);
        Assert.All(after.Graphs[0].Nodes, node =>
            Assert.Equal(ResourceAccess.ReadWrite, Assert.Single(after.Graphs[0].Bindings(node)).Access));
    }

    [Fact]
    public void ImplicitCallInstanceResourcesProtectTheTargetTemporary()
    {
        var child = Builder("child");
        Add(child, "a", "a");
        Add(child, "b", "b", dependencies: ["a"]);
        var root = Builder();
        root.AddResource(new(new("a"), "a", GraphDefinitionResourceKind.Temporary, F),
            GraphBindingOwner.Instance(new("call")));
        root.AddCall(new(new("call"), new("child"), GraphCompileBoundary.Atomic, new([])),
            [Binding("input", "input", ResourceAccess.Read, "call"),
             Binding("output", "output", ResourceAccess.Write, "call"),
             Binding("left-output", "left-output", ResourceAccess.Write, "call"),
             Binding("right-output", "right-output", ResourceAccess.Write, "call")]);
        root.AddOperation(Operation("consume", "core.copy"),
            [new("input", new(GraphBindingOwner.Instance(new("call")), new("a")),
                ResourceAccess.Read, "consume:input"),
             Binding("output", "output", ResourceAccess.Write, "consume")], [new("call")]);
        var result = new TemporaryResourceReusePostprocessor().Process(
            new([root.Build(), child.Build()]), Context);
        Assert.Equal(6, result.Graphs.Single(graph => graph.Id.Value == "child").Resources.Count);
    }

    [Fact]
    public void ReductionPreservesReachabilityOnVariedDags()
    {
        var random = new Random(417);
        for (var iteration = 0; iteration < 20; iteration++)
        {
            var builder = new ExecutionGraphBuilder(new("random"));
            for (var index = 0; index < 12; index++)
            {
                builder.AddOperation(Operation($"n{index}", "custom.operation"), [],
                    Enumerable.Range(0, index).Where(_ => random.Next(3) == 0)
                        .Select(predecessor => new ExecutionNodeId($"n{predecessor}")));
            }

            var before = builder.Build();
            var after = Process(new DependencyReductionPostprocessor(), before).Graphs[0];
            foreach (var first in before.Nodes)
            {
                foreach (var second in before.Nodes)
                {
                    Assert.Equal(
                        before.Structure.DependencyView().IsReachable(first.GraphId, second.GraphId),
                        after.Structure.DependencyView().IsReachable(first.GraphId, second.GraphId));
                }
            }
        }
    }

    private static ExecutionGraphBuilder Builder(
        string id = "root",
        GraphDefinitionResourceKind secondKind = GraphDefinitionResourceKind.Temporary,
        TensorDescriptor? secondTensor = null,
        string? bindingKey = null)
    {
        var builder = new ExecutionGraphBuilder(new(id));
        foreach (var name in new[] { "input", "output", "left-output", "right-output", "a", "b" })
        {
            builder.AddResource(new(new(name), name,
                name == "input" ? GraphDefinitionResourceKind.Input :
                name is "output" or "left-output" or "right-output" ? GraphDefinitionResourceKind.Output :
                name == "b" ? secondKind : GraphDefinitionResourceKind.Temporary,
                name == "b" ? secondTensor ?? F : F, name == "b" ? bindingKey : null));
        }

        return builder;
    }

    private static void Add(
        ExecutionGraphBuilder builder, string id, string output, string? input = null,
        string[]? dependencies = null)
    {
        var bindings = new List<ModelResourceBinding>();
        if (input is not null)
        {
            bindings.Add(Binding("input", input, ResourceAccess.Read, id));
        }

        bindings.Add(Binding("output", output, ResourceAccess.Write, id));
        builder.AddOperation(Operation(id, input is null ? "core.tensor.fill" : "core.copy"),
            bindings, (dependencies ?? []).Select(value => new ExecutionNodeId(value)));
    }

    private static ExecutionOperationNode Operation(string id, string name, int version = 1) =>
        new(new(id), new(name, version),
            name == "core.tensor.fill" ? new Dictionary<string, string> { ["value"] = "1" } : new Dictionary<string, string>(),
            new(ElementType.Float32, ElementType.Float32), new([new(id)]));

    private static ModelResourceBinding Binding(
        string port, string resource, ResourceAccess access, string id) =>
        new(port, new(GraphBindingOwner.Parent, new(resource)), access, $"{id}:{port}");

    private static ExecutionGraphProgram Process(IExecutionGraphPostprocessor pass, ExecutionGraph graph) =>
        pass.Process(new([graph]), Context);

    private static ExecutionGraph WithEdges(ExecutionGraph graph, IEnumerable<ExecutionGraphEdge> edges) =>
        new(graph.Id, new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(
            graph.Structure.Nodes, edges), graph.DefaultCompileBoundary, graph.Architecture, graph.Source);

    private static void AssertIdempotent(IExecutionGraphPostprocessor pass, ExecutionGraphProgram original)
    {
        var once = pass.Process(original, Context);
        Assert.Equal(ExecutionGraphXml.Serialize(once), ExecutionGraphXml.Serialize(pass.Process(once, Context)));
    }
}
