using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Execution;
using SharpInference.Instructions;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Tests;

public sealed class ExecutionGraphPostprocessorTests
{
    private static readonly TensorDescriptor Vector = new(ElementType.Float32, [4]);
    private static readonly ExecutionPassContext Context = new(new("test", "cpu-vm"), 0, "test");

    [Fact]
    public void DependencyReductionPreservesReachabilityBindingsAndOneParallelEdge()
    {
        var random = new Random(417);
        for (var iteration = 0; iteration < 20; iteration++)
        {
            var graph = Create();
            for (var index = 0; index < 12; index++)
            {
                var action = Add(graph, $"n{index}", "output");
                foreach (var predecessor in Enumerable.Range(0, index).Where(_ => random.Next(3) == 0))
                {
                    graph.Actions[predecessor].Then(action);
                }
            }
            graph.Actions[0].Then(graph.Actions[1]);
            graph.Actions[0].Then(graph.Actions[1]);
            var before = graph.DependencyGraph();
            var after = Process(new DependencyReductionPostprocessor(), graph).MainGraph!;
            var structure = after.DependencyGraph();
            foreach (var first in before.Nodes)
            {
                foreach (var second in before.Nodes)
                {
                    Assert.Equal(before.IsReachable(first.Id, second.Id), structure.IsReachable(first.Id, second.Id));
                }
            }

            Assert.Single(after.GraphFlow, flow => flow.Source.Action == "n0" && flow.Target.Action == "n1");
            Assert.Equal(graph.Actions.SelectMany(action => action.Bindings), after.Actions.SelectMany(action => action.Bindings));
            AssertIdempotent(new DependencyReductionPostprocessor(), new(graph));
        }
    }

    [Fact]
    public void SerialTemporariesShareStorageWithoutChangingDeclarationsActionsOrFlow()
    {
        var graph = Create();
        Add(graph, "write", "a");
        Add(graph, "read", "output", "a", ["write"]);
        Add(graph, "next", "b", dependencies: ["read"]);
        var before = GraphPackageXml.Serialize(new GraphPackage<ExecutionGraph>(graph));
        var after = Process(new TemporaryResourceReusePostprocessor(), graph).MainGraph!;
        Assert.Equal("a", after.Storage["b"].Slot);
        Assert.Equal(ResourceStorageKind.LocalBuffer, after.Storage["b"].Kind);
        Assert.Equal(graph.Resources.Select(resource => resource.Id), after.Resources.Select(resource => resource.Id));
        Assert.Equal(graph.GraphFlow, after.GraphFlow);
        Assert.Equal(graph.Actions.SelectMany(action => action.Bindings), after.Actions.SelectMany(action => action.Bindings));
        Assert.Equal(before, GraphPackageXml.Serialize(new GraphPackage<ExecutionGraph>(graph)));
        AssertIdempotent(new TemporaryResourceReusePostprocessor(), new(graph));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ReuseRequiresAllLastReadersToPrecedeTheNextWriter(bool diamond, bool joined)
    {
        var graph = Create();
        Add(graph, "write", "a");
        Add(graph, "left", "left-output", "a", ["write"]);
        if (diamond)
        {
            Add(graph, "right", "right-output", "a", ["write"]);
        }

        Add(graph, "next", "b", dependencies: joined ? ["left", "right"] : diamond ? ["left"] : []);
        var after = Process(new TemporaryResourceReusePostprocessor(), graph).MainGraph!;
        Assert.Equal(joined ? "a" : null, after.Storage["b"].Slot);
    }

    [Theory]
    [InlineData("initial-read")]
    [InlineData("read-write")]
    [InlineData("unknown-write")]
    [InlineData("unknown-version")]
    [InlineData("same-operation")]
    [InlineData("multiple-writes")]
    public void UnprovenOrOverlappingLifetimesNeverShareStorage(string scenario)
    {
        var graph = Create();
        Add(graph, "first", "a");
        if (scenario == "initial-read")
        {
            Add(graph, "second", "output", "b", ["first"]);
        }
        else
        {
            var operation = Add(graph, "second", "b", scenario == "same-operation" ? "a" : null, ["first"]);
            if (scenario == "read-write")
            {
                operation.Bindings[0] = operation.Bindings[0] with { Access = ResourceAccess.ReadWrite };
            }

            if (scenario == "unknown-write")
            {
                operation.Operation = new("custom.partial");
            }

            if (scenario == "unknown-version")
            {
                operation.Operation = new(Operations.Copy, 2);
            }

            if (scenario == "multiple-writes")
            {
                Add(graph, "last", "a", dependencies: ["second"]);
            }
        }
        Assert.Null(Process(new TemporaryResourceReusePostprocessor(), graph).MainGraph!.Storage["b"].Slot);
    }

    [Theory]
    [InlineData("state")]
    [InlineData("input")]
    [InlineData("output")]
    [InlineData("inout")]
    [InlineData("weight")]
    [InlineData("session")]
    [InlineData("global")]
    [InlineData("binding")]
    [InlineData("shape")]
    [InlineData("precision")]
    [InlineData("layout")]
    public void VisibleOrIncompatibleResourcesNeverShareStorage(string protection)
    {
        var graph = Create();
        var resource = graph.Resources.Single(resource => resource.Id == "b");
        switch (protection)
        {
            case "state": resource.IsState = true; break;
            case "input": resource.GraphIO = GraphIO.In; break;
            case "output": resource.GraphIO = GraphIO.Out; break;
            case "inout": resource.GraphIO = GraphIO.InOut; break;
            case "weight": resource.Usage = ResourceUsage.Weight; resource.Permission = ResourcePermission.ReadOnly; break;
            case "session": resource.Scope = ResourceScope.Session; break;
            case "global": resource.Scope = ResourceScope.Global; break;
            case "binding": resource.BindingKey = "external"; break;
            case "shape": resource.Tensor = new(ElementType.Float32, [2, 2]); break;
            case "precision": resource.Tensor = new(ElementType.Float16, [4]); break;
            case "layout": resource.Tensor = new(ElementType.Float32, [4], "other"); break;
        }
        Add(graph, "first", "a");
        if (protection is "weight" or "input")
        {
            Add(graph, "second", "output", "b", ["first"]);
        }
        else
        {
            Add(graph, "second", "b", dependencies: ["first"]);
        }

        Assert.Null(Process(new TemporaryResourceReusePostprocessor(), graph).MainGraph!.Storage["b"].Slot);
    }

    [Fact]
    public void DeclarationOrderDoesNotChangeReuseSafety()
    {
        var graph = Create();
        Add(graph, "z-last-name", "a");
        Add(graph, "a-first-name", "output", "a", ["z-last-name"]);
        Add(graph, "middle", "b", dependencies: ["a-first-name"]);
        graph.Actions.Reverse();
        graph.Resources.Reverse();
        graph.GraphFlow.Reverse();
        Assert.Equal("a", Process(new TemporaryResourceReusePostprocessor(), graph).MainGraph!.Storage["b"].Slot);
    }

    [Fact]
    public void TemplateSharingPreservesCallsBindingsSourcesAndReadWriteAccess()
    {
        foreach (var readWrite in new[] { false, true })
        {
            var graph = Create();
            foreach (var id in new[] { "a", "b" })
            {
                var operation = readWrite ? graph.Operation(id, "custom.update") : Add(graph, id, id, "input");
                if (readWrite)
                {
                    operation.Bind("state", id, ResourceAccess.ReadWrite);
                }

                operation.Source("test", graph.Id, id);
            }
            graph.Actions[0].Then(graph.Actions[1]);
            var result = Process(new OperationTemplateSharingPostprocessor(), graph);
            Assert.Equal(2, result.Count);
            var calls = result.MainGraph!.Actions.Cast<ExecutionCall>().ToArray();
            Assert.Equal(calls[0].Target, calls[1].Target);
            Assert.Equal(graph.GraphFlow, result.MainGraph.GraphFlow);
            Assert.Equal(new SourceOrigin("test", "main", "a"), Assert.Single(calls[0].Sources));
            var template = result.Single(item => item != result.MainGraph);
            Assert.All(template.Resources, resource => Assert.Null(resource.BindingKey));
            if (readWrite)
            {
                Assert.Equal(GraphIO.InOut, Assert.Single(template.Resources).GraphIO);
                Assert.Equal(ResourceAccess.ReadWrite, Assert.Single(template.Actions[0].Bindings).Access);
            }
            AssertIdempotent(new OperationTemplateSharingPostprocessor(), new(graph));
        }
    }

    [Theory]
    [InlineData("attributes")]
    [InlineData("precision")]
    [InlineData("version")]
    [InlineData("shape")]
    [InlineData("alias")]
    [InlineData("category")]
    public void DifferentOrAliasedOperationContractsAreNotShared(string difference)
    {
        var graph = Create();
        Add(graph, "first", "a", "input");
        var second = Add(graph, "second", "b", difference == "alias" ? "b" : "input");
        switch (difference)
        {
            case "attributes": second.Attributes["mode"] = "other"; break;
            case "precision": second.RequirePrecision(ElementType.Float16, ElementType.Float16); break;
            case "version": second.Operation = new(Operations.Copy, 2); break;
            case "shape": graph.Resources.Single(resource => resource.Id == "b").Tensor = new(ElementType.Float32, [2]); break;
            case "category": second.Category = "custom.operation"; break;
        }
        Assert.Single(Process(new OperationTemplateSharingPostprocessor(), graph));
    }

    [Fact]
    public void CallInstanceResourceReferencesProtectCalleeStorage()
    {
        var child = Create("child");
        Add(child, "first", "a");
        Add(child, "second", "b", dependencies: ["first"]);
        var root = Create();
        var call = root.Call("invoke", child);
        foreach (var resource in child.GraphIOs)
        {
            call.Bind(resource.Id, resource.Id, resource.GraphIO == GraphIO.In ? ResourceAccess.Read : ResourceAccess.Write);
        }

        var consume = Add(root, "consume", "output", dependencies: ["invoke"]);
        consume.Bind("input", ResourceRef.InCall("invoke", "a"), ResourceAccess.Read);
        var result = new TemporaryResourceReusePostprocessor().Process(new(root, child), Context);
        Assert.Null(result.Single(graph => graph.Id == "child").Storage["b"].Slot);
    }

    private static ExecutionGraph Create(string id = "main")
    {
        var graph = new ExecutionGraph { Id = id };
        foreach (var name in new[] { "input", "output", "left-output", "right-output", "a", "b" })
        {
            graph.Resource(name, Vector, new ExecutionStorage(ResourceStorageKind.LocalBuffer),
                name == "input" ? GraphIO.In : name.Contains("output", StringComparison.Ordinal) ? GraphIO.Out : GraphIO.Internal);
        }

        return graph;
    }

    private static ExecutionOperation Add(ExecutionGraph graph, string id, string output,
        string? input = null, string[]? dependencies = null)
    {
        var action = graph.Operation(id, Operations.Copy);
        if (input is not null)
        {
            action.Bind("input", input, ResourceAccess.Read);
        }

        action.Bind("output", output, ResourceAccess.Write);
        foreach (var dependency in dependencies ?? [])
        {
            graph.Actions.Single(operation => operation.Id == dependency).Then(action);
        }

        return action;
    }

    private static GraphPackage<ExecutionGraph> Process(IExecutionGraphPostprocessor processor, ExecutionGraph graph) =>
        processor.Process(new(graph), Context);

    private static void AssertIdempotent(IExecutionGraphPostprocessor processor, GraphPackage<ExecutionGraph> package)
    {
        var once = processor.Process(package, Context);
        Assert.Equal(GraphPackageXml.Serialize(once), GraphPackageXml.Serialize(processor.Process(once, Context)));
    }
}
