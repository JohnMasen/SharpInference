using SharpInference.Graphs;
using SharpInference.Graphs.Logical;
using SharpInference.Instructions;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class GraphDependencyAnalyzerTests
{
    private static readonly TensorDescriptor Vector = new(ElementType.Float32, [8]);

    [Fact]
    public void IntermediateValuesInferDependenciesWithoutChangingTheSource()
    {
        var graph = new LogicalGraph { Id = "main" };
        var input = graph.Resource("input", Vector, GraphIO.In);
        var middle = graph.Resource("middle", Vector);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        var finish = graph.Operation("finish", Operations.Copy);
        finish.Bind("input", middle, ResourceAccess.Read);
        finish.Bind("output", output, ResourceAccess.Write);
        var start = graph.Operation("start", Operations.Copy);
        start.Bind("input", input, ResourceAccess.Read);
        start.Bind("output", middle, ResourceAccess.Write);

        var result = Analyze(graph);
        Assert.Empty(result.Diagnostics);
        Assert.Equal(["start", "finish"], result.Order.Select(action => action.Id));
        Assert.Single(result.Dependencies);
        Assert.Empty(graph.GraphFlow);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void StateReadsUseTheDeclaredOldOrNewValueOrder(bool oldValue)
    {
        var graph = new LogicalGraph { Id = "main" };
        var state = graph.Resource("state", Vector, GraphIO.InOut, ResourceScope.Session, isState: true);
        var read = graph.Operation("read", Operations.Copy);
        read.Bind("input", state, ResourceAccess.Read);
        var update = graph.Operation("update", Operations.Copy);
        update.Bind("output", state, ResourceAccess.Write);
        if (oldValue)
        {
            read.Then(update);
        }
        else
        {
            update.Then(read);
        }

        var result = Analyze(graph);
        Assert.Empty(result.Diagnostics);
        Assert.Equal(oldValue ? ["read", "update"] : new[] { "update", "read" },
            result.Order.Select(action => action.Id));
    }

    [Fact]
    public void StateConflictRemainsAnErrorWhenNeitherValueOrderIsSpecified()
    {
        var graph = new LogicalGraph { Id = "main" };
        var state = graph.Resource("state", Vector, GraphIO.InOut, ResourceScope.Session, isState: true);
        graph.Operation("read", Operations.Copy).Bind("input", state, ResourceAccess.Read);
        graph.Operation("write", Operations.Copy).Bind("output", state, ResourceAccess.Write);
        Assert.Contains(Analyze(graph).Diagnostics, diagnostic => diagnostic.Code == "A0003");
        Assert.Empty(graph.GraphFlow);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MultipleWritersRequireAProvableOrder(bool ordered)
    {
        var graph = new LogicalGraph { Id = "main" };
        var output = graph.Resource("output", Vector, GraphIO.Out);
        var first = graph.Operation("first", Operations.Copy);
        first.Bind("output", output, ResourceAccess.Write);
        var second = graph.Operation("second", Operations.Copy);
        second.Bind("output", output, ResourceAccess.Write);
        if (ordered)
        {
            first.Then(second);
        }

        Assert.Equal(!ordered, Analyze(graph).Diagnostics.Any(item => item.Code == "A0003"));
    }

    [Fact]
    public void InOutCanPassThroughButOutNeedsAnAvailableValue()
    {
        var graph = new LogicalGraph { Id = "main" };
        var value = graph.Resource("value", Vector, GraphIO.InOut);
        Assert.Empty(Analyze(graph).Diagnostics);
        value.GraphIO = GraphIO.Out;
        Assert.Contains(Analyze(graph).Diagnostics, item => item.Code == "A0004");
    }

    [Fact]
    public void ReadWriteRequiresInitializationRatherThanInitializingItself()
    {
        var graph = new LogicalGraph { Id = "main" };
        var value = graph.Resource("value", Vector);
        graph.Operation("update", Operations.Add).Bind("value", value, ResourceAccess.ReadWrite);
        Assert.Contains(Analyze(graph).Diagnostics, item => item.Code == "A0002");
        value.GraphIO = GraphIO.InOut;
        Assert.Empty(Analyze(graph).Diagnostics);
    }

    [Fact]
    public void PrivateCallResourceRequiresAnUnambiguousInstance()
    {
        var child = new LogicalGraph { Id = "child" };
        child.Resource("temporary", Vector);
        var main = new LogicalGraph { Id = "main" };
        main.Call("first", child);
        main.Call("second", child);
        var read = main.Operation("read", Operations.Copy);
        read.Bind("input", ResourceRef.InGraph("child", "temporary"), ResourceAccess.Read);
        var package = new LogicalGraphPackage(main, child);
        Assert.Contains(package.Analyze(main).Diagnostics, item => item.Code == "A0006");
        read.Bindings.Clear();
        read.Bind("input", ResourceRef.InCall("first", "temporary"), ResourceAccess.Read);
        var result = package.Analyze(main);
        Assert.DoesNotContain(result.Diagnostics, item => item.Code == "A0006");
        Assert.Contains(result.Diagnostics, item => item.Code == "A0002");
    }

    [Fact]
    public void NestedCallsExposeSharedSideEffectsToTheirCaller()
    {
        var leaf = new LogicalGraph { Id = "leaf" };
        var state = leaf.Resource("state", Vector, scope: ResourceScope.Session, isState: true, bindingKey: "state");
        leaf.Operation("write", Operations.Copy).Bind("output", state, ResourceAccess.Write);
        var middle = new LogicalGraph { Id = "middle" };
        middle.Call("inner", leaf);
        var main = new LogicalGraph { Id = "main" };
        var call = main.Call("outer", middle);
        var read = main.Operation("read", Operations.Copy);
        read.Bind("input", state, ResourceAccess.Read);
        var package = new LogicalGraphPackage(main, middle, leaf);
        Assert.Contains(package.Analyze(main).Diagnostics, item => item.Code == "A0003");
        call.Then(read);
        Assert.Empty(package.Analyze(main).Diagnostics);
    }

    [Fact]
    public void SharedBindingIdentityCannotCollideWithAGraphName()
    {
        var weights = new LogicalGraph { Id = "weights" };
        var weight = weights.Resource("weight", Vector, scope: ResourceScope.Global, bindingKey: "same");
        var main = new LogicalGraph { Id = "Global" };
        var local = main.Resource("same", Vector);
        main.Operation("write-local", Operations.Copy).Bind("output", local, ResourceAccess.Write);
        main.Operation("read-shared", Operations.Copy).Bind("input", weight, ResourceAccess.Read);
        var result = new LogicalGraphPackage(main, weights).Analyze(main);
        Assert.Empty(result.Diagnostics);
        Assert.Empty(result.Dependencies);
    }

    private static GraphAnalysis Analyze(LogicalGraph graph) =>
        new LogicalGraphPackage(graph).Analyze(graph);
}
