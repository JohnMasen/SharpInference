using System.Xml.Linq;
using SharpInference.Graphs;
using SharpInference.Graphs.Logical;
using SharpInference.Instructions;

namespace SharpInference.Tests;

public sealed class ComputationGraphTests
{
    private static readonly TensorDescriptor Vector = new(ElementType.Float32, [8]);

    [Fact]
    public void BuilderReturnsEditableDeclarationsWithoutAnalysis()
    {
        var graph = InferenceGraphBuilder.Builder<LogicalGraph>("main", g =>
        {
            var read = g.Operation("read", Operations.Copy);
            read.Bind("input", "future", ResourceAccess.Read);
            read.Then("future-action");
        });

        Assert.Single(graph.Actions);
        Assert.Single(graph.GraphFlow);
        Assert.Null(Assert.IsType<OperationAction>(graph.Actions[0]).Precision);
        Assert.Contains(graph.Validate(), item => item.Code == "G0012");
        graph.Resource("future", Vector);
        graph.Operation("future-action", Operations.Copy);
        Assert.Empty(graph.Validate());
        Assert.Single(graph.GraphFlow);
    }

    [Fact]
    public void GenericBuilderExposesCovariantStageOperations()
    {
        var graph = InferenceGraphBuilder.Builder<SpecialGraph>("main", g =>
        {
            var operation = g.Operation("copy", PrimitiveOperations.Copy);
            operation.Implementation = "test.copy";
        });

        Assert.Equal("test.copy", Assert.IsType<SpecialOperation>(graph.Actions[0]).Implementation);
    }

    [Fact]
    public void ResourceDefaultsAndInterfaceEnumeratorsReflectEdits()
    {
        var graph = new Graph { Id = "main" };
        var inputs = graph.Inputs;
        var outputs = graph.Outputs;
        var interfaces = graph.GraphIOs;
        var resource = graph.Resource("value", Vector);
        Assert.Equal(GraphIO.Internal, resource.GraphIO);
        Assert.Equal(ResourceScope.Call, resource.Scope);
        Assert.False(resource.IsState);
        Assert.Null(resource.Usage);
        Assert.Empty(interfaces);
        resource.GraphIO = GraphIO.InOut;
        Assert.Same(resource, Assert.Single(inputs));
        Assert.Same(resource, Assert.Single(outputs));
        Assert.Same(resource, Assert.Single(interfaces));
        resource.GraphIO = GraphIO.In;
        Assert.Empty(outputs);
    }

    [Fact]
    public void ObjectAndStringReferencesResolveTheSameDeclaration()
    {
        var graph = new Graph { Id = "main" };
        var value = graph.Resource("value.with.dots", Vector);
        var left = graph.Operation("left", Operations.Copy);
        left.Bind("input", value, ResourceAccess.Read);
        var right = graph.Operation("right", Operations.Copy);
        right.Bind("input", "value.with.dots", ResourceAccess.Read);
        left.Then(right);
        var index = new Dictionary<string, Graph> { [graph.Id] = graph };
        Assert.True(GraphValidation.TryResolve(graph, left.Bindings[0].Target, index, out var first));
        Assert.True(GraphValidation.TryResolve(graph, right.Bindings[0].Target, index, out var second));
        Assert.Same(first, second);
        Assert.Equal(new ActionRef(right), graph.GraphFlow[0].Target);
        Assert.Empty(graph.Validate());
    }

    [Fact]
    public void ObjectReferencesCaptureIdentityWithoutRegisteringTargets()
    {
        var child = new Graph { Id = "child" };
        var main = new Graph { Id = "main" };
        var call = main.Call("invoke", child);
        child.Id = "renamed";
        var package = new GraphPackage<Graph>(main);
        Assert.Equal("child", call.Target);
        Assert.Contains(package.Validate(), item => item.Code == "P0004");
        Assert.Single(package);
    }

    [Fact]
    public void SnapshotIsIndependentOfEditableCollectionsAndAttributes()
    {
        var graph = new LogicalGraph { Id = "main", CompileBoundary = GraphCompileBoundary.Atomic };
        var resource = graph.Resource("value", Vector);
        var operation = graph.Operation("copy", Operations.Copy);
        operation.Bind("input", resource, ResourceAccess.Read);
        operation.Attributes.Add("mode", "first");
        graph.Metadata.Add("name", "original");
        var package = new GraphPackage<LogicalGraph>(graph);
        var snapshot = package.Snapshot();

        operation.Attributes["mode"] = "second";
        operation.Bindings.Clear();
        resource.GraphIO = GraphIO.Out;
        graph.Metadata.Clear();
        graph.Actions.Clear();
        package.Clear();

        var saved = Assert.Single(snapshot);
        Assert.Equal(GraphCompileBoundary.Atomic, saved.CompileBoundary);
        Assert.Equal("original", saved.Metadata["name"]);
        Assert.Equal(GraphIO.Internal, Assert.Single(saved.Resources).GraphIO);
        var savedOperation = Assert.IsType<OperationAction>(Assert.Single(saved.Actions));
        Assert.Equal("first", savedOperation.Attributes["mode"]);
        Assert.Single(savedOperation.Bindings);
        Assert.Same(saved, savedOperation.Graph);
    }

    [Theory]
    [InlineData(GraphIO.In)]
    [InlineData(GraphIO.Internal)]
    public void WritesRespectInterfaceAndPermission(GraphIO graphIO)
    {
        var graph = new Graph { Id = "main" };
        var resource = graph.Resource("value", Vector, graphIO,
            permission: ResourcePermission.ReadOnly);
        graph.Operation("update", Operations.Copy).Bind("output", resource, ResourceAccess.Write);
        Assert.Contains(graph.Validate(), item => item.Code == "G0016");
        resource.Permission = ResourcePermission.ReadWrite;
        Assert.Equal(graphIO == GraphIO.In, graph.Validate().Any(item => item.Code == "G0016"));
    }

    [Fact]
    public void ExplicitMainAllowsUnusedGraphsButReportsTheirErrors()
    {
        var main = new Graph { Id = "main" };
        var unused = new Graph { Id = "unused" };
        var package = new GraphPackage<Graph>(main, unused);
        Assert.DoesNotContain(package.Validate(), item => item.Severity == GraphDiagnosticSeverity.Error);
        Assert.Contains(package.Validate(), item => item.Graph == "unused" && item.Code == "P0002");
        unused.Operation("bad", Operations.Copy).Bind("input", "missing", ResourceAccess.Read);
        Assert.Throws<InvalidDataException>(() => GraphValidation.ThrowIfErrors(package.Validate()));
    }

    [Fact]
    public void RepeatedCallsValidateEachInterfaceAndKeepDistinctOwners()
    {
        var child = Pass();
        var main = new LogicalGraph { Id = "main" };
        var input = main.Resource("input", Vector, GraphIO.In);
        var middle = main.Resource("middle", Vector);
        var output = main.Resource("output", Vector, GraphIO.Out);
        var first = main.Call("first", child);
        first.Bind("input", input, ResourceAccess.Read);
        first.Bind("output", middle, ResourceAccess.Write);
        var second = main.Call("second", "pass");
        second.Bind("input", middle, ResourceAccess.Read);
        second.Bind("output", output, ResourceAccess.Write);
        first.Then(second);
        var package = new GraphPackage<LogicalGraph>(main, child);
        Assert.DoesNotContain(package.Validate(), item => item.Severity == GraphDiagnosticSeverity.Error);
        second.Bindings.RemoveAt(1);
        var missing = Assert.Single(package.Validate(), item => item.Code == "P0005");
        Assert.Equal("second", missing.Node);
        Assert.Equal("output", missing.Port);
    }

    [Fact]
    public void PrivateResourceReferencesParticipateInUsageClosure()
    {
        var catalog = new LogicalGraph { Id = "catalog" };
        var privateWeight = catalog.Resource("private.weight", Vector, scope: ResourceScope.Global,
            usage: ResourceUsage.Weight, permission: ResourcePermission.ReadOnly);
        var main = new LogicalGraph { Id = "main" };
        main.Operation("read", Operations.Copy).Bind("input", privateWeight, ResourceAccess.Read);
        var package = new GraphPackage<LogicalGraph>(main, catalog);
        Assert.Empty(package.Validate());
    }

    [Fact]
    public void ExplicitAndCallCyclesAreDiagnosedWithoutMutatingSources()
    {
        var main = new Graph { Id = "main" };
        var a = main.Operation("a", Operations.Copy);
        var b = main.Operation("b", Operations.Copy);
        a.Then(b);
        b.Then(a);
        Assert.Contains(main.Validate(), item => item.Code == "G0015");
        Assert.Equal(2, main.GraphFlow.Count);
        main.GraphFlow.Clear();
        var helper = new Graph { Id = "helper" };
        main.Call("invoke", helper);
        helper.Call("again", helper);
        Assert.Contains(new GraphPackage<Graph>(main, helper).Validate(), item => item.Code == "P0010");
        helper.Actions.Clear();
        helper.Call("invalid-entry", main);
        Assert.Contains(new GraphPackage<Graph>(main, helper).Validate(), item => item.Code == "P0003");
    }

    [Fact]
    public void XmlRoundTripsEditableSourcesAndNullablePrecision()
    {
        var graph = Pass();
        graph.CompileBoundary = GraphCompileBoundary.Transparent;
        graph.Metadata.Add("formula", "output = Copy(input)");
        graph.Resource("state", Vector, GraphIO.InOut, ResourceScope.Session, isState: true);
        var call = graph.Call("later", "missing-target");
        call.Bind("input", "future", ResourceAccess.ReadWrite);
        var package = new GraphPackage<LogicalGraph>(graph);
        var xml = GraphPackageXml.Serialize(package);
        var loaded = GraphPackageXml.Deserialize<LogicalGraph>(xml);
        var saved = Assert.Single(loaded);
        Assert.Equal(xml, GraphPackageXml.Serialize(loaded));
        Assert.Equal(GraphCompileBoundary.Transparent, saved.CompileBoundary);
        Assert.Null(Assert.IsType<OperationAction>(saved.Actions[0]).Precision);
        Assert.True(saved.Resources.Single(resource => resource.Id == "state").IsState);
        Assert.Contains(loaded.Validate(), item => item.Code == "P0004");
        Assert.Single(XDocument.Parse(xml).Descendants("Binding"), binding =>
            (string?)binding.Attribute("access") == "ReadWrite");
    }

    [Theory]
    [InlineData("<!DOCTYPE GraphPackage [<!ENTITY x SYSTEM 'file:///ignored'>]><GraphPackage/>")]
    [InlineData("<Graph kind='logical' version='4'/>")]
    [InlineData("<GraphPackage version='2' stage='business'><Graphs/></GraphPackage>")]
    [InlineData("<GraphPackage version='1' stage='business'><Unknown/></GraphPackage>")]
    public void XmlRejectsUnsafeOrUnsupportedDocuments(string xml) =>
        Assert.Throws<InvalidDataException>(() => GraphPackageXml.Deserialize<Graph>(xml));

    private static LogicalGraph Pass()
    {
        var graph = new LogicalGraph { Id = "pass" };
        var input = graph.Resource("input", Vector, GraphIO.In);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        var copy = graph.Operation("copy", Operations.Copy);
        copy.Bind("input", input, ResourceAccess.Read);
        copy.Bind("output", output, ResourceAccess.Write);
        return graph;
    }

    private sealed class SpecialGraph : Graph
    {
        public SpecialGraph() { }

        public override SpecialOperation Operation(string id, OperationId operation)
        {
            var action = new SpecialOperation(this, id, operation);
            Actions.Add(action);
            return action;
        }
    }

    private sealed class SpecialOperation(Graph graph, string id, OperationId operation)
        : OperationAction(graph, id, operation)
    {
        public string? Implementation { get; set; }
    }
}
