using SharpInference.Graphs.Core;

namespace SharpInference.Tests;

public sealed class GraphCoreTests
{
    private interface INode : IGraphNode;
    private interface IEdge : IGraphEdge;
    private sealed record Node(string Name) : INode
    {
        public GraphNodeId Id => new(Name);
    }
    private sealed record Resource(string Name) : INode
    {
        public GraphNodeId Id => new(Name);
    }
    private sealed record Dependency(string Name, string From, string To) : IEdge
    {
        public GraphEdgeId Id => new(Name);
        public GraphNodeId Source => new(From);
        public GraphNodeId Target => new(To);
    }
    private sealed record Read(string Name, string From, string To) : IEdge
    {
        public GraphEdgeId Id => new(Name);
        public GraphNodeId Source => new(From);
        public GraphNodeId Target => new(To);
    }
    private sealed record Write(string Name, string From, string To) : IEdge
    {
        public GraphEdgeId Id => new(Name);
        public GraphNodeId Source => new(From);
        public GraphNodeId Target => new(To);
    }

    [Fact]
    public void GraphIndexesTypedParallelEdges()
    {
        var graph = new Graph<INode, IEdge>(
            [new Node("operation"), new Resource("resource")],
            [
                new Read("read", "resource", "operation"),
                new Write("write", "operation", "resource"),
            ]);

        Assert.Single(graph.InEdges(new("operation")).OfType<Read>());
        Assert.Single(graph.OutEdges(new("operation")).OfType<Write>());
        Assert.Equal(2, graph.Edges.Count);
    }

    [Fact]
    public void DependencyViewIgnoresResourceAccessCycle()
    {
        var graph = new Graph<INode, IEdge>(
            [new Node("first"), new Node("update"), new Resource("state")],
            [
                new Dependency("dependency", "first", "update"),
                new Read("read", "state", "update"),
                new Write("write", "update", "state"),
            ]);

        var dependencyView = graph.View(
            node => node is Node,
            edge => edge is Dependency);

        Assert.Equal(
            ["first", "update"],
            dependencyView.TopologicalOrder().Select(node => node.Id.Value));
        Assert.True(dependencyView.IsReachable(new("first"), new("update")));
    }

    [Fact]
    public void GraphRejectsUnknownEdgeEndpoint()
    {
        Assert.Throws<InvalidDataException>(() =>
            new Graph<INode, IEdge>(
                [new Node("known")],
                [new Dependency("dependency", "known", "missing")]));
    }

    [Fact]
    public void PatternReturnsNamedNodeAndEdgeGroup()
    {
        var graph = new Graph<INode, IEdge>(
            [new Node("matvec"), new Resource("temp"), new Node("add")],
            [
                new Write("write", "matvec", "temp"),
                new Read("read", "temp", "add"),
            ]);
        var pattern = new GraphPattern<INode, IEdge>()
            .Node<Node>("producer", node => node.Name == "matvec")
            .Node<Resource>("temporary")
            .Node<Node>("consumer", node => node.Name == "add")
            .Edge<Write>("write", "producer", "temporary")
            .Edge<Read>("read", "temporary", "consumer");

        var match = Assert.Single(pattern.Find(graph));

        Assert.Equal("matvec", match.Node<Node>("producer").Name);
        Assert.Equal("temp", match.Node<Resource>("temporary").Name);
        Assert.Equal("add", match.Node<Node>("consumer").Name);
        Assert.Equal("write", match.Edge<Write>("write").Name);
        Assert.Equal("read", match.Edge<Read>("read").Name);
    }

    [Fact]
    public void RewriteTargetCapturesBoundaryAndPlanAppliesBatch()
    {
        var graph = new Graph<INode, IEdge>(
            [new Node("a"), new Node("b"), new Node("c")],
            [
                new Dependency("ab", "a", "b"),
                new Dependency("bc", "b", "c"),
            ]);
        var target = GraphRewriteTarget<INode, IEdge>.Capture(
            graph, [new GraphNodeId("b")]);
        var rewritten = new GraphRewritePlan<INode, IEdge>()
            .Remove(target)
            .AddNode(new Resource("replacement"))
            .AddEdge(new Dependency("ac", "a", "c"))
            .Apply(graph);

        Assert.Empty(target.InternalEdges);
        Assert.Equal("ab", Assert.Single(target.IncomingEdges).Id.Value);
        Assert.Equal("bc", Assert.Single(target.OutgoingEdges).Id.Value);
        Assert.Equal(["a", "c", "replacement"],
            rewritten.Nodes.Select(node => node.Id.Value));
        Assert.Equal("ac", Assert.Single(rewritten.Edges).Id.Value);
    }

    [Fact]
    public void RewriteTargetsReportOverlap()
    {
        var graph = new Graph<INode, IEdge>(
            [new Node("a"), new Node("b"), new Node("c")],
            []);
        var first = GraphRewriteTarget<INode, IEdge>.Capture(
            graph, [new GraphNodeId("a"), new GraphNodeId("b")]);
        var second = GraphRewriteTarget<INode, IEdge>.Capture(
            graph, [new GraphNodeId("b"), new GraphNodeId("c")]);

        Assert.True(first.Overlaps(second));
    }

    [Fact]
    public void InvalidRewriteDoesNotModifySourceGraph()
    {
        var graph = new Graph<INode, IEdge>(
            [new Node("a"), new Node("b")],
            [new Dependency("ab", "a", "b")]);
        var plan = new GraphRewritePlan<INode, IEdge>()
            .AddEdge(new Dependency("missing", "a", "missing"));

        Assert.Throws<InvalidDataException>(() => plan.Apply(graph));
        Assert.Equal(["a", "b"], graph.Nodes.Select(node => node.Id.Value));
        Assert.Equal("ab", Assert.Single(graph.Edges).Id.Value);
    }
}
