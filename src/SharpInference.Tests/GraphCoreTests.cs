using SharpInference.Graphs.Core;

namespace SharpInference.Tests;

/// <summary>Verifies graph structure, typed linear matching, and atomic graph rewrites.</summary>
public sealed class GraphCoreTests
{
    /// <summary>The common node contract for the heterogeneous test graph.</summary>
    private interface INode : IGraphNode;
    /// <summary>The common edge contract for the heterogeneous test graph.</summary>
    private interface IEdge : IGraphEdge;
    /// <summary>A test operation node.</summary>
    /// <param name="Name">The operation's structural identifier.</param>
    private sealed record Node(string Name) : INode
    {
        /// <inheritdoc/>
        public GraphNodeId Id => new(Name);
    }
    /// <summary>A test resource node.</summary>
    /// <param name="Name">The resource's structural identifier.</param>
    private sealed record Resource(string Name) : INode
    {
        /// <inheritdoc/>
        public GraphNodeId Id => new(Name);
    }
    /// <summary>A direct test dependency edge.</summary>
    /// <param name="Name">The unique edge identifier.</param>
    /// <param name="From">The source operation identifier.</param>
    /// <param name="To">The target operation identifier.</param>
    private sealed record Dependency(string Name, string From, string To) : IEdge
    {
        /// <inheritdoc/>
        public GraphEdgeId Id => new(Name);
        /// <inheritdoc/>
        public GraphNodeId Source => new(From);
        /// <inheritdoc/>
        public GraphNodeId Target => new(To);
    }
    /// <summary>A test edge from a resource to its reader.</summary>
    /// <param name="Name">The unique edge identifier.</param>
    /// <param name="From">The source resource identifier.</param>
    /// <param name="To">The reader operation identifier.</param>
    private sealed record Read(string Name, string From, string To) : IEdge
    {
        /// <inheritdoc/>
        public GraphEdgeId Id => new(Name);
        /// <inheritdoc/>
        public GraphNodeId Source => new(From);
        /// <inheritdoc/>
        public GraphNodeId Target => new(To);
    }
    /// <summary>A test edge from a writer to its resource.</summary>
    /// <param name="Name">The unique edge identifier.</param>
    /// <param name="From">The writer operation identifier.</param>
    /// <param name="To">The target resource identifier.</param>
    private sealed record Write(string Name, string From, string To) : IEdge
    {
        /// <inheritdoc/>
        public GraphEdgeId Id => new(Name);
        /// <inheritdoc/>
        public GraphNodeId Source => new(From);
        /// <inheritdoc/>
        public GraphNodeId Target => new(To);
    }

    /// <summary>Checks direct typed adjacency lookup in a heterogeneous graph.</summary>
    [Fact]
    public void GraphIndexesTypedParallelEdges()
    {
        var graph = new Graph<INode, IEdge>(
            [new Node("operation"), new Resource("resource")],
            [
                new Read("read", "resource", "operation"),
                new Write("write", "operation", "resource"),
            ]);

        Assert.Single(graph.EdgesToNode(new("operation")).OfType<Read>());
        Assert.Single(graph.EdgesFromNode(new("operation")).OfType<Write>());
        Assert.Equal(2, graph.Edges.Count);
    }

    /// <summary>Checks that dependency filtering excludes a resource-access cycle.</summary>
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

        var dependencyView = graph.Filter(
            node => node is Node,
            edge => edge is Dependency);

        Assert.Equal(
            ["first", "update"],
            dependencyView.TopologicalOrder().Select(node => node.Id.Value));
        Assert.True(dependencyView.IsReachable(new("first"), new("update")));
    }

    /// <summary>Checks structural validation of unknown edge endpoints.</summary>
    [Fact]
    public void GraphRejectsUnknownEdgeEndpoint()
    {
        Assert.Throws<InvalidDataException>(() =>
            new Graph<INode, IEdge>(
                [new Node("known")],
                [new Dependency("dependency", "known", "missing")]));
    }

    /// <summary>Checks typed producer/resource/consumer matching with identity-anchored endpoints.</summary>
    [Fact]
    public void PatternReturnsExactTypedNodeAndEdgeView()
    {
        var graph = new Graph<INode, IEdge>(
            [new Node("matvec"), new Resource("temp"), new Node("add")],
            [
                new Write("write", "matvec", "temp"),
                new Read("read", "temp", "add"),
            ]);
        var pattern = Pattern.Create(PatternElements.Node(new GraphNodeId("matvec")).Where<Node>(_ => true))
            - PatternElements.Edge().Where<Write>(_ => true)
            - PatternElements.Node().Where<Resource>(_ => true)
            - PatternElements.Edge().Where<Read>(_ => true)
            - PatternElements.Node(new GraphNodeId("add")).Where<Node>(_ => true);

        var match = Assert.Single(graph.Match(pattern));

        Assert.Equal("matvec", Assert.IsType<Node>(match.GetNode(new("matvec"))).Name);
        Assert.Equal("temp", Assert.IsType<Resource>(match.GetNode(new("temp"))).Name);
        Assert.Equal("add", Assert.IsType<Node>(match.GetNode(new("add"))).Name);
        Assert.Equal("write", Assert.IsType<Write>(match.Edges.First()).Name);
        Assert.Equal("read", Assert.IsType<Read>(match.Edges.Last()).Name);
    }

    /// <summary>Checks boundary capture and batch replacement without changing the original graph.</summary>
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

    /// <summary>Checks overlap detection by captured node identities.</summary>
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

    /// <summary>Checks that failed rewrite validation leaves the source graph intact.</summary>
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
