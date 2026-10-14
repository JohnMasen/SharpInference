namespace SharpInference.Graphs.Core;

/// <summary>A filtered non-owning projection over a read-only graph view.</summary>
/// <typeparam name="TNode">The type of nodes exposed by the source view.</typeparam>
/// <typeparam name="TEdge">The type of directed edges exposed by the source view.</typeparam>
/// <remarks>Predicates are evaluated during access; this view is not a materialized snapshot.</remarks>
public sealed class GraphView<TNode, TEdge> : IGraphView<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    private readonly IGraphView<TNode, TEdge> graph;
    private readonly Func<TNode, bool> nodePredicate;
    private readonly Func<TEdge, bool> edgePredicate;

    /// <summary>Creates a filtered view without copying graph elements.</summary>
    /// <param name="graph">The source view defining the maximum visible scope.</param>
    /// <param name="nodePredicate">The additional node condition, or null to accept all source nodes.</param>
    /// <param name="edgePredicate">The additional edge condition, or null to accept all source edges.</param>
    internal GraphView(
        IGraphView<TNode, TEdge> graph,
        Func<TNode, bool>? nodePredicate,
        Func<TEdge, bool>? edgePredicate)
    {
        this.graph = graph;
        this.nodePredicate = nodePredicate ?? (_ => true);
        this.edgePredicate = edgePredicate ?? (_ => true);
    }

    /// <inheritdoc/>
    public IEnumerable<TNode> Nodes => graph.Nodes.Where(nodePredicate);
    /// <inheritdoc/>
    public IEnumerable<TEdge> Edges => graph.Edges.Where(IsVisible);

    /// <summary>Gets a source node that satisfies this view's node condition.</summary>
    /// <param name="id">The structural identity of the node to retrieve.</param>
    /// <returns>The visible node with the supplied identity.</returns>
    /// <exception cref="KeyNotFoundException">The node is unknown or hidden in either view.</exception>
    public TNode GetNode(GraphNodeId id)
    {
        var node = graph.GetNode(id);
        return nodePredicate(node)
            ? node
            : throw new KeyNotFoundException($"Graph node '{id}' is not visible in this view.");
    }

    /// <summary>Determines whether a source node satisfies this view's node condition.</summary>
    /// <param name="id">The structural identity to look up.</param>
    /// <returns>Whether the node exists and is visible in both views.</returns>
    public bool ContainsNode(GraphNodeId id) =>
        graph.ContainsNode(id) && nodePredicate(graph.GetNode(id));

    /// <summary>Gets visible edges pointing to a visible node.</summary>
    /// <param name="node">The identity of the target node.</param>
    /// <returns>A deferred sequence retaining the source view's edge order.</returns>
    /// <exception cref="KeyNotFoundException">The node is unknown or hidden.</exception>
    public IEnumerable<TEdge> EdgesToNode(GraphNodeId node)
    {
        _ = GetNode(node);
        return graph.EdgesToNode(node).Where(IsVisible);
    }

    /// <summary>Gets visible edges originating from a visible node.</summary>
    /// <param name="node">The identity of the source node.</param>
    /// <returns>A deferred sequence retaining the source view's edge order.</returns>
    /// <exception cref="KeyNotFoundException">The node is unknown or hidden.</exception>
    public IEnumerable<TEdge> EdgesFromNode(GraphNodeId node)
    {
        _ = GetNode(node);
        return graph.EdgesFromNode(node).Where(IsVisible);
    }

    /// <summary>Checks the edge condition and the visibility of both endpoints.</summary>
    /// <param name="edge">A candidate edge from the source view.</param>
    /// <returns>Whether the edge and both endpoint nodes satisfy the additional conditions.</returns>
    private bool IsVisible(TEdge edge) =>
        edgePredicate(edge) &&
        nodePredicate(graph.GetNode(edge.Source)) &&
        nodePredicate(graph.GetNode(edge.Target));
}
