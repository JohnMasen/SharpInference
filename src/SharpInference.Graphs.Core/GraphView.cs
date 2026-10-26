using System.Diagnostics.CodeAnalysis;

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
        return TryGetNode(id, out var node)
            ? node
            : throw new KeyNotFoundException($"Graph node '{id}' is not visible in this view.");
    }

    /// <summary>Determines whether a source node satisfies this view's node condition.</summary>
    /// <param name="id">The structural identity to look up.</param>
    /// <returns>Whether the node exists and is visible in both views.</returns>
    public bool ContainsNode(GraphNodeId id) =>
        TryGetNode(id, out _);

    /// <inheritdoc/>
    public bool TryGetNode(GraphNodeId id, [MaybeNullWhen(false)] out TNode node)
    {
        if (graph.TryGetNode(id, out node) && nodePredicate(node))
        {
            return true;
        }

        node = default;
        return false;
    }

    /// <inheritdoc/>
    public TEdge GetEdge(GraphEdgeId id) =>
        TryGetEdge(id, out var edge) ? edge
            : throw new KeyNotFoundException($"Graph edge '{id}' is not visible in this view.");

    /// <inheritdoc/>
    public bool TryGetEdge(GraphEdgeId id, [MaybeNullWhen(false)] out TEdge edge)
    {
        if (graph.TryGetEdge(id, out edge) && IsVisible(edge))
        {
            return true;
        }

        edge = default;
        return false;
    }

    /// <summary>Gets visible edges pointing to a visible node.</summary>
    /// <param name="node">The identity of the target node.</param>
    /// <returns>Visible incoming edges retaining the source view's edge order.</returns>
    /// <exception cref="KeyNotFoundException">No visible incoming edges were found.</exception>
    public IEnumerable<TEdge> EdgesToNode(GraphNodeId node)
    {
        var edges = ContainsNode(node) ? EnumerateEdgesToNode(node).ToArray() : [];
        return edges.Length > 0 ? edges
            : throw new KeyNotFoundException($"No incoming edges found for graph node '{node}' in this view.");
    }

    /// <summary>Gets visible edges originating from a visible node.</summary>
    /// <param name="node">The identity of the source node.</param>
    /// <returns>Visible outgoing edges retaining the source view's edge order.</returns>
    /// <exception cref="KeyNotFoundException">No visible outgoing edges were found.</exception>
    public IEnumerable<TEdge> EdgesFromNode(GraphNodeId node)
    {
        var edges = ContainsNode(node) ? EnumerateEdgesFromNode(node).ToArray() : [];
        return edges.Length > 0 ? edges
            : throw new KeyNotFoundException($"No outgoing edges found for graph node '{node}' in this view.");
    }

    /// <inheritdoc/>
    public IEnumerable<TEdge> EnumerateEdgesToNode(GraphNodeId id)
    {
        _ = GetNode(id);
        return graph.EnumerateEdgesToNode(id).Where(IsVisible);
    }

    /// <inheritdoc/>
    public IEnumerable<TEdge> EnumerateEdgesFromNode(GraphNodeId id)
    {
        _ = GetNode(id);
        return graph.EnumerateEdgesFromNode(id).Where(IsVisible);
    }

    /// <summary>Checks the edge condition and the visibility of both endpoints.</summary>
    /// <param name="edge">A candidate edge from the source view.</param>
    /// <returns>Whether the edge and both endpoint nodes satisfy the additional conditions.</returns>
    private bool IsVisible(TEdge edge) =>
        edgePredicate(edge) &&
        nodePredicate(graph.GetNode(edge.Source)) &&
        nodePredicate(graph.GetNode(edge.Target));
}
