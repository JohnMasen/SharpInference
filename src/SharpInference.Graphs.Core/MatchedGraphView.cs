namespace SharpInference.Graphs.Core;

/// <summary>A stable view containing only the exact nodes and edges of one matched path.</summary>
/// <typeparam name="TNode">The type of matched nodes.</typeparam>
/// <typeparam name="TEdge">The type of matched directed edges.</typeparam>
internal sealed class MatchedGraphView<TNode, TEdge> : IGraphView<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    private readonly IReadOnlyList<TNode> nodes;
    private readonly IReadOnlyList<TEdge> edges;

    /// <summary>Copies the current match buffers so later backtracking cannot change this result.</summary>
    /// <param name="nodes">The matched nodes in path order.</param>
    /// <param name="edges">The matched edges in path order.</param>
    internal MatchedGraphView(TNode[] nodes, TEdge[] edges)
    {
        this.nodes = Array.AsReadOnly((TNode[])nodes.Clone());
        this.edges = Array.AsReadOnly((TEdge[])edges.Clone());
    }

    /// <inheritdoc/>
    public IEnumerable<TNode> Nodes => nodes;
    /// <inheritdoc/>
    public IEnumerable<TEdge> Edges => edges;
    /// <summary>Determines whether a node belongs to this matched path.</summary>
    /// <param name="id">The structural identity to look up.</param>
    /// <returns>Whether the matched node sequence contains the identity.</returns>
    public bool ContainsNode(GraphNodeId id) => nodes.Any(node => node.Id == id);

    /// <summary>Gets a node from this matched path.</summary>
    /// <param name="id">The structural identity of the node to retrieve.</param>
    /// <returns>The matching path node.</returns>
    /// <exception cref="KeyNotFoundException">The node is not part of this match.</exception>
    public TNode GetNode(GraphNodeId id)
    {
        foreach (var node in nodes)
            if (node.Id == id)
                return node;
        throw new KeyNotFoundException($"Unknown graph node '{id}'.");
    }

    /// <summary>Gets only matched edges pointing to a path node.</summary>
    /// <param name="id">The identity of the matched target node.</param>
    /// <returns>Matched incoming edges in path order; source-graph edges are not restored.</returns>
    /// <exception cref="KeyNotFoundException">The node is not part of this match.</exception>
    public IEnumerable<TEdge> EdgesToNode(GraphNodeId id)
    {
        _ = GetNode(id);
        return edges.Where(edge => edge.Target == id);
    }

    /// <summary>Gets only matched edges originating from a path node.</summary>
    /// <param name="id">The identity of the matched source node.</param>
    /// <returns>Matched outgoing edges in path order; source-graph edges are not restored.</returns>
    /// <exception cref="KeyNotFoundException">The node is not part of this match.</exception>
    public IEnumerable<TEdge> EdgesFromNode(GraphNodeId id)
    {
        _ = GetNode(id);
        return edges.Where(edge => edge.Source == id);
    }
}
