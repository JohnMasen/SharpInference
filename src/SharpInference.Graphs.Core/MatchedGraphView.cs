using System.Diagnostics.CodeAnalysis;

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
        return TryGetNode(id, out var node) ? node
            : throw new KeyNotFoundException($"Unknown graph node '{id}'.");
    }

    /// <inheritdoc/>
    public bool TryGetNode(GraphNodeId id, [MaybeNullWhen(false)] out TNode node)
    {
        foreach (var candidate in nodes)
            if (candidate.Id == id)
            {
                node = candidate;
                return true;
            }
        node = default;
        return false;
    }

    /// <inheritdoc/>
    public TEdge GetEdge(GraphEdgeId id) =>
        TryGetEdge(id, out var edge) ? edge
            : throw new KeyNotFoundException($"Unknown graph edge '{id}'.");

    /// <inheritdoc/>
    public bool TryGetEdge(GraphEdgeId id, [MaybeNullWhen(false)] out TEdge edge)
    {
        foreach (var candidate in edges)
            if (candidate.Id == id)
            {
                edge = candidate;
                return true;
            }
        edge = default;
        return false;
    }

    /// <summary>Gets only matched edges pointing to a path node.</summary>
    /// <param name="id">The identity of the matched target node.</param>
    /// <returns>Matched incoming edges in path order; source-graph edges are not restored.</returns>
    /// <exception cref="KeyNotFoundException">No matched incoming edges were found.</exception>
    public IEnumerable<TEdge> EdgesToNode(GraphNodeId id)
    {
        var found = edges.Where(edge => edge.Target == id).ToArray();
        return found.Length > 0 ? found
            : throw new KeyNotFoundException($"No incoming edges found for graph node '{id}' in this match.");
    }

    /// <summary>Gets only matched edges originating from a path node.</summary>
    /// <param name="id">The identity of the matched source node.</param>
    /// <returns>Matched outgoing edges in path order; source-graph edges are not restored.</returns>
    /// <exception cref="KeyNotFoundException">No matched outgoing edges were found.</exception>
    public IEnumerable<TEdge> EdgesFromNode(GraphNodeId id)
    {
        var found = edges.Where(edge => edge.Source == id).ToArray();
        return found.Length > 0 ? found
            : throw new KeyNotFoundException($"No outgoing edges found for graph node '{id}' in this match.");
    }

    /// <inheritdoc/>
    public IEnumerable<TEdge> EnumerateEdgesToNode(GraphNodeId id)
    {
        _ = GetNode(id);
        return edges.Where(edge => edge.Target == id);
    }

    /// <inheritdoc/>
    public IEnumerable<TEdge> EnumerateEdgesFromNode(GraphNodeId id)
    {
        _ = GetNode(id);
        return edges.Where(edge => edge.Source == id);
    }
}
