using System.Diagnostics.CodeAnalysis;

namespace SharpInference.Graphs.Core;

/// <summary>Read-only access to the nodes and exact edges visible in a graph.</summary>
/// <typeparam name="TNode">The type of visible nodes.</typeparam>
/// <typeparam name="TEdge">The type of visible directed edges.</typeparam>
public interface IGraphView<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    /// <summary>Gets visible nodes in the view's deterministic order.</summary>
    IEnumerable<TNode> Nodes
    {
        get;
    }
    /// <summary>Gets the exact visible edges, whose endpoints must also be visible.</summary>
    IEnumerable<TEdge> Edges
    {
        get;
    }

    /// <summary>Gets a visible node by its structural identity.</summary>
    /// <param name="id">The identity of the node to retrieve.</param>
    /// <returns>The visible node with the supplied identity.</returns>
    /// <exception cref="KeyNotFoundException">The node is unknown or hidden by this view.</exception>
    TNode GetNode(GraphNodeId id);

    /// <summary>Tries to retrieve a visible node without throwing when it is absent.</summary>
    /// <param name="id">The structural node identity.</param>
    /// <param name="node">The visible node on success; otherwise the default value.</param>
    /// <returns>Whether the node exists and is visible.</returns>
    bool TryGetNode(GraphNodeId id, [MaybeNullWhen(false)] out TNode node);

    /// <summary>Gets a visible edge by its structural identity.</summary>
    /// <param name="id">The structural edge identity.</param>
    /// <returns>The visible edge.</returns>
    /// <exception cref="KeyNotFoundException">The edge is unknown or hidden.</exception>
    TEdge GetEdge(GraphEdgeId id);

    /// <summary>Tries to retrieve a visible edge without throwing when it is absent.</summary>
    /// <param name="id">The structural edge identity.</param>
    /// <param name="edge">The visible edge on success; otherwise the default value.</param>
    /// <returns>Whether the edge exists and is visible.</returns>
    bool TryGetEdge(GraphEdgeId id, [MaybeNullWhen(false)] out TEdge edge);

    /// <summary>Determines whether a node is visible in this view.</summary>
    /// <param name="id">The structural identity to look up.</param>
    /// <returns>Whether the node exists and is visible.</returns>
    bool ContainsNode(GraphNodeId id);

    /// <summary>Gets visible edges that directly point to a node.</summary>
    /// <param name="id">The identity of the visible target node.</param>
    /// <returns>Incoming edges in the view's edge order, retaining parallel edges.</returns>
    /// <exception cref="KeyNotFoundException">No visible incoming edges were found.</exception>
    IEnumerable<TEdge> EdgesToNode(GraphNodeId id);

    /// <summary>Gets visible edges that directly originate from a node.</summary>
    /// <param name="id">The identity of the visible source node.</param>
    /// <returns>Outgoing edges in the view's edge order, retaining parallel edges.</returns>
    /// <exception cref="KeyNotFoundException">No visible outgoing edges were found.</exception>
    IEnumerable<TEdge> EdgesFromNode(GraphNodeId id);

    /// <summary>Enumerates incoming edges, returning an empty sequence for a visible node without any.</summary>
    /// <param name="id">The identity of the visible target node.</param>
    /// <returns>Visible incoming edges in declaration order, or an empty sequence.</returns>
    /// <exception cref="KeyNotFoundException">The node is unknown or hidden.</exception>
    IEnumerable<TEdge> EnumerateEdgesToNode(GraphNodeId id);

    /// <summary>Enumerates outgoing edges, returning an empty sequence for a visible node without any.</summary>
    /// <param name="id">The identity of the visible source node.</param>
    /// <returns>Visible outgoing edges in declaration order, or an empty sequence.</returns>
    /// <exception cref="KeyNotFoundException">The node is unknown or hidden.</exception>
    IEnumerable<TEdge> EnumerateEdgesFromNode(GraphNodeId id);
}
