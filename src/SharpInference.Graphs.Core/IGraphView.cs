namespace SharpInference.Graphs.Core;

/// <summary>Read-only access to the nodes and exact edges visible in a graph.</summary>
/// <typeparam name="TNode">The type of visible nodes.</typeparam>
/// <typeparam name="TEdge">The type of visible directed edges.</typeparam>
public interface IGraphView<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    /// <summary>Gets visible nodes in the view's deterministic order.</summary>
    IEnumerable<TNode> Nodes { get; }
    /// <summary>Gets the exact visible edges, whose endpoints must also be visible.</summary>
    IEnumerable<TEdge> Edges { get; }

    /// <summary>Gets a visible node by its structural identity.</summary>
    /// <param name="id">The identity of the node to retrieve.</param>
    /// <returns>The visible node with the supplied identity.</returns>
    /// <exception cref="KeyNotFoundException">The node is unknown or hidden by this view.</exception>
    TNode GetNode(GraphNodeId id);

    /// <summary>Determines whether a node is visible in this view.</summary>
    /// <param name="id">The structural identity to look up.</param>
    /// <returns>Whether the node exists and is visible.</returns>
    bool ContainsNode(GraphNodeId id);

    /// <summary>Gets visible edges that directly point to a node.</summary>
    /// <param name="id">The identity of the visible target node.</param>
    /// <returns>Incoming edges in the view's edge order, retaining parallel edges.</returns>
    /// <exception cref="KeyNotFoundException">The node is unknown or hidden by this view.</exception>
    IEnumerable<TEdge> EdgesToNode(GraphNodeId id);

    /// <summary>Gets visible edges that directly originate from a node.</summary>
    /// <param name="id">The identity of the visible source node.</param>
    /// <returns>Outgoing edges in the view's edge order, retaining parallel edges.</returns>
    /// <exception cref="KeyNotFoundException">The node is unknown or hidden by this view.</exception>
    IEnumerable<TEdge> EdgesFromNode(GraphNodeId id);
}
