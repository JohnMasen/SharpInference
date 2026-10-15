using System.Collections.Immutable;
using System.Collections.ObjectModel;

namespace SharpInference.Graphs.Core;

/// <summary>An immutable directed typed multigraph.</summary>
/// <typeparam name="TNode">The node type, with unique structural identities.</typeparam>
/// <typeparam name="TEdge">The directed edge type; parallel edges are allowed.</typeparam>
/// <remarks>The graph freezes collection membership and order, not the state of its elements.</remarks>
public sealed class Graph<TNode, TEdge> : IGraphView<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    private readonly IReadOnlyDictionary<GraphNodeId, TNode> nodesById;
    private readonly IReadOnlyDictionary<GraphEdgeId, TEdge> edgesById;
    private readonly IReadOnlyDictionary<GraphNodeId, IReadOnlyList<TEdge>> incoming;
    private readonly IReadOnlyDictionary<GraphNodeId, IReadOnlyList<TEdge>> outgoing;

    /// <summary>Creates and validates a graph from its complete node and edge sets.</summary>
    /// <param name="nodes">Nodes in declaration order; the sequence is consumed once.</param>
    /// <param name="edges">Edges in declaration order; the sequence is consumed once.</param>
    /// <exception cref="ArgumentNullException">A sequence or one of its elements is null.</exception>
    /// <exception cref="InvalidDataException">
    /// An identity is invalid or duplicated, or an edge references an unknown endpoint.
    /// </exception>
    public Graph(IEnumerable<TNode> nodes, IEnumerable<TEdge> edges)
    {
        ArgumentNullException.ThrowIfNull(nodes);
        ArgumentNullException.ThrowIfNull(edges);

        var nodeArray = nodes.ToArray();
        var edgeArray = edges.ToArray();
        var mutableNodes = new Dictionary<GraphNodeId, TNode>();
        foreach (var node in nodeArray)
        {
            ArgumentNullException.ThrowIfNull(node);
            if (string.IsNullOrWhiteSpace(node.Id.Value))
            {
                throw new InvalidDataException("A graph node identifier is required.");
            }

            if (!mutableNodes.TryAdd(node.Id, node))
            {
                throw new InvalidDataException($"Duplicate graph node '{node.Id}'.");
            }
        }

        var mutableEdges = new Dictionary<GraphEdgeId, TEdge>();
        var mutableIncoming = mutableNodes.Keys.ToDictionary(
            id => id, _ => new List<TEdge>());
        var mutableOutgoing = mutableNodes.Keys.ToDictionary(
            id => id, _ => new List<TEdge>());
        foreach (var edge in edgeArray)
        {
            ArgumentNullException.ThrowIfNull(edge);
            if (string.IsNullOrWhiteSpace(edge.Id.Value))
            {
                throw new InvalidDataException("A graph edge identifier is required.");
            }

            if (!mutableEdges.TryAdd(edge.Id, edge))
            {
                throw new InvalidDataException($"Duplicate graph edge '{edge.Id}'.");
            }

            if (!mutableNodes.ContainsKey(edge.Source))
            {
                throw new InvalidDataException(
                    $"Graph edge '{edge.Id}' references unknown source node '{edge.Source}'.");
            }

            if (!mutableNodes.ContainsKey(edge.Target))
            {
                throw new InvalidDataException(
                    $"Graph edge '{edge.Id}' references unknown target node '{edge.Target}'.");
            }

            mutableOutgoing[edge.Source].Add(edge);
            mutableIncoming[edge.Target].Add(edge);
        }
        Nodes = Array.AsReadOnly(nodeArray);
        Edges = Array.AsReadOnly(edgeArray);
        nodesById = new ReadOnlyDictionary<GraphNodeId, TNode>(mutableNodes);
        edgesById = new ReadOnlyDictionary<GraphEdgeId, TEdge>(mutableEdges);
        incoming = Freeze(mutableIncoming);
        outgoing = Freeze(mutableOutgoing);
    }

    /// <summary>Gets graph nodes in deterministic declaration order.</summary>
    public IReadOnlyList<TNode> Nodes { get; }

    /// <summary>Gets graph edges in deterministic declaration order.</summary>
    public IReadOnlyList<TEdge> Edges { get; }

    /// <inheritdoc/>
    IEnumerable<TNode> IGraphView<TNode, TEdge>.Nodes => Nodes;
    /// <inheritdoc/>
    IEnumerable<TEdge> IGraphView<TNode, TEdge>.Edges => Edges;

    /// <summary>Gets one node by identity.</summary>
    /// <param name="id">The structural identity of the node to retrieve.</param>
    /// <returns>The node with the supplied identity.</returns>
    /// <exception cref="KeyNotFoundException">The node is not in the graph.</exception>
    public TNode GetNode(GraphNodeId id) =>
        nodesById.TryGetValue(id, out var node)
            ? node
            : throw new KeyNotFoundException($"Unknown graph node '{id}'.");

    /// <summary>Gets one edge by identity.</summary>
    /// <param name="id">The structural identity of the edge to retrieve.</param>
    /// <returns>The edge with the supplied identity.</returns>
    /// <exception cref="KeyNotFoundException">The edge is not in the graph.</exception>
    public TEdge GetEdge(GraphEdgeId id) =>
        edgesById.TryGetValue(id, out var edge)
            ? edge
            : throw new KeyNotFoundException($"Unknown graph edge '{id}'.");

    /// <summary>Returns whether the graph contains a node.</summary>
    /// <param name="id">The structural identity to look up.</param>
    /// <returns>Whether a node with this identity exists.</returns>
    public bool ContainsNode(GraphNodeId id) => nodesById.ContainsKey(id);

    /// <summary>Gets all incoming edges in deterministic declaration order.</summary>
    /// <param name="node">The identity of the target node.</param>
    /// <returns>All edges pointing to the node, including parallel edges.</returns>
    /// <exception cref="KeyNotFoundException">The node is not in the graph.</exception>
    public IReadOnlyList<TEdge> EdgesToNode(GraphNodeId node) =>
        incoming.TryGetValue(node, out var edges)
            ? edges
            : throw new KeyNotFoundException($"Unknown graph node '{node}'.");

    /// <summary>Gets all outgoing edges in deterministic declaration order.</summary>
    /// <param name="node">The identity of the source node.</param>
    /// <returns>All edges originating from the node, including parallel edges.</returns>
    /// <exception cref="KeyNotFoundException">The node is not in the graph.</exception>
    public IReadOnlyList<TEdge> EdgesFromNode(GraphNodeId node) =>
        outgoing.TryGetValue(node, out var edges)
            ? edges
            : throw new KeyNotFoundException($"Unknown graph node '{node}'.");

    /// <summary>Exposes incoming edges through the read-only view contract.</summary>
    /// <param name="node">The identity of the target node.</param>
    /// <returns>Incoming edges in declaration order.</returns>
    IEnumerable<TEdge> IGraphView<TNode, TEdge>.EdgesToNode(GraphNodeId node) => EdgesToNode(node);
    /// <summary>Exposes outgoing edges through the read-only view contract.</summary>
    /// <param name="node">The identity of the source node.</param>
    /// <returns>Outgoing edges in declaration order.</returns>
    IEnumerable<TEdge> IGraphView<TNode, TEdge>.EdgesFromNode(GraphNodeId node) => EdgesFromNode(node);

    /// <summary>Gets nodes assignable to the requested type.</summary>
    /// <typeparam name="TRequested">The node type to retain.</typeparam>
    /// <returns>A deferred sequence of matching nodes in declaration order.</returns>
    public IEnumerable<TRequested> NodesOfType<TRequested>() where TRequested : TNode =>
        this.Filter(node: node => node is TRequested).Nodes.Cast<TRequested>();

    /// <summary>Gets edges assignable to the requested type.</summary>
    /// <typeparam name="TRequested">The edge type to retain.</typeparam>
    /// <returns>A deferred sequence of matching edges in declaration order.</returns>
    public IEnumerable<TRequested> EdgesOfType<TRequested>() where TRequested : TEdge =>
        this.Filter(edge: edge => edge is TRequested).Edges.Cast<TRequested>();

    /// <summary>Copies an adjacency index into read-only dictionaries and edge lists.</summary>
    /// <param name="source">The mutable index whose per-node edge order must be preserved.</param>
    /// <returns>A frozen adjacency index independent of the original lists.</returns>
    private static IReadOnlyDictionary<GraphNodeId, IReadOnlyList<TEdge>> Freeze(
        IReadOnlyDictionary<GraphNodeId, List<TEdge>> source) =>
        new ReadOnlyDictionary<GraphNodeId, IReadOnlyList<TEdge>>(
            source.ToDictionary(
                pair => pair.Key,
                pair => (IReadOnlyList<TEdge>)Array.AsReadOnly(pair.Value.ToArray())));
}
