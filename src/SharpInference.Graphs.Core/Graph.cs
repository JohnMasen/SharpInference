using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.Diagnostics.CodeAnalysis;

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

        //copy the value to the readonly properties, so that the graph is immutable
        Nodes = nodes.ToArray();
        Edges = edges.ToArray();


        var edgesWithSource = from edge in edges
                              group edge by edge.Source into g
                              select new { Source = g.Key, Value = g };
        outgoing = edgesWithSource.ToImmutableDictionary(x => x.Source, x => x.Value.ToList() as IReadOnlyList<TEdge>);
        var edgesWithTarget = from edge in edges
                              group edge by edge.Target into g
                              select new { Target = g.Key, Value = g };
        incoming = edgesWithTarget.ToImmutableDictionary(x => x.Target, x => x.Value.ToList() as IReadOnlyList<TEdge>);
        nodesById=new ReadOnlyDictionary<GraphNodeId, TNode>(nodes.ToDictionary(n=>n.Id));
        edgesById = new ReadOnlyDictionary<GraphEdgeId, TEdge>(edges.ToDictionary(e => e.Id));

        //validate edges
        foreach (var edge in edges)
        {
            if (!nodes.Any(x=>x.Id==edge.Source))
            {
                throw new InvalidDataException($"Graph edge '{edge.Id}' references unknown source node '{edge.Source}'.");
            }
            if (!nodes.Any(x=>x.Id==edge.Target))
            {
                throw new InvalidDataException($"Graph edge '{edge.Id}' references unknown target node '{edge.Source}'.");
            }
        }

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

    /// <inheritdoc/>
    public bool TryGetNode(GraphNodeId id, [MaybeNullWhen(false)] out TNode node) =>
        nodesById.TryGetValue(id, out node);

    /// <inheritdoc/>
    public bool TryGetEdge(GraphEdgeId id, [MaybeNullWhen(false)] out TEdge edge) =>
        edgesById.TryGetValue(id, out edge);

    /// <summary>Returns whether the graph contains a node.</summary>
    /// <param name="id">The structural identity to look up.</param>
    /// <returns>Whether a node with this identity exists.</returns>
    public bool ContainsNode(GraphNodeId id) => nodesById.ContainsKey(id);



    ///// <summary>Gets all outgoing edges in deterministic declaration order.</summary>
    ///// <param name="node">The identity of the source node.</param>
    ///// <returns>All edges originating from the node, including parallel edges.</returns>
    ///// <exception cref="KeyNotFoundException">No outgoing edges were found.</exception>
    //public IReadOnlyList<TEdge> EdgesFromNode(GraphNodeId node) =>
    //    outgoing.TryGetValue(node, out var edges)
    //        ? edges
    //        : throw new KeyNotFoundException($"No outgoing edges found for graph node '{node}'.");

    /// <inheritdoc/>
    public IEnumerable<TEdge> EnumerateEdgesToNode(GraphNodeId id)
    {
        _ = GetNode(id);    //check if node exists
        return incoming.TryGetValue(id, out var edges) ? edges : Array.Empty<TEdge>();
    }

    /// <inheritdoc/>
    public IEnumerable<TEdge> EnumerateEdgesFromNode(GraphNodeId id)
    {
        _ = GetNode(id);    //check if node exists
        return outgoing.TryGetValue(id, out var edges) ? edges : Array.Empty<TEdge>();
    }

    /// <summary>Exposes incoming edges through the read-only view contract.</summary>
    /// <param name="node">The identity of the target node.</param>
    /// <returns>Incoming edges in declaration order.</returns>
    public IEnumerable<TEdge> EdgesToNode(GraphNodeId node)
    {
        return incoming.TryGetValue(node, out var edges)
            ? edges
            : throw new KeyNotFoundException($"No incoming edges found for graph node '{node}'.");
    }
    /// <summary>Exposes outgoing edges through the read-only view contract.</summary>
    /// <param name="node">The identity of the source node.</param>
    /// <returns>Outgoing edges in declaration order.</returns>
    public IEnumerable<TEdge> EdgesFromNode(GraphNodeId node)
    {
        return outgoing.TryGetValue(node, out var edges)
            ? edges
            : throw new KeyNotFoundException($"No outgoing edges found for graph node '{node}'.");
    }

    /// <summary>Gets nodes assignable to the requested type.</summary>
    /// <typeparam name="TRequested">The node type to retain.</typeparam>
    /// <returns>A deferred sequence of matching nodes in declaration order.</returns>
    public IEnumerable<TRequested> NodesOfType<TRequested>() where TRequested : TNode
    {
        return this.Filter(node: node => node is TRequested).Nodes.Cast<TRequested>();
    }

    /// <summary>Gets edges assignable to the requested type.</summary>
    /// <typeparam name="TRequested">The edge type to retain.</typeparam>
    /// <returns>A deferred sequence of matching edges in declaration order.</returns>
    public IEnumerable<TRequested> EdgesOfType<TRequested>() where TRequested : TEdge
    {
        return this.Filter(edge: edge => edge is TRequested).Edges.Cast<TRequested>();
    }
        

    
}
