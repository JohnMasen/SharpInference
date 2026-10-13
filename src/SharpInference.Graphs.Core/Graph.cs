using System.Collections.ObjectModel;

namespace SharpInference.Graphs.Core;

/// <summary>Identifies one node within a graph.</summary>
public readonly record struct GraphNodeId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Identifies one edge within a graph.</summary>
public readonly record struct GraphEdgeId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Provides the structural identity required by a graph node.</summary>
public interface IGraphNode
{
    /// <summary>Gets the node identity.</summary>
    GraphNodeId Id { get; }
}

/// <summary>Provides the structural identity and endpoints required by a directed graph edge.</summary>
public interface IGraphEdge
{
    /// <summary>Gets the edge identity.</summary>
    GraphEdgeId Id { get; }

    /// <summary>Gets the source node.</summary>
    GraphNodeId Source { get; }

    /// <summary>Gets the target node.</summary>
    GraphNodeId Target { get; }
}

/// <summary>Optional base implementation for graph nodes.</summary>
public abstract record GraphNodeBase(GraphNodeId Id) : IGraphNode;

/// <summary>Optional base implementation for directed graph edges.</summary>
public abstract record GraphEdgeBase(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target) : IGraphEdge;

/// <summary>An immutable directed typed multigraph.</summary>
public sealed class Graph<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    private readonly IReadOnlyDictionary<GraphNodeId, TNode> nodesById;
    private readonly IReadOnlyDictionary<GraphEdgeId, TEdge> edgesById;
    private readonly IReadOnlyDictionary<GraphNodeId, IReadOnlyList<TEdge>> incoming;
    private readonly IReadOnlyDictionary<GraphNodeId, IReadOnlyList<TEdge>> outgoing;

    /// <summary>Creates and validates a graph from its complete node and edge sets.</summary>
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

    /// <summary>Gets one node by identity.</summary>
    public TNode GetNode(GraphNodeId id) =>
        nodesById.TryGetValue(id, out var node)
            ? node
            : throw new KeyNotFoundException($"Unknown graph node '{id}'.");

    /// <summary>Gets one edge by identity.</summary>
    public TEdge GetEdge(GraphEdgeId id) =>
        edgesById.TryGetValue(id, out var edge)
            ? edge
            : throw new KeyNotFoundException($"Unknown graph edge '{id}'.");

    /// <summary>Returns whether the graph contains a node.</summary>
    public bool ContainsNode(GraphNodeId id) => nodesById.ContainsKey(id);

    /// <summary>Gets all incoming edges in deterministic declaration order.</summary>
    public IReadOnlyList<TEdge> InEdges(GraphNodeId node) =>
        incoming.TryGetValue(node, out var edges)
            ? edges
            : throw new KeyNotFoundException($"Unknown graph node '{node}'.");

    /// <summary>Gets all outgoing edges in deterministic declaration order.</summary>
    public IReadOnlyList<TEdge> OutEdges(GraphNodeId node) =>
        outgoing.TryGetValue(node, out var edges)
            ? edges
            : throw new KeyNotFoundException($"Unknown graph node '{node}'.");

    /// <summary>Gets nodes assignable to the requested type.</summary>
    public IEnumerable<TRequested> NodesOfType<TRequested>() where TRequested : TNode =>
        Nodes.OfType<TRequested>();

    /// <summary>Gets edges assignable to the requested type.</summary>
    public IEnumerable<TRequested> EdgesOfType<TRequested>() where TRequested : TEdge =>
        Edges.OfType<TRequested>();

    /// <summary>Creates a filtered non-owning view over this graph.</summary>
    public GraphView<TNode, TEdge> View(
        Func<TNode, bool>? nodePredicate = null,
        Func<TEdge, bool>? edgePredicate = null) =>
        new(this, nodePredicate, edgePredicate);

    private static IReadOnlyDictionary<GraphNodeId, IReadOnlyList<TEdge>> Freeze(
        IReadOnlyDictionary<GraphNodeId, List<TEdge>> source) =>
        new ReadOnlyDictionary<GraphNodeId, IReadOnlyList<TEdge>>(
            source.ToDictionary(
                pair => pair.Key,
                pair => (IReadOnlyList<TEdge>)Array.AsReadOnly(pair.Value.ToArray())));
}

/// <summary>A filtered non-owning projection over an immutable graph.</summary>
public sealed class GraphView<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    private readonly Graph<TNode, TEdge> graph;
    private readonly Func<TNode, bool> nodePredicate;
    private readonly Func<TEdge, bool> edgePredicate;

    internal GraphView(
        Graph<TNode, TEdge> graph,
        Func<TNode, bool>? nodePredicate,
        Func<TEdge, bool>? edgePredicate)
    {
        this.graph = graph;
        this.nodePredicate = nodePredicate ?? (_ => true);
        this.edgePredicate = edgePredicate ?? (_ => true);
    }

    /// <summary>Gets visible nodes in source graph order.</summary>
    public IEnumerable<TNode> Nodes => graph.Nodes.Where(nodePredicate);

    /// <summary>Gets visible edges whose endpoints are both visible.</summary>
    public IEnumerable<TEdge> Edges => graph.Edges.Where(IsVisible);

    /// <summary>Gets one visible node by identity.</summary>
    public TNode GetNode(GraphNodeId id)
    {
        var node = graph.GetNode(id);
        return nodePredicate(node)
            ? node
            : throw new KeyNotFoundException($"Graph node '{id}' is not visible in this view.");
    }

    /// <summary>Returns whether the view contains a node.</summary>
    public bool ContainsNode(GraphNodeId id) =>
        graph.ContainsNode(id) && nodePredicate(graph.GetNode(id));

    /// <summary>Gets visible incoming edges.</summary>
    public IEnumerable<TEdge> InEdges(GraphNodeId node)
    {
        _ = GetNode(node);
        return graph.InEdges(node).Where(IsVisible);
    }

    /// <summary>Gets visible outgoing edges.</summary>
    public IEnumerable<TEdge> OutEdges(GraphNodeId node)
    {
        _ = GetNode(node);
        return graph.OutEdges(node).Where(IsVisible);
    }

    private bool IsVisible(TEdge edge) =>
        edgePredicate(edge) &&
        nodePredicate(graph.GetNode(edge.Source)) &&
        nodePredicate(graph.GetNode(edge.Target));
}
