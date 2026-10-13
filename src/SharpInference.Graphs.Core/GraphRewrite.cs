namespace SharpInference.Graphs.Core;

/// <summary>The complete node, edge, and boundary capture for one rewrite target.</summary>
public sealed class GraphRewriteTarget<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    private GraphRewriteTarget(
        IReadOnlyList<TNode> nodes,
        IReadOnlyList<TEdge> internalEdges,
        IReadOnlyList<TEdge> incomingEdges,
        IReadOnlyList<TEdge> outgoingEdges)
    {
        Nodes = nodes;
        InternalEdges = internalEdges;
        IncomingEdges = incomingEdges;
        OutgoingEdges = outgoingEdges;
    }

    /// <summary>Gets captured nodes in graph declaration order.</summary>
    public IReadOnlyList<TNode> Nodes { get; }

    /// <summary>Gets edges whose two endpoints are captured.</summary>
    public IReadOnlyList<TEdge> InternalEdges { get; }

    /// <summary>Gets edges entering the captured node set.</summary>
    public IReadOnlyList<TEdge> IncomingEdges { get; }

    /// <summary>Gets edges leaving the captured node set.</summary>
    public IReadOnlyList<TEdge> OutgoingEdges { get; }

    /// <summary>Captures a nonempty target and all of its structural boundaries.</summary>
    public static GraphRewriteTarget<TNode, TEdge> Capture(
        Graph<TNode, TEdge> graph,
        IEnumerable<GraphNodeId> nodeIds)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(nodeIds);
        var selected = nodeIds.ToHashSet();
        if (selected.Count == 0)
        {
            throw new ArgumentException(
                "A rewrite target requires at least one node.", nameof(nodeIds));
        }

        var nodes = graph.Nodes.Where(node => selected.Contains(node.Id)).ToArray();
        if (nodes.Length != selected.Count)
        {
            var missing = selected.Where(id => !graph.ContainsNode(id));
            throw new ArgumentException(
                $"Rewrite target contains unknown nodes: {string.Join(", ", missing)}.",
                nameof(nodeIds));
        }
        return new(
            nodes,
            graph.Edges.Where(edge =>
                selected.Contains(edge.Source) &&
                selected.Contains(edge.Target)).ToArray(),
            graph.Edges.Where(edge =>
                !selected.Contains(edge.Source) &&
                selected.Contains(edge.Target)).ToArray(),
            graph.Edges.Where(edge =>
                selected.Contains(edge.Source) &&
                !selected.Contains(edge.Target)).ToArray());
    }

    /// <summary>Returns whether this target shares a node with another target.</summary>
    public bool Overlaps(GraphRewriteTarget<TNode, TEdge> other)
    {
        ArgumentNullException.ThrowIfNull(other);
        var ids = Nodes.Select(node => node.Id).ToHashSet();
        return other.Nodes.Any(node => ids.Contains(node.Id));
    }
}

/// <summary>A deterministic batch of graph removals and additions.</summary>
public sealed class GraphRewritePlan<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    private readonly HashSet<GraphNodeId> removedNodes = [];
    private readonly HashSet<GraphEdgeId> removedEdges = [];
    private readonly List<TNode> addedNodes = [];
    private readonly List<TEdge> addedEdges = [];

    /// <summary>Removes a captured target and every edge incident to its nodes.</summary>
    public GraphRewritePlan<TNode, TEdge> Remove(
        GraphRewriteTarget<TNode, TEdge> target)
    {
        ArgumentNullException.ThrowIfNull(target);
        foreach (var node in target.Nodes)
        {
            removedNodes.Add(node.Id);
        }

        return this;
    }

    /// <summary>Removes one edge while retaining its endpoint nodes.</summary>
    public GraphRewritePlan<TNode, TEdge> RemoveEdge(GraphEdgeId edge)
    {
        removedEdges.Add(edge);
        return this;
    }

    /// <summary>Adds or replaces one node in the resulting graph.</summary>
    public GraphRewritePlan<TNode, TEdge> AddNode(TNode node)
    {
        ArgumentNullException.ThrowIfNull(node);
        addedNodes.Add(node);
        return this;
    }

    /// <summary>Adds one edge to the resulting graph.</summary>
    public GraphRewritePlan<TNode, TEdge> AddEdge(TEdge edge)
    {
        ArgumentNullException.ThrowIfNull(edge);
        addedEdges.Add(edge);
        return this;
    }

    /// <summary>Applies the complete batch and validates the resulting graph atomically.</summary>
    public Graph<TNode, TEdge> Apply(Graph<TNode, TEdge> graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var replacements = addedNodes.Select(node => node.Id).ToHashSet();
        var nodes = graph.Nodes.Where(node =>
                !removedNodes.Contains(node.Id) &&
                !replacements.Contains(node.Id))
            .Concat(addedNodes)
            .ToArray();
        var edges = graph.Edges.Where(edge =>
                !removedEdges.Contains(edge.Id) &&
                !removedNodes.Contains(edge.Source) &&
                !removedNodes.Contains(edge.Target))
            .Concat(addedEdges)
            .ToArray();
        return new Graph<TNode, TEdge>(nodes, edges);
    }
}
