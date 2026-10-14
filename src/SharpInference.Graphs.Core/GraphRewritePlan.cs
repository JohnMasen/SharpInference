namespace SharpInference.Graphs.Core;

/// <summary>A deterministic batch of graph removals and additions.</summary>
/// <typeparam name="TNode">The type of nodes removed, added, or replaced.</typeparam>
/// <typeparam name="TEdge">The type of directed edges removed or added.</typeparam>
public sealed class GraphRewritePlan<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    private readonly HashSet<GraphNodeId> removedNodes = [];
    private readonly HashSet<GraphEdgeId> removedEdges = [];
    private readonly List<TNode> addedNodes = [];
    private readonly List<TEdge> addedEdges = [];

    /// <summary>Removes a captured target and every edge incident to its nodes.</summary>
    /// <param name="target">The capture identifying nodes to remove when the plan is applied.</param>
    /// <returns>This plan for further batch composition.</returns>
    /// <exception cref="ArgumentNullException">The target is null.</exception>
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
    /// <param name="edge">The identity of the edge to remove, if present in the source graph.</param>
    /// <returns>This plan for further batch composition.</returns>
    public GraphRewritePlan<TNode, TEdge> RemoveEdge(GraphEdgeId edge)
    {
        removedEdges.Add(edge);
        return this;
    }

    /// <summary>Adds or replaces one node in the resulting graph.</summary>
    /// <param name="node">The new node, replacing an existing source node with the same identity.</param>
    /// <returns>This plan for further batch composition.</returns>
    /// <exception cref="ArgumentNullException">The node is null.</exception>
    public GraphRewritePlan<TNode, TEdge> AddNode(TNode node)
    {
        ArgumentNullException.ThrowIfNull(node);
        addedNodes.Add(node);
        return this;
    }

    /// <summary>Adds one edge to the resulting graph.</summary>
    /// <param name="edge">The edge to append; its identity and endpoints are validated during application.</param>
    /// <returns>This plan for further batch composition.</returns>
    /// <exception cref="ArgumentNullException">The edge is null.</exception>
    public GraphRewritePlan<TNode, TEdge> AddEdge(TEdge edge)
    {
        ArgumentNullException.ThrowIfNull(edge);
        addedEdges.Add(edge);
        return this;
    }

    /// <summary>Applies the complete batch and validates the resulting graph atomically.</summary>
    /// <param name="graph">The source graph, which is not modified even if validation fails.</param>
    /// <returns>A validated graph containing retained elements followed by additions.</returns>
    /// <exception cref="ArgumentNullException">The source graph is null.</exception>
    /// <exception cref="InvalidDataException">The resulting graph has invalid identities or endpoints.</exception>
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
