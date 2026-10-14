namespace SharpInference.Graphs.Core;

/// <summary>The complete node, edge, and boundary capture for one rewrite target.</summary>
/// <typeparam name="TNode">The captured node type.</typeparam>
/// <typeparam name="TEdge">The captured internal and boundary edge type.</typeparam>
public sealed class GraphRewriteTarget<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    /// <summary>Creates a target from already classified nodes and edges.</summary>
    /// <param name="nodes">The captured nodes in source graph order.</param>
    /// <param name="internalEdges">Edges with both endpoints in the captured set.</param>
    /// <param name="incomingEdges">Edges entering the captured set from outside.</param>
    /// <param name="outgoingEdges">Edges leaving the captured set.</param>
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
    /// <param name="graph">The graph from which all nodes and boundary edges are captured.</param>
    /// <param name="nodeIds">The node identities to select, consumed once and deduplicated.</param>
    /// <returns>A capture preserving source declaration order within each collection.</returns>
    /// <exception cref="ArgumentNullException">The graph or identity sequence is null.</exception>
    /// <exception cref="ArgumentException">The selection is empty or contains an unknown identity.</exception>
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
    /// <param name="other">The target whose node identities are compared with this target.</param>
    /// <returns>Whether the two targets contain at least one common node identity.</returns>
    /// <exception cref="ArgumentNullException">The other target is null.</exception>
    public bool Overlaps(GraphRewriteTarget<TNode, TEdge> other)
    {
        ArgumentNullException.ThrowIfNull(other);
        var ids = Nodes.Select(node => node.Id).ToHashSet();
        return other.Nodes.Any(node => ids.Contains(node.Id));
    }
}
