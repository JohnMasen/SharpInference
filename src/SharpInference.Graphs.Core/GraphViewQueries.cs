namespace SharpInference.Graphs.Core;

/// <summary>Composable queries shared by all read-only graph views.</summary>
public static class GraphViewQueries
{
    /// <summary>Restricts a view with deferred node and edge conditions.</summary>
    /// <typeparam name="TNode">The source node type.</typeparam>
    /// <typeparam name="TEdge">The source edge type.</typeparam>
    /// <param name="view">The source view whose visible scope cannot be expanded.</param>
    /// <param name="node">The node condition, or null to retain every source node.</param>
    /// <param name="edge">The edge condition, or null to retain every source edge with visible endpoints.</param>
    /// <returns>A non-owning view that preserves source order and permits isolated nodes.</returns>
    /// <exception cref="ArgumentNullException">The source view is null.</exception>
    public static GraphView<TNode, TEdge> Filter<TNode, TEdge>(
        this IGraphView<TNode, TEdge> view,
        Func<TNode, bool>? node = null,
        Func<TEdge, bool>? edge = null)
        where TNode : IGraphNode
        where TEdge : IGraphEdge
    {
        ArgumentNullException.ThrowIfNull(view);
        return new(view, node, edge);
    }

    /// <summary>Enumerates simple linear matches in starting-node and edge declaration order.</summary>
    /// <typeparam name="TNode">The source node type.</typeparam>
    /// <typeparam name="TEdge">The source edge type.</typeparam>
    /// <param name="view">The view defining the nodes and direct edges available for matching.</param>
    /// <param name="pattern">A complete alternating sequence ending with a node constraint.</param>
    /// <returns>A deferred sequence of stable views containing each match's exact path content.</returns>
    /// <remarks>
    /// Paths never repeat a node. An unknown identity anchor yields no matches.
    /// Each enumeration searches again; materialize results to reuse one search.
    /// </remarks>
    /// <exception cref="ArgumentNullException">The view or pattern is null.</exception>
    /// <exception cref="ArgumentException">The pattern ends with an edge constraint.</exception>
    public static IEnumerable<IGraphView<TNode, TEdge>> Match<TNode, TEdge>(
        this IGraphView<TNode, TEdge> view, MatchContext pattern)
        where TNode : IGraphNode
        where TEdge : IGraphEdge
    {
        ArgumentNullException.ThrowIfNull(view);
        ArgumentNullException.ThrowIfNull(pattern);
        if (!pattern.Complete)
            throw new ArgumentException("A complete pattern must end with a node.", nameof(pattern));
        return Enumerate();

        IEnumerable<IGraphView<TNode, TEdge>> Enumerate()
        {
            var nodes = new TNode[pattern.Nodes.Count];
            var edges = new TEdge[pattern.Edges.Count];
            var visited = new HashSet<GraphNodeId>();
            var first = pattern.Nodes[0];
            IEnumerable<TNode> candidates = first.Id is { } id
                ? view.ContainsNode(id) ? [view.GetNode(id)] : []
                : view.Nodes;
            foreach (var candidate in candidates)
            {
                if (!first.Predicate(candidate))
                    continue;
                nodes[0] = candidate;
                visited.Add(candidate.Id);
                foreach (var match in Extend(0))
                    yield return match;
                visited.Remove(candidate.Id);
            }

            IEnumerable<IGraphView<TNode, TEdge>> Extend(int index)
            {
                if (index == edges.Length)
                {
                    yield return new MatchedGraphView<TNode, TEdge>(nodes, edges);
                    yield break;
                }
                foreach (var edge in view.EdgesFromNode(nodes[index].Id))
                {
                    var next = pattern.Nodes[index + 1];
                    if (visited.Contains(edge.Target) ||
                        next.Id is { } targetId && edge.Target != targetId ||
                        !pattern.Edges[index].Predicate(edge))
                        continue;
                    var target = view.GetNode(edge.Target);
                    if (!next.Predicate(target))
                        continue;
                    nodes[index + 1] = target;
                    edges[index] = edge;
                    visited.Add(target.Id);
                    foreach (var match in Extend(index + 1))
                        yield return match;
                    visited.Remove(target.Id);
                }
            }
        }
    }

    /// <summary>Materializes the exact visible structure of one view as a new graph.</summary>
    /// <typeparam name="TNode">The visible node type.</typeparam>
    /// <typeparam name="TEdge">The visible edge type.</typeparam>
    /// <param name="view">The view to consume without restoring hidden edges.</param>
    /// <returns>A graph with copied collection membership and the same node and edge instances.</returns>
    /// <exception cref="ArgumentNullException">The view or a visible element is null.</exception>
    /// <exception cref="InvalidDataException">The visible structure contains invalid identities or endpoints.</exception>
    public static Graph<TNode, TEdge> ToGraph<TNode, TEdge>(
        this IGraphView<TNode, TEdge> view)
        where TNode : IGraphNode
        where TEdge : IGraphEdge
    {
        ArgumentNullException.ThrowIfNull(view);
        return new(view.Nodes, view.Edges);
    }

    /// <summary>Consumes and merges exact visible content; conflicting identities are rejected.</summary>
    /// <typeparam name="TNode">The node type shared by the views.</typeparam>
    /// <typeparam name="TEdge">The edge type shared by the views.</typeparam>
    /// <param name="views">The view sequence to enumerate once, usually the results of a match query.</param>
    /// <returns>A graph deduplicated by identity in first-occurrence order, or an empty graph.</returns>
    /// <remarks>
    /// Merging discards match grouping. Subsequent queries may combine content from different matches.
    /// Elements sharing an identity must compare equal; edge endpoints must also agree.
    /// </remarks>
    /// <exception cref="ArgumentNullException">The sequence or one of its views is null.</exception>
    /// <exception cref="InvalidDataException">Identities conflict or the merged graph is invalid.</exception>
    public static Graph<TNode, TEdge> ToGraph<TNode, TEdge>(
        this IEnumerable<IGraphView<TNode, TEdge>> views)
        where TNode : IGraphNode
        where TEdge : IGraphEdge
    {
        ArgumentNullException.ThrowIfNull(views);
        var nodes = new Dictionary<GraphNodeId, TNode>();
        var edges = new Dictionary<GraphEdgeId, TEdge>();
        foreach (var view in views)
        {
            ArgumentNullException.ThrowIfNull(view);
            foreach (var node in view.Nodes)
            {
                if (nodes.TryGetValue(node.Id, out var previous) &&
                    !EqualityComparer<TNode>.Default.Equals(previous, node))
                    throw new InvalidDataException($"Conflicting graph node '{node.Id}'.");
                nodes.TryAdd(node.Id, node);
            }
            foreach (var edge in view.Edges)
            {
                if (edges.TryGetValue(edge.Id, out var previous) &&
                    (previous.Source != edge.Source || previous.Target != edge.Target ||
                     !EqualityComparer<TEdge>.Default.Equals(previous, edge)))
                    throw new InvalidDataException($"Conflicting graph edge '{edge.Id}'.");
                edges.TryAdd(edge.Id, edge);
            }
        }
        return new(nodes.Values, edges.Values);
    }
}
