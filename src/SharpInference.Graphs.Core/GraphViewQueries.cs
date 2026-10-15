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

    /// <summary>Enumerates simple linear matches using the more selective endpoint when sufficiently smaller.</summary>
    /// <typeparam name="TNode">The source node type.</typeparam>
    /// <typeparam name="TEdge">The source edge type.</typeparam>
    /// <param name="view">The view defining the nodes and direct edges available for matching.</param>
    /// <param name="pattern">A complete alternating sequence ending with a node constraint.</param>
    /// <returns>A deferred sequence of stable views containing each match's exact path content.</returns>
    /// <remarks>
    /// Paths never repeat a node. An unknown identity anchor yields no matches.
    /// Each enumeration searches again; materialize results to reuse one search.
    /// Result order is unspecified; each result's nodes and edges remain in forward path order.
    /// Search proceeds backward when target candidates are fewer than half the source candidates.
    /// General endpoint conditions may require a candidate scan before the first result.
    /// Predicates must not depend on invocation order or side effects.
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
            var last = pattern.Nodes[^1];
            var reverse = false;
            HashSet<GraphNodeId>? sourceIds = null;
            HashSet<GraphNodeId>? targetIds = null;
            IEnumerable<TNode> candidates;
            if (edges.Length == 0 || first.Id is not null || first.Unrestricted && last.Unrestricted)
            {
                candidates = Candidates(first);
            }
            else if (last.Id is not null && first.Unrestricted)
            {
                var targets = Candidates(last).ToArray();
                if (targets.Length == 0)
                    yield break;
                var sources = view.Nodes.Take(3).ToArray();
                reverse = sources.Length > 2;
                candidates = reverse ? targets : sources;
                targetIds = targets.Select(node => node.Id).ToHashSet();
            }
            else
            {
                var sources = new List<TNode>();
                var targets = new List<TNode>();
                if (last.Id is not null)
                    targets.AddRange(Candidates(last));
                foreach (var node in view.Nodes)
                {
                    if (first.Predicate(node))
                        sources.Add(node);
                    if (last.Id is null && last.Predicate(node))
                        targets.Add(node);
                }
                if (sources.Count == 0 || targets.Count == 0)
                    yield break;
                sourceIds = sources.Select(node => node.Id).ToHashSet();
                targetIds = targets.Select(node => node.Id).ToHashSet();
                reverse = 2L * targets.Count < sources.Count;
                candidates = reverse ? targets : sources;
            }
            var initial = reverse ? nodes.Length - 1 : 0;
            var finish = reverse ? 0 : nodes.Length - 1;
            foreach (var candidate in candidates)
            {
                nodes[initial] = candidate;
                visited.Add(candidate.Id);
                foreach (var match in Extend(initial))
                    yield return match;
                visited.Remove(candidate.Id);
            }

            IEnumerable<TNode> Candidates(NodePattern constraint)
            {
                if (constraint.Id is { } id)
                {
                    if (view.ContainsNode(id))
                    {
                        var node = view.GetNode(id);
                        if (constraint.Predicate(node))
                            yield return node;
                    }
                }
                else
                {
                    foreach (var node in view.Nodes)
                        if (constraint.Predicate(node))
                            yield return node;
                }
            }

            IEnumerable<IGraphView<TNode, TEdge>> Extend(int index)
            {
                if (index == finish)
                {
                    yield return new MatchedGraphView<TNode, TEdge>(nodes, edges);
                    yield break;
                }
                var edgeIndex = reverse ? index - 1 : index;
                var nextIndex = reverse ? index - 1 : index + 1;
                var adjacent = reverse
                    ? view.EdgesToNode(nodes[index].Id)
                    : view.EdgesFromNode(nodes[index].Id);
                foreach (var edge in adjacent)
                {
                    var next = pattern.Nodes[nextIndex];
                    var nextId = reverse ? edge.Source : edge.Target;
                    if (visited.Contains(nextId) ||
                        next.Id is { } targetId && nextId != targetId ||
                        !pattern.Edges[edgeIndex].Predicate(edge))
                        continue;
                    var target = view.GetNode(nextId);
                    var cached = nextIndex == 0 ? sourceIds
                        : nextIndex == nodes.Length - 1 ? targetIds : null;
                    if (cached is not null ? !cached.Contains(nextId) : !next.Predicate(target))
                        continue;
                    nodes[nextIndex] = target;
                    edges[edgeIndex] = edge;
                    visited.Add(target.Id);
                    foreach (var match in Extend(nextIndex))
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
