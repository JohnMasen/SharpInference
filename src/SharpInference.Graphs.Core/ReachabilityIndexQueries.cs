namespace SharpInference.Graphs.Core;

/// <summary>Builds reusable reachability indexes without changing the source graph.</summary>
public static class ReachabilityIndexQueries
{
    /// <summary>Builds a fixed reachability index for selected source and target identities.</summary>
    /// <typeparam name="TNode">The node type used for endpoint selection and instance queries.</typeparam>
    /// <typeparam name="TEdge">The visible directed edge type.</typeparam>
    /// <param name="view">The view defining the complete scope of possible paths.</param>
    /// <param name="sources">Source nodes, enumerated once and deduplicated by identity.</param>
    /// <param name="targets">Target nodes, enumerated once and deduplicated by identity.</param>
    /// <returns>An index accepting only the selected endpoints, including empty scopes.</returns>
    /// <remarks>
    /// Intermediate path nodes may be any visible nodes, not just endpoints.
    /// Filtered views are materialized to freeze their visible structure. A node reaches itself.
    /// Dense DAG queries use reverse-topological propagation; sparse or cyclic graphs use traversal.
    /// </remarks>
    /// <exception cref="ArgumentNullException">The view, an endpoint sequence, or an endpoint node is null.</exception>
    /// <exception cref="KeyNotFoundException">A selected endpoint is unknown or hidden in the view.</exception>
    public static ReachabilityIndex<TNode> CreateReachabilityIndex<TNode, TEdge>(
        this IGraphView<TNode, TEdge> view,
        IEnumerable<TNode> sources,
        IEnumerable<TNode> targets)
        where TNode : IGraphNode
        where TEdge : IGraphEdge
    {
        ArgumentNullException.ThrowIfNull(view);
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(targets);
        var from = EndpointIds(sources);
        var to = EndpointIds(targets);
        // Filter predicates may depend on mutable external state: freeze the visible graph once.
        var snapshot = view is Graph<TNode, TEdge> graph ? graph : view.ToGraph();
        foreach (var id in from.Concat(to))
        {
            _ = snapshot.GetNode(id);
        }

        var columns = to.Select((id, index) => (id, index))
            .ToDictionary(item => item.id, item => item.index);
        var words = (to.Length + 63) / 64;
        var rows = new Dictionary<GraphNodeId, ulong[]>();
        if (from.Length == 0 || to.Length == 0)
        {
            foreach (var id in from)
            {
                rows.Add(id, new ulong[words]);
            }

            return new(rows, columns);
        }

        // Full DAG queries amortize reverse-topological propagation; sparse queries avoid V*T storage.
        var order = from.Length >= Math.Max(1, snapshot.Nodes.Count / 4)
            ? GraphAlgorithms.TryTopologicalOrder(snapshot) : null;
        if (order is not null)
        {
            var reachable = new Dictionary<GraphNodeId, ulong[]>(order.Count);
            for (var index = order.Count - 1; index >= 0; index--)
            {
                var node = order[index];
                var row = new ulong[words];
                if (columns.TryGetValue(node.Id, out var column))
                {
                    row[column / 64] |= 1UL << (column % 64);
                }

                foreach (var edge in snapshot.EnumerateEdgesFromNode(node.Id))
                {
                    var successor = reachable[edge.Target];
                    for (var word = 0; word < words; word++)
                    {
                        row[word] |= successor[word];
                    }
                }
                reachable.Add(node.Id, row);
            }
            foreach (var id in from)
            {
                rows.Add(id, reachable[id]);
            }
        }
        else
        {
            foreach (var source in from)
            {
                var row = new ulong[words];
                var remaining = to.Length;
                var visited = new HashSet<GraphNodeId> { source };
                var pending = new Queue<GraphNodeId>();
                pending.Enqueue(source);
                while (pending.TryDequeue(out var current))
                {
                    if (columns.TryGetValue(current, out var column))
                    {
                        row[column / 64] |= 1UL << (column % 64);
                        if (--remaining == 0)
                        {
                            break;
                        }
                    }
                    foreach (var edge in snapshot.EnumerateEdgesFromNode(current))
                    {
                        if (visited.Add(edge.Target))
                        {
                            pending.Enqueue(edge.Target);
                        }
                    }
                }
                rows.Add(source, row);
            }
        }
        return new(rows, columns);
    }

    /// <summary>Consumes an endpoint sequence once and extracts its distinct structural identities.</summary>
    /// <typeparam name="TNode">The endpoint node type.</typeparam>
    /// <param name="nodes">The non-null endpoint sequence to consume.</param>
    /// <returns>The distinct endpoint identities.</returns>
    /// <exception cref="ArgumentNullException">An endpoint node is null.</exception>
    private static GraphNodeId[] EndpointIds<TNode>(IEnumerable<TNode> nodes)
        where TNode : IGraphNode
    {
        var ids = new HashSet<GraphNodeId>();
        foreach (var node in nodes)
        {
            ArgumentNullException.ThrowIfNull(node);
            ids.Add(node.Id);
        }
        return ids.ToArray();
    }
}
