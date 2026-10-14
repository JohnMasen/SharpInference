namespace SharpInference.Graphs.Core;

/// <summary>Algorithms over filtered graph views.</summary>
public static class GraphAlgorithms
{
    /// <summary>Returns a stable topological order or rejects a directed cycle.</summary>
    /// <typeparam name="TNode">The visible node type.</typeparam>
    /// <typeparam name="TEdge">The visible edge type.</typeparam>
    /// <param name="view">The view whose visible edges define ordering constraints.</param>
    /// <returns>All visible nodes, using view order to break ties among ready nodes.</returns>
    /// <exception cref="ArgumentNullException">The view is null.</exception>
    /// <exception cref="InvalidDataException">The visible edges contain a directed cycle.</exception>
    public static IReadOnlyList<TNode> TopologicalOrder<TNode, TEdge>(
        this IGraphView<TNode, TEdge> view)
        where TNode : IGraphNode
        where TEdge : IGraphEdge
    {
        return TryTopologicalOrder(view) ??
            throw new InvalidDataException("The graph view contains a directed cycle.");
    }

    /// <summary>Attempts a stable topological ordering without treating a cycle as an exception.</summary>
    /// <typeparam name="TNode">The visible node type.</typeparam>
    /// <typeparam name="TEdge">The visible edge type.</typeparam>
    /// <param name="view">The view whose visible edges define ordering constraints.</param>
    /// <returns>The ordered nodes, or null if the visible graph contains a directed cycle.</returns>
    /// <exception cref="ArgumentNullException">The view is null.</exception>
    internal static IReadOnlyList<TNode>? TryTopologicalOrder<TNode, TEdge>(
        IGraphView<TNode, TEdge> view)
        where TNode : IGraphNode
        where TEdge : IGraphEdge
    {
        ArgumentNullException.ThrowIfNull(view);
        var nodes = view.Nodes.ToArray();
        var positions = nodes.Select((node, index) => (node.Id, index))
            .ToDictionary(item => item.Id, item => item.index);
        var indegree = nodes.ToDictionary(
            node => node.Id,
            node => view.EdgesToNode(node.Id).Count());
        var ready = new PriorityQueue<TNode, int>();
        foreach (var node in nodes.Where(node => indegree[node.Id] == 0))
        {
            ready.Enqueue(node, positions[node.Id]);
        }

        var result = new List<TNode>(nodes.Length);
        while (ready.TryDequeue(out var node, out _))
        {
            result.Add(node);
            foreach (var edge in view.EdgesFromNode(node.Id))
            {
                indegree[edge.Target]--;
                if (indegree[edge.Target] == 0)
                {
                    ready.Enqueue(view.GetNode(edge.Target), positions[edge.Target]);
                }
            }
        }

        return result.Count == nodes.Length ? result : null;
    }

    /// <summary>Returns whether a directed path exists between two visible nodes.</summary>
    /// <typeparam name="TNode">The visible node type.</typeparam>
    /// <typeparam name="TEdge">The visible edge type.</typeparam>
    /// <param name="view">The view limiting all nodes and edges of the path.</param>
    /// <param name="source">The identity of the visible starting node.</param>
    /// <param name="target">The identity of the visible destination node.</param>
    /// <returns>Whether a zero-or-more-edge path exists; a visible node reaches itself.</returns>
    /// <remarks>Uses breadth-first traversal and terminates on cyclic graphs.</remarks>
    /// <exception cref="ArgumentNullException">The view is null.</exception>
    /// <exception cref="KeyNotFoundException">Either endpoint is unknown or hidden.</exception>
    public static bool IsReachable<TNode, TEdge>(
        this IGraphView<TNode, TEdge> view,
        GraphNodeId source,
        GraphNodeId target)
        where TNode : IGraphNode
        where TEdge : IGraphEdge
    {
        ArgumentNullException.ThrowIfNull(view);
        _ = view.GetNode(source);
        _ = view.GetNode(target);
        if (source == target)
        {
            return true;
        }

        var visited = new HashSet<GraphNodeId> { source };
        var pending = new Queue<GraphNodeId>();
        pending.Enqueue(source);
        while (pending.TryDequeue(out var current))
        {
            foreach (var edge in view.EdgesFromNode(current))
            {
                if (edge.Target == target)
                {
                    return true;
                }

                if (visited.Add(edge.Target))
                {
                    pending.Enqueue(edge.Target);
                }
            }
        }

        return false;
    }

    /// <summary>Returns all nodes reachable from a visible source, including the source.</summary>
    /// <typeparam name="TNode">The visible node type.</typeparam>
    /// <typeparam name="TEdge">The visible edge type.</typeparam>
    /// <param name="view">The view limiting the traversal.</param>
    /// <param name="source">The identity of the visible starting node.</param>
    /// <returns>Reachable nodes in view order, not breadth-first discovery order.</returns>
    /// <exception cref="ArgumentNullException">The view is null.</exception>
    /// <exception cref="KeyNotFoundException">The source is unknown or hidden.</exception>
    public static IReadOnlyList<TNode> ReachableFrom<TNode, TEdge>(
        this IGraphView<TNode, TEdge> view,
        GraphNodeId source)
        where TNode : IGraphNode
        where TEdge : IGraphEdge
    {
        ArgumentNullException.ThrowIfNull(view);
        _ = view.GetNode(source);
        var visited = new HashSet<GraphNodeId> { source };
        var pending = new Queue<GraphNodeId>();
        pending.Enqueue(source);
        while (pending.TryDequeue(out var current))
        {
            foreach (var edge in view.EdgesFromNode(current))
            {
                if (visited.Add(edge.Target))
                {
                    pending.Enqueue(edge.Target);
                }
            }
        }

        return view.Nodes.Where(node => visited.Contains(node.Id)).ToArray();
    }
}
