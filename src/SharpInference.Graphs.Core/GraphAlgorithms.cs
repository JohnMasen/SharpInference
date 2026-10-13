namespace SharpInference.Graphs.Core;

/// <summary>Algorithms over filtered graph views.</summary>
public static class GraphAlgorithms
{
    /// <summary>Returns a stable topological order or rejects a directed cycle.</summary>
    public static IReadOnlyList<TNode> TopologicalOrder<TNode, TEdge>(
        this GraphView<TNode, TEdge> view)
        where TNode : IGraphNode
        where TEdge : IGraphEdge
    {
        ArgumentNullException.ThrowIfNull(view);
        var nodes = view.Nodes.ToArray();
        var positions = nodes.Select((node, index) => (node.Id, index))
            .ToDictionary(item => item.Id, item => item.index);
        var indegree = nodes.ToDictionary(
            node => node.Id,
            node => view.InEdges(node.Id).Count());
        var ready = new PriorityQueue<TNode, int>();
        foreach (var node in nodes.Where(node => indegree[node.Id] == 0))
        {
            ready.Enqueue(node, positions[node.Id]);
        }

        var result = new List<TNode>(nodes.Length);
        while (ready.TryDequeue(out var node, out _))
        {
            result.Add(node);
            foreach (var edge in view.OutEdges(node.Id))
            {
                indegree[edge.Target]--;
                if (indegree[edge.Target] == 0)
                {
                    ready.Enqueue(view.GetNode(edge.Target), positions[edge.Target]);
                }
            }
        }

        if (result.Count != nodes.Length)
        {
            throw new InvalidDataException("The graph view contains a directed cycle.");
        }

        return result;
    }

    /// <summary>Returns whether a directed path exists between two visible nodes.</summary>
    public static bool IsReachable<TNode, TEdge>(
        this GraphView<TNode, TEdge> view,
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
            foreach (var edge in view.OutEdges(current))
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
    public static IReadOnlyList<TNode> ReachableFrom<TNode, TEdge>(
        this GraphView<TNode, TEdge> view,
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
            foreach (var edge in view.OutEdges(current))
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
