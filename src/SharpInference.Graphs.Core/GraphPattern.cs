namespace SharpInference.Graphs.Core;

/// <summary>A named typed graph-search pattern.</summary>
public sealed class GraphPattern<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    private readonly List<NodePattern> nodes = [];
    private readonly List<EdgePattern> edges = [];

    /// <summary>Adds one named node variable.</summary>
    public GraphPattern<TNode, TEdge> Node<TRequested>(
        string name,
        Func<TRequested, bool>? predicate = null)
        where TRequested : TNode
    {
        RequireUnique(name, nodes.Select(node => node.Name));
        nodes.Add(new(
            name,
            node => node is TRequested requested &&
                (predicate is null || predicate(requested))));
        return this;
    }

    /// <summary>Adds one named directed edge variable between node variables.</summary>
    public GraphPattern<TNode, TEdge> Edge<TRequested>(
        string name,
        string source,
        string target,
        Func<TRequested, bool>? predicate = null)
        where TRequested : TEdge
    {
        RequireUnique(name, edges.Select(edge => edge.Name));
        if (nodes.All(node => node.Name != source) ||
            nodes.All(node => node.Name != target))
        {
            throw new ArgumentException(
                "Pattern edge endpoints must reference declared node variables.");
        }

        edges.Add(new(
            name,
            source,
            target,
            edge => edge is TRequested requested &&
                (predicate is null || predicate(requested))));
        return this;
    }

    /// <summary>Finds every complete named match in deterministic graph order.</summary>
    public IReadOnlyList<GraphMatch<TNode, TEdge>> Find(
        Graph<TNode, TEdge> graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var results = new List<GraphMatch<TNode, TEdge>>();
        BindNode(0, new Dictionary<string, TNode>(StringComparer.Ordinal));
        return results;

        void BindNode(int index, Dictionary<string, TNode> bindings)
        {
            if (index == nodes.Count)
            {
                BindEdge(
                    0,
                    bindings,
                    new Dictionary<string, TEdge>(StringComparer.Ordinal));
                return;
            }

            var pattern = nodes[index];
            foreach (var candidate in graph.Nodes.Where(pattern.Predicate))
            {
                if (bindings.Values.Any(bound => bound.Id == candidate.Id))
                {
                    continue;
                }

                bindings.Add(pattern.Name, candidate);
                if (CanStillMatch(graph, bindings))
                {
                    BindNode(index + 1, bindings);
                }

                bindings.Remove(pattern.Name);
            }
        }

        void BindEdge(
            int index,
            IReadOnlyDictionary<string, TNode> nodeBindings,
            Dictionary<string, TEdge> edgeBindings)
        {
            if (index == edges.Count)
            {
                results.Add(new(
                    new Dictionary<string, TNode>(
                        nodeBindings, StringComparer.Ordinal),
                    new Dictionary<string, TEdge>(
                        edgeBindings, StringComparer.Ordinal)));
                return;
            }
            var pattern = edges[index];
            var source = nodeBindings[pattern.Source].Id;
            var target = nodeBindings[pattern.Target].Id;
            foreach (var candidate in graph.OutEdges(source).Where(edge =>
                edge.Target == target && pattern.Predicate(edge)))
            {
                if (edgeBindings.Values.Any(bound => bound.Id == candidate.Id))
                {
                    continue;
                }

                edgeBindings.Add(pattern.Name, candidate);
                BindEdge(index + 1, nodeBindings, edgeBindings);
                edgeBindings.Remove(pattern.Name);
            }
        }
    }

    private bool CanStillMatch(
        Graph<TNode, TEdge> graph,
        IReadOnlyDictionary<string, TNode> bindings)
    {
        foreach (var edge in edges)
        {
            if (!bindings.TryGetValue(edge.Source, out var source) ||
                !bindings.TryGetValue(edge.Target, out var target))
            {
                continue;
            }

            if (!graph.OutEdges(source.Id).Any(candidate =>
                candidate.Target == target.Id && edge.Predicate(candidate)))
            {
                return false;
            }
        }
        return true;
    }

    private static void RequireUnique(string name, IEnumerable<string> existing)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("A pattern variable name is required.", nameof(name));
        }

        if (existing.Contains(name, StringComparer.Ordinal))
        {
            throw new ArgumentException(
                $"Pattern variable '{name}' is already declared.", nameof(name));
        }
    }

    private sealed record NodePattern(string Name, Func<TNode, bool> Predicate);
    private sealed record EdgePattern(
        string Name,
        string Source,
        string Target,
        Func<TEdge, bool> Predicate);
}

/// <summary>One complete mapping from named pattern variables to graph elements.</summary>
public sealed class GraphMatch<TNode, TEdge>
    where TNode : IGraphNode
    where TEdge : IGraphEdge
{
    internal GraphMatch(
        IReadOnlyDictionary<string, TNode> nodes,
        IReadOnlyDictionary<string, TEdge> edges)
    {
        Nodes = nodes;
        Edges = edges;
    }

    /// <summary>Gets all named node captures.</summary>
    public IReadOnlyDictionary<string, TNode> Nodes { get; }

    /// <summary>Gets all named edge captures.</summary>
    public IReadOnlyDictionary<string, TEdge> Edges { get; }

    /// <summary>Gets one named node capture with its requested type.</summary>
    public TRequested Node<TRequested>(string name) where TRequested : TNode =>
        Nodes.TryGetValue(name, out var node) && node is TRequested requested
            ? requested
            : throw new KeyNotFoundException(
                $"Pattern node '{name}' is not a {typeof(TRequested).Name}.");

    /// <summary>Gets one named edge capture with its requested type.</summary>
    public TRequested Edge<TRequested>(string name) where TRequested : TEdge =>
        Edges.TryGetValue(name, out var edge) && edge is TRequested requested
            ? requested
            : throw new KeyNotFoundException(
                $"Pattern edge '{name}' is not a {typeof(TRequested).Name}.");
}
