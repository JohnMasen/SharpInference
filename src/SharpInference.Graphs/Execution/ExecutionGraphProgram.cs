using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Execution;

/// <summary>Stores execution graphs and derived call sites forming a single-rooted, reachable call DAG.</summary>
/// <remarks>Multiple call instances may target a shared graph. This validates graph relationships,
/// not callee interface compatibility or compilation-boundary policies.</remarks>
public sealed class ExecutionGraphProgram
{
    /// <summary>Snapshots the graphs, derives call relations, and validates a unique root and call DAG.</summary>
    /// <param name="graphs">The nonempty graph sequence, retained in enumeration order.</param>
    /// <exception cref="ArgumentNullException">The graph sequence is null.</exception>
    /// <exception cref="ArgumentException">The graph sequence is empty.</exception>
    /// <exception cref="InvalidDataException">Graph identities repeat, a call target is missing, there is not exactly
    /// one unreferenced root, a reachable call cycle exists, or some graph is unreachable from that root.</exception>
    public ExecutionGraphProgram(IEnumerable<ExecutionGraph> graphs)
    {
        var values = graphs?.ToArray() ?? throw new ArgumentNullException(nameof(graphs));
        if (values.Length == 0)
        {
            throw new ArgumentException(
                "An execution graph program requires at least one graph.", nameof(graphs));
        }

        var map = new Dictionary<ExecutionGraphId, ExecutionGraph>();
        foreach (var graph in values)
        {
            if (!map.TryAdd(graph.Id, graph))
            {
                throw new InvalidDataException($"Duplicate execution graph '{graph.Id}'.");
            }
        }

        var relations = values.SelectMany(graph =>
            graph.Structure.Filter(node: node => node is ExecutionGraphCallNode)
                .Nodes.Cast<ExecutionGraphCallNode>().Select(call =>
                new ExecutionGraphCallRelation(
                    $"{graph.Id.Value}:{call.Id.Value}",
                    graph.Id,
                    call.Target,
                    call.Id))).ToArray();
        foreach (var relation in relations)
        {
            if (!map.ContainsKey(relation.Target))
            {
                throw new InvalidDataException(
                    $"Graph call '{relation.Source}/{relation.Call}' targets unknown graph " +
                    $"'{relation.Target}'.");
            }
        }

        var referenced = relations.Select(relation => relation.Target).ToHashSet();
        var roots = values.Where(graph => !referenced.Contains(graph.Id)).ToArray();
        if (roots.Length != 1)
        {
            throw new InvalidDataException(
                $"An execution graph program requires exactly one root; found {roots.Length}.");
        }

        ValidateCallDag(roots[0].Id, map, relations);
        Graphs = Array.AsReadOnly(values);
        Calls = Array.AsReadOnly(relations);
        RootGraph = roots[0].Id;
    }

    /// <summary>Gets the read-only graph snapshot in supplied order.</summary>
    public IReadOnlyList<ExecutionGraph> Graphs { get; }
    /// <summary>Gets call-site relations in graph order and then call-node declaration order.</summary>
    public IReadOnlyList<ExecutionGraphCallRelation> Calls { get; }
    /// <summary>Gets the identity of the single graph that is not targeted by any call.</summary>
    public ExecutionGraphId RootGraph { get; }

    /// <summary>Gets graph-call instances originating in one graph.</summary>
    /// <param name="source">The caller graph identity to match.</param>
    /// <returns>Matching call relations in stored order, or an empty list for no matches.</returns>
    public IReadOnlyList<ExecutionGraphCallRelation> Callees(
        ExecutionGraphId source) =>
        Calls.Where(call => call.Source == source).ToArray();

    /// <summary>Gets graph-call instances targeting one graph.</summary>
    /// <param name="target">The callee graph identity to match.</param>
    /// <returns>Matching call relations in stored order, including distinct calls from the same caller.</returns>
    public IReadOnlyList<ExecutionGraphCallRelation> Callers(
        ExecutionGraphId target) =>
        Calls.Where(call => call.Target == target).ToArray();

    /// <summary>Gets non-root graphs that currently have no callers.</summary>
    /// <returns>Matching graphs in stored order; empty for a successfully validated immutable program.</returns>
    public IReadOnlyList<ExecutionGraph> UnreferencedGraphs() =>
        Graphs.Where(graph =>
            graph.Id != RootGraph && Callers(graph.Id).Count == 0).ToArray();

    /// <summary>Returns whether a call is the target graph's only call site.</summary>
    /// <param name="caller">The caller graph identity.</param>
    /// <param name="call">The local call node identity.</param>
    /// <returns>True when exactly one call relation targets the selected call's callee.</returns>
    /// <exception cref="ArgumentException">The caller and call identities do not identify a call site.</exception>
    public bool IsExclusiveCall(ExecutionGraphId caller, ExecutionNodeId call)
    {
        var relation = Calls.SingleOrDefault(value =>
            value.Source == caller && value.Call == call)
            ?? throw new ArgumentException(
                $"Execution graph program has no call '{caller}/{call}'.",
                nameof(call));
        return Callers(relation.Target).Count == 1;
    }

    /// <summary>Traverses distinct call targets from the root, rejecting reachable cycles and unreachable definitions.</summary>
    /// <param name="root">The inferred root identity.</param>
    /// <param name="graphs">All graph definitions keyed by identity.</param>
    /// <param name="relations">The derived call-site relations with already resolved targets.</param>
    /// <exception cref="InvalidDataException">Traversal encounters a cycle or does not reach every graph.</exception>
    private static void ValidateCallDag(
        ExecutionGraphId root,
        IReadOnlyDictionary<ExecutionGraphId, ExecutionGraph> graphs,
        IReadOnlyList<ExecutionGraphCallRelation> relations)
    {
        var states = new Dictionary<ExecutionGraphId, int>();
        var reachable = new HashSet<ExecutionGraphId>();
        var path = new Stack<ExecutionGraphId>();
        Visit(root);
        if (reachable.Count != graphs.Count)
        {
            var missing = graphs.Keys.Where(id => !reachable.Contains(id));
            throw new InvalidDataException(
                $"Execution graph program contains unreachable graphs: " +
                $"{string.Join(", ", missing)}.");
        }

        void Visit(ExecutionGraphId id)
        {
            if (states.TryGetValue(id, out var state))
            {
                if (state == 1)
                {
                    throw new InvalidDataException(
                        $"Execution graph call cycle detected: " +
                        $"{string.Join(" -> ", path.Reverse())} -> {id}.");
                }

                return;
            }
            states[id] = 1;
            reachable.Add(id);
            path.Push(id);
            foreach (var target in relations.Where(relation => relation.Source == id)
                .Select(relation => relation.Target).Distinct())
            {
                Visit(target);
            }

            path.Pop();
            states[id] = 2;
        }
    }
}
