using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

internal sealed record TierOneGraphSelection(ExecutionGraph Graph,
    IReadOnlyDictionary<ExecutionNodeId, InstructionOptimizationCapability> Implementations,
    TierOneOptimizationReport Report)
{
    public static TierOneGraphSelection Apply(ExecutionGraph graph, VmTarget target, VmOptimizationOptions options,
        IReadOnlyList<InstructionOptimizationCapability> capabilities,
        IReadOnlyDictionary<ResourceId, ResourceId> aliases)
    {
        const string model = "Additive conservative measured pointwise-segment savings; cache and whole-plan interactions unmodeled";
        var settings = options.TierOne;
        TierOneGraphSelection Retain(string diagnostic, int matched = 0, int trusted = 0, int states = 0) =>
            new(graph, new Dictionary<ExecutionNodeId, InstructionOptimizationCapability>(),
                new(TierOneSelectionStatus.BaselineRetained, model, diagnostic, matched, trusted, 0, states, 0,
                    graph.Nodes.Count, graph.Nodes.Count));
        if (settings is null) return Retain("No offline cost profile supplied; T0 retained.");
        if (settings.Profile.EnvironmentFingerprint != settings.EnvironmentFingerprint)
            return Retain("Offline profile environment fingerprint does not match; T0 retained.");
        var age = DateTimeOffset.UtcNow - settings.Profile.MeasuredAt;
        if (age > settings.MaximumProfileAge || age < -TimeSpan.FromMinutes(5))
            return Retain("Offline profile is expired or has a future timestamp; T0 retained.");
        var matches = TierOnePatternMatcher.Find(graph, capabilities, target, aliases);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var measurements = settings.Profile.Measurements.ToDictionary(measurement => measurement.Key, StringComparer.Ordinal);
        var dependencies = Dependencies(graph);
        var candidates = new List<CostedCandidate>();
        foreach (var candidate in matches)
        {
            var key = $"{candidate.Capability.Name}|{candidate.Capability.ImplementationFingerprint}|" +
                $"{string.Join("x", resources[candidate.Output].Tensor.Dimensions)}|{candidate.InputAliases}|" +
                $"{options.ThreadsPerGroup}|{options.ReuseLocalStorage}";
            if (measurements.TryGetValue(key, out var measurement) && measurement.ConservativeSavingMicroseconds > 0 &&
                Contract(graph, dependencies, [candidate]) is not null)
                candidates.Add(new(candidate, measurement.ConservativeSavingMicroseconds));
        }
        if (candidates.Count == 0)
            return Retain("No matching, legal, conservatively profitable measurements; T0 retained.", matches.Count);

        var budget = new SearchBudget(settings.MaximumSearchStates);
        var relaxed = new List<CostedCandidate>();
        var remaining = candidates.ToHashSet();
        while (remaining.Count != 0)
        {
            var first = remaining.OrderBy(candidate => candidate.Fusion.Nodes[0]).ThenBy(candidate => candidate.Fusion.Capability.Name,
                StringComparer.Ordinal).First();
            remaining.Remove(first);
            var component = new List<CostedCandidate> { first };
            for (var index = 0; index < component.Count; index++)
                foreach (var next in remaining.Where(next => Overlap(component[index], next)).ToArray())
                {
                    remaining.Remove(next);
                    component.Add(next);
                }
            relaxed.AddRange(component.All(candidate => candidate.Fusion.Nodes.Count ==
                    candidate.Fusion.Nodes[^1] - candidate.Fusion.Nodes[0] + 1)
                ? SolveIntervals(component) : Search(component, budget, null));
        }
        var chosen = relaxed;
        var plan = Contract(graph, dependencies, chosen.Select(candidate => candidate.Fusion).ToArray());
        if (plan is null)
        {
            chosen = Search(candidates, budget, selection =>
                Contract(graph, dependencies, selection.Select(candidate => candidate.Fusion).ToArray()) is not null);
            plan = Contract(graph, dependencies, chosen.Select(candidate => candidate.Fusion).ToArray());
        }
        if (chosen.Count == 0)
            return Retain("No profitable legal plan found within the search budget; T0 retained.",
                matches.Count, candidates.Count, budget.Used);
        if (plan is null) throw new InvalidOperationException("T1 search produced an illegal contraction.");
        var selected = chosen.ToDictionary(candidate => candidate.Fusion.Nodes[^1], candidate => candidate.Fusion);
        var implementations = new Dictionary<ExecutionNodeId, InstructionOptimizationCapability>();
        var removed = chosen.SelectMany(candidate => candidate.Fusion.Intermediates).ToHashSet();
        var nodes = plan.Order.Select(index =>
        {
            var node = graph.Nodes[index];
            var predecessors = plan.Dependencies[index].Select(dependency => graph.Nodes[dependency].Id).ToArray();
            if (!selected.TryGetValue(index, out var candidate)) return node with { Dependencies = predecessors };
            implementations.Add(node.Id, candidate.Capability);
            return node with
            {
                Operation = new("t1." + candidate.Capability.Name),
                Dependencies = predecessors,
                Resources = candidate.Inputs.Select((input, port) =>
                        new NodeResourceBinding(candidate.Capability.Definition.Inputs[port], input, GraphResourceAccess.Read))
                    .Append(new("output", candidate.Output, GraphResourceAccess.Write)).ToArray(),
                Source = new(candidate.Nodes.SelectMany(source => graph.Nodes[source].Source.LogicalNodes).ToArray(), null),
                InternalResources = [],
                FirstWriteResources = [],
            };
        }).ToArray();
        var optimized = new ExecutionGraph(graph.Identity, graph.Model, graph.Resources.Where(resource => !removed.Contains(resource.Id)),
            graph.Regions, nodes, graph.Inputs, graph.Outputs, graph.GraphState);
        return new(optimized, implementations,
            new(budget.Exhausted ? TierOneSelectionStatus.BestFoundWithinBudget : TierOneSelectionStatus.ModelOptimal,
                model, budget.Exhausted ? "Legal profitable incumbent; search budget exhausted, optimality not proved." :
                    "Optimal in the registered candidate space under the stated additive cost model, not hardware-global optimality.",
                matches.Count, candidates.Count, chosen.Count, budget.Used, chosen.Sum(candidate => candidate.Saving),
                graph.Nodes.Count, nodes.Length));
    }

    private static bool Overlap(CostedCandidate left, CostedCandidate right) =>
        left.Fusion.Nodes.Intersect(right.Fusion.Nodes).Any();

    private static List<CostedCandidate> SolveIntervals(List<CostedCandidate> component)
    {
        var ordered = component.OrderBy(candidate => candidate.Fusion.Nodes[^1]).ThenBy(candidate => candidate.Fusion.Nodes[0])
            .ThenBy(candidate => candidate.Fusion.Capability.Name, StringComparer.Ordinal).ToArray();
        var plans = new List<CostedCandidate>[ordered.Length + 1];
        var values = new double[ordered.Length + 1];
        plans[0] = [];
        for (var index = 0; index < ordered.Length; index++)
        {
            var previous = index - 1;
            while (previous >= 0 && ordered[previous].Fusion.Nodes[^1] >= ordered[index].Fusion.Nodes[0]) previous--;
            var value = values[previous + 1] + ordered[index].Saving;
            if (value > values[index])
            {
                values[index + 1] = value;
                plans[index + 1] = [.. plans[previous + 1], ordered[index]];
            }
            else
            {
                values[index + 1] = values[index];
                plans[index + 1] = plans[index];
            }
        }
        return plans[^1];
    }

    private static List<CostedCandidate> Search(List<CostedCandidate> candidates, SearchBudget budget,
        Func<IReadOnlyList<CostedCandidate>, bool>? legal)
    {
        var ordered = candidates.OrderByDescending(candidate => candidate.Saving)
            .ThenBy(candidate => candidate.Fusion.Nodes[0]).ToArray();
        var best = new List<CostedCandidate>();
        foreach (var candidate in ordered)
        {
            if (best.Any(selected => Overlap(selected, candidate))) continue;
            var next = best.Append(candidate).ToArray();
            if (legal is null || legal(next)) best.Add(candidate);
        }
        var bestValue = best.Sum(candidate => candidate.Saving);
        var selected = new List<CostedCandidate>();
        var suffix = new double[ordered.Length + 1];
        for (var index = ordered.Length - 1; index >= 0; index--) suffix[index] = suffix[index + 1] + ordered[index].Saving;
        Visit(0, 0);
        return best;

        void Visit(int index, double value)
        {
            if (!budget.Take()) return;
            if (value > bestValue && (legal is null || legal(selected)))
            {
                bestValue = value;
                best = selected.ToList();
            }
            if (index == ordered.Length || value + suffix[index] <= bestValue) return;
            var candidate = ordered[index];
            if (!selected.Any(other => Overlap(other, candidate)))
            {
                selected.Add(candidate);
                Visit(index + 1, value + candidate.Saving);
                selected.RemoveAt(selected.Count - 1);
            }
            if (!budget.Exhausted) Visit(index + 1, value);
        }
    }

    private static HashSet<int>[] Dependencies(ExecutionGraph graph)
    {
        var indices = graph.Nodes.Select((node, index) => (node.Id, index)).ToDictionary(pair => pair.Id, pair => pair.index);
        var dependencies = graph.Nodes.Select(node => node.Dependencies.Select(dependency => indices[dependency]).ToHashSet()).ToArray();
        var writers = new Dictionary<ResourceId, int>();
        var readers = new Dictionary<ResourceId, HashSet<int>>();
        for (var index = 0; index < graph.Nodes.Count; index++)
        {
            foreach (var binding in graph.Nodes[index].Resources.Where(binding => binding.Access != GraphResourceAccess.Write))
            {
                if (writers.TryGetValue(binding.Resource, out var writer) && writer != index) dependencies[index].Add(writer);
                if (!readers.TryGetValue(binding.Resource, out var uses)) readers.Add(binding.Resource, uses = []);
                uses.Add(index);
            }
            foreach (var binding in graph.Nodes[index].Resources.Where(binding => binding.Access != GraphResourceAccess.Read))
            {
                if (writers.TryGetValue(binding.Resource, out var writer) && writer != index) dependencies[index].Add(writer);
                if (readers.TryGetValue(binding.Resource, out var uses))
                {
                    dependencies[index].UnionWith(uses.Where(reader => reader != index));
                    uses.Clear();
                }
                writers[binding.Resource] = index;
            }
        }
        return dependencies;
    }

    private static ContractedPlan? Contract(ExecutionGraph graph, HashSet<int>[] dependencies,
        IReadOnlyList<TierOneFusionCandidate> candidates)
    {
        var owners = Enumerable.Range(0, graph.Nodes.Count).ToArray();
        var occupied = new HashSet<int>();
        foreach (var candidate in candidates)
            foreach (var index in candidate.Nodes)
            {
                if (!occupied.Add(index)) return null;
                owners[index] = candidate.Nodes[^1];
            }
        var rewritten = Enumerable.Range(0, graph.Nodes.Count).Select(_ => new HashSet<int>()).ToArray();
        var successors = Enumerable.Range(0, graph.Nodes.Count).Select(_ => new List<int>()).ToArray();
        for (var index = 0; index < graph.Nodes.Count; index++)
            foreach (var dependency in dependencies[index])
                if (owners[index] != owners[dependency]) rewritten[owners[index]].Add(owners[dependency]);
        foreach (var index in owners.Distinct())
            foreach (var dependency in rewritten[index]) successors[dependency].Add(index);
        var pending = rewritten.Select(set => set.Count).ToArray();
        var ready = new PriorityQueue<int, int>();
        foreach (var index in owners.Distinct().Where(index => pending[index] == 0)) ready.Enqueue(index, index);
        var order = new List<int>();
        while (ready.TryDequeue(out var index, out _))
        {
            order.Add(index);
            foreach (var successor in successors[index])
                if (--pending[successor] == 0) ready.Enqueue(successor, successor);
        }
        return order.Count == owners.Distinct().Count() ? new(order.ToArray(), rewritten) : null;
    }

    private sealed record CostedCandidate(TierOneFusionCandidate Fusion, double Saving);
    private sealed record ContractedPlan(int[] Order, HashSet<int>[] Dependencies);
    private sealed class SearchBudget(int maximum)
    {
        public int Used { get; private set; }
        public bool Exhausted { get; private set; }
        public bool Take()
        {
            if (Used >= maximum) { Exhausted = true; return false; }
            Used++;
            return true;
        }
    }
}
