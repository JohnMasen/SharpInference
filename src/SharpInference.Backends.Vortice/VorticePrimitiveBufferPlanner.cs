using SharpInference.Graphs;

namespace SharpInference.Backends.Vortice;

internal sealed class VorticePrimitiveBufferPlanner
{
    private VorticePrimitiveBufferPlanner(
        IReadOnlyDictionary<ResourceId, int> slots,
        IReadOnlyList<ulong> capacities,
        IReadOnlySet<ExecutionNodeId> elidedSteps)
    {
        Slots = slots;
        Capacities = capacities;
        ElidedSteps = elidedSteps;
    }

    public IReadOnlyDictionary<ResourceId, int> Slots { get; }
    public IReadOnlyList<ulong> Capacities { get; }
    public IReadOnlySet<ExecutionNodeId> ElidedSteps { get; }

    public static VorticePrimitiveBufferPlanner Create(VorticePrimitiveGraphPlan plan)
    {
        var graph = plan.Graph;
        var endpoints = graph.Inputs.Concat(graph.Outputs).ToHashSet();
        var local = graph.Resources.Where(resource =>
            resource.Scope == GraphResourceScope.Local && !endpoints.Contains(resource.Id))
            .ToDictionary(resource => resource.Id);
        var intervals = new Dictionary<ResourceId, (int First, int Last)>();
        for (var index = 0; index < plan.Steps.Count; index++)
        {
            var step = plan.Steps[index];
            var accessed = step.FusedInputs is { } fused
                ? fused.Append(step.Output)
                : new[] { step.Input0, step.Input1 ?? default, step.Output };
            foreach (var id in accessed)
            {
                if (!local.ContainsKey(id)) continue;
                if (intervals.TryGetValue(id, out var current))
                    intervals[id] = (current.First, index);
                else
                    intervals[id] = (index, index);
            }
        }
        var writeCounts = plan.Steps.GroupBy(step => step.Output)
            .ToDictionary(group => group.Key, group => group.Count());
        var aliases = new Dictionary<ResourceId, (ResourceId Source, ExecutionNodeId Node)>();
        var elidedSteps = new HashSet<ExecutionNodeId>();
        for (var index = 0; index < plan.Steps.Count; index++)
        {
            var step = plan.Steps[index];
            if (step.Kernel != "PortableCopy" ||
                !local.TryGetValue(step.Input0, out var source) ||
                !local.TryGetValue(step.Output, out var target) ||
                source.Tensor.ElementType != target.Tensor.ElementType ||
                source.Tensor.Dimensions.Aggregate(1L, (size, dim) => checked(size * dim)) !=
                    target.Tensor.Dimensions.Aggregate(1L, (size, dim) => checked(size * dim)) ||
                !writeCounts.TryGetValue(step.Input0, out var sourceWrites) || sourceWrites != 1 ||
                writeCounts[step.Output] != 1 ||
                intervals[step.Input0].Last != index || intervals[step.Output].First != index)
                continue;
            aliases.Add(step.Output, (step.Input0, step.Node));
        }
        var assignments = new Dictionary<ResourceId, int>();
        var capacities = new List<ulong>();
        var types = new List<GraphElementType>();
        var lastUse = new List<int>();
        foreach (var (id, range) in intervals.OrderBy(pair => pair.Value.First))
        {
            var resource = local[id];
            var size = checked((ulong)resource.Tensor.Dimensions.Aggregate(
                1L, (count, dimension) => checked(count * dimension)) *
                (uint)(resource.Tensor.ElementType == GraphElementType.Float16 ? sizeof(ushort) : sizeof(float)));
            if (aliases.TryGetValue(id, out var alias) &&
                assignments.TryGetValue(alias.Source, out var sourceSlot) &&
                lastUse[sourceSlot] == range.First &&
                capacities[sourceSlot] >= size)
            {
                assignments.Add(id, sourceSlot);
                lastUse[sourceSlot] = range.Last;
                elidedSteps.Add(alias.Node);
                continue;
            }
            var best = -1;
            for (var index = 0; index < capacities.Count; index++)
            {
                if (lastUse[index] >= range.First || types[index] != resource.Tensor.ElementType ||
                    capacities[index] < size ||
                    best >= 0 && capacities[index] >= capacities[best])
                    continue;
                best = index;
            }
            if (best < 0)
            {
                best = capacities.Count;
                capacities.Add(size);
                types.Add(resource.Tensor.ElementType);
                lastUse.Add(range.Last);
            }
            else
                lastUse[best] = range.Last;
            assignments.Add(id, best);
        }
        return new VorticePrimitiveBufferPlanner(assignments, capacities, elidedSteps);
    }
}
