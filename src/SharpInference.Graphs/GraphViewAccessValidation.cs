namespace SharpInference.Graphs;

internal sealed record GraphViewAccessNode(string Id, IReadOnlyList<NodeResourceBinding> Bindings,
    IReadOnlyList<string> Dependencies, IReadOnlyList<ResourceId> FirstWrites);

internal static class GraphViewAccessValidation
{
    internal static void Validate(IReadOnlyList<GraphResource> resources, IReadOnlyList<ResourceId> inputs,
        IReadOnlyList<ResourceId> outputs, IEnumerable<GraphViewAccessNode> nodes)
    {
        var values = nodes.ToArray();
        if (!values.Any(node => node.Bindings.Any(binding => binding.View is not null))) return;
        var byId = values.ToDictionary(node => node.Id, StringComparer.Ordinal);
        var parents = resources.ToDictionary(resource => resource.Id);
        var ancestors = values.ToDictionary(node => node.Id, node => Ancestors(node), StringComparer.Ordinal);
        var initialized = inputs.Concat(resources.Where(resource => resource.Scope == GraphResourceScope.Global ||
            resource.Kind == GraphResourceKind.SessionState).Select(resource => resource.Id)).ToHashSet();
        var accesses = values.SelectMany(node => node.Bindings.Select(binding => (Node: node, Binding: binding)))
            .GroupBy(item => item.Binding.Resource).ToDictionary(group => group.Key, group => group.ToArray());
        foreach (var (resourceId, uses) in accesses)
        {
            var parent = parents[resourceId];
            for (var i = 0; i < uses.Length; i++)
            {
                var access = uses[i];
                var range = Range(access.Binding, parent);
                for (var j = i + 1; j < uses.Length; j++)
                {
                    var other = uses[j];
                    if (access.Node.Id == other.Node.Id ||
                        access.Binding.Access == GraphResourceAccess.Read && other.Binding.Access == GraphResourceAccess.Read ||
                        ancestors[access.Node.Id].Contains(other.Node.Id) || ancestors[other.Node.Id].Contains(access.Node.Id))
                        continue;
                    var otherRange = Range(other.Binding, parent);
                    if (range.Start < otherRange.End && otherRange.Start < range.End)
                        throw new InvalidDataException($"Resource '{resourceId}' has unordered overlapping view accesses at '{access.Node.Id}' and '{other.Node.Id}'.");
                }
                if (access.Binding.Access == GraphResourceAccess.Write || initialized.Contains(resourceId) ||
                    access.Binding.Access == GraphResourceAccess.ReadWrite && access.Node.FirstWrites.Contains(resourceId))
                    continue;
                var writes = uses.Where(candidate => candidate.Binding.Access != GraphResourceAccess.Read &&
                    ancestors[access.Node.Id].Contains(candidate.Node.Id)).Select(candidate => Range(candidate.Binding, parent));
                if (!Covered(range, writes))
                    throw new InvalidDataException($"Resource '{resourceId}' view at '{access.Node.Id}' reads an uninitialized byte range.");
            }
        }
        foreach (var output in outputs.Where(output => !initialized.Contains(output)))
        {
            if (!accesses.TryGetValue(output, out var uses) ||
                !Covered((0, GraphTensorView.ByteLengthOf(parents[output].Tensor)),
                    uses.Where(access => access.Binding.Access != GraphResourceAccess.Read)
                        .Select(access => Range(access.Binding, parents[output]))))
                throw new InvalidDataException($"Output '{output}' has an uninitialized byte range.");
        }

        HashSet<string> Ancestors(GraphViewAccessNode node)
        {
            var result = new HashSet<string>(StringComparer.Ordinal);
            var pending = new Stack<string>(node.Dependencies);
            while (pending.TryPop(out var id))
                if (result.Add(id))
                    foreach (var dependency in byId[id].Dependencies) pending.Push(dependency);
            return result;
        }
    }

    private static (ulong Start, ulong End) Range(NodeResourceBinding binding, GraphResource parent) =>
        binding.View is { } view ? (view.ByteOffset, checked(view.ByteOffset + view.ByteLength)) :
        (0, GraphTensorView.ByteLengthOf(parent.Tensor));

    private static bool Covered((ulong Start, ulong End) requested, IEnumerable<(ulong Start, ulong End)> ranges)
    {
        var cursor = requested.Start;
        foreach (var range in ranges.OrderBy(range => range.Start))
        {
            if (range.End <= cursor) continue;
            if (range.Start > cursor) return false;
            cursor = Math.Max(cursor, range.End);
            if (cursor >= requested.End) return true;
        }
        return false;
    }
}
