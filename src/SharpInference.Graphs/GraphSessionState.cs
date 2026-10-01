namespace SharpInference.Graphs;

public sealed record GraphStateValue(string Name, IReadOnlyList<int> Dimensions, float[] Values);

public interface IProcessorStateExecutor
{
    IReadOnlyList<GraphStateValue> ReadState(GraphState graph);
    void WriteState(GraphState graph, IReadOnlyList<GraphStateValue> values);
}

public static class GraphSessionStateAccess
{
    public static IReadOnlyList<GraphStateValue> Read(
        GraphState graph, IReadOnlyList<GraphResource> resources, INamedRwkvState state)
    {
        var views = Match(graph, resources, state);
        return views.Select(view =>
            new GraphStateValue(view.Name, view.Dimensions.ToArray(), (float[])view.Values.Clone())).ToArray();
    }

    public static void Write(
        GraphState graph, IReadOnlyList<GraphResource> resources, INamedRwkvState state,
        IReadOnlyList<GraphStateValue> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        var views = Match(graph, resources, state);
        if (values.Count != views.Length)
            throw new InvalidDataException("The state tensor count does not match GraphState.");
        for (var index = 0; index < views.Length; index++)
        {
            var view = views[index];
            var value = values[index];
            if (value is null || value.Name != view.Name ||
                !value.Dimensions.SequenceEqual(view.Dimensions) ||
                value.Values.Length != view.Values.Length)
                throw new InvalidDataException($"State tensor '{view.Name}' does not match GraphState.");
        }
        for (var index = 0; index < views.Length; index++)
            values[index].Values.CopyTo(views[index].Values, 0);
        state.CommitViews();
    }

    private static RwkvStateView[] Match(
        GraphState graph, IReadOnlyList<GraphResource> resources, INamedRwkvState state)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(resources);
        ArgumentNullException.ThrowIfNull(state);
        var views = state.Views.ToDictionary(view => view.Name, StringComparer.Ordinal);
        var descriptions = resources.ToDictionary(resource => resource.Id);
        if (graph.Slots.Count != views.Count ||
            resources.Count(resource => resource.Kind == GraphResourceKind.SessionState) != graph.Slots.Count)
            throw new InvalidDataException("GraphState must cover all session-state tensors.");
        var matched = new RwkvStateView[graph.Slots.Count];
        for (var index = 0; index < matched.Length; index++)
        {
            var slot = graph.Slots[index];
            if (!descriptions.TryGetValue(slot.Resource, out var resource) ||
                resource.Kind != GraphResourceKind.SessionState ||
                resource.Tensor.ElementType != GraphElementType.Float32 ||
                resource.Tensor.Layout != "dense" ||
                !views.TryGetValue(slot.Name, out var view) ||
                !resource.Tensor.Dimensions.SequenceEqual(view.Dimensions) ||
                view.Values.Length != resource.Tensor.Dimensions.Aggregate(
                    1L, (count, size) => checked(count * size)))
                throw new InvalidDataException($"State tensor '{slot.Name}' does not match GraphState.");
            matched[index] = view;
        }
        return matched;
    }
}
