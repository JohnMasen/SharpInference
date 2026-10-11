namespace SharpInference.Graphs;

/// <summary>Contains the dimensions and FP32 values of a named graph state entry.</summary>
/// <param name="Name">The state entry name.</param>
/// <param name="Dimensions">The tensor dimension sizes in axis order.</param>
/// <param name="Values">The state tensor values.</param>
public sealed record GraphStateValue(string Name, IReadOnlyList<int> Dimensions, float[] Values);

/// <summary>Reads and restores named graph state values for a processor session.</summary>
public interface IProcessorStateExecutor
{
    /// <summary>Reads the named graph state entries as dimensioned FP32 values.</summary>
    /// <param name="graph">The ordered graph state entries and their compatibility schema.</param>
    IReadOnlyList<GraphStateValue> ReadState(GraphState graph);
    /// <summary>Restores named graph state entries from dimensioned FP32 values.</summary>
    /// <param name="graph">The ordered graph state entries and their compatibility schema.</param>
    /// <param name="values">The state tensor values.</param>
    void WriteState(GraphState graph, IReadOnlyList<GraphStateValue> values);
}

/// <summary>Transfers named FP32 state between graph bindings and a model state object.</summary>
public static class GraphSessionStateAccess
{
    /// <summary>Reads named graph state entries as dimensioned FP32 values.</summary>
    /// <param name="graph">The ordered graph state entries and their compatibility schema.</param>
    /// <param name="resources">The graph resource descriptors.</param>
    /// <param name="state">The model state exposing named FP32 host views.</param>
    public static IReadOnlyList<GraphStateValue> Read(
        GraphState graph, IReadOnlyList<GraphResource> resources, INamedFloat32ModelState state)
    {
        var views = Match(graph, resources, state);
        return views.Select(view =>
            new GraphStateValue(view.Name, view.Dimensions.ToArray(), (float[])view.Values.Clone())).ToArray();
    }

    /// <summary>Validates and restores named graph state entries from dimensioned FP32 values.</summary>
    /// <param name="graph">The ordered graph state entries and their compatibility schema.</param>
    /// <param name="resources">The graph resource descriptors.</param>
    /// <param name="state">The model state exposing named FP32 host views.</param>
    /// <param name="values">The state tensor values.</param>
    public static void Write(
        GraphState graph, IReadOnlyList<GraphResource> resources, INamedFloat32ModelState state,
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

    private static Float32StateView[] Match(
        GraphState graph, IReadOnlyList<GraphResource> resources, INamedFloat32ModelState state)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(resources);
        ArgumentNullException.ThrowIfNull(state);
        var views = state.Views.ToDictionary(view => view.Name, StringComparer.Ordinal);
        var descriptions = resources.ToDictionary(resource => resource.Id);
        if (graph.Slots.Count != views.Count ||
            resources.Count(resource => resource.Kind == GraphResourceKind.SessionState) != graph.Slots.Count)
            throw new InvalidDataException("GraphState must cover all session-state tensors.");
        var matched = new Float32StateView[graph.Slots.Count];
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
