namespace SharpInference.Graphs;

/// <summary>Expands model-owned operators into a graph accepted by an explicitly selected execution target.</summary>
public interface IGraphOperationLowerer
{
    LogicalGraph Lower(LogicalGraph graph);
}

public static class GraphOperationLowering
{
    public static LogicalGraph Apply(LogicalGraph graph, IGraphOperationLowerer? lowerer)
    {
        ArgumentNullException.ThrowIfNull(graph);
        GraphValidator.Validate(graph);
        if (lowerer is null) return graph;
        var lowered = lowerer.Lower(graph) ?? throw new InvalidDataException("The operation lowerer returned no graph.");
        GraphValidator.Validate(lowered);
        if (lowered.Identity != graph.Identity || !lowered.Model.Equals(graph.Model) ||
            !lowered.Inputs.SequenceEqual(graph.Inputs) || !lowered.Outputs.SequenceEqual(graph.Outputs) ||
            !lowered.GraphState.Schema.Equals(graph.GraphState.Schema) ||
            !lowered.GraphState.Entries.SequenceEqual(graph.GraphState.Entries))
            throw new InvalidDataException("Operation lowering must preserve model identity, graph IO and state ABI.");
        var originals = graph.Resources.ToDictionary(resource => resource.Id);
        var rewritten = lowered.Resources.ToDictionary(resource => resource.Id);
        foreach (var resource in graph.Resources.Where(resource => resource.Scope != GraphResourceScope.Local ||
            graph.Inputs.Contains(resource.Id) || graph.Outputs.Contains(resource.Id)))
        {
            if (!rewritten.TryGetValue(resource.Id, out var replacement) ||
                replacement.Kind != resource.Kind || replacement.Scope != resource.Scope ||
                replacement.Lifetime != resource.Lifetime || replacement.BindingKey != resource.BindingKey ||
                replacement.Tensor.ElementType != resource.Tensor.ElementType ||
                replacement.Tensor.Layout != resource.Tensor.Layout ||
                !replacement.Tensor.Dimensions.SequenceEqual(resource.Tensor.Dimensions))
                throw new InvalidDataException($"Lowering changed externally bound resource '{resource.Id}'.");
        }
        if (lowered.Resources.Any(resource => !originals.ContainsKey(resource.Id) &&
            resource.Scope != GraphResourceScope.Local))
            throw new InvalidDataException("Lowering cannot introduce unbound global or session resources.");
        return lowered;
    }
}
