namespace SharpInference.Graphs;

public interface IModelGraphModule : ILogicalGraphProvider
{
    bool CanLoad(IModelTensorCatalog tensors);
    ModelMetadata ReadMetadata(IModelTensorCatalog tensors);
    IGraphOperationValidator? OperationValidator => null;
    IGraphOperationLowerer? OperationLowerer => null;
    IGraphModelSignatureReader? LegacySignatureReader => null;
}

public sealed record ModelGraphSelection(IModelGraphModule Module, ModelMetadata Metadata, LogicalGraph Graph);

/// <summary>Resolves explicitly registered modules without assuming a model family or backend.</summary>
public sealed class ModelGraphModuleRegistry
{
    private readonly Dictionary<string, IModelGraphModule> modules = new(StringComparer.Ordinal);

    public void Register(IModelGraphModule module)
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentException.ThrowIfNullOrWhiteSpace(module.ArchitectureId);
        if (!modules.TryAdd(module.ArchitectureId, module))
            throw new InvalidOperationException($"Model module '{module.ArchitectureId}' is already registered.");
    }

    public IModelGraphModule GetRequired(string architectureId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(architectureId);
        return modules.TryGetValue(architectureId, out var module) ? module :
            throw new NotSupportedException($"No model module is registered for '{architectureId}'.");
    }

    public IModelGraphModule Resolve(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var matches = modules.Values.Where(module => module.CanLoad(tensors)).ToArray();
        return matches.Length switch
        {
            1 => matches[0],
            0 => throw new NotSupportedException("No registered model module recognizes the tensor catalog."),
            _ => throw new InvalidDataException($"Multiple model modules recognize the catalog: {string.Join(", ", matches.Select(module => module.ArchitectureId).Order(StringComparer.Ordinal))}."),
        };
    }

    public ModelGraphSelection Build(IModelTensorCatalog tensors)
    {
        var module = Resolve(tensors);
        var metadata = module.ReadMetadata(tensors) ?? throw new InvalidDataException("The model module returned no metadata.");
        var graph = module.Build(tensors) ?? throw new InvalidDataException("The model module returned no graph.");
        if (metadata.ArchitectureId != module.ArchitectureId || graph.Identity.ArchitectureId != module.ArchitectureId)
            throw new InvalidDataException("The model module, metadata and logical graph have different architecture identifiers.");
        foreach (var dimension in graph.Model.Dimensions)
            if (metadata.Dimensions.TryGetValue(dimension.Key, out var value) && value != dimension.Value)
                throw new InvalidDataException($"Graph dimension '{dimension.Key}' differs from the model metadata.");
        GraphValidator.Validate(graph, module.OperationValidator);
        return new(module, metadata, graph);
    }
}
