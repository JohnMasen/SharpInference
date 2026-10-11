namespace SharpInference.Graphs;

/// <summary>Recognizes tensor catalogs and supplies architecture-specific metadata, graphs, validation, and lowering.</summary>
public interface IModelGraphModule : ILogicalGraphProvider
{
    /// <summary>Determines whether this module recognizes the supplied model tensor catalog.</summary>
    /// <param name="tensors">The model tensor catalog.</param>
    bool CanLoad(IModelTensorCatalog tensors);
    /// <summary>Reads architecture-specific model metadata from a recognized tensor catalog.</summary>
    /// <param name="tensors">The model tensor catalog.</param>
    ModelMetadata ReadMetadata(IModelTensorCatalog tensors);
    /// <summary>Gets the optional architecture-specific operation validator.</summary>
    IGraphOperationValidator? OperationValidator => null;
    /// <summary>Gets the optional architecture-specific operation lowerer.</summary>
    IGraphOperationLowerer? OperationLowerer => null;
    /// <summary>Gets the optional architecture-specific legacy model signature reader.</summary>
    IGraphModelSignatureReader? LegacySignatureReader => null;
}

/// <summary>Contains the selected architecture module, model metadata, and validated logical graph.</summary>
/// <param name="Module">The architecture-specific graph module.</param>
/// <param name="Metadata">The model metadata.</param>
/// <param name="Graph">The graph supplying operations and resource binding metadata.</param>
public sealed record ModelGraphSelection(IModelGraphModule Module, ModelMetadata Metadata, LogicalGraph Graph);

/// <summary>Resolves explicitly registered modules without assuming a model family or backend.</summary>
public sealed class ModelGraphModuleRegistry
{
    private readonly Dictionary<string, IModelGraphModule> modules = new(StringComparer.Ordinal);

    /// <summary>Registers a module under its architecture identifier, rejecting duplicate registrations.</summary>
    /// <param name="module">The architecture-specific graph module.</param>
    public void Register(IModelGraphModule module)
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentException.ThrowIfNullOrWhiteSpace(module.ArchitectureId);
        if (!modules.TryAdd(module.ArchitectureId, module))
            throw new InvalidOperationException($"Model module '{module.ArchitectureId}' is already registered.");
    }

    /// <summary>Returns the registered module for an architecture, rejecting unknown identifiers.</summary>
    /// <param name="architectureId">The model architecture identifier.</param>
    public IModelGraphModule GetRequired(string architectureId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(architectureId);
        return modules.TryGetValue(architectureId, out var module) ? module :
            throw new NotSupportedException($"No model module is registered for '{architectureId}'.");
    }

    /// <summary>Resolves the single registered module recognizing the tensor catalog, rejecting missing or ambiguous matches.</summary>
    /// <param name="tensors">The model tensor catalog.</param>
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

    /// <summary>Resolves a module, builds its graph, and validates model metadata and operation contracts.</summary>
    /// <param name="tensors">The model tensor catalog.</param>
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
