using SharpInference.Graphs;

namespace SharpInference.Runtime;

/// <summary>Adapts a compiled portable VM graph to the RWKV architecture contract.</summary>
public sealed class PortableGraphArchitecture : IRwkvArchitecture, IModelWeightOwnershipPolicy
{
    private readonly VmGraphBackend backend;
    private readonly VmCompiledPlan plan;

    /// <summary>Creates an architecture adapter for a backend and compiled graph plan.</summary>
    /// <param name="backend">The VM graph backend used for execution and weight preparation.</param>
    /// <param name="plan">The compiled plan whose binding graph defines the model contract.</param>
    public PortableGraphArchitecture(VmGraphBackend backend, VmCompiledPlan plan)
    {
        this.backend = backend ?? throw new ArgumentNullException(nameof(backend));
        this.plan = plan ?? throw new ArgumentNullException(nameof(plan));
    }

    public string Id => plan.BindingGraph.Identity.ArchitectureId;
    public bool RequiresCpuWeightCopy => false;

    /// <summary>Checks model dimensions and weight tensor descriptors against the binding graph.</summary>
    public bool CanLoad(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var model = plan.BindingGraph.Model;
        return model.VocabularySize == tensors.VocabularySize &&
               model.EmbeddingSize == tensors.EmbeddingSize &&
               model.LayerCount == tensors.LayerCount &&
               plan.BindingGraph.Resources.Where(resource => resource.Kind == GraphResourceKind.Weight)
                   .All(resource => tensors.TryGet(resource.BindingKey
                       ?? throw new InvalidDataException($"Weight resource '{resource.Id}' has no binding key."), out var tensor) &&
                       resource.Tensor.Dimensions.SequenceEqual(tensor.Dimensions) &&
                       resource.Tensor.ElementType == (tensor.DataType switch
                       {
                           RwkvTensorDataType.Float16 => GraphElementType.Float16,
                           RwkvTensorDataType.Float32 => GraphElementType.Float32,
                           _ => throw new InvalidDataException($"Unsupported tensor type for '{tensor.Name}'."),
                       }));
    }

    /// <summary>Binds a compatible tensor catalog to a portable graph model and prepares its weights.</summary>
    public IRwkvModel Bind(IModelTensorCatalog tensors)
    {
        if (!CanLoad(tensors))
            throw new InvalidDataException("The graph weights or model shape do not match the catalog.");
        var signature = plan.BindingGraph.Model;
        var model = new PortableGraphModel(
            new RwkvModelMetadata(signature.VocabularySize, signature.EmbeddingSize,
                signature.LayerCount, signature.HeadCount, signature.HeadSize, Id),
            tensors, plan.BindingGraph);
        backend.PrepareModelWeights(model);
        return model;
    }

    /// <summary>Creates state for a model bound to this adapter's graph.</summary>
    public IRwkvState CreateState(IRwkvModel model)
    {
        if (model is not PortableGraphModel portable || !ReferenceEquals(portable.Graph, plan.BindingGraph))
            throw new ArgumentException("The model does not belong to this graph.", nameof(model));
        return new PortableGraphState(plan.BindingGraph);
    }

    /// <summary>Executes one token through a temporary session and writes its logits.</summary>
    public void ForwardToken(IRwkvModel model, int token, IRwkvState state, Span<float> logits)
    {
        using var session = backend.CreateSessionExecutor(model, state, plan);
        session.ForwardToken(token, logits);
        _ = ((PortableGraphState)state).Views;
    }
}
