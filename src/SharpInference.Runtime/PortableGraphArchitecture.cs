using SharpInference.Graphs;

namespace SharpInference.Runtime;

public sealed class PortableGraphArchitecture : IRwkvArchitecture, IModelWeightOwnershipPolicy
{
    private readonly IExecutionGraphBackend backend;
    private readonly IBackendExecutablePlan plan;

    public PortableGraphArchitecture(IExecutionGraphBackend backend, IBackendExecutablePlan plan)
    {
        this.backend = backend ?? throw new ArgumentNullException(nameof(backend));
        this.plan = plan ?? throw new ArgumentNullException(nameof(plan));
    }

    public string Id => plan.Graph.Identity.ArchitectureId;
    public bool RequiresCpuWeightCopy =>
        backend is not IGraphModelWeightBackend { RequiresCpuWeightCopy: false };

    public bool CanLoad(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var model = plan.Graph.Model;
        return model.VocabularySize == tensors.VocabularySize &&
               model.EmbeddingSize == tensors.EmbeddingSize &&
               model.LayerCount == tensors.LayerCount &&
               plan.Graph.Resources.Where(resource => resource.Kind == GraphResourceKind.Weight)
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

    public IRwkvModel Bind(IModelTensorCatalog tensors)
    {
        if (!CanLoad(tensors))
            throw new InvalidDataException("The graph weights or model shape do not match the catalog.");
        var signature = plan.Graph.Model;
        var model = new PortableGraphModel(
            new RwkvModelMetadata(signature.VocabularySize, signature.EmbeddingSize,
                signature.LayerCount, signature.HeadCount, signature.HeadSize, Id),
            tensors, plan.Graph);
        if (backend is IGraphModelWeightBackend weightBackend)
            weightBackend.PrepareModelWeights(model);
        return model;
    }

    public IRwkvState CreateState(IRwkvModel model)
    {
        if (model is not PortableGraphModel portable || !ReferenceEquals(portable.Graph, plan.Graph))
            throw new ArgumentException("The model does not belong to this graph.", nameof(model));
        return new PortableGraphState(plan.Graph);
    }

    public void ForwardToken(IRwkvModel model, int token, IRwkvState state, Span<float> logits)
    {
        using var session = backend.CreateSessionExecutor(model, state, plan);
        session.ForwardToken(token, logits);
    }
}
