using SharpInference.Graphs;

namespace SharpInference.Runtime;

/// <summary>Adapts a compiled portable VM graph to the model architecture contract.</summary>
public sealed class PortableGraphArchitecture : IModelArchitecture, IModelWeightOwnershipPolicy
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

    /// <inheritdoc/>
    public string Id => plan.BindingGraph.Identity.ArchitectureId;
    /// <summary>Gets a value indicating that this adapter does not require a separate CPU copy of model weights.</summary>
    public bool RequiresCpuWeightCopy => false;

    /// <summary>Checks model dimensions and weight tensor descriptors against the binding graph.</summary>
    /// <param name="tensors">The tensor catalog supplying model weights and their descriptors.</param>
    public bool CanLoad(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        return plan.BindingGraph.Resources.Where(resource => resource.Kind == GraphResourceKind.Weight)
                   .All(resource => tensors.TryGet(resource.BindingKey
                       ?? throw new InvalidDataException($"Weight resource '{resource.Id}' has no binding key."), out var tensor) &&
                       resource.Tensor.Dimensions.SequenceEqual(tensor.Dimensions) &&
                       resource.Tensor.ElementType == (tensor.DataType switch
                       {
                           TensorDataType.Float16 => GraphElementType.Float16,
                           TensorDataType.Float32 => GraphElementType.Float32,
                           _ => throw new InvalidDataException($"Unsupported tensor type for '{tensor.Name}'."),
                       }));
    }

    /// <summary>Binds a compatible tensor catalog to a portable graph model and prepares its weights.</summary>
    /// <param name="tensors">The tensor catalog supplying model weights and their descriptors.</param>
    public IModel Bind(IModelTensorCatalog tensors)
    {
        if (!CanLoad(tensors))
            throw new InvalidDataException("The graph weights or model shape do not match the catalog.");
        var signature = plan.BindingGraph.Model;
        var model = new PortableGraphModel(
            new ModelMetadata(Id, signature.Dimensions.ToDictionary(
                dimension => dimension.Key, dimension => (long)dimension.Value, StringComparer.Ordinal),
                signature.Attributes),
            tensors, plan.BindingGraph);
        backend.PrepareModelWeights(model);
        return model;
    }

    /// <summary>Creates state for a model bound to this adapter's graph.</summary>
    /// <param name="model">The model bound to the prepared graph and backend.</param>
    public IModelState CreateState(IModel model)
    {
        if (model is not PortableGraphModel portable || !ReferenceEquals(portable.Graph, plan.BindingGraph))
            throw new ArgumentException("The model does not belong to this graph.", nameof(model));
        return backend.SupportsTokenSessions
            ? new PortableGraphState(plan.BindingGraph)
            : new GraphTensorState(plan.BindingGraph);
    }

    /// <summary>Executes one token through a temporary session and writes its logits.</summary>
    /// <param name="model">The model bound to the prepared graph and backend.</param>
    /// <param name="token">The input token identifier to process.</param>
    /// <param name="state">The compatible model state updated by the temporary VM session.</param>
    /// <param name="logits">The destination span receiving output logits for the token.</param>
    public void ForwardToken(IModel model, int token, IModelState state, Span<float> logits)
    {
        using var session = backend.CreateSessionExecutor(model, state, plan);
        session.ForwardToken(token, logits);
        _ = ((PortableGraphState)state).Views;
    }
}
