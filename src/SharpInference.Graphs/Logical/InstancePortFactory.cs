
namespace SharpInference.Graphs.Logical;

/// <summary>Recreates a child contract with caller-facing input handles and bound or generated outputs.</summary>
/// <param name="owner">The containing graph identity used for remapped inputs and generated outputs.</param>
/// <param name="call">The call identity used to prefix generated output identifiers and names.</param>
/// <param name="definition">The child definition whose public port kinds and tensor contracts must match.</param>
/// <param name="bindings">The required input and input/output bindings, indexed by ordinal port name.</param>
/// <param name="outputTargets">Explicit output and input/output targets, including their binding scopes.</param>
internal sealed class InstancePortFactory(
    GraphDefinitionId owner,
    GraphDefinitionNodeId call,
    LogicalGraphDefinition definition,
    IReadOnlyList<PendingBinding> bindings,
    IReadOnlyDictionary<ResourceId, GraphCallBoundResource> outputTargets) : IGraphPortFactory
{
    /// <summary>Input and input/output bindings keyed by child port name.</summary>
    private readonly IReadOnlyDictionary<string, PendingBinding> inputs =
        bindings.ToDictionary(binding => binding.Port, StringComparer.Ordinal);
    /// <summary>Parent-scoped declarations to add to the containing graph.</summary>
    private readonly Dictionary<ResourceId, GraphDefinitionResource> resources = [];
    /// <summary>Write bindings emitted only for pure output ports.</summary>
    private readonly List<PendingBinding> outputBindings = [];

    /// <summary>Gets the parent-scoped declarations collected while recreating the contract.</summary>
    internal IReadOnlyCollection<GraphDefinitionResource> Resources => resources.Values;
    /// <summary>Gets output writes; input/output read/write bindings already come from the input list.</summary>
    internal IReadOnlyList<PendingBinding> OutputBindings => outputBindings;

    /// <summary>Resolves an input and returns its bound declaration with the containing graph as owner.</summary>
    /// <typeparam name="T">The unchecked resource type marker.</typeparam>
    /// <param name="id">The child input port identifier.</param>
    /// <param name="tensor">The child port's exact tensor contract.</param>
    /// <returns>An input handle; only parent-scoped bindings register a containing-graph resource.</returns>
    /// <exception cref="InvalidOperationException">The child input is unknown or its tensor contract differs.</exception>
    /// <exception cref="KeyNotFoundException">No input binding exists for the requested port.</exception>
    public GraphInput<T> AddInput<T>(string id, TensorDescriptor tensor)
    {
        var definitionResource = Find(id, GraphDefinitionResourceKind.Input, tensor);
        var binding = inputs[id];
        var resource = binding.Resource with { Id = binding.Resource.Id, Owner = owner };
        if (binding.Owner.IsParent)
        {
            resources.TryAdd(resource.Id, resource);
        }

        return new GraphInput<T>(resource);
    }

    /// <summary>Uses an explicit output target or generates a parent-owned resource prefixed by the call identifier.</summary>
    /// <typeparam name="T">The unchecked resource type marker.</typeparam>
    /// <param name="id">The child output port identifier.</param>
    /// <param name="tensor">The child port's exact tensor contract.</param>
    /// <returns>The output handle for the selected or generated target.</returns>
    /// <remarks>Every request appends a write binding. Instance-scoped targets are not registered as parent resources.</remarks>
    /// <exception cref="InvalidOperationException">The child output is unknown or its tensor contract differs.</exception>
    public GraphOutput<T> AddOutput<T>(string id, TensorDescriptor tensor)
    {
        var definitionResource = Find(id, GraphDefinitionResourceKind.Output, tensor);
        var target = outputTargets.GetValueOrDefault(definitionResource.Id);
        var resource = target?.Resource ?? definitionResource with
        {
            Id = new ResourceId($"{call.Value}/{id}"),
            Name = $"{call.Value}/{definitionResource.Name}",
            Owner = owner,
        };
        if (target is null || target.Owner.IsParent)
        {
            resources.TryAdd(resource.Id, resource);
        }

        outputBindings.Add(new PendingBinding(
            id,
            resource,
            GraphResourceAccess.Write,
            target?.Owner ?? GraphBindingOwner.Parent));
        return new GraphOutput<T>(resource);
    }

    /// <summary>Resolves a required input/output target without creating an additional write binding.</summary>
    /// <typeparam name="T">The unchecked resource type marker.</typeparam>
    /// <param name="id">The child input/output port identifier.</param>
    /// <param name="tensor">The child port's exact tensor contract.</param>
    /// <returns>A handle over the bound declaration without rewriting its graph owner.</returns>
    /// <exception cref="InvalidOperationException">The port is unknown, incompatible, or unbound.</exception>
    public GraphInOut<T> AddInOut<T>(string id, TensorDescriptor tensor)
    {
        var definitionResource = Find(id, GraphDefinitionResourceKind.InOut, tensor);
        if (!outputTargets.TryGetValue(definitionResource.Id, out var target))
        {
            throw new InvalidOperationException($"InOut port '{id}' is not bound.");
        }

        if (target.Owner.IsParent)
        {
            resources.TryAdd(target.Resource.Id, target.Resource);
        }

        return new GraphInOut<T>(target.Resource);
    }

    /// <summary>Finds a child public port of the requested kind and verifies its tensor contract.</summary>
    /// <param name="id">The exact definition-local port name.</param>
    /// <param name="kind">The required public-port kind.</param>
    /// <param name="tensor">The expected element type, layout, and dimensions.</param>
    /// <returns>The matching child declaration.</returns>
    /// <exception cref="InvalidOperationException">No port matches or its tensor descriptor differs.</exception>
    private GraphDefinitionResource Find(
        string id,
        GraphDefinitionResourceKind kind,
        TensorDescriptor tensor)
    {
        var resource = definition.Resources.SingleOrDefault(resource =>
            resource.Id.Value == id && resource.Kind == kind)
            ?? throw new InvalidOperationException(
                $"Unknown {kind} port '{id}' in graph '{definition.Id}'.");
        if (resource.Tensor.ElementType != tensor.ElementType ||
            resource.Tensor.Layout != tensor.Layout ||
            !resource.Tensor.Dimensions.SequenceEqual(tensor.Dimensions))
        {
            throw new InvalidOperationException(
                $"Port '{id}' has a conflicting tensor contract.");
        }

        return resource;
    }
}
