
namespace SharpInference.Graphs.Logical;

/// <summary>Configures a child's public ports against parent resources or explicitly scoped call-instance resources.</summary>
/// <typeparam name="TContract">The called definition's public port contract.</typeparam>
/// <remarks>
/// Inputs require read access, outputs write access, and input/output ports read/write access.
/// Compatibility compares tensor element type, layout, and dimensions, not the generic type marker.
/// A declaration's graph owner and the binding's parent-or-instance scope are distinct.
/// </remarks>
public sealed class GraphCallBindings<TContract> where TContract : IGraphPortContract<TContract>
{
    /// <summary>The containing definition identity required for parent-scoped bindings.</summary>
    private readonly GraphDefinitionId owner;
    /// <summary>Input and input/output targets indexed by child port identifier.</summary>
    private readonly Dictionary<ResourceId, GraphCallBoundResource> values = [];
    /// <summary>Explicit output and input/output targets indexed by child port identifier.</summary>
    private readonly Dictionary<ResourceId, GraphCallBoundResource> outputs = [];

    /// <summary>Creates an empty set of bindings against a child definition's contract.</summary>
    /// <param name="owner">The containing definition identity.</param>
    /// <param name="ports">The uninstantiated child port contract.</param>
    internal GraphCallBindings(GraphDefinitionId owner, TContract ports)
    {
        this.owner = owner;
        Ports = ports;
    }

    /// <summary>Gets the child definition's ports used as binding keys.</summary>
    public TContract Ports { get; }

    /// <summary>Gets distinct parent-scoped declarations from inputs and outputs, deduplicated by identifier.</summary>
    internal IEnumerable<GraphDefinitionResource> Resources =>
        values.Values.Concat(outputs.Values)
            .Where(value => value.Owner.IsParent)
            .Select(value => value.Resource)
            .DistinctBy(resource => resource.Id);
    /// <summary>Gets explicitly bound output and input/output targets and their scopes.</summary>
    internal IReadOnlyDictionary<ResourceId, GraphCallBoundResource> Outputs => outputs;

    /// <summary>Binds an input for reading a resource owned by the containing graph.</summary>
    /// <typeparam name="T">The common port and resource type marker.</typeparam>
    /// <param name="port">The child input port.</param>
    /// <param name="resource">The parent-owned resource to read.</param>
    /// <exception cref="ArgumentNullException">The port or resource is null.</exception>
    /// <exception cref="ArgumentException">Tensor contracts differ or the resource has another graph owner.</exception>
    /// <exception cref="InvalidOperationException">The input is already bound.</exception>
    public void Bind<T>(GraphInput<T> port, GraphResource<T> resource) =>
        BindInput(port, resource, GraphBindingOwner.Parent);

    /// <summary>Binds an input for reading a resource in an explicitly identified call instance.</summary>
    /// <typeparam name="TOwnerContract">The resource-owning instance's contract.</typeparam>
    /// <typeparam name="T">The common port and resource type marker.</typeparam>
    /// <param name="port">The child input port.</param>
    /// <param name="resource">A declaration from the owning instance's definition.</param>
    /// <param name="instance">The instance that scopes the resource access.</param>
    /// <exception cref="ArgumentNullException">The port, resource, or instance is null.</exception>
    /// <exception cref="ArgumentException">The resource is not declared by the instance's definition or the tensors differ.</exception>
    /// <exception cref="InvalidOperationException">The input is already bound.</exception>
    public void Bind<TOwnerContract, T>(
        GraphInput<T> port,
        GraphResource<T> resource,
        GraphCallInstance<TOwnerContract> instance)
        where TOwnerContract : IGraphPortContract<TOwnerContract>
    {
        RequireInstanceResource(resource, instance);
        BindInput(port, resource, GraphBindingOwner.Instance(instance.Id));
    }

    /// <summary>Binds an input/output port for reading and writing a parent-owned resource.</summary>
    /// <typeparam name="T">The common port and resource type marker.</typeparam>
    /// <param name="port">The child input/output port.</param>
    /// <param name="resource">The parent-owned resource to read and write.</param>
    /// <exception cref="ArgumentNullException">The port or resource is null.</exception>
    /// <exception cref="ArgumentException">Tensor contracts differ or the resource has another graph owner.</exception>
    /// <exception cref="InvalidOperationException">The input/output port is already bound.</exception>
    public void Bind<T>(GraphInOut<T> port, GraphResource<T> resource) =>
        BindInOut(port, resource, GraphBindingOwner.Parent);

    /// <summary>Binds an input/output port to read and write a declaration in an explicit call-instance scope.</summary>
    /// <typeparam name="TOwnerContract">The resource-owning instance's contract.</typeparam>
    /// <typeparam name="T">The common port and resource type marker.</typeparam>
    /// <param name="port">The child input/output port.</param>
    /// <param name="resource">A declaration from the owning instance's definition.</param>
    /// <param name="instance">The instance that scopes the resource access.</param>
    /// <remarks>Non-parent writes are retained and later produce the LGW003 validation warning.</remarks>
    /// <exception cref="ArgumentNullException">The port, resource, or instance is null.</exception>
    /// <exception cref="ArgumentException">The resource is not declared by the instance's definition or the tensors differ.</exception>
    /// <exception cref="InvalidOperationException">The input/output port is already bound.</exception>
    public void Bind<TOwnerContract, T>(
        GraphInOut<T> port,
        GraphResource<T> resource,
        GraphCallInstance<TOwnerContract> instance)
        where TOwnerContract : IGraphPortContract<TOwnerContract>
    {
        RequireInstanceResource(resource, instance);
        BindInOut(port, resource, GraphBindingOwner.Instance(instance.Id));
    }

    /// <summary>Binds an output for writing a parent-owned resource instead of generating an output target.</summary>
    /// <typeparam name="T">The common port and resource type marker.</typeparam>
    /// <param name="port">The child output port.</param>
    /// <param name="resource">The parent-owned target resource.</param>
    /// <exception cref="ArgumentNullException">The port or resource is null.</exception>
    /// <exception cref="ArgumentException">Tensor contracts differ or the resource has another graph owner.</exception>
    /// <exception cref="InvalidOperationException">The output is already bound.</exception>
    public void Bind<T>(GraphOutput<T> port, GraphResource<T> resource) =>
        BindOutput(port, resource, GraphBindingOwner.Parent);

    /// <summary>Binds an output to write a declaration in an explicit call-instance scope.</summary>
    /// <typeparam name="TOwnerContract">The resource-owning instance's contract.</typeparam>
    /// <typeparam name="T">The common port and resource type marker.</typeparam>
    /// <param name="port">The child output port.</param>
    /// <param name="resource">A declaration from the owning instance's definition.</param>
    /// <param name="instance">The instance that scopes the resource access.</param>
    /// <remarks>Non-parent writes are retained and later produce the LGW003 validation warning.</remarks>
    /// <exception cref="ArgumentNullException">The port, resource, or instance is null.</exception>
    /// <exception cref="ArgumentException">The resource is not declared by the instance's definition or the tensors differ.</exception>
    /// <exception cref="InvalidOperationException">The output is already bound.</exception>
    public void Bind<TOwnerContract, T>(
        GraphOutput<T> port,
        GraphResource<T> resource,
        GraphCallInstance<TOwnerContract> instance)
        where TOwnerContract : IGraphPortContract<TOwnerContract>
    {
        RequireInstanceResource(resource, instance);
        BindOutput(port, resource, GraphBindingOwner.Instance(instance.Id));
    }

    /// <summary>Requires every declared input and input/output port and emits their scoped access bindings.</summary>
    /// <param name="definition">The child definition that determines required ports and access modes.</param>
    /// <returns>Read bindings for inputs and read/write bindings for input/output ports, in declaration order.</returns>
    /// <exception cref="InvalidOperationException">One or more required ports have not been bound.</exception>
    internal IReadOnlyList<PendingBinding> BuildInputs(LogicalGraphDefinition definition)
    {
        var inputs = definition.Resources.Where(resource =>
            resource.Kind is GraphDefinitionResourceKind.Input
                or GraphDefinitionResourceKind.InOut).ToArray();
        var missing = inputs.Where(input => !values.ContainsKey(input.Id)).ToArray();
        if (missing.Length != 0)
        {
            throw new InvalidOperationException(
                $"Graph call '{definition.Id}' is missing inputs: " +
                $"{string.Join(", ", missing.Select(input => input.Id))}.");
        }

        return inputs.Select(input => new PendingBinding(
            input.Id.Value,
            values[input.Id].Resource,
            input.Kind == GraphDefinitionResourceKind.InOut
                ? GraphResourceAccess.ReadWrite
                : GraphResourceAccess.Read,
            values[input.Id].Owner)).ToArray();
    }

    /// <summary>Checks tensor compatibility and inserts one scoped input binding.</summary>
    /// <typeparam name="T">The common port and resource type marker.</typeparam>
    /// <param name="port">The input port used as the binding key.</param>
    /// <param name="resource">The resource to read.</param>
    /// <param name="resourceOwner">The parent or call-instance access scope.</param>
    /// <exception cref="ArgumentNullException">The port or resource is null.</exception>
    /// <exception cref="ArgumentException">The tensors differ or a parent binding references another graph's declaration.</exception>
    /// <exception cref="InvalidOperationException">The input is already bound.</exception>
    private void BindInput<T>(
        GraphInput<T> port,
        GraphResource<T> resource,
        GraphBindingOwner resourceOwner)
    {
        RequireCompatible(port, resource);
        if (!values.TryAdd(port.Id, Resolve(resource.Resource, resourceOwner)))
        {
            throw new InvalidOperationException($"Input '{port.Id}' is already bound.");
        }
    }

    /// <summary>Checks compatibility and registers the same scoped target in both input and output maps.</summary>
    /// <typeparam name="T">The common port and resource type marker.</typeparam>
    /// <param name="port">The input/output port used as the binding key.</param>
    /// <param name="resource">The resource to read and write.</param>
    /// <param name="resourceOwner">The parent or call-instance access scope.</param>
    /// <exception cref="ArgumentNullException">The port or resource is null.</exception>
    /// <exception cref="ArgumentException">The tensors differ or a parent binding references another graph's declaration.</exception>
    /// <exception cref="InvalidOperationException">Either map already contains the port; earlier insertions are not rolled back.</exception>
    private void BindInOut<T>(
        GraphInOut<T> port,
        GraphResource<T> resource,
        GraphBindingOwner resourceOwner)
    {
        RequireCompatible(port, resource);
        var normalized = Resolve(resource.Resource, resourceOwner);
        if (!values.TryAdd(port.Id, normalized) || !outputs.TryAdd(port.Id, normalized))
        {
            throw new InvalidOperationException($"InOut port '{port.Id}' is already bound.");
        }
    }

    /// <summary>Checks compatibility and inserts one explicit scoped output target.</summary>
    /// <typeparam name="T">The common port and resource type marker.</typeparam>
    /// <param name="port">The output port used as the binding key.</param>
    /// <param name="resource">The resource to write.</param>
    /// <param name="resourceOwner">The parent or call-instance access scope.</param>
    /// <exception cref="ArgumentNullException">The port or resource is null.</exception>
    /// <exception cref="ArgumentException">The tensors differ or a parent binding references another graph's declaration.</exception>
    /// <exception cref="InvalidOperationException">The output is already bound.</exception>
    private void BindOutput<T>(
        GraphOutput<T> port,
        GraphResource<T> resource,
        GraphBindingOwner resourceOwner)
    {
        RequireCompatible(port, resource);
        if (!outputs.TryAdd(port.Id, Resolve(resource.Resource, resourceOwner)))
        {
            throw new InvalidOperationException($"Output '{port.Id}' is already bound.");
        }
    }

    /// <summary>Preserves the declaration and pairs it with its access scope after checking parent ownership.</summary>
    /// <param name="resource">The unchanged resource declaration.</param>
    /// <param name="resourceOwner">The parent or call-instance scope for this binding.</param>
    /// <returns>The declaration and binding owner as a pair.</returns>
    /// <exception cref="ArgumentException">A parent-scoped declaration belongs to another graph.</exception>
    private GraphCallBoundResource Resolve(
        GraphDefinitionResource resource,
        GraphBindingOwner resourceOwner)
    {
        if (resourceOwner.IsParent && resource.Owner != owner)
        {
            throw new ArgumentException(
                $"Parent binding cannot reference resource '{resource.Id}' owned by graph '{resource.Owner}'.");
        }

        return new GraphCallBoundResource(resource, resourceOwner);
    }

    /// <summary>Checks nonnull handles and exact tensor element type, layout, and dimension equality.</summary>
    /// <typeparam name="T">The common handle type marker, not independently validated.</typeparam>
    /// <param name="port">The port defining the required tensor contract.</param>
    /// <param name="resource">The proposed target resource.</param>
    /// <exception cref="ArgumentNullException">Either handle is null.</exception>
    /// <exception cref="ArgumentException">The tensor contracts differ.</exception>
    private static void RequireCompatible<T>(
        GraphResource<T> port,
        GraphResource<T> resource)
    {
        ArgumentNullException.ThrowIfNull(port);
        ArgumentNullException.ThrowIfNull(resource);
        if (port.Tensor.ElementType != resource.Tensor.ElementType ||
            port.Tensor.Layout != resource.Tensor.Layout ||
            !port.Tensor.Dimensions.SequenceEqual(resource.Tensor.Dimensions))
        {
            throw new ArgumentException(
                $"Resource '{resource.Id}' is incompatible with port '{port.Id}'.",
                nameof(resource));
        }
    }

    /// <summary>Checks the declaration's graph owner and membership in the specified instance's definition.</summary>
    /// <typeparam name="TOwnerContract">The owning instance's contract.</typeparam>
    /// <typeparam name="T">The resource handle's type marker.</typeparam>
    /// <param name="resource">The child declaration to access.</param>
    /// <param name="instance">The instance that will scope the binding.</param>
    /// <exception cref="ArgumentNullException">The resource or instance is null.</exception>
    /// <exception cref="ArgumentException">The owner or declaration membership does not match the instance's definition.</exception>
    private static void RequireInstanceResource<TOwnerContract, T>(
        GraphResource<T> resource,
        GraphCallInstance<TOwnerContract> instance)
        where TOwnerContract : IGraphPortContract<TOwnerContract>
    {
        ArgumentNullException.ThrowIfNull(resource);
        ArgumentNullException.ThrowIfNull(instance);
        if (resource.Resource.Owner != instance.Definition.Id ||
            !instance.Definition.Resources.Contains(resource.Resource))
        {
            throw new ArgumentException(
                $"Resource '{resource.Id}' is not declared by graph instance '{instance.Id}'.",
                nameof(resource));
        }
    }
}
