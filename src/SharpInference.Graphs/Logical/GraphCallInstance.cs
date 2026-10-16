
namespace SharpInference.Graphs.Logical;

/// <summary>Identifies one graph call and exposes its remapped ports and child resource declarations.</summary>
/// <typeparam name="TContract">The called graph's public port contract.</typeparam>
public sealed class GraphCallInstance<TContract> where TContract : IGraphPortContract<TContract>
{
    /// <summary>Stores the call identity, called definition, and instantiated port handles.</summary>
    /// <param name="id">The call identity in the containing graph.</param>
    /// <param name="definition">The called definition.</param>
    /// <param name="ports">The ports remapped by the instance port factory.</param>
    internal GraphCallInstance(
        GraphDefinitionNodeId id,
        LogicalGraphDefinition definition,
        TContract ports)
    {
        Id = id;
        Definition = definition;
        Ports = ports;
    }

    /// <summary>Gets the call identity used to scope instance-owned bindings.</summary>
    public GraphDefinitionNodeId Id { get; }
    /// <summary>Gets the ports remapped to this call's inputs and output targets.</summary>
    public TContract Ports { get; }
    /// <summary>Gets the called definition used to resolve declared resources.</summary>
    internal LogicalGraphDefinition Definition { get; }

    /// <summary>Wraps a resource declared by the child definition without remapping its identity or owner.</summary>
    /// <typeparam name="T">The unchecked type marker for the returned resource handle.</typeparam>
    /// <param name="id">The resource identifier in the called definition.</param>
    /// <returns>A handle that can be bound with this instance to select its resource scope.</returns>
    /// <exception cref="ArgumentException">The identifier is blank or the called definition does not declare it.</exception>
    public GraphResource<T> Resource<T>(string id)
    {
        var resourceId = new ResourceId(LogicalGraphDefinition.RequireId(id, nameof(id)));
        var resource = Definition.Resources.SingleOrDefault(resource => resource.Id == resourceId)
            ?? throw new ArgumentException(
                $"Graph definition '{Definition.Id}' does not declare resource '{id}'.",
                nameof(id));
        return new GraphResource<T>(resource);
    }
}
