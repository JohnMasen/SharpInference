
namespace SharpInference.Graphs.Logical;

/// <summary>Declares graph-owned public ports and reuses compatible repeated declarations by identifier.</summary>
/// <param name="owner">The definition identity assigned to every new port resource.</param>
internal sealed class DefinitionPortFactory(GraphDefinitionId owner) : IGraphPortFactory
{
    /// <summary>Resource declarations indexed by their definition-local identifiers.</summary>
    private readonly Dictionary<ResourceId, GraphDefinitionResource> resources = [];
    /// <summary>Gets the currently declared ports in dictionary insertion order.</summary>
    internal IReadOnlyCollection<GraphDefinitionResource> Resources => resources.Values;

    /// <summary>Declares or reuses a compatible input resource.</summary>
    /// <typeparam name="T">The unchecked resource type marker.</typeparam>
    /// <param name="id">The nonblank port identifier, also used as the resource name.</param>
    /// <param name="tensor">The tensor contract to declare or compare.</param>
    /// <returns>A typed input handle over the shared declaration.</returns>
    /// <exception cref="ArgumentException">The identifier is blank.</exception>
    /// <exception cref="ArgumentNullException">A new port's tensor descriptor is null.</exception>
    /// <exception cref="InvalidOperationException">An existing port has a different kind or tensor contract.</exception>
    public GraphInput<T> AddInput<T>(string id, TensorDescriptor tensor) =>
        new(GetOrAdd(id, GraphDefinitionResourceKind.Input, tensor));

    /// <summary>Declares or reuses a compatible output resource.</summary>
    /// <typeparam name="T">The unchecked resource type marker.</typeparam>
    /// <param name="id">The nonblank port identifier, also used as the resource name.</param>
    /// <param name="tensor">The tensor contract to declare or compare.</param>
    /// <returns>A typed output handle over the shared declaration.</returns>
    /// <exception cref="ArgumentException">The identifier is blank.</exception>
    /// <exception cref="ArgumentNullException">A new port's tensor descriptor is null.</exception>
    /// <exception cref="InvalidOperationException">An existing port has a different kind or tensor contract.</exception>
    public GraphOutput<T> AddOutput<T>(string id, TensorDescriptor tensor) =>
        new(GetOrAdd(id, GraphDefinitionResourceKind.Output, tensor));

    /// <summary>Declares or reuses a compatible input/output resource.</summary>
    /// <typeparam name="T">The unchecked resource type marker.</typeparam>
    /// <param name="id">The nonblank port identifier, also used as the resource name.</param>
    /// <param name="tensor">The tensor contract to declare or compare.</param>
    /// <returns>A typed input/output handle over the shared declaration.</returns>
    /// <exception cref="ArgumentException">The identifier is blank.</exception>
    /// <exception cref="ArgumentNullException">A new port's tensor descriptor is null.</exception>
    /// <exception cref="InvalidOperationException">An existing port has a different kind or tensor contract.</exception>
    public GraphInOut<T> AddInOut<T>(string id, TensorDescriptor tensor) =>
        new(GetOrAdd(id, GraphDefinitionResourceKind.InOut, tensor));

    /// <summary>Looks up a compatible declaration or creates one owned by the definition.</summary>
    /// <param name="id">The required identifier and new declaration's name.</param>
    /// <param name="kind">The exact public-port kind to match.</param>
    /// <param name="tensor">The element type, layout, and dimensions to match.</param>
    /// <returns>The existing or newly registered declaration.</returns>
    /// <exception cref="ArgumentException">The identifier is blank.</exception>
    /// <exception cref="ArgumentNullException">A new declaration's tensor is null.</exception>
    /// <exception cref="InvalidOperationException">The identifier already has a different kind or tensor descriptor.</exception>
    private GraphDefinitionResource GetOrAdd(
        string id,
        GraphDefinitionResourceKind kind,
        TensorDescriptor tensor)
    {
        var resourceId = new ResourceId(LogicalGraphDefinition.RequireId(id, nameof(id)));
        if (resources.TryGetValue(resourceId, out var existing))
        {
            if (existing.Kind != kind ||
                existing.Tensor.ElementType != tensor.ElementType ||
                existing.Tensor.Layout != tensor.Layout ||
                !existing.Tensor.Dimensions.SequenceEqual(tensor.Dimensions))
            {
                throw new InvalidOperationException($"Port '{id}' has conflicting declarations.");
            }

            return existing;
        }
        var resource = new GraphDefinitionResource(
            resourceId,
            id,
            kind,
            tensor ?? throw new ArgumentNullException(nameof(tensor)),
            owner);
        resources.Add(resourceId, resource);
        return resource;
    }
}
