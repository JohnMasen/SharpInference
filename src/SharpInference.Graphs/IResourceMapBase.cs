namespace SharpInference.Graphs;

/// <summary>
/// A typed resource map. Metadata is defined by the graph, whereas values are supplied by a
/// backend. Implementations must reject unknown identifiers and values incompatible with their
/// backend's device and tensor descriptor.
/// </summary>
public interface IResourceMapBase<TResource> where TResource : class
{
    /// <summary>Gets graph resource metadata known to the map.</summary>
    IReadOnlyList<GraphResource> Resources { get; }

    /// <summary>Attempts to retrieve graph metadata by resource identifier.</summary>
    bool TryGetMetadata(ResourceId id, out GraphResource metadata);

    /// <summary>Attempts to retrieve a bound backend resource by identifier.</summary>
    bool TryGetResource(ResourceId id, out TResource? resource);

    /// <summary>Binds a compatible backend resource to a graph resource.</summary>
    void SetResource(ResourceId id, TResource resource);
}

/// <summary>
/// A metadata-checked, typed resource map without boxing or per-invocation copies.
/// The compatibility predicate must check the backend value's device, element type, layout, and
/// dimensions before it can be stored; graph metadata alone cannot inspect an opaque backend value.
/// </summary>
public sealed class GraphResourceMap<TResource> : IResourceMapBase<TResource> where TResource : class
{
    private readonly IReadOnlyDictionary<ResourceId, GraphResource> metadata;
    private readonly IReadOnlyDictionary<ResourceId, ExecutionNodeId> privateOwners;
    private readonly Func<GraphResource, TResource, bool> isCompatible;
    private readonly Dictionary<ResourceId, TResource> values = [];
    private readonly object gate = new();
    private bool bindingsSealed;

    /// <summary>Creates a resource map from graph metadata and a compatibility predicate.</summary>
    /// <param name="resources">The graph resources accepted by the map.</param>
    /// <param name="isCompatible">Checks whether a backend value matches resource metadata.</param>
    public GraphResourceMap(IEnumerable<GraphResource> resources, Func<GraphResource, TResource, bool> isCompatible)
    {
        ArgumentNullException.ThrowIfNull(resources);
        this.isCompatible = isCompatible ?? throw new ArgumentNullException(nameof(isCompatible));
        var entries = resources.ToArray();
        metadata = entries.ToDictionary(resource => resource.Id);
        privateOwners = new Dictionary<ResourceId, ExecutionNodeId>();
        Resources = Array.AsReadOnly(entries);
    }

    /// <summary>Creates a resource map using resources and private ownership from an execution graph.</summary>
    /// <param name="graph">The graph defining resource metadata and private owners.</param>
    /// <param name="isCompatible">Checks whether a backend value matches resource metadata.</param>
    public GraphResourceMap(ExecutionGraph graph, Func<GraphResource, TResource, bool> isCompatible)
        : this((graph ?? throw new ArgumentNullException(nameof(graph))).Resources, isCompatible)
    {
        privateOwners = graph.InternalResourceOwners;
    }

    public IReadOnlyList<GraphResource> Resources { get; }

    public bool TryGetMetadata(ResourceId id, out GraphResource metadata) =>
        this.metadata.TryGetValue(id, out metadata!);

    public bool TryGetResource(ResourceId id, out TResource? resource)
    {
        if (!metadata.ContainsKey(id))
        {
            resource = null;
            return false;
        }
        lock (gate)
        {
            bindingsSealed = true;
            return values.TryGetValue(id, out resource);
        }
    }

    /// <summary>Binds a public resource.</summary>
    public void SetResource(ResourceId id, TResource resource) => Bind(id, resource, null);

    /// <summary>Binds a resource that is private to the specified execution node.</summary>
    public void SetPrivateResource(ExecutionNodeId owner, ResourceId id, TResource resource)
    {
        if (!privateOwners.TryGetValue(id, out var actualOwner) || actualOwner != owner)
            throw new InvalidOperationException($"Resource '{id}' is not private to node '{owner}'.");
        Bind(id, resource, owner);
    }

    /// <summary>Prevents further resource bindings while allowing existing resources to be read.</summary>
    public void SealBindings()
    {
        lock (gate)
            bindingsSealed = true;
    }

    private void Bind(ResourceId id, TResource resource, ExecutionNodeId? owner)
    {
        ArgumentNullException.ThrowIfNull(resource);
        if (!metadata.TryGetValue(id, out var descriptor))
            throw new KeyNotFoundException($"Unknown graph resource '{id}'.");
        if (privateOwners.TryGetValue(id, out var actualOwner) && owner != actualOwner)
            throw new InvalidOperationException($"Private resource '{id}' belongs to node '{actualOwner}'.");
        lock (gate)
        {
            if (bindingsSealed)
                throw new InvalidOperationException("Resource bindings are sealed.");
            if (!isCompatible(descriptor, resource))
                throw new ArgumentException($"Resource value is incompatible with graph resource '{id}'.", nameof(resource));
            values[id] = resource;
        }
    }
}
