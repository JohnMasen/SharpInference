namespace SharpInference.Graphs;

/// <summary>Builds graph regions, resources, operations, and state bindings through a fluent API.</summary>
/// <param name="identity">The graph architecture, IR version, and name.</param>
/// <param name="model">The model dimensions, attributes, and state ABI signature.</param>
public sealed class LogicalGraphBuilder(GraphIdentity identity, GraphModelSignature model)
{
    private readonly List<GraphResource> resources = [];
    private readonly List<GraphRegion> regions = [];
    private readonly List<LogicalNode> nodes = [];
    private readonly List<ResourceId> inputs = [];
    private readonly List<ResourceId> outputs = [];
    private readonly List<GraphStateSlot> stateSlots = [];
    private StateSchema? stateSchema;
    private string? currentRegion;

    /// <summary>Gets the graph&apos;s architecture identifier, IR version, and name.</summary>
    public GraphIdentity Identity { get; } = identity ?? throw new ArgumentNullException(nameof(identity));
    /// <summary>Gets the model dimensions, attributes, and state ABI signature.</summary>
    public GraphModelSignature Model { get; } = model ?? throw new ArgumentNullException(nameof(model));

    /// <summary>Adds a named graph region with optional parent and architecture metadata.</summary>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="type">The graph region type or numeric element type.</param>
    /// <param name="name">The name identifying this graph element.</param>
    /// <param name="parentId">The parent region identifier, or null for the graph root.</param>
    /// <param name="role">The optional semantic role of the region.</param>
    /// <param name="attributes">The operation or model attributes keyed by name.</param>
    /// <param name="architecture">The optional architecture presentation metadata.</param>
    public LogicalGraphBuilder AddRegion(
        string id,
        string type,
        string name,
        string? parentId = null,
        string? role = null,
        IReadOnlyDictionary<string, string>? attributes = null,
        GraphRegionArchitecture? architecture = null)
    {
        regions.Add(new GraphRegion(
            new RegionId(RequireId(id, nameof(id))),
            parentId is null ? null : new RegionId(RequireId(parentId, nameof(parentId))),
            RequireId(type, nameof(type)),
            role,
            string.IsNullOrWhiteSpace(name) ? throw new ArgumentException("A region name is required.", nameof(name)) : name,
            attributes ?? EmptyAttributes)
        { Architecture = architecture });
        return this;
    }

    /// <summary>Adds a region, executes the callback in its scope, and restores the previous active region.</summary>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="type">The graph region type or numeric element type.</param>
    /// <param name="name">The name identifying this graph element.</param>
    /// <param name="build">The callback executed while the new region is active.</param>
    /// <param name="parentId">The parent region identifier, or null for the graph root.</param>
    /// <param name="role">The optional semantic role of the region.</param>
    /// <param name="attributes">The operation or model attributes keyed by name.</param>
    /// <param name="architecture">The optional architecture presentation metadata.</param>
    public LogicalGraphBuilder WithRegion(
        string id,
        string type,
        string name,
        Action<LogicalGraphBuilder> build,
        string? parentId = null,
        string? role = null,
        IReadOnlyDictionary<string, string>? attributes = null,
        GraphRegionArchitecture? architecture = null)
    {
        ArgumentNullException.ThrowIfNull(build);
        WithRegion(id, type, name, scoped =>
        {
            build(scoped);
            return true;
        }, parentId, role, attributes, architecture);
        return this;
    }

    /// <summary>Adds a region, executes the callback in its scope, and restores the previous active region.</summary>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="type">The graph region type or numeric element type.</param>
    /// <param name="name">The name identifying this graph element.</param>
    /// <param name="build">The callback executed while the new region is active.</param>
    /// <param name="parentId">The parent region identifier, or null for the graph root.</param>
    /// <param name="role">The optional semantic role of the region.</param>
    /// <param name="attributes">The operation or model attributes keyed by name.</param>
    /// <param name="architecture">The optional architecture presentation metadata.</param>
    /// <typeparam name="T">The region callback result type.</typeparam>
    public T WithRegion<T>(
        string id,
        string type,
        string name,
        Func<LogicalGraphBuilder, T> build,
        string? parentId = null,
        string? role = null,
        IReadOnlyDictionary<string, string>? attributes = null,
        GraphRegionArchitecture? architecture = null)
    {
        ArgumentNullException.ThrowIfNull(build);
        var previous = currentRegion;
        AddRegion(id, type, name, parentId ?? previous, role, attributes, architecture);
        currentRegion = id;
        try
        {
            return build(this);
        }
        finally
        {
            currentRegion = previous;
        }
    }

    /// <summary>Adds a tensor resource and optionally exposes it as a graph input or output.</summary>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="name">The name identifying this graph element.</param>
    /// <param name="kind">The resource role or execution definition category.</param>
    /// <param name="lifetime">The required backing-storage lifetime.</param>
    /// <param name="tensor">The tensor element type, dimensions, and layout.</param>
    /// <param name="bindingKey">The external resource binding key, or null for an unbound resource.</param>
    /// <param name="graphInput">Whether the resource is exposed as a graph input.</param>
    /// <param name="graphOutput">Whether the resource is exposed as a graph output.</param>
    /// <param name="scope">The resource sharing scope.</param>
    public LogicalGraphBuilder AddResource(
        string id,
        string name,
        GraphResourceKind kind,
        GraphResourceLifetime lifetime,
        TensorDescriptor tensor,
        string? bindingKey = null,
        bool graphInput = false,
        bool graphOutput = false,
        GraphResourceScope? scope = null)
    {
        var resourceId = new ResourceId(RequireId(id, nameof(id)));
        resources.Add(new GraphResource(
            resourceId,
            string.IsNullOrWhiteSpace(name) ? throw new ArgumentException("A resource name is required.", nameof(name)) : name,
            kind,
            lifetime,
            tensor ?? throw new ArgumentNullException(nameof(tensor)),
            bindingKey,
            DeclaredScope: scope));
        if (graphInput) inputs.Add(resourceId);
        if (graphOutput) outputs.Add(resourceId);
        return this;
    }

    /// <summary>Adds a logical operation node with bindings, dependencies, attributes, and precision requirements.</summary>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="operation">The operation contract identifier.</param>
    /// <param name="bindings">The node or call port resource bindings.</param>
    /// <param name="dependencies">The predecessor node identifiers.</param>
    /// <param name="attributes">The operation or model attributes keyed by name.</param>
    /// <param name="requirements">The operation precision requirements.</param>
    public LogicalGraphBuilder AddNode(
        string id,
        GraphOperationId operation,
        IEnumerable<NodeResourceBinding> bindings,
        IEnumerable<string>? dependencies = null,
        IReadOnlyDictionary<string, string>? attributes = null,
        PrecisionRequirement? requirements = null) =>
        AddNode(id, operation, currentRegion ??
            throw new InvalidOperationException("A region scope is required when the node's region is omitted."),
            bindings, dependencies, attributes, requirements);

    /// <summary>Adds a logical operation node with bindings, dependencies, attributes, and precision requirements.</summary>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="operation">The operation contract identifier.</param>
    /// <param name="regionId">The containing region identifier, or null to use the active builder region.</param>
    /// <param name="bindings">The node or call port resource bindings.</param>
    /// <param name="dependencies">The predecessor node identifiers.</param>
    /// <param name="attributes">The operation or model attributes keyed by name.</param>
    /// <param name="requirements">The operation precision requirements.</param>
    public LogicalGraphBuilder AddNode(
        string id,
        GraphOperationId operation,
        string regionId,
        IEnumerable<NodeResourceBinding> bindings,
        IEnumerable<string>? dependencies = null,
        IReadOnlyDictionary<string, string>? attributes = null,
        PrecisionRequirement? requirements = null)
    {
        var bindingValues = bindings?.ToArray() ?? throw new ArgumentNullException(nameof(bindings));
        nodes.Add(new LogicalNode(
            new LogicalNodeId(RequireId(id, nameof(id))),
            operation,
            new RegionId(RequireId(regionId, nameof(regionId))),
            bindingValues,
            dependencies?.Select(value => new LogicalNodeId(RequireId(value, nameof(dependencies)))).ToArray() ?? [],
            attributes ?? EmptyAttributes,
            requirements ?? InferRequirements(operation, bindingValues)));
        return this;
    }

    /// <summary>Sets the schema governing the graph&apos;s named state entries.</summary>
    /// <param name="schema">The state ABI schema.</param>
    public LogicalGraphBuilder SetStateSchema(StateSchema schema)
    {
        stateSchema = schema ?? throw new ArgumentNullException(nameof(schema));
        return this;
    }

    /// <summary>Adds a named resource to the graph&apos;s state schema.</summary>
    /// <param name="name">The name identifying this graph element.</param>
    /// <param name="resourceId">The graph resource identifier.</param>
    public LogicalGraphBuilder AddStateSlot(string name, string resourceId)
    {
        stateSlots.Add(new GraphStateSlot(RequireId(name, nameof(name)),
            new ResourceId(RequireId(resourceId, nameof(resourceId)))));
        return this;
    }

    /// <summary>Adds a named resource to the graph&apos;s state schema.</summary>
    /// <param name="name">The name identifying this graph element.</param>
    /// <param name="resourceId">The graph resource identifier.</param>
    public LogicalGraphBuilder AddGraphState(string name, string resourceId) =>
        AddStateSlot(name, resourceId);

    /// <summary>Builds and validates the logical graph from the accumulated resources, regions, and nodes.</summary>
    public LogicalGraph Build()
    {
        if (nodes.All(node => node.Dependencies.Count == 0)) return BuildSequential();
        GraphState? state = null;
        if (stateSchema is not null || stateSlots.Count > 0)
        {
            state = new GraphState(stateSchema ?? new StateSchema(Model.StateAbiId), stateSlots);
        }
        return new LogicalGraph(Identity, Model, resources, regions, nodes, inputs, outputs, state);
    }

    /// <summary>Builds and validates the logical graph from the accumulated resources, regions, and nodes.</summary>
    /// <param name="structure">The ordered graph structure and reusable definitions.</param>
    public LogicalGraph Build(GraphStructure structure)
    {
        GraphState? state = stateSchema is null && stateSlots.Count == 0 ? null :
            new GraphState(stateSchema ?? new StateSchema(Model.StateAbiId), stateSlots);
        return new LogicalGraph(Identity, Model, resources, structure, inputs, outputs, state);
    }

    /// <summary>Builds and validates a graph with each node depending on its immediate predecessor.</summary>
    public LogicalGraph BuildSequential()
    {
        // Compatibility migration for model providers which previously emitted serial predecessor chains.
        var positions = nodes.Select((node, index) => (node.Id, index)).ToDictionary(x => x.Id, x => x.index);
        for (var i = 0; i < nodes.Count; i++)
            foreach (var dependency in nodes[i].Dependencies)
                if (!positions.TryGetValue(dependency, out var position) || position >= i)
                    throw new InvalidDataException("Sequential dependencies must precede the dependent node.");
        var serial = nodes.Select(node => node with { Dependencies = Array.Empty<LogicalNodeId>() }).ToArray();
        var structure = GraphStructure.FromExpanded(regions, serial);
        GraphState? state = stateSchema is null && stateSlots.Count == 0 ? null :
            new GraphState(stateSchema ?? new StateSchema(Model.StateAbiId), stateSlots);
        return new LogicalGraph(Identity, Model, resources, structure, inputs, outputs, state);
    }

    private static readonly IReadOnlyDictionary<string, string> EmptyAttributes =
        new Dictionary<string, string>(StringComparer.Ordinal);

    private static string RequireId(string value, string parameterName) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException("An identifier is required.", parameterName)
            : value;

    private PrecisionRequirement InferRequirements(
        GraphOperationId operation,
        IEnumerable<NodeResourceBinding> bindings)
    {
        var resourceMap = resources.ToDictionary(resource => resource.Id);
        var floatingTypes = bindings
            .Select(binding => resourceMap[binding.Resource].Tensor.ElementType)
            .Where(NumericTypeCompatibility.IsFloatingPoint)
            .ToArray();
        var minimum = floatingTypes.Length == 0
            ? GraphElementType.Float32
            : NumericTypeCompatibility.Maximum(floatingTypes);
        var accumulator = minimum == GraphElementType.Float16 &&
                          (operation == PrimitiveGraphOperations.ReduceSum ||
                           operation == PrimitiveGraphOperations.ReduceMean ||
                           operation == PrimitiveGraphOperations.MatVec)
            ? GraphElementType.Float32
            : minimum;
        return new PrecisionRequirement(minimum, accumulator);
    }
}

/// <summary>Creates read, write, and read-write bindings between node ports and graph resources.</summary>
public static class GraphBindings
{
    /// <summary>Creates a read-only binding from a node port to a graph resource.</summary>
    /// <param name="port">The node port name.</param>
    /// <param name="resource">The graph resource identifier.</param>
    public static NodeResourceBinding Read(string port, string resource) =>
        Create(port, resource, GraphResourceAccess.Read);

    /// <summary>Creates a write-only binding from a node port to a graph resource.</summary>
    /// <param name="port">The node port name.</param>
    /// <param name="resource">The graph resource identifier.</param>
    public static NodeResourceBinding Write(string port, string resource) =>
        Create(port, resource, GraphResourceAccess.Write);

    /// <summary>Creates a binding that reads and updates a graph resource.</summary>
    /// <param name="port">The node port name.</param>
    /// <param name="resource">The graph resource identifier.</param>
    public static NodeResourceBinding ReadWrite(string port, string resource) =>
        Create(port, resource, GraphResourceAccess.ReadWrite);

    private static NodeResourceBinding Create(string port, string resource, GraphResourceAccess access) =>
        new(
            string.IsNullOrWhiteSpace(port) ? throw new ArgumentException("A port name is required.", nameof(port)) : port,
            new ResourceId(string.IsNullOrWhiteSpace(resource) ? throw new ArgumentException("A resource identifier is required.", nameof(resource)) : resource),
            access);
}
