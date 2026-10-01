namespace SharpInference.Graphs;

public sealed class LogicalGraphBuilder(GraphIdentity identity, GraphModelSignature model)
{
    private readonly List<GraphResource> resources = [];
    private readonly List<GraphRegion> regions = [];
    private readonly List<LogicalNode> nodes = [];
    private readonly List<ResourceId> inputs = [];
    private readonly List<ResourceId> outputs = [];
    private readonly List<GraphStateSlot> stateSlots = [];
    private StateSchema? stateSchema;

    public GraphIdentity Identity { get; } = identity ?? throw new ArgumentNullException(nameof(identity));
    public GraphModelSignature Model { get; } = model ?? throw new ArgumentNullException(nameof(model));

    public LogicalGraphBuilder AddRegion(
        string id,
        string type,
        string name,
        string? parentId = null,
        string? role = null,
        IReadOnlyDictionary<string, string>? attributes = null)
    {
        regions.Add(new GraphRegion(
            new RegionId(RequireId(id, nameof(id))),
            parentId is null ? null : new RegionId(RequireId(parentId, nameof(parentId))),
            RequireId(type, nameof(type)),
            role,
            string.IsNullOrWhiteSpace(name) ? throw new ArgumentException("A region name is required.", nameof(name)) : name,
            attributes ?? EmptyAttributes));
        return this;
    }

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

    public LogicalGraphBuilder SetStateSchema(StateSchema schema)
    {
        stateSchema = schema ?? throw new ArgumentNullException(nameof(schema));
        return this;
    }

    public LogicalGraphBuilder AddStateSlot(string name, string resourceId)
    {
        stateSlots.Add(new GraphStateSlot(RequireId(name, nameof(name)),
            new ResourceId(RequireId(resourceId, nameof(resourceId)))));
        return this;
    }

    public LogicalGraphBuilder AddGraphState(string name, string resourceId) =>
        AddStateSlot(name, resourceId);

    public LogicalGraph Build()
    {
        GraphState? state = null;
        if (stateSchema is not null || stateSlots.Count > 0)
        {
            state = new GraphState(stateSchema ?? new StateSchema(Model.StateAbiId), stateSlots);
        }
        return new LogicalGraph(Identity, Model, resources, regions, nodes, inputs, outputs, state);
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

public static class GraphBindings
{
    public static NodeResourceBinding Read(string port, string resource) =>
        Create(port, resource, GraphResourceAccess.Read);

    public static NodeResourceBinding Write(string port, string resource) =>
        Create(port, resource, GraphResourceAccess.Write);

    public static NodeResourceBinding ReadWrite(string port, string resource) =>
        Create(port, resource, GraphResourceAccess.ReadWrite);

    private static NodeResourceBinding Create(string port, string resource, GraphResourceAccess access) =>
        new(
            string.IsNullOrWhiteSpace(port) ? throw new ArgumentException("A port name is required.", nameof(port)) : port,
            new ResourceId(string.IsNullOrWhiteSpace(resource) ? throw new ArgumentException("A resource identifier is required.", nameof(resource)) : resource),
            access);
}
