namespace SharpInference.Graphs;

public readonly record struct ResourceId(string Value)
{
    public override string ToString() => Value;
}

public readonly record struct RegionId(string Value)
{
    public override string ToString() => Value;
}

public readonly record struct LogicalNodeId(string Value)
{
    public override string ToString() => Value;
}

public readonly record struct ExecutionNodeId(string Value)
{
    public override string ToString() => Value;
}

public readonly record struct GraphOperationId(string Name, int Version = 1)
{
    public override string ToString() => $"{Name}@{Version}";
}

public static class GraphRegionTypes
{
    public const string Graph = "graph";
    public const string Pipeline = "pipeline";
    public const string Layer = "layer";
    public const string Stage = "stage";
}

public enum GraphResourceKind
{
    Input,
    Output,
    Weight,
    SessionState,
    TokenTransient,
    Temporary,
    Constant,
}

public enum GraphResourceLifetime
{
    External,
    Model,
    Session,
    Token,
    Invocation,
}

public enum GraphResourceScope
{
    Global,
    Session,
    Local,
}

public enum GraphElementType
{
    Byte,
    Int32,
    UInt32,
    Float16,
    Float32,
}

public enum GraphResourceAccess
{
    Read,
    Write,
    ReadWrite,
}

public sealed record TensorDescriptor
{
    public TensorDescriptor(GraphElementType elementType, IEnumerable<int> dimensions, string layout = "dense")
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        var values = dimensions.ToArray();
        if (values.Any(value => value <= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions), "Tensor dimensions must be positive.");
        }

        ElementType = elementType;
        Dimensions = Array.AsReadOnly(values);
        Layout = string.IsNullOrWhiteSpace(layout)
            ? throw new ArgumentException("A tensor layout is required.", nameof(layout))
            : layout;
    }

    public GraphElementType ElementType { get; }
    public IReadOnlyList<int> Dimensions { get; }
    public string Layout { get; }
}

public sealed class StateSchema : IEquatable<StateSchema>
{
    public StateSchema(string name)
    {
        Name = string.IsNullOrWhiteSpace(name)
            ? throw new ArgumentException("A state schema name is required.", nameof(name)) : name;
    }

    public string Name { get; }
    public bool IsCompatibleWith(StateSchema? other) => Equals(other);
    public bool Equals(StateSchema? other) => other is not null &&
        string.Equals(Name, other.Name, StringComparison.Ordinal);
    public override bool Equals(object? obj) => Equals(obj as StateSchema);
    public override int GetHashCode() => StringComparer.Ordinal.GetHashCode(Name);
}

public record GraphStateEntry(string Name, ResourceId Resource);

public sealed record GraphStateSlot(string Name, ResourceId Resource) : GraphStateEntry(Name, Resource);

public sealed class GraphState : IReadOnlyList<GraphStateEntry>
{
    public GraphState(StateSchema schema, IEnumerable<GraphStateEntry> entries)
    {
        Schema = schema ?? throw new ArgumentNullException(nameof(schema));
        Entries = entries?.ToArray() ?? throw new ArgumentNullException(nameof(entries));
    }

    public StateSchema Schema { get; }
    public IReadOnlyList<GraphStateEntry> Entries { get; }
    public IReadOnlyList<GraphStateEntry> Slots => Entries;
    public int Count => Entries.Count;
    public GraphStateEntry this[int index] => Entries[index];
    public IEnumerator<GraphStateEntry> GetEnumerator() => Entries.GetEnumerator();
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
}

public sealed record GraphIdentity(string ArchitectureId, int IrVersion, string Name);

public sealed class GraphModelSignature : IEquatable<GraphModelSignature>
{
    public GraphModelSignature(
        string modelType,
        string stateAbiId,
        IReadOnlyDictionary<string, int> dimensions,
        IReadOnlyDictionary<string, string>? attributes = null)
    {
        ModelType = string.IsNullOrWhiteSpace(modelType)
            ? throw new ArgumentException("A graph model type is required.", nameof(modelType))
            : modelType;
        StateAbiId = string.IsNullOrWhiteSpace(stateAbiId)
            ? throw new ArgumentException("A state ABI identifier is required.", nameof(stateAbiId))
            : stateAbiId;
        ArgumentNullException.ThrowIfNull(dimensions);
        if (dimensions.Any(value => string.IsNullOrWhiteSpace(value.Key) || value.Value <= 0))
            throw new ArgumentException("Graph model dimensions require names and positive values.", nameof(dimensions));
        if (attributes?.Any(value => string.IsNullOrWhiteSpace(value.Key) || value.Value is null) == true)
            throw new ArgumentException("Graph model attributes require names and values.", nameof(attributes));
        Dimensions = new System.Collections.ObjectModel.ReadOnlyDictionary<string, int>(
            new Dictionary<string, int>(dimensions, StringComparer.Ordinal));
        Attributes = new System.Collections.ObjectModel.ReadOnlyDictionary<string, string>(
            new Dictionary<string, string>(attributes ?? new Dictionary<string, string>(), StringComparer.Ordinal));
    }

    public string ModelType { get; }
    public string StateAbiId { get; }
    public IReadOnlyDictionary<string, int> Dimensions { get; }
    public IReadOnlyDictionary<string, string> Attributes { get; }

    public bool Equals(GraphModelSignature? other) =>
        other is not null &&
        string.Equals(ModelType, other.ModelType, StringComparison.Ordinal) &&
        string.Equals(StateAbiId, other.StateAbiId, StringComparison.Ordinal) &&
        Dimensions.OrderBy(value => value.Key, StringComparer.Ordinal)
            .SequenceEqual(other.Dimensions.OrderBy(value => value.Key, StringComparer.Ordinal)) &&
        Attributes.OrderBy(value => value.Key, StringComparer.Ordinal)
            .SequenceEqual(other.Attributes.OrderBy(value => value.Key, StringComparer.Ordinal));

    public override bool Equals(object? obj) => Equals(obj as GraphModelSignature);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(ModelType, StringComparer.Ordinal);
        hash.Add(StateAbiId, StringComparer.Ordinal);
        foreach (var value in Dimensions.OrderBy(value => value.Key, StringComparer.Ordinal))
        {
            hash.Add(value.Key, StringComparer.Ordinal);
            hash.Add(value.Value);
        }
        foreach (var value in Attributes.OrderBy(value => value.Key, StringComparer.Ordinal))
        {
            hash.Add(value.Key, StringComparer.Ordinal);
            hash.Add(value.Value, StringComparer.Ordinal);
        }
        return hash.ToHashCode();
    }

}

public sealed record GraphResource(
    ResourceId Id,
    string Name,
    GraphResourceKind Kind,
    GraphResourceLifetime Lifetime,
    TensorDescriptor Tensor,
    string? BindingKey = null,
    string? DeviceId = null,
    GraphResourceScope? DeclaredScope = null)
{
    public GraphResourceScope Scope => DeclaredScope ?? (Lifetime switch
    {
        GraphResourceLifetime.External when Kind is GraphResourceKind.Input or GraphResourceKind.Output =>
            GraphResourceScope.Local,
        GraphResourceLifetime.External or GraphResourceLifetime.Model => GraphResourceScope.Global,
        GraphResourceLifetime.Session => GraphResourceScope.Session,
        GraphResourceLifetime.Token or GraphResourceLifetime.Invocation => GraphResourceScope.Local,
        _ => throw new InvalidDataException($"Invalid lifetime for resource '{Id}'."),
    });
}

public sealed record GraphRegion(
    RegionId Id,
    RegionId? ParentId,
    string Type,
    string? Role,
    string Name,
    IReadOnlyDictionary<string, string> Attributes);

public sealed record NodeResourceBinding(
    string Port,
    ResourceId Resource,
    GraphResourceAccess Access,
    bool InitializedBeforeRead = false,
    GraphTensorView? View = null);

public sealed record LogicalNode(
    LogicalNodeId Id,
    GraphOperationId Operation,
    RegionId Region,
    IReadOnlyList<NodeResourceBinding> Resources,
    IReadOnlyList<LogicalNodeId> Dependencies,
    IReadOnlyDictionary<string, string> Attributes,
    PrecisionRequirement Requirements);

public sealed class LogicalGraph
{
    public LogicalGraph(
        GraphIdentity identity,
        GraphModelSignature model,
        IEnumerable<GraphResource> resources,
        IEnumerable<GraphRegion> regions,
        IEnumerable<LogicalNode> nodes,
        IEnumerable<ResourceId> inputs,
        IEnumerable<ResourceId> outputs,
        IEnumerable<GraphStateEntry>? graphState = null)
    {
        Identity = identity ?? throw new ArgumentNullException(nameof(identity));
        Model = model ?? throw new ArgumentNullException(nameof(model));
        Resources = resources?.ToArray() ?? throw new ArgumentNullException(nameof(resources));
        Regions = regions?.ToArray() ?? throw new ArgumentNullException(nameof(regions));
        Nodes = nodes?.ToArray() ?? throw new ArgumentNullException(nameof(nodes));
        Inputs = inputs?.ToArray() ?? throw new ArgumentNullException(nameof(inputs));
        Outputs = outputs?.ToArray() ?? throw new ArgumentNullException(nameof(outputs));
        GraphState = CreateGraphState(Model, Resources, graphState);
        GraphValidator.Validate(this);
    }

    public GraphIdentity Identity { get; }
    public GraphModelSignature Model { get; }
    public IReadOnlyList<GraphResource> Resources { get; }
    public IReadOnlyList<GraphRegion> Regions { get; }
    public IReadOnlyList<LogicalNode> Nodes { get; }
    public IReadOnlyList<ResourceId> Inputs { get; }
    public IReadOnlyList<ResourceId> Outputs { get; }
    public GraphState GraphState { get; }

    internal static GraphState CreateGraphState(
        GraphModelSignature model, IReadOnlyList<GraphResource> resources, IEnumerable<GraphStateEntry>? entries)
    {
        var values = entries?.ToArray() ?? [];
        var resourceMap = resources.GroupBy(resource => resource.Id).ToDictionary(group => group.Key, group => group.First());
        foreach (var entry in values)
        {
            if (entry is null || !resourceMap.TryGetValue(entry.Resource, out var resource))
                throw new InvalidDataException($"Unknown graph state resource '{entry?.Resource}'.");
        }
        return new GraphState(entries is GraphState original ? original.Schema : new StateSchema(model.StateAbiId), values);
    }
}

public interface ILogicalGraphProvider
{
    string ArchitectureId { get; }
    LogicalGraph Build(IModelTensorCatalog tensors);
}

public sealed record ExecutionSourceMap(
    IReadOnlyList<LogicalNodeId> LogicalNodes,
    string? AppliedRuleId)
{
    public static ExecutionSourceMap XmlOnly { get; } = new([], null);
}

public sealed record ExecutionNode(
    ExecutionNodeId Id,
    GraphOperationId Operation,
    RegionId Region,
    IReadOnlyList<NodeResourceBinding> Resources,
    IReadOnlyList<ExecutionNodeId> Dependencies,
    IReadOnlyDictionary<string, string> Attributes,
    PrecisionRequirement Requirements,
    ExecutionSourceMap Source)
{
    public IReadOnlyList<NodeResourceBinding> InternalResources { get; init; } = [];
    public IReadOnlyList<ResourceId> FirstWriteResources { get; init; } = [];
}

public sealed class ExecutionGraph
{
    public ExecutionGraph(
        GraphIdentity identity,
        GraphModelSignature model,
        IEnumerable<GraphResource> resources,
        IEnumerable<GraphRegion> regions,
        IEnumerable<ExecutionNode> nodes,
        IEnumerable<ResourceId> inputs,
        IEnumerable<ResourceId> outputs,
        IEnumerable<GraphStateEntry>? graphState = null)
    {
        Identity = identity ?? throw new ArgumentNullException(nameof(identity));
        Model = model ?? throw new ArgumentNullException(nameof(model));
        Resources = resources?.ToArray() ?? throw new ArgumentNullException(nameof(resources));
        Regions = regions?.ToArray() ?? throw new ArgumentNullException(nameof(regions));
        Nodes = nodes?.ToArray() ?? throw new ArgumentNullException(nameof(nodes));
        Inputs = inputs?.ToArray() ?? throw new ArgumentNullException(nameof(inputs));
        Outputs = outputs?.ToArray() ?? throw new ArgumentNullException(nameof(outputs));
        GraphState = LogicalGraph.CreateGraphState(Model, Resources, graphState);
        GraphValidator.Validate(this);
        InternalResourceOwners = new System.Collections.ObjectModel.ReadOnlyDictionary<ResourceId, ExecutionNodeId>(
            Nodes.SelectMany(node => node.InternalResources.Select(binding => (binding.Resource, node.Id)))
                .ToDictionary(item => item.Resource, item => item.Id));
    }

    public GraphIdentity Identity { get; }
    public GraphModelSignature Model { get; }
    public IReadOnlyList<GraphResource> Resources { get; }
    public IReadOnlyList<GraphRegion> Regions { get; }
    public IReadOnlyList<ExecutionNode> Nodes { get; }
    public IReadOnlyList<ResourceId> Inputs { get; }
    public IReadOnlyList<ResourceId> Outputs { get; }
    public GraphState GraphState { get; }
    public IReadOnlyDictionary<ResourceId, ExecutionNodeId> InternalResourceOwners { get; }

    public bool CanReuseInternalBacking(ResourceId first, ResourceId second)
    {
        if (first == second ||
            !InternalResourceOwners.TryGetValue(first, out var firstOwner) ||
            !InternalResourceOwners.TryGetValue(second, out var secondOwner) ||
            firstOwner == secondOwner)
        {
            return false;
        }

        var left = Resources.First(resource => resource.Id == first);
        var right = Resources.First(resource => resource.Id == second);
        if (left.Lifetime != right.Lifetime ||
            !string.Equals(left.DeviceId, right.DeviceId, StringComparison.Ordinal) ||
            left.Tensor.ElementType != right.Tensor.ElementType ||
            left.Tensor.Layout != right.Tensor.Layout ||
            !left.Tensor.Dimensions.SequenceEqual(right.Tensor.Dimensions))
        {
            return false;
        }

        return DependsOn(firstOwner, secondOwner) || DependsOn(secondOwner, firstOwner);

        bool DependsOn(ExecutionNodeId nodeId, ExecutionNodeId predecessor)
        {
            var visited = new HashSet<ExecutionNodeId>();
            var pending = new Stack<ExecutionNodeId>();
            pending.Push(nodeId);
            while (pending.TryPop(out var current))
            {
                if (!visited.Add(current)) continue;
                foreach (var dependency in Nodes.First(node => node.Id == current).Dependencies)
                {
                    if (dependency == predecessor) return true;
                    pending.Push(dependency);
                }
            }
            return false;
        }
    }
}
