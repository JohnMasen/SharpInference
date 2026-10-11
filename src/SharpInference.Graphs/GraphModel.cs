namespace SharpInference.Graphs;

/// <summary>Identifies a tensor resource within a graph.</summary>
/// <param name="Value">The resource identifier text.</param>
public readonly record struct ResourceId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Identifies a region in the graph hierarchy.</summary>
/// <param name="Value">The region identifier text.</param>
public readonly record struct RegionId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Identifies an operation node in a logical graph.</summary>
/// <param name="Value">The logical node identifier text.</param>
public readonly record struct LogicalNodeId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Identifies a scheduled operation node in an execution graph.</summary>
/// <param name="Value">The execution node identifier text.</param>
public readonly record struct ExecutionNodeId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Identifies an operation contract by name and version.</summary>
/// <param name="Name">The operation contract name.</param>
/// <param name="Version">The operation contract version.</param>
public readonly record struct GraphOperationId(string Name, int Version = 1)
{
    /// <inheritdoc/>
    public override string ToString() => $"{Name}@{Version}";
}

/// <summary>Defines the standard region type names used in graph hierarchies.</summary>
public static class GraphRegionTypes
{
    /// <summary>The standard type name for a graph region.</summary>
    public const string Graph = "graph";
    /// <summary>The standard type name for a pipeline region.</summary>
    public const string Pipeline = "pipeline";
    /// <summary>The standard type name for a layer region.</summary>
    public const string Layer = "layer";
    /// <summary>The standard type name for a stage region.</summary>
    public const string Stage = "stage";
    /// <summary>The standard type name for an architecture region.</summary>
    public const string Architecture = "architecture";
}

/// <summary>Describes the role of a tensor resource in a graph.</summary>
public enum GraphResourceKind
{
    /// <summary>An externally supplied graph input.</summary>
    Input,
    /// <summary>A graph output returned to the caller.</summary>
    Output,
    /// <summary>A model weight tensor.</summary>
    Weight,
    /// <summary>A tensor holding persistent session state.</summary>
    SessionState,
    /// <summary>A transient tensor used during token processing.</summary>
    TokenTransient,
    /// <summary>A temporary operation tensor.</summary>
    Temporary,
    /// <summary>An immutable constant tensor.</summary>
    Constant,
}

/// <summary>Describes how long a resource&apos;s backing storage remains valid.</summary>
public enum GraphResourceLifetime
{
    /// <summary>Storage is supplied externally.</summary>
    External,
    /// <summary>Storage remains valid for the model lifetime.</summary>
    Model,
    /// <summary>Storage remains valid for the session lifetime.</summary>
    Session,
    /// <summary>Storage remains valid for one token step.</summary>
    Token,
    /// <summary>Storage remains valid for one graph invocation.</summary>
    Invocation,
}

/// <summary>Describes the sharing scope of a graph resource.</summary>
public enum GraphResourceScope
{
    /// <summary>Storage is shared at model or global scope.</summary>
    Global,
    /// <summary>Storage is private to a model session.</summary>
    Session,
    /// <summary>Storage is local to graph execution.</summary>
    Local,
}

/// <summary>Specifies the scalar element representation of a graph tensor.</summary>
public enum GraphElementType
{
    /// <summary>An unsigned eight-bit integer element.</summary>
    Byte,
    /// <summary>A signed 32-bit integer element.</summary>
    Int32,
    /// <summary>An unsigned 32-bit integer element.</summary>
    UInt32,
    /// <summary>A 16-bit floating-point element.</summary>
    Float16,
    /// <summary>A 32-bit floating-point element.</summary>
    Float32,
}

/// <summary>Specifies whether a node reads, writes, or updates a resource.</summary>
public enum GraphResourceAccess
{
    /// <summary>Reads existing resource contents.</summary>
    Read,
    /// <summary>Writes resource contents without requiring previous values.</summary>
    Write,
    /// <summary>Reads and updates resource contents.</summary>
    ReadWrite,
}

/// <summary>Describes a tensor&apos;s element type, positive dimensions, and storage layout.</summary>
public sealed record TensorDescriptor
{
    /// <summary>Creates a tensor descriptor with positive dimensions and a nonempty layout name.</summary>
    /// <param name="elementType">The tensor scalar element type.</param>
    /// <param name="dimensions">The tensor dimension sizes in axis order.</param>
    /// <param name="layout">The tensor storage layout name.</param>
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

    /// <summary>Gets the tensor scalar element type.</summary>
    public GraphElementType ElementType { get; }
    /// <summary>Gets the tensor dimensions in axis order.</summary>
    public IReadOnlyList<int> Dimensions { get; }
    /// <summary>Gets the tensor storage layout name.</summary>
    public string Layout { get; }
}

/// <summary>Identifies a state ABI schema using an ordinal name.</summary>
public sealed class StateSchema : IEquatable<StateSchema>
{
    /// <summary>Creates a state compatibility schema with a nonempty ordinal name.</summary>
    /// <param name="name">The state schema name used for ordinal compatibility checks.</param>
    public StateSchema(string name)
    {
        Name = string.IsNullOrWhiteSpace(name)
            ? throw new ArgumentException("A state schema name is required.", nameof(name)) : name;
    }

    /// <summary>Gets the name of this element.</summary>
    public string Name { get; }
    /// <summary>Determines whether another state schema has the same ordinal schema name.</summary>
    /// <param name="other">The value to compare.</param>
    public bool IsCompatibleWith(StateSchema? other) => Equals(other);
    /// <inheritdoc/>
    public bool Equals(StateSchema? other) => other is not null &&
        string.Equals(Name, other.Name, StringComparison.Ordinal);
    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as StateSchema);
    /// <inheritdoc/>
    public override int GetHashCode() => StringComparer.Ordinal.GetHashCode(Name);
}

/// <summary>Associates a named state entry with a graph resource.</summary>
/// <param name="Name">The state entry name.</param>
/// <param name="Resource">The referenced graph resource identifier.</param>
public record GraphStateEntry(string Name, ResourceId Resource);

/// <summary>Represents a named resource slot in a graph state schema.</summary>
/// <param name="Name">The state entry name.</param>
/// <param name="Resource">The referenced graph resource identifier.</param>
public sealed record GraphStateSlot(string Name, ResourceId Resource) : GraphStateEntry(Name, Resource);

/// <summary>Provides ordered state entries and the schema governing their compatibility.</summary>
public sealed class GraphState : IReadOnlyList<GraphStateEntry>
{
    /// <summary>Creates ordered graph state entries governed by the supplied schema.</summary>
    /// <param name="schema">The state ABI schema.</param>
    /// <param name="entries">The ordered named entries.</param>
    public GraphState(StateSchema schema, IEnumerable<GraphStateEntry> entries)
    {
        Schema = schema ?? throw new ArgumentNullException(nameof(schema));
        Entries = entries?.ToArray() ?? throw new ArgumentNullException(nameof(entries));
    }

    /// <summary>Gets the state ABI compatibility schema.</summary>
    public StateSchema Schema { get; }
    /// <summary>Gets the ordered named entries.</summary>
    public IReadOnlyList<GraphStateEntry> Entries { get; }
    /// <summary>Gets the ordered state entries as resource slots.</summary>
    public IReadOnlyList<GraphStateEntry> Slots => Entries;
    /// <summary>Gets the number of entries currently stored.</summary>
    public int Count => Entries.Count;
    /// <summary>Gets the state entry at the specified zero-based index.</summary>
    /// <param name="index">The zero-based entry index.</param>
    public GraphStateEntry this[int index] => Entries[index];
    /// <inheritdoc/>
    public IEnumerator<GraphStateEntry> GetEnumerator() => Entries.GetEnumerator();
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
}

/// <summary>Describes a graph&apos;s architecture, intermediate representation version, and name.</summary>
/// <param name="ArchitectureId">The model architecture identifier.</param>
/// <param name="IrVersion">The graph intermediate representation version.</param>
/// <param name="Name">The graph name.</param>
public sealed record GraphIdentity(string ArchitectureId, int IrVersion, string Name);

/// <summary>Describes model dimensions, attributes, and the state ABI required by a graph.</summary>
public sealed class GraphModelSignature : IEquatable<GraphModelSignature>
{
    /// <summary>Creates a model signature with positive named dimensions and optional attributes.</summary>
    /// <param name="modelType">The architecture-specific model type name.</param>
    /// <param name="stateAbiId">The state ABI identifier.</param>
    /// <param name="dimensions">The tensor dimension sizes in axis order.</param>
    /// <param name="attributes">The operation or model attributes keyed by name.</param>
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

    /// <summary>Gets the architecture-specific model type name.</summary>
    public string ModelType { get; }
    /// <summary>Gets the graph state ABI identifier.</summary>
    public string StateAbiId { get; }
    /// <summary>Gets the tensor dimensions in axis order.</summary>
    public IReadOnlyDictionary<string, int> Dimensions { get; }
    /// <summary>Gets attributes keyed by name.</summary>
    public IReadOnlyDictionary<string, string> Attributes { get; }

    /// <inheritdoc/>
    public bool Equals(GraphModelSignature? other) =>
        other is not null &&
        string.Equals(ModelType, other.ModelType, StringComparison.Ordinal) &&
        string.Equals(StateAbiId, other.StateAbiId, StringComparison.Ordinal) &&
        Dimensions.OrderBy(value => value.Key, StringComparer.Ordinal)
            .SequenceEqual(other.Dimensions.OrderBy(value => value.Key, StringComparer.Ordinal)) &&
        Attributes.OrderBy(value => value.Key, StringComparer.Ordinal)
            .SequenceEqual(other.Attributes.OrderBy(value => value.Key, StringComparer.Ordinal));

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as GraphModelSignature);

    /// <inheritdoc/>
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

/// <summary>Describes a tensor resource, its lifetime, and its external binding metadata.</summary>
/// <param name="Id">The graph resource identifier.</param>
/// <param name="Name">The name identifying this graph element.</param>
/// <param name="Kind">The resource role or execution definition category.</param>
/// <param name="Lifetime">The required backing-storage lifetime.</param>
/// <param name="Tensor">The tensor element type, dimensions, and layout.</param>
/// <param name="BindingKey">The external resource binding key, or null for an unbound resource.</param>
/// <param name="DeviceId">The optional device identifier for the backing storage.</param>
/// <param name="DeclaredScope">The explicit sharing scope, or null to infer it from the resource lifetime and kind.</param>
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
    /// <summary>Gets the declared sharing scope, or the scope inferred from resource lifetime and kind.</summary>
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

/// <summary>Describes a named region in the graph hierarchy and its architecture metadata.</summary>
/// <param name="Id">The graph region identifier.</param>
/// <param name="ParentId">The parent region identifier, or null for the graph root.</param>
/// <param name="Type">The graph region type or numeric element type.</param>
/// <param name="Role">The optional semantic role of the region.</param>
/// <param name="Name">The name identifying this graph element.</param>
/// <param name="Attributes">The operation or model attributes keyed by name.</param>
public sealed record GraphRegion(
    RegionId Id,
    RegionId? ParentId,
    string Type,
    string? Role,
    string Name,
    IReadOnlyDictionary<string, string> Attributes)
{
    /// <summary>Gets the optional region architecture metadata.</summary>
    public GraphRegionArchitecture? Architecture { get; init; }
}

/// <summary>Binds a node port to a resource or tensor view with an access contract.</summary>
/// <param name="Port">The node port name.</param>
/// <param name="Resource">The referenced graph resource identifier.</param>
/// <param name="Access">The permitted read and write access.</param>
/// <param name="InitializedBeforeRead">Whether resource contents are initialized before their first read.</param>
/// <param name="View">The optional contiguous tensor view within the resource.</param>
public sealed record NodeResourceBinding(
    string Port,
    ResourceId Resource,
    GraphResourceAccess Access,
    bool InitializedBeforeRead = false,
    GraphTensorView? View = null);

/// <summary>Describes a logical operation, its resource bindings, dependencies, and precision requirements.</summary>
/// <param name="Id">The logical node identifier.</param>
/// <param name="Operation">The operation contract identifier.</param>
/// <param name="Region">The region containing the node.</param>
/// <param name="Resources">The node port resource bindings.</param>
/// <param name="Dependencies">The predecessor node identifiers.</param>
/// <param name="Attributes">The operation or model attributes keyed by name.</param>
/// <param name="Requirements">The operation precision requirements.</param>
public sealed record LogicalNode(
    LogicalNodeId Id,
    GraphOperationId Operation,
    RegionId Region,
    IReadOnlyList<NodeResourceBinding> Resources,
    IReadOnlyList<LogicalNodeId> Dependencies,
    IReadOnlyDictionary<string, string> Attributes,
    PrecisionRequirement Requirements);

/// <summary>Exposes model identity, resources, graph inputs and outputs, and state bindings.</summary>
public interface IGraphBindingMetadata
{
    /// <summary>Gets the graph&apos;s architecture identifier, IR version, and name.</summary>
    GraphIdentity Identity { get; }
    /// <summary>Gets the model dimensions, attributes, and state ABI signature.</summary>
    GraphModelSignature Model { get; }
    /// <summary>Gets the graph resource descriptors.</summary>
    IReadOnlyList<GraphResource> Resources { get; }
    /// <summary>Gets the graph input resource identifiers.</summary>
    IReadOnlyList<ResourceId> Inputs { get; }
    /// <summary>Gets the graph output resource identifiers.</summary>
    IReadOnlyList<ResourceId> Outputs { get; }
    /// <summary>Gets the named state entries and their compatibility schema.</summary>
    GraphState GraphState { get; }
}

/// <summary>Contains and validates a backend-independent graph of tensor operations.</summary>
public sealed class LogicalGraph : IGraphBindingMetadata
{
    /// <summary>Creates and validates a logical graph with the supplied structure, resources, IO, and state.</summary>
    /// <param name="identity">The graph architecture, IR version, and name.</param>
    /// <param name="model">The model dimensions, attributes, and state ABI signature.</param>
    /// <param name="resources">The graph resource descriptors.</param>
    /// <param name="structure">The ordered graph structure and reusable definitions.</param>
    /// <param name="inputs">The graph input resource identifiers.</param>
    /// <param name="outputs">The graph output resource identifiers.</param>
    /// <param name="graphState">The optional named graph state resource entries.</param>
    public LogicalGraph(GraphIdentity identity, GraphModelSignature model, IEnumerable<GraphResource> resources,
        GraphStructure structure, IEnumerable<ResourceId> inputs, IEnumerable<ResourceId> outputs,
        IEnumerable<GraphStateEntry>? graphState = null)
    {
        Identity = identity ?? throw new ArgumentNullException(nameof(identity));
        Model = model ?? throw new ArgumentNullException(nameof(model));
        declaredResources = resources?.ToArray() ?? throw new ArgumentNullException(nameof(resources));
        Structure = structure ?? throw new ArgumentNullException(nameof(structure));
        if (Identity.IrVersion < 2 && structure.LayerDefinitions.Any(definition =>
            GraphStructure.Walk(definition.Body).OfType<GraphNodeElement>().Any(node => node.Node.Resources.Any(binding => binding.View is not null))))
            throw new InvalidDataException("Tensor views require graph IR version 2 or newer, including unused definitions.");
        var expanded = structure.Expand(DeclaredResources);
        Resources = expanded.Resources;
        Regions = expanded.Regions;
        Nodes = expanded.Nodes;
        Inputs = inputs?.ToArray() ?? throw new ArgumentNullException(nameof(inputs));
        Outputs = outputs?.ToArray() ?? throw new ArgumentNullException(nameof(outputs));
        var externalIds = DeclaredResources.Select(r => r.Id).ToHashSet();
        if (Inputs.Concat(Outputs).Any(id => !externalIds.Contains(id)))
            throw new InvalidDataException("Definition-local resources cannot be graph IO.");
        GraphState = CreateGraphState(Model, Resources, graphState);
        GraphValidator.Validate(this);
    }

    /// <summary>Gets the retained ordered graph structure, or null when only expanded nodes are available.</summary>
    public GraphStructure? Structure { get; }
    /// <summary>Gets resources declared outside reusable definitions, or all resources for an expanded graph.</summary>
    public IReadOnlyList<GraphResource> DeclaredResources => declaredResources ?? Resources;
    private readonly IReadOnlyList<GraphResource>? declaredResources;

    /// <summary>Creates and validates a logical graph with the supplied structure, resources, IO, and state.</summary>
    /// <param name="identity">The graph architecture, IR version, and name.</param>
    /// <param name="model">The model dimensions, attributes, and state ABI signature.</param>
    /// <param name="resources">The graph resource descriptors.</param>
    /// <param name="regions">The graph regions in hierarchy order.</param>
    /// <param name="nodes">The graph or plan operation nodes.</param>
    /// <param name="inputs">The graph input resource identifiers.</param>
    /// <param name="outputs">The graph output resource identifiers.</param>
    /// <param name="graphState">The optional named graph state resource entries.</param>
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

    /// <summary>Gets the graph&apos;s architecture identifier, IR version, and name.</summary>
    public GraphIdentity Identity { get; }
    /// <summary>Gets the model dimensions, attributes, and state ABI signature.</summary>
    public GraphModelSignature Model { get; }
    /// <summary>Gets the graph resource descriptors.</summary>
    public IReadOnlyList<GraphResource> Resources { get; }
    /// <summary>Gets the graph regions and their hierarchy metadata.</summary>
    public IReadOnlyList<GraphRegion> Regions { get; }
    /// <summary>Gets the operation nodes in execution order.</summary>
    public IReadOnlyList<LogicalNode> Nodes { get; }
    /// <summary>Gets the graph input resource identifiers.</summary>
    public IReadOnlyList<ResourceId> Inputs { get; }
    /// <summary>Gets the graph output resource identifiers.</summary>
    public IReadOnlyList<ResourceId> Outputs { get; }
    /// <summary>Gets the named state entries and their compatibility schema.</summary>
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

/// <summary>Builds logical graphs for a model architecture from a tensor catalog.</summary>
public interface ILogicalGraphProvider
{
    /// <summary>Gets the model architecture identifier.</summary>
    string ArchitectureId { get; }
    /// <summary>Builds a logical graph for the supplied model tensor catalog.</summary>
    /// <param name="tensors">The model tensor catalog.</param>
    LogicalGraph Build(IModelTensorCatalog tensors);
}

/// <summary>Tracks the logical nodes and fusion rule that produced an execution node.</summary>
/// <param name="LogicalNodes">The logical nodes represented by the execution node.</param>
/// <param name="AppliedRuleId">The applied fusion rule identifier, or null when no rule was applied.</param>
public sealed record ExecutionSourceMap(
    IReadOnlyList<LogicalNodeId> LogicalNodes,
    string? AppliedRuleId)
{
    /// <summary>Gets a source map without logical-node provenance for plans loaded only from XML.</summary>
    public static ExecutionSourceMap XmlOnly { get; } = new([], null);
}

/// <summary>Describes a scheduled operation and its logical provenance and private resources.</summary>
/// <param name="Id">The execution node identifier.</param>
/// <param name="Operation">The operation contract identifier.</param>
/// <param name="Region">The region containing the node.</param>
/// <param name="Resources">The node port resource bindings.</param>
/// <param name="Dependencies">The predecessor node identifiers.</param>
/// <param name="Attributes">The operation or model attributes keyed by name.</param>
/// <param name="Requirements">The operation precision requirements.</param>
/// <param name="Source">The logical-node provenance and applied fusion rule.</param>
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
    /// <summary>Gets resource bindings private to this execution node.</summary>
    public IReadOnlyList<NodeResourceBinding> InternalResources { get; init; } = [];
    /// <summary>Gets resources whose contents are initialized by their first write in this node.</summary>
    public IReadOnlyList<ResourceId> FirstWriteResources { get; init; } = [];
}

/// <summary>Exposes the nodes, regions, and private resource ownership needed for execution planning.</summary>
public interface IExecutionPlanningGraph : IGraphBindingMetadata
{
    /// <summary>Gets the graph regions and their hierarchy metadata.</summary>
    IReadOnlyList<GraphRegion> Regions { get; }
    /// <summary>Gets the operation nodes in execution order.</summary>
    IReadOnlyList<ExecutionNode> Nodes { get; }
    /// <summary>Gets the execution node owning each private resource.</summary>
    IReadOnlyDictionary<ResourceId, ExecutionNodeId> InternalResourceOwners { get; }
}

/// <summary>Contains and validates an execution graph with explicit dependencies and private resource ownership.</summary>
public sealed class ExecutionGraph : IExecutionPlanningGraph
{
    /// <summary>Creates and validates an execution graph and indexes private resource ownership.</summary>
    /// <param name="identity">The graph architecture, IR version, and name.</param>
    /// <param name="model">The model dimensions, attributes, and state ABI signature.</param>
    /// <param name="resources">The graph resource descriptors.</param>
    /// <param name="regions">The graph regions in hierarchy order.</param>
    /// <param name="nodes">The graph or plan operation nodes.</param>
    /// <param name="inputs">The graph input resource identifiers.</param>
    /// <param name="outputs">The graph output resource identifiers.</param>
    /// <param name="graphState">The optional named graph state resource entries.</param>
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

    /// <summary>Gets the graph&apos;s architecture identifier, IR version, and name.</summary>
    public GraphIdentity Identity { get; }
    /// <summary>Gets the model dimensions, attributes, and state ABI signature.</summary>
    public GraphModelSignature Model { get; }
    /// <summary>Gets the graph resource descriptors.</summary>
    public IReadOnlyList<GraphResource> Resources { get; }
    /// <summary>Gets the graph regions and their hierarchy metadata.</summary>
    public IReadOnlyList<GraphRegion> Regions { get; }
    /// <summary>Gets the operation nodes in execution order.</summary>
    public IReadOnlyList<ExecutionNode> Nodes { get; }
    /// <summary>Gets the graph input resource identifiers.</summary>
    public IReadOnlyList<ResourceId> Inputs { get; }
    /// <summary>Gets the graph output resource identifiers.</summary>
    public IReadOnlyList<ResourceId> Outputs { get; }
    /// <summary>Gets the named state entries and their compatibility schema.</summary>
    public GraphState GraphState { get; }
    /// <summary>Gets the execution node owning each private resource.</summary>
    public IReadOnlyDictionary<ResourceId, ExecutionNodeId> InternalResourceOwners { get; }

    /// <summary>Determines whether compatible private resources have dependency-ordered owners and can share backing storage.</summary>
    /// <param name="first">The first private resource to compare.</param>
    /// <param name="second">The second private resource to compare.</param>
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
