using System.Collections.ObjectModel;

namespace SharpInference.Graphs;

/// <summary>Controls whether a graph definition creates an execution completion boundary when compiled.</summary>
public enum GraphCompileBoundary
{
    /// <summary>The compiler selects the safest applicable boundary.</summary>
    Auto,
    /// <summary>The compiler preserves one completion boundary for the entire graph call.</summary>
    Atomic,
    /// <summary>The compiler preserves architecture metadata but removes the graph call as an execution boundary.</summary>
    Transparent,
}

/// <summary>Describes a graph definition or call for architecture presentation.</summary>
/// <param name="Name">The optional display name.</param>
/// <param name="Role">The optional architecture role.</param>
/// <param name="Description">The optional explanatory text.</param>
/// <param name="Attributes">Additional presentation metadata.</param>
public sealed record GraphAnnotation(
    string? Name = null,
    string? Role = null,
    string? Description = null,
    IReadOnlyDictionary<string, string>? Attributes = null);

/// <summary>Identifies one reusable graph definition.</summary>
/// <param name="Value">The definition identifier.</param>
public readonly record struct GraphDefinitionId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Identifies one node within a graph definition.</summary>
/// <param name="Value">The node identifier.</param>
public readonly record struct GraphDefinitionNodeId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Identifies the graph instance that owns a resource used by a call binding.</summary>
/// <param name="Call">The owning call in the current definition, or null for the parent instance.</param>
public readonly record struct GraphBindingOwner(GraphDefinitionNodeId? Call)
{
    /// <summary>References a resource owned by the current parent graph instance.</summary>
    public static GraphBindingOwner Parent => new(null);

    /// <summary>Creates a reference to a direct child graph-call instance.</summary>
    public static GraphBindingOwner Instance(GraphDefinitionNodeId call) => new(call);

    /// <summary>Gets whether this owner is the current parent graph instance.</summary>
    public bool IsParent => Call is null;

    /// <inheritdoc/>
    public override string ToString() => Call?.Value ?? "parent";
}

/// <summary>Specifies the role of a resource declared by a graph definition.</summary>
public enum GraphDefinitionResourceKind
{
    /// <summary>A required graph input.</summary>
    Input,
    /// <summary>A graph result.</summary>
    Output,
    /// <summary>A graph input that is returned after possible updates.</summary>
    InOut,
    /// <summary>An internal logical tensor.</summary>
    Temporary,
    /// <summary>A model weight.</summary>
    Weight,
    /// <summary>Persistent model-session state.</summary>
    State,
    /// <summary>An immutable constant.</summary>
    Constant,
}

/// <summary>Describes one symbolic resource owned or referenced by a graph definition.</summary>
/// <param name="Id">The resource identifier.</param>
/// <param name="Name">The display name.</param>
/// <param name="Kind">The logical resource role.</param>
/// <param name="Tensor">The tensor contract.</param>
/// <param name="Owner">The graph definition that declared the resource.</param>
/// <param name="BindingKey">The optional external binding key.</param>
public sealed record GraphDefinitionResource(
    ResourceId Id,
    string Name,
    GraphDefinitionResourceKind Kind,
    TensorDescriptor Tensor,
    GraphDefinitionId Owner,
    string? BindingKey = null);

/// <summary>A strongly typed handle to one logical graph resource.</summary>
/// <typeparam name="T">The user-defined tensor contract marker.</typeparam>
public class GraphResource<T>
{
    internal GraphResource(GraphDefinitionResource resource)
    {
        Resource = resource;
    }

    /// <summary>Gets the untyped logical resource definition.</summary>
    public GraphDefinitionResource Resource { get; }

    /// <summary>Gets the logical resource identifier.</summary>
    public ResourceId Id => Resource.Id;

    /// <summary>Gets the tensor contract.</summary>
    public TensorDescriptor Tensor => Resource.Tensor;
}

/// <summary>A strongly typed graph input port.</summary>
public sealed class GraphInput<T> : GraphResource<T>
{
    internal GraphInput(GraphDefinitionResource resource) : base(resource) { }
}

/// <summary>A strongly typed graph output port.</summary>
public sealed class GraphOutput<T> : GraphResource<T>
{
    internal GraphOutput(GraphDefinitionResource resource) : base(resource) { }
}

/// <summary>A strongly typed graph read-write port.</summary>
public sealed class GraphInOut<T> : GraphResource<T>
{
    internal GraphInOut(GraphDefinitionResource resource) : base(resource) { }
}

/// <summary>Creates the strongly typed input and output contract of a reusable graph definition.</summary>
/// <typeparam name="TSelf">The concrete contract type.</typeparam>
public interface IGraphPortContract<TSelf> where TSelf : IGraphPortContract<TSelf>
{
    /// <summary>Creates a contract using resources supplied by the port factory.</summary>
    static abstract TSelf Create(IGraphPortFactory ports);
}

/// <summary>Creates strongly typed graph input and output ports.</summary>
public interface IGraphPortFactory
{
    /// <summary>Creates or resolves an input port.</summary>
    GraphInput<T> AddInput<T>(string id, TensorDescriptor tensor);

    /// <summary>Creates or resolves an output port.</summary>
    GraphOutput<T> AddOutput<T>(string id, TensorDescriptor tensor);

    /// <summary>Creates or resolves a read-write port.</summary>
    GraphInOut<T> AddInOut<T>(string id, TensorDescriptor tensor);
}

/// <summary>Describes the severity of a logical graph diagnostic.</summary>
public enum LogicalGraphDiagnosticSeverity
{
    /// <summary>The graph is valid but may be surprising or nondeterministic.</summary>
    Warning,
    /// <summary>The graph definition is invalid.</summary>
    Error,
}

/// <summary>Reports a logical graph validation or architecture concern.</summary>
/// <param name="Severity">The diagnostic severity.</param>
/// <param name="Code">The stable diagnostic code.</param>
/// <param name="Message">The diagnostic text.</param>
/// <param name="Node">The optional related node.</param>
/// <param name="Resource">The optional related resource.</param>
public sealed record LogicalGraphDiagnostic(
    LogicalGraphDiagnosticSeverity Severity,
    string Code,
    string Message,
    GraphDefinitionNodeId? Node = null,
    ResourceId? Resource = null);

/// <summary>Binds one operation or graph-call port to a logical resource.</summary>
/// <param name="Port">The node port.</param>
/// <param name="Resource">The logical resource.</param>
/// <param name="Access">The required access.</param>
public sealed record GraphDefinitionBinding(
    string Port,
    ResourceId Resource,
    GraphResourceAccess Access,
    GraphBindingOwner Instance = default)
{
    /// <summary>Gets the normalized owner, treating the default value as the parent instance.</summary>
    public GraphBindingOwner ResourceOwner => Instance.IsParent ? GraphBindingOwner.Parent : Instance;
}

/// <summary>Base model for nodes contained by a graph definition.</summary>
/// <param name="Id">The node identifier within the definition.</param>
/// <param name="Bindings">The resource bindings.</param>
/// <param name="Dependencies">Explicit predecessor nodes.</param>
/// <param name="Annotation">Optional architecture metadata.</param>
public abstract record GraphDefinitionNode(
    GraphDefinitionNodeId Id,
    IReadOnlyList<GraphDefinitionBinding> Bindings,
    IReadOnlyList<GraphDefinitionNodeId> Dependencies,
    GraphAnnotation? Annotation);

/// <summary>One directly executable operation in a graph definition.</summary>
/// <param name="Id">The node identifier.</param>
/// <param name="Operation">The operation contract.</param>
/// <param name="Bindings">The operation resource bindings.</param>
/// <param name="Dependencies">Explicit predecessor nodes.</param>
/// <param name="Attributes">Operation attributes.</param>
/// <param name="Requirements">Precision requirements.</param>
/// <param name="Annotation">Optional architecture metadata.</param>
public sealed record GraphOperationNode(
    GraphDefinitionNodeId Id,
    GraphOperationId Operation,
    IReadOnlyList<GraphDefinitionBinding> Bindings,
    IReadOnlyList<GraphDefinitionNodeId> Dependencies,
    IReadOnlyDictionary<string, string> Attributes,
    PrecisionRequirement Requirements,
    GraphAnnotation? Annotation = null)
    : GraphDefinitionNode(Id, Bindings, Dependencies, Annotation);

/// <summary>One call to another reusable graph definition.</summary>
/// <param name="Id">The call identifier.</param>
/// <param name="Definition">The called definition.</param>
/// <param name="Bindings">Bindings from called ports to caller resources.</param>
/// <param name="Dependencies">Explicit predecessor nodes.</param>
/// <param name="Annotation">Optional architecture metadata for this call.</param>
public sealed record GraphCallNode(
    GraphDefinitionNodeId Id,
    GraphDefinitionId Definition,
    IReadOnlyList<GraphDefinitionBinding> Bindings,
    IReadOnlyList<GraphDefinitionNodeId> Dependencies,
    GraphAnnotation? Annotation = null)
    : GraphDefinitionNode(Id, Bindings, Dependencies, Annotation);

/// <summary>An immutable reusable logical graph definition without execution or storage assignments.</summary>
public sealed class LogicalGraphDefinition
{
    internal LogicalGraphDefinition(
        GraphDefinitionId id,
        GraphCompileBoundary compileBoundary,
        GraphAnnotation? annotation,
        IEnumerable<GraphDefinitionResource> resources,
        IEnumerable<GraphDefinitionNode> nodes,
        IEnumerable<LogicalGraphDefinition> definitions)
    {
        Id = id;
        CompileBoundary = compileBoundary;
        Annotation = annotation;
        Resources = Array.AsReadOnly(resources.ToArray());
        Nodes = Array.AsReadOnly(nodes.ToArray());
        Definitions = Array.AsReadOnly(definitions.ToArray());
        var analysis = LogicalGraphDefinitionValidator.Analyze(this);
        CompletionSinks = Array.AsReadOnly(analysis.CompletionSinks.ToArray());
        TerminalWriters = new ReadOnlyDictionary<ResourceId, IReadOnlyList<GraphDefinitionNodeId>>(
            analysis.TerminalWriters.ToDictionary(
                pair => pair.Key,
                pair => (IReadOnlyList<GraphDefinitionNodeId>)Array.AsReadOnly(pair.Value.ToArray())));
        Diagnostics = Array.AsReadOnly(analysis.Diagnostics.ToArray());
        var errors = Diagnostics.Where(diagnostic =>
            diagnostic.Severity == LogicalGraphDiagnosticSeverity.Error).ToArray();
        if (errors.Length != 0)
            throw new InvalidDataException(string.Join(Environment.NewLine, errors.Select(error => error.Message)));
    }

    /// <summary>Gets the definition identifier.</summary>
    public GraphDefinitionId Id { get; }
    /// <summary>Gets the requested compile boundary.</summary>
    public GraphCompileBoundary CompileBoundary { get; }
    /// <summary>Gets optional architecture metadata.</summary>
    public GraphAnnotation? Annotation { get; }
    /// <summary>Gets the resources declared or referenced by this definition.</summary>
    public IReadOnlyList<GraphDefinitionResource> Resources { get; }
    /// <summary>Gets operation and graph-call nodes.</summary>
    public IReadOnlyList<GraphDefinitionNode> Nodes { get; }
    /// <summary>Gets directly called graph definitions.</summary>
    public IReadOnlyList<LogicalGraphDefinition> Definitions { get; }
    /// <summary>Gets nodes with no successor in this definition.</summary>
    public IReadOnlyList<GraphDefinitionNodeId> CompletionSinks { get; }
    /// <summary>Gets maximal writers for every graph output.</summary>
    public IReadOnlyDictionary<ResourceId, IReadOnlyList<GraphDefinitionNodeId>> TerminalWriters { get; }
    /// <summary>Gets non-fatal and fatal analysis diagnostics.</summary>
    public IReadOnlyList<LogicalGraphDiagnostic> Diagnostics { get; }

    /// <summary>Creates a strongly typed reusable graph definition.</summary>
    public static GraphDefinition<TContract> Define<TContract>(
        string id,
        Action<GraphDefinitionBuilder<TContract>> build,
        GraphCompileBoundary compileBoundary = GraphCompileBoundary.Auto,
        GraphAnnotation? annotation = null)
        where TContract : IGraphPortContract<TContract>
    {
        ArgumentNullException.ThrowIfNull(build);
        var definitionId = new GraphDefinitionId(RequireId(id, nameof(id)));
        var ports = new DefinitionPortFactory(definitionId);
        var contract = TContract.Create(ports);
        var builder = new GraphDefinitionBuilder<TContract>(
            definitionId, contract, ports, compileBoundary, annotation);
        build(builder);
        return builder.Build();
    }

    /// <summary>Creates a single-input, single-output graph whose operations form one continuous chain.</summary>
    public static GraphDefinition<TContract> CreateListGraph<TContract>(
        string id,
        IEnumerable<GraphOperationId> operations,
        GraphCompileBoundary compileBoundary = GraphCompileBoundary.Auto,
        GraphAnnotation? annotation = null)
        where TContract : IGraphPortContract<TContract>
    {
        ArgumentNullException.ThrowIfNull(operations);
        var definitionId = new GraphDefinitionId(RequireId(id, nameof(id)));
        var ports = new DefinitionPortFactory(definitionId);
        var contract = TContract.Create(ports);
        var input = ports.Resources.SingleOrDefault(resource =>
            resource.Kind == GraphDefinitionResourceKind.Input)
            ?? throw new InvalidOperationException("A list graph requires exactly one input.");
        if (ports.Resources.Count(resource => resource.Kind == GraphDefinitionResourceKind.Input) != 1)
            throw new InvalidOperationException("A list graph requires exactly one input.");
        var output = ports.Resources.SingleOrDefault(resource =>
            resource.Kind == GraphDefinitionResourceKind.Output)
            ?? throw new InvalidOperationException("A list graph requires exactly one output.");
        if (ports.Resources.Count(resource => resource.Kind == GraphDefinitionResourceKind.Output) != 1)
            throw new InvalidOperationException("A list graph requires exactly one output.");
        var values = operations.ToArray();
        if (values.Length == 0)
            throw new ArgumentException("A list graph requires at least one operation.", nameof(operations));

        var resources = ports.Resources.ToList();
        var nodes = new List<GraphDefinitionNode>();
        var source = input;
        for (var index = 0; index < values.Length; index++)
        {
            var target = index == values.Length - 1
                ? output
                : new GraphDefinitionResource(
                    new ResourceId($"step-{index}-output"),
                    $"Step {index} output",
                    GraphDefinitionResourceKind.Temporary,
                    output.Tensor,
                    definitionId);
            if (index != values.Length - 1) resources.Add(target);
            var nodeId = new GraphDefinitionNodeId($"step-{index}");
            nodes.Add(new GraphOperationNode(
                nodeId,
                values[index],
                [
                    new GraphDefinitionBinding("input", source.Id, GraphResourceAccess.Read),
                    new GraphDefinitionBinding("output", target.Id, GraphResourceAccess.Write),
                ],
                index == 0 ? [] : [nodes[^1].Id],
                new Dictionary<string, string>(),
                new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32)));
            source = target;
        }
        var definition = new LogicalGraphDefinition(
            definitionId, compileBoundary, annotation, resources, nodes, []);
        return new GraphDefinition<TContract>(definition, contract);
    }

    internal static string RequireId(string value, string parameterName) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException("An identifier is required.", parameterName)
            : value;
}

/// <summary>A strongly typed reusable graph definition.</summary>
public sealed class GraphDefinition<TContract> where TContract : IGraphPortContract<TContract>
{
    internal GraphDefinition(LogicalGraphDefinition definition, TContract ports)
    {
        Definition = definition;
        Ports = ports;
    }

    /// <summary>Gets the serializable untyped definition.</summary>
    public LogicalGraphDefinition Definition { get; }
    /// <summary>Gets the strongly typed symbolic ports.</summary>
    public TContract Ports { get; }
}

/// <summary>Represents one graph-call instance and its strongly typed public ports.</summary>
public sealed class GraphCallInstance<TContract> where TContract : IGraphPortContract<TContract>
{
    private readonly LogicalGraphDefinition definition;

    internal GraphCallInstance(
        GraphDefinitionNodeId id,
        LogicalGraphDefinition definition,
        TContract ports)
    {
        Id = id;
        this.definition = definition;
        Ports = ports;
    }

    /// <summary>Gets the call identifier in the containing definition.</summary>
    public GraphDefinitionNodeId Id { get; }

    /// <summary>Gets the called graph's strongly typed public ports.</summary>
    public TContract Ports { get; }

    internal LogicalGraphDefinition Definition => definition;

    /// <summary>Gets a strongly typed reference to a resource declared by this instance's definition.</summary>
    public GraphResource<T> Resource<T>(string id)
    {
        var resourceId = new ResourceId(LogicalGraphDefinition.RequireId(id, nameof(id)));
        var resource = definition.Resources.SingleOrDefault(resource => resource.Id == resourceId)
            ?? throw new ArgumentException(
                $"Graph definition '{definition.Id}' does not declare resource '{id}'.",
                nameof(id));
        return new GraphResource<T>(resource);
    }
}

/// <summary>Builds one reusable graph definition.</summary>
public sealed class GraphDefinitionBuilder<TContract> where TContract : IGraphPortContract<TContract>
{
    private readonly GraphDefinitionId id;
    private readonly DefinitionPortFactory ports;
    private readonly GraphCompileBoundary compileBoundary;
    private readonly GraphAnnotation? annotation;
    private readonly List<GraphDefinitionResource> resources;
    private readonly List<GraphDefinitionNode> nodes = [];
    private readonly Dictionary<GraphDefinitionId, LogicalGraphDefinition> definitions = [];

    internal GraphDefinitionBuilder(
        GraphDefinitionId id,
        TContract contract,
        DefinitionPortFactory ports,
        GraphCompileBoundary compileBoundary,
        GraphAnnotation? annotation)
    {
        this.id = id;
        this.ports = ports;
        this.compileBoundary = compileBoundary;
        this.annotation = annotation;
        Ports = contract;
        resources = [.. ports.Resources];
    }

    /// <summary>Gets the definition's strongly typed ports.</summary>
    public TContract Ports { get; }

    /// <summary>Declares an internal logical resource.</summary>
    public GraphResource<T> AddResource<T>(
        string id,
        string name,
        TensorDescriptor tensor,
        GraphDefinitionResourceKind kind = GraphDefinitionResourceKind.Temporary,
        string? bindingKey = null)
    {
        if (kind is GraphDefinitionResourceKind.Input or GraphDefinitionResourceKind.Output)
            throw new ArgumentException("Use the graph port contract to declare inputs and outputs.", nameof(kind));
        var resource = new GraphDefinitionResource(
            new ResourceId(LogicalGraphDefinition.RequireId(id, nameof(id))),
            string.IsNullOrWhiteSpace(name) ? throw new ArgumentException("A resource name is required.", nameof(name)) : name,
            kind,
            tensor ?? throw new ArgumentNullException(nameof(tensor)),
            this.id,
            bindingKey);
        resources.Add(resource);
        return new GraphResource<T>(resource);
    }

    /// <summary>Adds and configures one operation node.</summary>
    public TResult AddNode<TResult>(
        string id,
        GraphOperationId operation,
        Func<GraphOperationNodeBuilder, TResult> configure,
        GraphAnnotation? annotation = null)
    {
        ArgumentNullException.ThrowIfNull(configure);
        var nodeId = new GraphDefinitionNodeId(LogicalGraphDefinition.RequireId(id, nameof(id)));
        var builder = new GraphOperationNodeBuilder(this.id, nodeId);
        var result = configure(builder);
        foreach (var resource in builder.Resources)
            AddReferencedResource(resource);
        nodes.Add(builder.Build(operation, annotation));
        return result;
    }

    /// <summary>Adds one operation node without returning a resource.</summary>
    public void AddNode(
        string id,
        GraphOperationId operation,
        Action<GraphOperationNodeBuilder> configure,
        GraphAnnotation? annotation = null) =>
        AddNode(id, operation, builder =>
        {
            configure(builder);
            return true;
        }, annotation);

    /// <summary>Adds one call to a reusable child graph and returns strongly typed instance ports.</summary>
    public TChildContract AddNode<TChildContract>(
        string id,
        GraphDefinition<TChildContract> definition,
        Action<GraphCallBindings<TChildContract>> bind,
        GraphAnnotation? annotation = null)
        where TChildContract : IGraphPortContract<TChildContract>
        => AddCall(id, definition, bind, annotation).Ports;

    /// <summary>Adds one call and returns its instance identity and strongly typed public ports.</summary>
    public GraphCallInstance<TChildContract> AddCall<TChildContract>(
        string id,
        GraphDefinition<TChildContract> definition,
        Action<GraphCallBindings<TChildContract>> bind,
        GraphAnnotation? annotation = null)
        where TChildContract : IGraphPortContract<TChildContract>
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentNullException.ThrowIfNull(bind);
        var nodeId = new GraphDefinitionNodeId(LogicalGraphDefinition.RequireId(id, nameof(id)));
        var bindings = new GraphCallBindings<TChildContract>(this.id, definition.Ports);
        bind(bindings);
        var inputs = bindings.BuildInputs(definition.Definition);
        foreach (var resource in bindings.Resources)
            AddReferencedResource(resource);
        var instancePorts = new InstancePortFactory(
            this.id, nodeId, definition.Definition, inputs, bindings.Outputs);
        var result = TChildContract.Create(instancePorts);
        foreach (var resource in instancePorts.Resources)
            AddReferencedResource(resource);
        var values = inputs.Concat(instancePorts.OutputBindings).ToArray();
        nodes.Add(new GraphCallNode(nodeId, definition.Definition.Id, values, [], annotation));
        definitions.TryAdd(definition.Definition.Id, definition.Definition);
        return new GraphCallInstance<TChildContract>(nodeId, definition.Definition, result);
    }

    /// <summary>Adds a dynamically typed call while preserving the child definition.</summary>
    public GraphDefinitionNodeId AddCall(
        string id,
        LogicalGraphDefinition definition,
        IEnumerable<(string Port, GraphResource<object> Resource, GraphResourceAccess Access)> bindings,
        IEnumerable<GraphDefinitionNodeId>? dependencies = null,
        GraphAnnotation? annotation = null)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentNullException.ThrowIfNull(bindings);
        var nodeId = new GraphDefinitionNodeId(LogicalGraphDefinition.RequireId(id, nameof(id)));
        var ports = definition.Resources
            .Where(resource => resource.Kind is GraphDefinitionResourceKind.Input
                or GraphDefinitionResourceKind.Output
                or GraphDefinitionResourceKind.InOut)
            .ToDictionary(resource => resource.Id.Value, StringComparer.Ordinal);
        var values = bindings.Select(binding =>
        {
            if (!ports.TryGetValue(binding.Port, out var port))
                throw new InvalidOperationException(
                    $"Definition '{definition.Id}' has no public port '{binding.Port}'.");
            if (port.Tensor != binding.Resource.Resource.Tensor)
                throw new InvalidOperationException(
                    $"Call '{nodeId}' binds incompatible resource '{binding.Resource.Resource.Id}' to '{binding.Port}'.");
            var requiredAccess = port.Kind switch
            {
                GraphDefinitionResourceKind.Input => GraphResourceAccess.Read,
                GraphDefinitionResourceKind.Output => GraphResourceAccess.Write,
                GraphDefinitionResourceKind.InOut => GraphResourceAccess.ReadWrite,
                _ => throw new InvalidDataException(),
            };
            if (binding.Access != requiredAccess)
                throw new InvalidOperationException(
                    $"Call '{nodeId}' port '{binding.Port}' requires {requiredAccess} access.");
            AddReferencedResource(binding.Resource.Resource);
            return new GraphDefinitionBinding(
                binding.Port,
                binding.Resource.Resource.Id,
                binding.Access);
        }).ToArray();
        if (values.Select(binding => binding.Port).Distinct(StringComparer.Ordinal).Count() != ports.Count ||
            values.Length != ports.Count)
            throw new InvalidOperationException(
                $"Call '{nodeId}' must bind every public port of definition '{definition.Id}' exactly once.");
        nodes.Add(new GraphCallNode(
            nodeId,
            definition.Id,
            values,
            dependencies?.ToArray() ?? [],
            annotation));
        definitions.TryAdd(definition.Id, definition);
        return nodeId;
    }

    internal GraphDefinition<TContract> Build()
    {
        foreach (var port in ports.Resources)
            if (!resources.Any(resource => resource.Id == port.Id))
                resources.Add(port);
        var inferred = LogicalGraphDefinitionValidator.InferDependencies(nodes);
        var definition = new LogicalGraphDefinition(
            id, compileBoundary, annotation, resources, inferred, definitions.Values);
        return new GraphDefinition<TContract>(definition, Ports);
    }

    private void AddReferencedResource(GraphDefinitionResource resource)
    {
        var existing = resources.SingleOrDefault(value => value.Id == resource.Id);
        if (existing is null)
        {
            resources.Add(resource);
            return;
        }
        if (existing != resource)
            throw new InvalidOperationException($"Resource '{resource.Id}' has conflicting declarations.");
    }
}

/// <summary>Configures one operation node.</summary>
public sealed class GraphOperationNodeBuilder
{
    private readonly GraphDefinitionId owner;
    private readonly GraphDefinitionNodeId id;
    private readonly List<GraphDefinitionBinding> bindings = [];
    private readonly List<GraphDefinitionNodeId> dependencies = [];
    private readonly Dictionary<string, string> attributes = new(StringComparer.Ordinal);
    private readonly Dictionary<ResourceId, GraphDefinitionResource> resources = [];
    private PrecisionRequirement? requirements;

    internal GraphOperationNodeBuilder(GraphDefinitionId owner, GraphDefinitionNodeId id)
    {
        this.owner = owner;
        this.id = id;
    }

    internal IReadOnlyCollection<GraphDefinitionResource> Resources => resources.Values;

    /// <summary>Adds a read-only input binding.</summary>
    public void AddInput<T>(string port, GraphResource<T> resource) =>
        AddBinding(port, resource, GraphResourceAccess.Read);

    /// <summary>Adds a write-only output binding and returns the same strongly typed resource.</summary>
    public GraphResource<T> AddOutput<T>(string port, GraphResource<T> resource)
    {
        AddBinding(port, resource, GraphResourceAccess.Write);
        return resource;
    }

    /// <summary>Adds a read-write binding and returns the same strongly typed resource.</summary>
    public GraphResource<T> AddInOut<T>(string port, GraphResource<T> resource)
    {
        AddBinding(port, resource, GraphResourceAccess.ReadWrite);
        return resource;
    }

    /// <summary>Adds an explicit predecessor.</summary>
    public void DependsOn(GraphDefinitionNodeId dependency) => dependencies.Add(dependency);

    /// <summary>Adds an operation attribute.</summary>
    public void AddAttribute(string name, string value) =>
        attributes.Add(
            LogicalGraphDefinition.RequireId(name, nameof(name)),
            value ?? throw new ArgumentNullException(nameof(value)));

    /// <summary>Overrides inferred operation precision requirements.</summary>
    public void RequirePrecision(PrecisionRequirement value) =>
        requirements = value ?? throw new ArgumentNullException(nameof(value));

    internal GraphOperationNode Build(GraphOperationId operation, GraphAnnotation? annotation)
    {
        var inferred = requirements ?? new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32);
        return new GraphOperationNode(id, operation, bindings.ToArray(), dependencies.Distinct().ToArray(),
            new ReadOnlyDictionary<string, string>(attributes), inferred, annotation);
    }

    private void AddBinding<T>(string port, GraphResource<T> resource, GraphResourceAccess access)
    {
        ArgumentNullException.ThrowIfNull(resource);
        if (resource.Resource.Owner != owner)
            throw new ArgumentException(
                $"Operation '{id}' cannot directly reference resource '{resource.Id}' owned by graph " +
                $"'{resource.Resource.Owner}'. Bind graph instances at their containing call instead.",
                nameof(resource));
        var definition = resource.Resource;
        bindings.Add(new GraphDefinitionBinding(
            LogicalGraphDefinition.RequireId(port, nameof(port)), definition.Id, access));
        resources.TryAdd(definition.Id, definition);
    }
}

internal sealed record GraphCallBoundResource(
    GraphDefinitionResource Resource,
    GraphBindingOwner Owner);

/// <summary>Strongly typed input bindings for one child graph call.</summary>
public sealed class GraphCallBindings<TContract> where TContract : IGraphPortContract<TContract>
{
    private readonly GraphDefinitionId owner;
    private readonly Dictionary<ResourceId, GraphCallBoundResource> values = [];
    private readonly Dictionary<ResourceId, GraphCallBoundResource> outputs = [];

    internal GraphCallBindings(GraphDefinitionId owner, TContract ports)
    {
        this.owner = owner;
        Ports = ports;
    }

    /// <summary>Gets the called graph's strongly typed symbolic ports.</summary>
    public TContract Ports { get; }

    internal IEnumerable<GraphDefinitionResource> Resources =>
        values.Values.Concat(outputs.Values)
            .Where(value => value.Owner.IsParent)
            .Select(value => value.Resource)
            .DistinctBy(resource => resource.Id);
    internal IReadOnlyDictionary<ResourceId, GraphCallBoundResource> Outputs => outputs;

    /// <summary>Binds one called input port to a caller resource.</summary>
    public void Bind<T>(GraphInput<T> port, GraphResource<T> resource)
        => Bind(port, resource, GraphBindingOwner.Parent);

    /// <summary>Binds one called input port to a resource owned by another graph-call instance.</summary>
    public void Bind<TOwnerContract, T>(
        GraphInput<T> port,
        GraphResource<T> resource,
        GraphCallInstance<TOwnerContract> instance)
        where TOwnerContract : IGraphPortContract<TOwnerContract>
    {
        ArgumentNullException.ThrowIfNull(instance);
        RequireInstanceResource(resource, instance);
        Bind(port, resource, GraphBindingOwner.Instance(instance.Id));
    }

    private void Bind<T>(
        GraphInput<T> port,
        GraphResource<T> resource,
        GraphBindingOwner resourceOwner)
    {
        ArgumentNullException.ThrowIfNull(port);
        ArgumentNullException.ThrowIfNull(resource);
        if (!SameTensor(port.Tensor, resource.Tensor))
            throw new ArgumentException($"Resource '{resource.Id}' is incompatible with input '{port.Id}'.", nameof(resource));
        if (!values.TryAdd(port.Id, Resolve(resource.Resource, resourceOwner)))
            throw new InvalidOperationException($"Input '{port.Id}' is already bound.");
    }

    /// <summary>Binds one called read-write port to a caller resource.</summary>
    public void Bind<T>(GraphInOut<T> port, GraphResource<T> resource)
        => Bind(port, resource, GraphBindingOwner.Parent);

    /// <summary>Binds one called read-write port to a resource owned by another graph-call instance.</summary>
    public void Bind<TOwnerContract, T>(
        GraphInOut<T> port,
        GraphResource<T> resource,
        GraphCallInstance<TOwnerContract> instance)
        where TOwnerContract : IGraphPortContract<TOwnerContract>
    {
        ArgumentNullException.ThrowIfNull(instance);
        RequireInstanceResource(resource, instance);
        Bind(port, resource, GraphBindingOwner.Instance(instance.Id));
    }

    private void Bind<T>(
        GraphInOut<T> port,
        GraphResource<T> resource,
        GraphBindingOwner resourceOwner)
    {
        ArgumentNullException.ThrowIfNull(port);
        ArgumentNullException.ThrowIfNull(resource);
        RequireCompatible(port, resource);
        var normalized = Resolve(resource.Resource, resourceOwner);
        if (!values.TryAdd(port.Id, normalized) ||
            !outputs.TryAdd(port.Id, normalized))
            throw new InvalidOperationException($"InOut port '{port.Id}' is already bound.");
    }

    /// <summary>Maps one called output directly to a caller resource.</summary>
    public void Bind<T>(GraphOutput<T> port, GraphResource<T> resource)
        => Bind(port, resource, GraphBindingOwner.Parent);

    /// <summary>Maps one called output to a resource owned by another graph-call instance.</summary>
    public void Bind<TOwnerContract, T>(
        GraphOutput<T> port,
        GraphResource<T> resource,
        GraphCallInstance<TOwnerContract> instance)
        where TOwnerContract : IGraphPortContract<TOwnerContract>
    {
        ArgumentNullException.ThrowIfNull(instance);
        RequireInstanceResource(resource, instance);
        Bind(port, resource, GraphBindingOwner.Instance(instance.Id));
    }

    private void Bind<T>(
        GraphOutput<T> port,
        GraphResource<T> resource,
        GraphBindingOwner resourceOwner)
    {
        ArgumentNullException.ThrowIfNull(port);
        ArgumentNullException.ThrowIfNull(resource);
        RequireCompatible(port, resource);
        if (!outputs.TryAdd(port.Id, Resolve(resource.Resource, resourceOwner)))
            throw new InvalidOperationException($"Output '{port.Id}' is already bound.");
    }

    internal IReadOnlyList<GraphDefinitionBinding> BuildInputs(LogicalGraphDefinition definition)
    {
        var inputs = definition.Resources.Where(resource =>
            resource.Kind is GraphDefinitionResourceKind.Input or GraphDefinitionResourceKind.InOut).ToArray();
        var missing = inputs.Where(input => !values.ContainsKey(input.Id)).ToArray();
        if (missing.Length != 0)
            throw new InvalidOperationException(
                $"Graph call '{definition.Id}' is missing inputs: {string.Join(", ", missing.Select(input => input.Id))}.");
        return inputs.Select(input => new GraphDefinitionBinding(
            input.Id.Value,
            values[input.Id].Resource.Id,
            input.Kind == GraphDefinitionResourceKind.InOut
                ? GraphResourceAccess.ReadWrite
                : GraphResourceAccess.Read,
            values[input.Id].Owner)).ToArray();
    }

    private static void RequireCompatible<T>(GraphResource<T> port, GraphResource<T> resource)
    {
        if (!SameTensor(port.Tensor, resource.Tensor))
            throw new ArgumentException(
                $"Resource '{resource.Id}' is incompatible with port '{port.Id}'.", nameof(resource));
    }

    private static bool SameTensor(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType &&
        string.Equals(left.Layout, right.Layout, StringComparison.Ordinal) &&
        left.Dimensions.SequenceEqual(right.Dimensions);

    private static void RequireInstanceResource<TOwnerContract, T>(
        GraphResource<T> resource,
        GraphCallInstance<TOwnerContract> instance)
        where TOwnerContract : IGraphPortContract<TOwnerContract>
    {
        ArgumentNullException.ThrowIfNull(resource);
        if (resource.Resource.Owner != instance.Definition.Id ||
            !instance.Definition.Resources.Any(candidate => candidate == resource.Resource))
            throw new ArgumentException(
                $"Resource '{resource.Id}' is not declared by graph instance '{instance.Id}'.",
                nameof(resource));
    }

    private GraphCallBoundResource Resolve(
        GraphDefinitionResource resource,
        GraphBindingOwner resourceOwner)
    {
        if (resourceOwner.IsParent)
        {
            if (resource.Owner != owner)
                throw new ArgumentException(
                    $"Parent binding cannot reference resource '{resource.Id}' owned by graph '{resource.Owner}'.");
        }
        return new GraphCallBoundResource(resource, resourceOwner);
    }
}

internal sealed class DefinitionPortFactory(GraphDefinitionId owner) : IGraphPortFactory
{
    private readonly Dictionary<ResourceId, GraphDefinitionResource> resources = [];

    internal IReadOnlyCollection<GraphDefinitionResource> Resources => resources.Values;

    public GraphInput<T> AddInput<T>(string id, TensorDescriptor tensor) =>
        new(GetOrAdd(id, GraphDefinitionResourceKind.Input, tensor));

    public GraphOutput<T> AddOutput<T>(string id, TensorDescriptor tensor) =>
        new(GetOrAdd(id, GraphDefinitionResourceKind.Output, tensor));

    public GraphInOut<T> AddInOut<T>(string id, TensorDescriptor tensor) =>
        new(GetOrAdd(id, GraphDefinitionResourceKind.InOut, tensor));

    private GraphDefinitionResource GetOrAdd(
        string id,
        GraphDefinitionResourceKind kind,
        TensorDescriptor tensor)
    {
        var resourceId = new ResourceId(LogicalGraphDefinition.RequireId(id, nameof(id)));
        if (resources.TryGetValue(resourceId, out var existing))
        {
            if (existing.Kind != kind || !SameTensor(existing.Tensor, tensor))
                throw new InvalidOperationException($"Port '{id}' has conflicting declarations.");
            return existing;
        }
        var resource = new GraphDefinitionResource(resourceId, id, kind,
            tensor ?? throw new ArgumentNullException(nameof(tensor)), owner);
        resources.Add(resourceId, resource);
        return resource;
    }

    private static bool SameTensor(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType &&
        string.Equals(left.Layout, right.Layout, StringComparison.Ordinal) &&
        left.Dimensions.SequenceEqual(right.Dimensions);
}

internal sealed class InstancePortFactory(
    GraphDefinitionId owner,
    GraphDefinitionNodeId call,
    LogicalGraphDefinition definition,
    IReadOnlyList<GraphDefinitionBinding> bindings,
    IReadOnlyDictionary<ResourceId, GraphCallBoundResource> outputTargets) : IGraphPortFactory
{
    private readonly IReadOnlyDictionary<string, ResourceId> inputs =
        bindings.ToDictionary(binding => binding.Port, binding => binding.Resource, StringComparer.Ordinal);
    private readonly Dictionary<ResourceId, GraphDefinitionResource> resources = [];
    private readonly List<GraphDefinitionBinding> outputBindings = [];

    internal IReadOnlyCollection<GraphDefinitionResource> Resources => resources.Values;
    internal IReadOnlyList<GraphDefinitionBinding> OutputBindings => outputBindings;

    public GraphInput<T> AddInput<T>(string id, TensorDescriptor tensor)
    {
        var definitionResource = Find(id, GraphDefinitionResourceKind.Input, tensor);
        var resource = definitionResource with { Id = inputs[id], Owner = owner };
        resources.TryAdd(resource.Id, resource);
        return new GraphInput<T>(resource);
    }

    public GraphOutput<T> AddOutput<T>(string id, TensorDescriptor tensor)
    {
        var definitionResource = Find(id, GraphDefinitionResourceKind.Output, tensor);
        var target = outputTargets.GetValueOrDefault(definitionResource.Id);
        var resource = target is not null
            ? target.Resource
            : definitionResource with
            {
                Id = new ResourceId($"{call.Value}/{id}"),
                Name = $"{call.Value}/{definitionResource.Name}",
                Owner = owner,
            };
        if (target is null || target.Owner.IsParent)
            resources.TryAdd(resource.Id, resource);
        outputBindings.Add(new GraphDefinitionBinding(
            id,
            resource.Id,
            GraphResourceAccess.Write,
            target?.Owner ?? GraphBindingOwner.Parent));
        return new GraphOutput<T>(resource);
    }

    public GraphInOut<T> AddInOut<T>(string id, TensorDescriptor tensor)
    {
        var definitionResource = Find(id, GraphDefinitionResourceKind.InOut, tensor);
        if (!outputTargets.TryGetValue(definitionResource.Id, out var target))
            throw new InvalidOperationException($"InOut port '{id}' is not bound.");
        if (target.Owner.IsParent)
            resources.TryAdd(target.Resource.Id, target.Resource);
        return new GraphInOut<T>(target.Resource);
    }

    private GraphDefinitionResource Find(
        string id,
        GraphDefinitionResourceKind kind,
        TensorDescriptor tensor)
    {
        var resource = definition.Resources.SingleOrDefault(resource =>
            resource.Id.Value == id && resource.Kind == kind)
            ?? throw new InvalidOperationException($"Unknown {kind} port '{id}' in graph '{definition.Id}'.");
        if (resource.Tensor.ElementType != tensor.ElementType ||
            resource.Tensor.Layout != tensor.Layout ||
            !resource.Tensor.Dimensions.SequenceEqual(tensor.Dimensions))
            throw new InvalidOperationException($"Port '{id}' has a conflicting tensor contract.");
        return resource;
    }
}

internal static class LogicalGraphDefinitionValidator
{
    internal sealed record Analysis(
        IReadOnlyList<GraphDefinitionNodeId> CompletionSinks,
        IReadOnlyDictionary<ResourceId, GraphDefinitionNodeId[]> TerminalWriters,
        IReadOnlyList<LogicalGraphDiagnostic> Diagnostics);

    internal static IReadOnlyList<GraphDefinitionNode> InferDependencies(
        IReadOnlyList<GraphDefinitionNode> nodes)
    {
        var writers = nodes
            .SelectMany(node => node.Bindings
                .Where(binding => binding.Access != GraphResourceAccess.Read)
                .Select(binding => (binding.ResourceOwner, binding.Resource, node.Id)))
            .GroupBy(item => (item.ResourceOwner, item.Resource))
            .ToDictionary(group => group.Key, group => group.Select(item => item.Id).Distinct().ToArray());
        return nodes.Select(node =>
        {
            var dependencies = node.Dependencies.ToHashSet();
            foreach (var input in node.Bindings.Where(binding =>
                binding.Access == GraphResourceAccess.Read))
            {
                if (!input.ResourceOwner.IsParent && input.ResourceOwner.Call is { } ownerCall &&
                    ownerCall != node.Id)
                    dependencies.Add(ownerCall);
                if (writers.TryGetValue((input.ResourceOwner, input.Resource), out var producers) &&
                    producers.Where(producer => producer != node.Id).ToArray() is [var producer])
                    dependencies.Add(producer);
            }
            return node switch
            {
                GraphOperationNode operation => operation with { Dependencies = dependencies.ToArray() },
                GraphCallNode call => call with { Dependencies = dependencies.ToArray() },
                _ => node,
            };
        }).ToArray();
    }

    internal static Analysis Analyze(LogicalGraphDefinition graph)
    {
        var diagnostics = new List<LogicalGraphDiagnostic>();
        if (string.IsNullOrWhiteSpace(graph.Id.Value))
            diagnostics.Add(Error("LG0001", "A graph definition identifier is required."));
        if (!Enum.IsDefined(graph.CompileBoundary))
            diagnostics.Add(Error("LG0002", $"Graph '{graph.Id}' has an invalid compile boundary."));

        var resources = Unique(graph.Resources, resource => resource.Id, "resource", diagnostics);
        var nodes = Unique(graph.Nodes, node => node.Id, "node", diagnostics);
        var definitions = Unique(graph.Definitions, definition => definition.Id, "called definition", diagnostics);
        var calls = graph.Nodes.OfType<GraphCallNode>()
            .ToDictionary(call => call.Id);
        foreach (var node in graph.Nodes)
        {
            LogicalGraphDefinition? calledDefinition = null;
            if (node is GraphCallNode graphCall)
                definitions.TryGetValue(graphCall.Definition, out calledDefinition);
            if (node.Dependencies.Any(dependency => !nodes.ContainsKey(dependency)))
                diagnostics.Add(Error("LG0010", $"Node '{node.Id}' depends on an unknown node.", node.Id));
            if (node.Dependencies.Contains(node.Id))
                diagnostics.Add(Error("LG0011", $"Node '{node.Id}' depends on itself.", node.Id));
            var ports = new HashSet<string>(StringComparer.Ordinal);
            foreach (var binding in node.Bindings)
            {
                if (string.IsNullOrWhiteSpace(binding.Port) || !ports.Add(binding.Port))
                    diagnostics.Add(Error("LG0012", $"Node '{node.Id}' has a duplicate or empty port '{binding.Port}'.", node.Id));
                if (node is GraphOperationNode && !binding.ResourceOwner.IsParent)
                {
                    diagnostics.Add(Error("LG0021",
                        $"Operation '{node.Id}' cannot bind graph-instance resource '{binding.Resource}'.",
                        node.Id, binding.Resource));
                    continue;
                }
                if (!binding.ResourceOwner.IsParent &&
                    binding.Access != GraphResourceAccess.Read)
                    diagnostics.Add(new(
                        LogicalGraphDiagnosticSeverity.Warning,
                        "LGW003",
                        $"Node '{node.Id}' writes resource '{binding.Resource}' owned by graph instance " +
                        $"'{binding.ResourceOwner}'. This may cause unexpected cross-instance behavior.",
                        node.Id,
                        binding.Resource));
                GraphDefinitionResource? resource;
                if (binding.ResourceOwner.IsParent)
                {
                    if (!resources.TryGetValue(binding.Resource, out resource))
                    {
                        diagnostics.Add(Error("LG0016",
                            $"Node '{node.Id}' references unknown parent resource '{binding.Resource}'.",
                            node.Id, binding.Resource));
                        continue;
                    }
                    if (resource.Owner != graph.Id)
                        diagnostics.Add(Error("LG0017",
                            $"Parent resource '{binding.Resource}' is not owned by graph '{graph.Id}'.",
                            node.Id, binding.Resource));
                }
                else
                {
                    var ownerCallId = binding.ResourceOwner.Call!.Value;
                    if (ownerCallId == node.Id)
                    {
                        diagnostics.Add(Error("LG0022",
                            $"Call '{node.Id}' cannot bind a resource from itself.",
                            node.Id, binding.Resource));
                        continue;
                    }
                    if (!calls.TryGetValue(ownerCallId, out var ownerCall))
                    {
                        diagnostics.Add(Error("LG0018",
                            $"Node '{node.Id}' references unknown graph instance '{ownerCallId}'.",
                            node.Id, binding.Resource));
                        continue;
                    }
                    if (!definitions.TryGetValue(ownerCall.Definition, out var ownerDefinition))
                    {
                        diagnostics.Add(Error("LG0014",
                            $"Call '{ownerCall.Id}' references unknown definition '{ownerCall.Definition}'.",
                            ownerCall.Id));
                        continue;
                    }
                    resource = ownerDefinition.Resources.SingleOrDefault(candidate =>
                        candidate.Id == binding.Resource);
                    if (resource is null)
                    {
                        diagnostics.Add(Error("LG0019",
                            $"Graph instance '{ownerCallId}' does not declare resource '{binding.Resource}'.",
                            node.Id, binding.Resource));
                        continue;
                    }
                }
                if (resource.Kind is GraphDefinitionResourceKind.Weight or GraphDefinitionResourceKind.Constant &&
                    binding.Access != GraphResourceAccess.Read)
                    diagnostics.Add(Error("LG0013",
                        $"Node '{node.Id}' writes read-only resource '{binding.Resource}'.", node.Id, binding.Resource));
                if (calledDefinition is not null)
                {
                    var port = calledDefinition.Resources.SingleOrDefault(candidate =>
                        candidate.Id.Value == binding.Port &&
                        candidate.Kind is GraphDefinitionResourceKind.Input
                            or GraphDefinitionResourceKind.Output
                            or GraphDefinitionResourceKind.InOut);
                    if (port is null)
                        diagnostics.Add(Error("LG0023",
                            $"Call '{node.Id}' binds unknown port '{binding.Port}'.",
                            node.Id, binding.Resource));
                    else
                    {
                        var expectedAccess = port.Kind switch
                        {
                            GraphDefinitionResourceKind.Input => GraphResourceAccess.Read,
                            GraphDefinitionResourceKind.Output => GraphResourceAccess.Write,
                            _ => GraphResourceAccess.ReadWrite,
                        };
                        if (binding.Access != expectedAccess)
                            diagnostics.Add(Error("LG0024",
                                $"Call '{node.Id}' port '{binding.Port}' requires {expectedAccess} access.",
                                node.Id, binding.Resource));
                        if (!SameTensor(port.Tensor, resource.Tensor))
                            diagnostics.Add(Error("LG0025",
                                $"Call '{node.Id}' port '{binding.Port}' has an incompatible tensor resource.",
                                node.Id, binding.Resource));
                    }
                }
            }
            if (node is GraphCallNode call && !definitions.ContainsKey(call.Definition))
                diagnostics.Add(Error("LG0014", $"Call '{call.Id}' references unknown definition '{call.Definition}'.", call.Id));
            if (calledDefinition is not null)
            {
                var requiredPorts = calledDefinition.Resources.Where(resource =>
                    resource.Kind is GraphDefinitionResourceKind.Input
                        or GraphDefinitionResourceKind.Output
                        or GraphDefinitionResourceKind.InOut);
                foreach (var requiredPort in requiredPorts)
                    if (!node.Bindings.Any(binding => binding.Port == requiredPort.Id.Value))
                        diagnostics.Add(Error("LG0026",
                            $"Call '{node.Id}' is missing port '{requiredPort.Id}'.",
                            node.Id, requiredPort.Id));
            }
        }

        ValidateAcyclic(graph.Nodes, diagnostics);
        var successors = graph.Nodes.ToDictionary(node => node.Id, _ => new HashSet<GraphDefinitionNodeId>());
        foreach (var node in graph.Nodes)
            foreach (var dependency in node.Dependencies)
                if (successors.TryGetValue(dependency, out var values)) values.Add(node.Id);
        var sinks = graph.Nodes.Where(node => successors[node.Id].Count == 0).Select(node => node.Id).ToArray();

        var ancestors = graph.Nodes.ToDictionary(node => node.Id,
            node => Ancestors(node.Id, nodes));
        var terminal = new Dictionary<ResourceId, GraphDefinitionNodeId[]>();
        foreach (var output in graph.Resources.Where(resource =>
            resource.Kind is GraphDefinitionResourceKind.Output or GraphDefinitionResourceKind.InOut &&
            resource.Owner == graph.Id))
        {
            var writers = graph.Nodes.Where(node => node.Bindings.Any(binding =>
                binding.ResourceOwner.IsParent &&
                binding.Resource == output.Id &&
                binding.Access != GraphResourceAccess.Read)).Select(node => node.Id).ToArray();
            if (writers.Length == 0)
            {
                diagnostics.Add(Error("LG0020", $"Graph output '{output.Id}' is never written.", resource: output.Id));
                terminal.Add(output.Id, []);
                continue;
            }
            var maximal = writers.Where(writer => !writers.Any(other =>
                other != writer && ancestors[other].Contains(writer))).ToArray();
            terminal.Add(output.Id, maximal);
            if (maximal.Length > 1)
                diagnostics.Add(new(LogicalGraphDiagnosticSeverity.Warning, "LGW002",
                    $"Graph output '{output.Id}' has unordered terminal writers: " +
                    string.Join(", ", maximal.Select(writer => $"'{writer}'")) + ".",
                    Resource: output.Id));
        }
        return new Analysis(sinks, terminal, diagnostics);
    }

    private static Dictionary<TKey, TValue> Unique<TValue, TKey>(
        IEnumerable<TValue> values,
        Func<TValue, TKey> key,
        string kind,
        ICollection<LogicalGraphDiagnostic> diagnostics)
        where TKey : notnull
    {
        var result = new Dictionary<TKey, TValue>();
        foreach (var value in values)
            if (!result.TryAdd(key(value), value))
                diagnostics.Add(Error("LG0003", $"Duplicate {kind} identifier '{key(value)}'."));
        return result;
    }

    private static void ValidateAcyclic(
        IReadOnlyList<GraphDefinitionNode> nodes,
        ICollection<LogicalGraphDiagnostic> diagnostics)
    {
        var map = nodes.ToDictionary(node => node.Id);
        var state = new Dictionary<GraphDefinitionNodeId, int>();
        foreach (var node in nodes)
            Visit(node.Id);

        void Visit(GraphDefinitionNodeId id)
        {
            if (state.TryGetValue(id, out var value))
            {
                if (value == 1)
                    diagnostics.Add(Error("LG0015", $"Graph dependency cycle detected at '{id}'.", id));
                return;
            }
            state[id] = 1;
            if (map.TryGetValue(id, out var node))
                foreach (var dependency in node.Dependencies)
                    if (map.ContainsKey(dependency)) Visit(dependency);
            state[id] = 2;
        }
    }

    private static HashSet<GraphDefinitionNodeId> Ancestors(
        GraphDefinitionNodeId id,
        IReadOnlyDictionary<GraphDefinitionNodeId, GraphDefinitionNode> nodes)
    {
        var result = new HashSet<GraphDefinitionNodeId>();
        var pending = new Stack<GraphDefinitionNodeId>(nodes[id].Dependencies);
        while (pending.TryPop(out var dependency))
            if (result.Add(dependency) && nodes.TryGetValue(dependency, out var node))
                foreach (var parent in node.Dependencies) pending.Push(parent);
        return result;
    }

    private static LogicalGraphDiagnostic Error(
        string code,
        string message,
        GraphDefinitionNodeId? node = null,
        ResourceId? resource = null) =>
        new(LogicalGraphDiagnosticSeverity.Error, code, message, node, resource);

    private static bool SameTensor(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType &&
        string.Equals(left.Layout, right.Layout, StringComparison.Ordinal) &&
        left.Dimensions.SequenceEqual(right.Dimensions);
}
