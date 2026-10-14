using System.Collections.ObjectModel;
using SharpInference.Graphs.Core;

namespace SharpInference.Graphs;

public enum GraphCompileBoundary
{
    Auto,
    Atomic,
    Transparent,
}

public sealed record GraphAnnotation(
    string? Name = null,
    string? Role = null,
    string? Description = null,
    IReadOnlyDictionary<string, string>? Attributes = null);

public readonly record struct GraphDefinitionId(string Value)
{
    public override string ToString() => Value;
}

public readonly record struct GraphDefinitionNodeId(string Value)
{
    public override string ToString() => Value;
}

public readonly record struct GraphBindingOwner(GraphDefinitionNodeId? Call)
{
    public static GraphBindingOwner Parent => new(null);
    public static GraphBindingOwner Instance(GraphDefinitionNodeId call) => new(call);
    public bool IsParent => Call is null;
    public override string ToString() => Call?.Value ?? "parent";
}

public enum GraphDefinitionResourceKind
{
    Input,
    Output,
    InOut,
    Temporary,
    Weight,
    State,
    Constant,
}

public sealed record GraphDefinitionResource(
    ResourceId Id,
    string Name,
    GraphDefinitionResourceKind Kind,
    TensorDescriptor Tensor,
    GraphDefinitionId Owner,
    string? BindingKey = null);

public class GraphResource<T>
{
    internal GraphResource(GraphDefinitionResource resource) => Resource = resource;
    public GraphDefinitionResource Resource { get; }
    public ResourceId Id => Resource.Id;
    public TensorDescriptor Tensor => Resource.Tensor;
}

public sealed class GraphInput<T>(GraphDefinitionResource resource) : GraphResource<T>(resource);
public sealed class GraphOutput<T>(GraphDefinitionResource resource) : GraphResource<T>(resource);
public sealed class GraphInOut<T>(GraphDefinitionResource resource) : GraphResource<T>(resource);

public interface IGraphPortContract<TSelf> where TSelf : IGraphPortContract<TSelf>
{
    static abstract TSelf Create(IGraphPortFactory ports);
}

public interface IGraphPortFactory
{
    GraphInput<T> AddInput<T>(string id, TensorDescriptor tensor);
    GraphOutput<T> AddOutput<T>(string id, TensorDescriptor tensor);
    GraphInOut<T> AddInOut<T>(string id, TensorDescriptor tensor);
}

public enum LogicalGraphDiagnosticSeverity
{
    Warning,
    Error,
}

public sealed record LogicalGraphDiagnostic(
    LogicalGraphDiagnosticSeverity Severity,
    string Code,
    string Message,
    GraphDefinitionNodeId? Node = null,
    ResourceId? Resource = null);

public abstract record LogicalGraphNode : IModelGraphNode
{
    public abstract GraphNodeId GraphId { get; }
    GraphNodeId IGraphNode.Id => GraphId;
}

public sealed record LogicalResourceNode(
    ModelResourceNodeId ResourceNodeId,
    GraphDefinitionResource Resource) : LogicalGraphNode, IModelResourceNode
{
    public override GraphNodeId GraphId => ResourceNodeId.GraphId;
    public TensorDescriptor Tensor => Resource.Tensor;
    public GraphDefinitionResourceKind Kind => Resource.Kind;
}

public abstract record GraphDefinitionNode(
    GraphDefinitionNodeId Id,
    GraphAnnotation? Annotation) : LogicalGraphNode, IModelOperationNode
{
    public override GraphNodeId GraphId => NodeGraphId(Id);

    internal static GraphNodeId NodeGraphId(GraphDefinitionNodeId id) =>
        new($"operation:{id.Value}");
}

public sealed record GraphOperationNode(
    GraphDefinitionNodeId Id,
    GraphOperationId Operation,
    IReadOnlyDictionary<string, string> Attributes,
    PrecisionRequirement Requirements,
    GraphAnnotation? Annotation = null)
    : GraphDefinitionNode(Id, Annotation);

public sealed record GraphCallNode(
    GraphDefinitionNodeId Id,
    GraphDefinitionId Definition,
    GraphAnnotation? Annotation = null)
    : GraphDefinitionNode(Id, Annotation);

public abstract record LogicalGraphEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target) : IModelGraphEdge;

public sealed record LogicalDependencyEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target) : LogicalGraphEdge(Id, Source, Target), IDependencyEdge;

public sealed record LogicalResourceReadEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target,
    string Port,
    string BindingId) : LogicalGraphEdge(Id, Source, Target), IResourceReadEdge;

public sealed record LogicalResourceWriteEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target,
    string Port,
    string BindingId) : LogicalGraphEdge(Id, Source, Target), IResourceWriteEdge;

public sealed class LogicalGraphDefinition
{
    internal LogicalGraphDefinition(
        GraphDefinitionId id,
        GraphCompileBoundary compileBoundary,
        GraphAnnotation? annotation,
        Graph<LogicalGraphNode, LogicalGraphEdge> structure,
        IEnumerable<LogicalGraphDefinition> definitions)
    {
        Id = id;
        CompileBoundary = compileBoundary;
        Annotation = annotation;
        Structure = structure ?? throw new ArgumentNullException(nameof(structure));
        Definitions = Array.AsReadOnly(definitions?.ToArray()
            ?? throw new ArgumentNullException(nameof(definitions)));
        var analysis = LogicalGraphDefinitionValidator.Analyze(this);
        CompletionSinks = Array.AsReadOnly(analysis.CompletionSinks.ToArray());
        TerminalWriters = new ReadOnlyDictionary<ResourceId, IReadOnlyList<GraphDefinitionNodeId>>(
            analysis.TerminalWriters.ToDictionary(
                pair => pair.Key,
                pair => (IReadOnlyList<GraphDefinitionNodeId>)Array.AsReadOnly(pair.Value)));
        Diagnostics = Array.AsReadOnly(analysis.Diagnostics.ToArray());
        var errors = Diagnostics.Where(diagnostic =>
            diagnostic.Severity == LogicalGraphDiagnosticSeverity.Error).ToArray();
        if (errors.Length != 0)
        {
            throw new InvalidDataException(
                string.Join(Environment.NewLine, errors.Select(error => error.Message)));
        }
    }

    public GraphDefinitionId Id { get; }
    public GraphCompileBoundary CompileBoundary { get; }
    public GraphAnnotation? Annotation { get; }
    public Graph<LogicalGraphNode, LogicalGraphEdge> Structure { get; }
    public IReadOnlyList<GraphDefinitionResource> Resources =>
        Structure.NodesOfType<LogicalResourceNode>()
            .Where(node => node.ResourceNodeId.Owner.IsParent)
            .Select(node => node.Resource).ToArray();
    public IReadOnlyList<GraphDefinitionNode> Nodes =>
        Structure.Nodes.OfType<GraphDefinitionNode>().ToArray();
    public IReadOnlyList<LogicalGraphDefinition> Definitions { get; }
    public IReadOnlyList<GraphDefinitionNodeId> CompletionSinks { get; }
    public IReadOnlyDictionary<ResourceId, IReadOnlyList<GraphDefinitionNodeId>> TerminalWriters { get; }
    public IReadOnlyList<LogicalGraphDiagnostic> Diagnostics { get; }

    public IReadOnlyList<ModelResourceBinding> Bindings(GraphDefinitionNode node) =>
        Structure.Bindings(node);

    public IReadOnlyList<GraphDefinitionNodeId> Dependencies(GraphDefinitionNode node) =>
        Structure.Predecessors(node).OfType<GraphDefinitionNode>().Select(value => value.Id).ToArray();

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

    public static GraphDefinition<TContract> CreateListGraph<TContract>(
        string id,
        IEnumerable<GraphOperationId> operations,
        GraphCompileBoundary compileBoundary = GraphCompileBoundary.Auto,
        GraphAnnotation? annotation = null)
        where TContract : IGraphPortContract<TContract>
    {
        ArgumentNullException.ThrowIfNull(operations);
        var values = operations.ToArray();
        if (values.Length == 0)
        {
            throw new ArgumentException("A list graph requires at least one operation.", nameof(operations));
        }

        return Define<TContract>(id, builder =>
        {
            var inputs = builder.Resources.Where(resource =>
                resource.Kind == GraphDefinitionResourceKind.Input).ToArray();
            var outputs = builder.Resources.Where(resource =>
                resource.Kind == GraphDefinitionResourceKind.Output).ToArray();
            if (inputs.Length != 1 || outputs.Length != 1)
            {
                throw new InvalidOperationException(
                    "A list graph requires exactly one input and one output.");
            }

            GraphResource<object> source = new(inputs[0]);
            GraphDefinitionNodeId? previous = null;
            for (var index = 0; index < values.Length; index++)
            {
                var target = index == values.Length - 1
                    ? new GraphResource<object>(outputs[0])
                    : builder.AddResource<object>(
                        $"step-{index}-output",
                        $"Step {index} output",
                        outputs[0].Tensor);
                var current = new GraphDefinitionNodeId($"step-{index}");
                builder.AddNode(current.Value, values[index], node =>
                {
                    node.AddInput("input", source);
                    node.AddOutput("output", target);
                    if (previous is { } dependency)
                    {
                        node.DependsOn(dependency);
                    }
                });
                source = target;
                previous = current;
            }
        }, compileBoundary, annotation);
    }

    internal static string RequireId(string value, string parameterName) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException("An identifier is required.", parameterName)
            : value;
}

public sealed class GraphDefinition<TContract> where TContract : IGraphPortContract<TContract>
{
    internal GraphDefinition(LogicalGraphDefinition definition, TContract ports)
    {
        Definition = definition;
        Ports = ports;
    }

    public LogicalGraphDefinition Definition { get; }
    public TContract Ports { get; }
}

public sealed class GraphCallInstance<TContract> where TContract : IGraphPortContract<TContract>
{
    internal GraphCallInstance(
        GraphDefinitionNodeId id,
        LogicalGraphDefinition definition,
        TContract ports)
    {
        Id = id;
        Definition = definition;
        Ports = ports;
    }

    public GraphDefinitionNodeId Id { get; }
    public TContract Ports { get; }
    internal LogicalGraphDefinition Definition { get; }

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

internal sealed record PendingBinding(
    string Port,
    GraphDefinitionResource Resource,
    GraphResourceAccess Access,
    GraphBindingOwner Owner);

public sealed class GraphDefinitionBuilder<TContract> where TContract : IGraphPortContract<TContract>
{
    private readonly GraphDefinitionId id;
    private readonly DefinitionPortFactory ports;
    private readonly GraphCompileBoundary compileBoundary;
    private readonly GraphAnnotation? annotation;
    private readonly List<GraphDefinitionResource> resources;
    private readonly List<GraphDefinitionNode> nodes = [];
    private readonly List<LogicalGraphEdge> edges = [];
    private readonly Dictionary<ModelResourceNodeId, GraphDefinitionResource> scopedResources = [];
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

    public TContract Ports { get; }
    internal IReadOnlyList<GraphDefinitionResource> Resources => resources;

    public GraphResource<T> AddResource<T>(
        string id,
        string name,
        TensorDescriptor tensor,
        GraphDefinitionResourceKind kind = GraphDefinitionResourceKind.Temporary,
        string? bindingKey = null)
    {
        if (kind is GraphDefinitionResourceKind.Input or GraphDefinitionResourceKind.Output)
        {
            throw new ArgumentException(
                "Use the graph port contract to declare inputs and outputs.", nameof(kind));
        }

        var resource = new GraphDefinitionResource(
            new ResourceId(LogicalGraphDefinition.RequireId(id, nameof(id))),
            string.IsNullOrWhiteSpace(name)
                ? throw new ArgumentException("A resource name is required.", nameof(name))
                : name,
            kind,
            tensor ?? throw new ArgumentNullException(nameof(tensor)),
            this.id,
            bindingKey);
        AddReferencedResource(resource);
        return new GraphResource<T>(resource);
    }

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
        {
            AddReferencedResource(resource);
        }

        var node = builder.Build(operation, annotation);
        AddNodeAndRelations(node, builder.Bindings, builder.Dependencies);
        return result;
    }

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

    public TChildContract AddNode<TChildContract>(
        string id,
        GraphDefinition<TChildContract> definition,
        Action<GraphCallBindings<TChildContract>> bind,
        GraphAnnotation? annotation = null)
        where TChildContract : IGraphPortContract<TChildContract> =>
        AddCall(id, definition, bind, annotation).Ports;

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
        {
            AddReferencedResource(resource);
        }

        var instancePorts = new InstancePortFactory(
            this.id, nodeId, definition.Definition, inputs, bindings.Outputs);
        var result = TChildContract.Create(instancePorts);
        foreach (var resource in instancePorts.Resources)
        {
            AddReferencedResource(resource);
        }

        var values = inputs.Concat(instancePorts.OutputBindings).ToArray();
        AddNodeAndRelations(
            new GraphCallNode(nodeId, definition.Definition.Id, annotation),
            values,
            []);
        definitions.TryAdd(definition.Definition.Id, definition.Definition);
        return new GraphCallInstance<TChildContract>(nodeId, definition.Definition, result);
    }

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
        var ports = definition.Resources.Where(resource =>
                resource.Kind is GraphDefinitionResourceKind.Input
                    or GraphDefinitionResourceKind.Output
                    or GraphDefinitionResourceKind.InOut)
            .ToDictionary(resource => resource.Id.Value, StringComparer.Ordinal);
        var values = bindings.Select(binding =>
        {
            if (!ports.TryGetValue(binding.Port, out var port))
            {
                throw new InvalidOperationException(
                    $"Definition '{definition.Id}' has no public port '{binding.Port}'.");
            }

            if (!SameTensor(port.Tensor, binding.Resource.Tensor))
            {
                throw new InvalidOperationException(
                    $"Call '{nodeId}' binds incompatible resource '{binding.Resource.Id}' to '{binding.Port}'.");
            }

            var requiredAccess = AccessFor(port.Kind);
            if (binding.Access != requiredAccess)
            {
                throw new InvalidOperationException(
                    $"Call '{nodeId}' port '{binding.Port}' requires {requiredAccess} access.");
            }

            AddReferencedResource(binding.Resource.Resource);
            return new PendingBinding(
                binding.Port, binding.Resource.Resource, binding.Access, GraphBindingOwner.Parent);
        }).ToArray();
        if (values.Select(binding => binding.Port).Distinct(StringComparer.Ordinal).Count() != ports.Count ||
            values.Length != ports.Count)
        {
            throw new InvalidOperationException(
                $"Call '{nodeId}' must bind every public port of definition '{definition.Id}' exactly once.");
        }

        AddNodeAndRelations(
            new GraphCallNode(nodeId, definition.Id, annotation),
            values,
            dependencies ?? []);
        definitions.TryAdd(definition.Id, definition);
        return nodeId;
    }

    internal GraphDefinition<TContract> Build()
    {
        foreach (var port in ports.Resources)
        {
            AddReferencedResource(port);
        }

        foreach (var resource in resources)
        {
            scopedResources.TryAdd(
                new ModelResourceNodeId(GraphBindingOwner.Parent, resource.Id),
                resource);
        }

        InferDependencies();
        var graphNodes = scopedResources
            .Select(pair => (LogicalGraphNode)new LogicalResourceNode(pair.Key, pair.Value))
            .Concat(nodes)
            .ToArray();
        var definition = new LogicalGraphDefinition(
            id,
            compileBoundary,
            annotation,
            new Graph<LogicalGraphNode, LogicalGraphEdge>(graphNodes, edges),
            definitions.Values);
        return new GraphDefinition<TContract>(definition, Ports);
    }

    private void AddNodeAndRelations(
        GraphDefinitionNode node,
        IEnumerable<PendingBinding> bindings,
        IEnumerable<GraphDefinitionNodeId> dependencies)
    {
        nodes.Add(node);
        foreach (var binding in bindings)
        {
            AddBindingEdges(node, binding);
        }

        foreach (var dependency in dependencies.Distinct())
        {
            AddDependency(dependency, node.Id);
        }
    }

    private void AddBindingEdges(GraphDefinitionNode node, PendingBinding binding)
    {
        var scoped = new ModelResourceNodeId(binding.Owner, binding.Resource.Id);
        scopedResources.TryAdd(scoped, binding.Resource);
        var bindingId = $"{node.Id.Value}:{binding.Port}";
        if (binding.Access != GraphResourceAccess.Write)
        {
            edges.Add(new LogicalResourceReadEdge(
                new($"binding:{bindingId}:read"),
                scoped.GraphId,
                node.GraphId,
                binding.Port,
                bindingId));
        }

        if (binding.Access != GraphResourceAccess.Read)
        {
            edges.Add(new LogicalResourceWriteEdge(
                new($"binding:{bindingId}:write"),
                node.GraphId,
                scoped.GraphId,
                binding.Port,
                bindingId));
        }
    }

    private void AddDependency(GraphDefinitionNodeId source, GraphDefinitionNodeId target)
    {
        if (source == target)
        {
            edges.Add(new LogicalDependencyEdge(
                new($"dependency:{source.Value}:{target.Value}"),
                GraphDefinitionNode.NodeGraphId(source),
                GraphDefinitionNode.NodeGraphId(target)));
            return;
        }
        if (edges.OfType<LogicalDependencyEdge>().Any(edge =>
            edge.Source == GraphDefinitionNode.NodeGraphId(source) &&
            edge.Target == GraphDefinitionNode.NodeGraphId(target)))
        {
            return;
        }

        edges.Add(new LogicalDependencyEdge(
            new($"dependency:{source.Value}:{target.Value}"),
            GraphDefinitionNode.NodeGraphId(source),
            GraphDefinitionNode.NodeGraphId(target)));
    }

    private void InferDependencies()
    {
        var writes = edges.OfType<LogicalResourceWriteEdge>()
            .GroupBy(edge => edge.Target)
            .ToDictionary(
                group => group.Key,
                group => group.Select(edge => edge.Source).Distinct().ToArray());
        var pairedWrites = edges.OfType<LogicalResourceWriteEdge>()
            .Select(edge => edge.BindingId).ToHashSet(StringComparer.Ordinal);
        foreach (var read in edges.OfType<LogicalResourceReadEdge>().ToArray())
        {
            var target = (GraphDefinitionNode)nodes.Single(node => node.GraphId == read.Target);
            var resource = scopedResources.Single(pair => pair.Key.GraphId == read.Source).Key;
            if (!resource.Owner.IsParent && resource.Owner.Call is { } ownerCall &&
                ownerCall != target.Id)
            {
                AddDependency(ownerCall, target.Id);
            }

            if (pairedWrites.Contains(read.BindingId))
            {
                continue;
            }

            if (writes.TryGetValue(read.Source, out var producers) &&
                producers.Where(producer => producer != read.Target).ToArray() is [var producer])
            {
                var source = (GraphDefinitionNode)nodes.Single(node => node.GraphId == producer);
                AddDependency(source.Id, target.Id);
            }
        }
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
        {
            throw new InvalidOperationException(
                $"Resource '{resource.Id}' has conflicting declarations.");
        }
    }

    private static GraphResourceAccess AccessFor(GraphDefinitionResourceKind kind) => kind switch
    {
        GraphDefinitionResourceKind.Input => GraphResourceAccess.Read,
        GraphDefinitionResourceKind.Output => GraphResourceAccess.Write,
        GraphDefinitionResourceKind.InOut => GraphResourceAccess.ReadWrite,
        _ => throw new InvalidDataException(),
    };

    private static bool SameTensor(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType &&
        left.Layout == right.Layout &&
        left.Dimensions.SequenceEqual(right.Dimensions);
}

public sealed class GraphOperationNodeBuilder
{
    private readonly GraphDefinitionId owner;
    private readonly GraphDefinitionNodeId id;
    private readonly List<PendingBinding> bindings = [];
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
    internal IReadOnlyList<PendingBinding> Bindings => bindings;
    internal IReadOnlyList<GraphDefinitionNodeId> Dependencies => dependencies;

    public void AddInput<T>(string port, GraphResource<T> resource) =>
        AddBinding(port, resource, GraphResourceAccess.Read);

    public GraphResource<T> AddOutput<T>(string port, GraphResource<T> resource)
    {
        AddBinding(port, resource, GraphResourceAccess.Write);
        return resource;
    }

    public GraphResource<T> AddInOut<T>(string port, GraphResource<T> resource)
    {
        AddBinding(port, resource, GraphResourceAccess.ReadWrite);
        return resource;
    }

    public void DependsOn(GraphDefinitionNodeId dependency) => dependencies.Add(dependency);

    public void AddAttribute(string name, string value) =>
        attributes.Add(
            LogicalGraphDefinition.RequireId(name, nameof(name)),
            value ?? throw new ArgumentNullException(nameof(value)));

    public void RequirePrecision(PrecisionRequirement value) =>
        requirements = value ?? throw new ArgumentNullException(nameof(value));

    internal GraphOperationNode Build(GraphOperationId operation, GraphAnnotation? annotation) =>
        new(
            id,
            operation,
            new ReadOnlyDictionary<string, string>(attributes),
            requirements ?? new PrecisionRequirement(
                GraphElementType.Float32, GraphElementType.Float32),
            annotation);

    private void AddBinding<T>(
        string port,
        GraphResource<T> resource,
        GraphResourceAccess access)
    {
        ArgumentNullException.ThrowIfNull(resource);
        if (resource.Resource.Owner != owner)
        {
            throw new ArgumentException(
                $"Operation '{id}' cannot directly reference resource '{resource.Id}' owned by graph " +
                $"'{resource.Resource.Owner}'. Bind graph instances at their containing call instead.",
                nameof(resource));
        }

        var name = LogicalGraphDefinition.RequireId(port, nameof(port));
        if (bindings.Any(binding => binding.Port == name))
        {
            throw new InvalidOperationException(
                $"Operation '{id}' already binds port '{name}'.");
        }

        bindings.Add(new PendingBinding(
            name, resource.Resource, access, GraphBindingOwner.Parent));
        resources.TryAdd(resource.Id, resource.Resource);
    }
}

internal sealed record GraphCallBoundResource(
    GraphDefinitionResource Resource,
    GraphBindingOwner Owner);

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

    public TContract Ports { get; }

    internal IEnumerable<GraphDefinitionResource> Resources =>
        values.Values.Concat(outputs.Values)
            .Where(value => value.Owner.IsParent)
            .Select(value => value.Resource)
            .DistinctBy(resource => resource.Id);
    internal IReadOnlyDictionary<ResourceId, GraphCallBoundResource> Outputs => outputs;

    public void Bind<T>(GraphInput<T> port, GraphResource<T> resource) =>
        BindInput(port, resource, GraphBindingOwner.Parent);

    public void Bind<TOwnerContract, T>(
        GraphInput<T> port,
        GraphResource<T> resource,
        GraphCallInstance<TOwnerContract> instance)
        where TOwnerContract : IGraphPortContract<TOwnerContract>
    {
        RequireInstanceResource(resource, instance);
        BindInput(port, resource, GraphBindingOwner.Instance(instance.Id));
    }

    public void Bind<T>(GraphInOut<T> port, GraphResource<T> resource) =>
        BindInOut(port, resource, GraphBindingOwner.Parent);

    public void Bind<TOwnerContract, T>(
        GraphInOut<T> port,
        GraphResource<T> resource,
        GraphCallInstance<TOwnerContract> instance)
        where TOwnerContract : IGraphPortContract<TOwnerContract>
    {
        RequireInstanceResource(resource, instance);
        BindInOut(port, resource, GraphBindingOwner.Instance(instance.Id));
    }

    public void Bind<T>(GraphOutput<T> port, GraphResource<T> resource) =>
        BindOutput(port, resource, GraphBindingOwner.Parent);

    public void Bind<TOwnerContract, T>(
        GraphOutput<T> port,
        GraphResource<T> resource,
        GraphCallInstance<TOwnerContract> instance)
        where TOwnerContract : IGraphPortContract<TOwnerContract>
    {
        RequireInstanceResource(resource, instance);
        BindOutput(port, resource, GraphBindingOwner.Instance(instance.Id));
    }

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

internal sealed class InstancePortFactory(
    GraphDefinitionId owner,
    GraphDefinitionNodeId call,
    LogicalGraphDefinition definition,
    IReadOnlyList<PendingBinding> bindings,
    IReadOnlyDictionary<ResourceId, GraphCallBoundResource> outputTargets) : IGraphPortFactory
{
    private readonly IReadOnlyDictionary<string, PendingBinding> inputs =
        bindings.ToDictionary(binding => binding.Port, StringComparer.Ordinal);
    private readonly Dictionary<ResourceId, GraphDefinitionResource> resources = [];
    private readonly List<PendingBinding> outputBindings = [];

    internal IReadOnlyCollection<GraphDefinitionResource> Resources => resources.Values;
    internal IReadOnlyList<PendingBinding> OutputBindings => outputBindings;

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

/// <summary>Validates logical graph relationships and derives dependency completion information.</summary>
internal static class LogicalGraphDefinitionValidator
{
    /// <summary>The completion information and diagnostics derived from one logical graph.</summary>
    /// <param name="CompletionSinks">Operations with no direct dependency successors.</param>
    /// <param name="TerminalWriters">Dependency-maximal writer identities for each output or in/out resource.</param>
    /// <param name="Diagnostics">Validation findings collected during analysis.</param>
    internal sealed record Analysis(
        IReadOnlyList<GraphDefinitionNodeId> CompletionSinks,
        IReadOnlyDictionary<ResourceId, GraphDefinitionNodeId[]> TerminalWriters,
        IReadOnlyList<LogicalGraphDiagnostic> Diagnostics);

    /// <summary>Validates a logical graph and derives completion sinks and terminal resource writers.</summary>
    /// <param name="graph">The definition whose nodes, bindings, resources, and dependencies are analyzed.</param>
    /// <returns>Completion information and validation diagnostics without modifying the definition.</returns>
    /// <remarks>Multiple-writer comparisons reuse one lazily constructed dependency reachability index.</remarks>
    internal static Analysis Analyze(LogicalGraphDefinition graph)
    {
        var diagnostics = new List<LogicalGraphDiagnostic>();
        if (string.IsNullOrWhiteSpace(graph.Id.Value))
        {
            diagnostics.Add(Error("LG0001", "A graph definition identifier is required."));
        }

        if (!Enum.IsDefined(graph.CompileBoundary))
        {
            diagnostics.Add(Error(
                "LG0002", $"Graph '{graph.Id}' has an invalid compile boundary."));
        }

        var nodes = graph.Nodes.ToDictionary(node => node.Id);
        var definitions = graph.Definitions.ToDictionary(definition => definition.Id);
        var calls = graph.Nodes.OfType<GraphCallNode>().ToDictionary(call => call.Id);
        foreach (var node in graph.Nodes)
        {
            var ports = new HashSet<string>(StringComparer.Ordinal);
            foreach (var binding in graph.Bindings(node))
            {
                if (string.IsNullOrWhiteSpace(binding.Port) || !ports.Add(binding.Port))
                {
                    diagnostics.Add(Error(
                        "LG0012",
                        $"Node '{node.Id}' has a duplicate or empty port '{binding.Port}'.",
                        node.Id));
                }

                var resourceNode = graph.Structure.GetNode(binding.Resource.GraphId)
                    as LogicalResourceNode;
                if (resourceNode is null)
                {
                    diagnostics.Add(Error(
                        "LG0016",
                        $"Node '{node.Id}' references unknown resource '{binding.Resource}'.",
                        node.Id,
                        binding.Resource.Resource));
                    continue;
                }
                if (!binding.Resource.Owner.IsParent &&
                    binding.Access != GraphResourceAccess.Read)
                {
                    diagnostics.Add(new(
                        LogicalGraphDiagnosticSeverity.Warning,
                        "LGW003",
                        $"Node '{node.Id}' writes resource '{binding.Resource.Resource}' owned by " +
                        $"graph instance '{binding.Resource.Owner}'. This may cause unexpected " +
                        "cross-instance behavior.",
                        node.Id,
                        binding.Resource.Resource));
                }

                if (resourceNode.Kind is GraphDefinitionResourceKind.Weight
                        or GraphDefinitionResourceKind.Constant &&
                    binding.Access != GraphResourceAccess.Read)
                {
                    diagnostics.Add(Error(
                        "LG0013",
                        $"Node '{node.Id}' writes read-only resource '{binding.Resource.Resource}'.",
                        node.Id,
                        binding.Resource.Resource));
                }
            }

            foreach (var dependency in graph.Dependencies(node))
            {
                if (!nodes.ContainsKey(dependency))
                {
                    diagnostics.Add(Error(
                        "LG0010", $"Node '{node.Id}' depends on an unknown node.", node.Id));
                }
            }

            if (node is not GraphCallNode call)
            {
                continue;
            }

            if (!definitions.TryGetValue(call.Definition, out var called))
            {
                diagnostics.Add(Error(
                    "LG0014",
                    $"Call '{call.Id}' references unknown definition '{call.Definition}'.",
                    call.Id));
                continue;
            }
            var bindings = graph.Bindings(call);
            foreach (var port in called.Resources.Where(resource =>
                resource.Kind is GraphDefinitionResourceKind.Input
                    or GraphDefinitionResourceKind.Output
                    or GraphDefinitionResourceKind.InOut))
            {
                var binding = bindings.SingleOrDefault(value => value.Port == port.Id.Value);
                if (binding is null)
                {
                    diagnostics.Add(Error(
                        "LG0026", $"Call '{call.Id}' is missing port '{port.Id}'.",
                        call.Id, port.Id));
                    continue;
                }
                var expected = port.Kind switch
                {
                    GraphDefinitionResourceKind.Input => GraphResourceAccess.Read,
                    GraphDefinitionResourceKind.Output => GraphResourceAccess.Write,
                    _ => GraphResourceAccess.ReadWrite,
                };
                if (binding.Access != expected)
                {
                    diagnostics.Add(Error(
                        "LG0024",
                        $"Call '{call.Id}' port '{binding.Port}' requires {expected} access.",
                        call.Id,
                        binding.Resource.Resource));
                }

                var actual = (LogicalResourceNode)graph.Structure.GetNode(
                    binding.Resource.GraphId);
                if (!SameTensor(port.Tensor, actual.Tensor))
                {
                    diagnostics.Add(Error(
                        "LG0025",
                        $"Call '{call.Id}' port '{binding.Port}' has an incompatible tensor resource.",
                        call.Id,
                        binding.Resource.Resource));
                }
            }
        }

        try
        {
            _ = graph.Structure.DependencyView().TopologicalOrder();
        }
        catch (InvalidDataException exception)
        {
            diagnostics.Add(Error("LG0015", exception.Message));
        }

        var sinks = graph.Nodes
            .Where(node => !graph.Structure.Successors(node).Any())
            .Select(node => node.Id)
            .ToArray();
        var terminal = new Dictionary<ResourceId, GraphDefinitionNodeId[]>();
        ReachabilityIndex<LogicalGraphNode>? reachability = null;
        foreach (var output in graph.Resources.Where(resource =>
            resource.Kind is GraphDefinitionResourceKind.Output
                or GraphDefinitionResourceKind.InOut))
        {
            var resourceNode = graph.Structure.NodesOfType<LogicalResourceNode>()
                .Single(node => node.ResourceNodeId ==
                    new ModelResourceNodeId(GraphBindingOwner.Parent, output.Id));
            var writers = graph.Structure.EdgesToNode(resourceNode.GraphId)
                .OfType<LogicalResourceWriteEdge>()
                .Select(edge => (GraphDefinitionNode)graph.Structure.GetNode(edge.Source))
                .Select(node => node.Id)
                .Distinct()
                .ToArray();
            if (writers.Length == 0)
            {
                diagnostics.Add(Error(
                    "LG0020",
                    $"Graph output '{output.Id}' is never written.",
                    resource: output.Id));
                terminal.Add(output.Id, []);
                continue;
            }
            var maximal = writers;
            if (writers.Length > 1)
            {
                reachability ??= graph.Structure.DependencyView()
                    .CreateReachabilityIndex(graph.Nodes, graph.Nodes);
                maximal = writers.Where(writer => !writers.Any(other =>
                    other != writer &&
                    reachability.IsReachable(
                        GraphDefinitionNode.NodeGraphId(writer),
                        GraphDefinitionNode.NodeGraphId(other)))).ToArray();
            }
            terminal.Add(output.Id, maximal);
            if (maximal.Length > 1)
            {
                diagnostics.Add(new(
                    LogicalGraphDiagnosticSeverity.Warning,
                    "LGW002",
                    $"Graph output '{output.Id}' has unordered terminal writers: " +
                    string.Join(", ", maximal.Select(writer => $"'{writer}'")) + ".",
                    Resource: output.Id));
            }
        }
        return new Analysis(sinks, terminal, diagnostics);
    }

    private static LogicalGraphDiagnostic Error(
        string code,
        string message,
        GraphDefinitionNodeId? node = null,
        ResourceId? resource = null) =>
        new(LogicalGraphDiagnosticSeverity.Error, code, message, node, resource);

    private static bool SameTensor(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType &&
        left.Layout == right.Layout &&
        left.Dimensions.SequenceEqual(right.Dimensions);
}
