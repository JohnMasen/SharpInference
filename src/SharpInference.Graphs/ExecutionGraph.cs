using System.Collections.ObjectModel;
using SharpInference.Graphs.Core;

namespace SharpInference.Graphs;

public readonly record struct ExecutionGraphId(string Value)
{
    public override string ToString() => Value;
}

public readonly record struct ExecutionNodeId(string Value)
{
    public override string ToString() => Value;
}

public sealed record GraphArchitecture(
    string? Name = null,
    string? Role = null,
    string? Description = null,
    IReadOnlyDictionary<string, string>? Attributes = null);

public sealed record ExecutionGraphSource(GraphDefinitionId Definition);

public sealed record ExecutionSourceMap(
    IReadOnlyList<GraphDefinitionNodeId> LogicalNodes,
    string? Transformation = null);

public sealed record ExecutionGraphResource(
    ResourceId Id,
    string Name,
    GraphDefinitionResourceKind Kind,
    TensorDescriptor Tensor,
    string? BindingKey = null);

public abstract record ExecutionGraphStructureNode : IModelGraphNode
{
    public abstract GraphNodeId GraphId { get; }
    GraphNodeId IGraphNode.Id => GraphId;
}

public sealed record ExecutionResourceNode(
    ModelResourceNodeId ResourceNodeId,
    ExecutionGraphResource Resource)
    : ExecutionGraphStructureNode, IModelResourceNode
{
    public override GraphNodeId GraphId => ResourceNodeId.GraphId;
    public TensorDescriptor Tensor => Resource.Tensor;
    public GraphDefinitionResourceKind Kind => Resource.Kind;
}

public abstract record ExecutionGraphNode(
    ExecutionNodeId Id,
    ExecutionSourceMap Source,
    GraphArchitecture? Architecture) : ExecutionGraphStructureNode, IModelOperationNode
{
    public override GraphNodeId GraphId => NodeGraphId(Id);

    internal static GraphNodeId NodeGraphId(ExecutionNodeId id) =>
        new($"operation:{id.Value}");
}

public sealed record ExecutionOperationNode(
    ExecutionNodeId Id,
    GraphOperationId Operation,
    IReadOnlyDictionary<string, string> Attributes,
    PrecisionRequirement Requirements,
    ExecutionSourceMap Source,
    GraphArchitecture? Architecture = null)
    : ExecutionGraphNode(Id, Source, Architecture);

public sealed record ExecutionGraphCallNode(
    ExecutionNodeId Id,
    ExecutionGraphId Target,
    GraphCompileBoundary CompileBoundary,
    ExecutionSourceMap Source,
    GraphArchitecture? Architecture = null)
    : ExecutionGraphNode(Id, Source, Architecture);

public abstract record ExecutionGraphEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target) : IModelGraphEdge;

public sealed record ExecutionDependencyEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target)
    : ExecutionGraphEdge(Id, Source, Target), IDependencyEdge;

public sealed record ExecutionResourceReadEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target,
    string Port,
    string BindingId)
    : ExecutionGraphEdge(Id, Source, Target), IResourceReadEdge;

public sealed record ExecutionResourceWriteEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target,
    string Port,
    string BindingId)
    : ExecutionGraphEdge(Id, Source, Target), IResourceWriteEdge;

public sealed class ExecutionGraph
{
    public ExecutionGraph(
        ExecutionGraphId id,
        Graph<ExecutionGraphStructureNode, ExecutionGraphEdge> structure,
        GraphCompileBoundary defaultCompileBoundary = GraphCompileBoundary.Auto,
        GraphArchitecture? architecture = null,
        ExecutionGraphSource? source = null)
    {
        if (string.IsNullOrWhiteSpace(id.Value))
        {
            throw new ArgumentException(
                "An execution graph identifier is required.", nameof(id));
        }

        if (!Enum.IsDefined(defaultCompileBoundary))
        {
            throw new ArgumentOutOfRangeException(nameof(defaultCompileBoundary));
        }

        Id = id;
        Structure = structure ?? throw new ArgumentNullException(nameof(structure));
        DefaultCompileBoundary = defaultCompileBoundary;
        Architecture = architecture;
        Source = source;
        Validate();
    }

    public ExecutionGraphId Id { get; }
    public Graph<ExecutionGraphStructureNode, ExecutionGraphEdge> Structure { get; }
    public IReadOnlyList<ExecutionGraphResource> Resources =>
        Structure.NodesOfType<ExecutionResourceNode>()
            .Where(node => node.ResourceNodeId.Owner.IsParent)
            .Select(node => node.Resource).ToArray();
    public IReadOnlyList<ExecutionGraphNode> Nodes =>
        Structure.Nodes.OfType<ExecutionGraphNode>().ToArray();
    public GraphCompileBoundary DefaultCompileBoundary { get; }
    public GraphArchitecture? Architecture { get; }
    public ExecutionGraphSource? Source { get; }

    public IReadOnlyList<ModelResourceBinding> Bindings(ExecutionGraphNode node) =>
        Structure.Bindings(node);

    public IReadOnlyList<ExecutionNodeId> Dependencies(ExecutionGraphNode node) =>
        Structure.Predecessors(node).OfType<ExecutionGraphNode>()
            .Select(value => value.Id).ToArray();

    private void Validate()
    {
        foreach (var node in Nodes)
        {
            var ports = new HashSet<string>(StringComparer.Ordinal);
            foreach (var binding in Bindings(node))
            {
                if (!ports.Add(binding.Port))
                {
                    throw new InvalidDataException(
                        $"Node '{Id}/{node.Id}' has duplicate port '{binding.Port}'.");
                }

                if (!binding.Resource.Owner.IsParent &&
                    Structure.GetNode(binding.Resource.Owner.Call is { } call
                        ? ExecutionGraphNode.NodeGraphId(new ExecutionNodeId(call.Value))
                        : throw new InvalidDataException()) is not ExecutionGraphCallNode)
                {
                    throw new InvalidDataException(
                        $"Node '{Id}/{node.Id}' references unknown graph-call instance " +
                        $"'{binding.Resource.Owner}'.");
                }
            }
        }
        try
        {
            _ = Structure.DependencyView().TopologicalOrder();
        }
        catch (InvalidDataException exception)
        {
            throw new InvalidDataException(
                $"Execution graph '{Id}' contains a dependency cycle.", exception);
        }
    }
}

/// <summary>Builds one immutable execution graph from typed nodes and edges.</summary>
public sealed class ExecutionGraphBuilder
{
    private readonly ExecutionGraphId id;
    private readonly GraphCompileBoundary boundary;
    private readonly GraphArchitecture? architecture;
    private readonly ExecutionGraphSource? source;
    private readonly List<ExecutionGraphStructureNode> nodes = [];
    private readonly List<ExecutionGraphEdge> edges = [];

    /// <summary>Creates an execution graph builder.</summary>
    public ExecutionGraphBuilder(
        ExecutionGraphId id,
        GraphCompileBoundary boundary = GraphCompileBoundary.Auto,
        GraphArchitecture? architecture = null,
        ExecutionGraphSource? source = null)
    {
        this.id = id;
        this.boundary = boundary;
        this.architecture = architecture;
        this.source = source;
    }

    /// <summary>Adds one parent-owned resource node.</summary>
    public void AddResource(ExecutionGraphResource resource) =>
        AddResource(resource, GraphBindingOwner.Parent);

    /// <summary>Adds one scoped resource node.</summary>
    public void AddResource(
        ExecutionGraphResource resource,
        GraphBindingOwner owner)
    {
        ArgumentNullException.ThrowIfNull(resource);
        nodes.Add(new ExecutionResourceNode(
            new ModelResourceNodeId(owner, resource.Id), resource));
    }

    /// <summary>Adds one operation with its complete bindings and dependencies.</summary>
    public void AddOperation(
        ExecutionOperationNode operation,
        IEnumerable<ModelResourceBinding> bindings,
        IEnumerable<ExecutionNodeId>? dependencies = null) =>
        AddNode(operation, bindings, dependencies);

    /// <summary>Adds one graph call with its complete bindings and dependencies.</summary>
    public void AddCall(
        ExecutionGraphCallNode call,
        IEnumerable<ModelResourceBinding> bindings,
        IEnumerable<ExecutionNodeId>? dependencies = null) =>
        AddNode(call, bindings, dependencies);

    /// <summary>Creates and validates the immutable graph.</summary>
    public ExecutionGraph Build() =>
        new(
            id,
            new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(
                nodes, edges),
            boundary,
            architecture,
            source);

    private void AddNode(
        ExecutionGraphNode node,
        IEnumerable<ModelResourceBinding> bindings,
        IEnumerable<ExecutionNodeId>? dependencies)
    {
        ArgumentNullException.ThrowIfNull(node);
        ArgumentNullException.ThrowIfNull(bindings);
        nodes.Add(node);
        foreach (var binding in bindings)
        {
            if (binding.Access != GraphResourceAccess.Write)
            {
                edges.Add(new ExecutionResourceReadEdge(
                    new($"binding:{binding.BindingId}:read"),
                    binding.Resource.GraphId,
                    node.GraphId,
                    binding.Port,
                    binding.BindingId));
            }

            if (binding.Access != GraphResourceAccess.Read)
            {
                edges.Add(new ExecutionResourceWriteEdge(
                    new($"binding:{binding.BindingId}:write"),
                    node.GraphId,
                    binding.Resource.GraphId,
                    binding.Port,
                    binding.BindingId));
            }
        }
        foreach (var dependency in dependencies ?? [])
        {
            edges.Add(new ExecutionDependencyEdge(
                new($"dependency:{dependency.Value}:{node.Id.Value}"),
                ExecutionGraphNode.NodeGraphId(dependency),
                node.GraphId));
        }
    }
}

public sealed record ExecutionGraphCallRelation(
    string Id,
    ExecutionGraphId Source,
    ExecutionGraphId Target,
    ExecutionNodeId Call);

public sealed class ExecutionGraphProgram
{
    public ExecutionGraphProgram(IEnumerable<ExecutionGraph> graphs)
    {
        var values = graphs?.ToArray() ?? throw new ArgumentNullException(nameof(graphs));
        if (values.Length == 0)
        {
            throw new ArgumentException(
                "An execution graph program requires at least one graph.", nameof(graphs));
        }

        var map = new Dictionary<ExecutionGraphId, ExecutionGraph>();
        foreach (var graph in values)
        {
            if (!map.TryAdd(graph.Id, graph))
            {
                throw new InvalidDataException($"Duplicate execution graph '{graph.Id}'.");
            }
        }

        var relations = values.SelectMany(graph =>
            graph.Nodes.OfType<ExecutionGraphCallNode>().Select(call =>
                new ExecutionGraphCallRelation(
                    $"{graph.Id.Value}:{call.Id.Value}",
                    graph.Id,
                    call.Target,
                    call.Id))).ToArray();
        foreach (var relation in relations)
        {
            if (!map.ContainsKey(relation.Target))
            {
                throw new InvalidDataException(
                    $"Graph call '{relation.Source}/{relation.Call}' targets unknown graph " +
                    $"'{relation.Target}'.");
            }
        }

        var referenced = relations.Select(relation => relation.Target).ToHashSet();
        var roots = values.Where(graph => !referenced.Contains(graph.Id)).ToArray();
        if (roots.Length != 1)
        {
            throw new InvalidDataException(
                $"An execution graph program requires exactly one root; found {roots.Length}.");
        }

        ValidateCallDag(roots[0].Id, map, relations);
        Graphs = Array.AsReadOnly(values);
        Calls = Array.AsReadOnly(relations);
        RootGraph = roots[0].Id;
    }

    public IReadOnlyList<ExecutionGraph> Graphs { get; }
    public IReadOnlyList<ExecutionGraphCallRelation> Calls { get; }
    public ExecutionGraphId RootGraph { get; }

    /// <summary>Gets graph-call instances originating in one graph.</summary>
    public IReadOnlyList<ExecutionGraphCallRelation> Callees(
        ExecutionGraphId source) =>
        Calls.Where(call => call.Source == source).ToArray();

    /// <summary>Gets graph-call instances targeting one graph.</summary>
    public IReadOnlyList<ExecutionGraphCallRelation> Callers(
        ExecutionGraphId target) =>
        Calls.Where(call => call.Target == target).ToArray();

    /// <summary>Gets non-root graphs that currently have no callers.</summary>
    public IReadOnlyList<ExecutionGraph> UnreferencedGraphs() =>
        Graphs.Where(graph =>
            graph.Id != RootGraph && Callers(graph.Id).Count == 0).ToArray();

    /// <summary>Returns whether a call is the target graph's only call site.</summary>
    public bool IsExclusiveCall(ExecutionGraphId caller, ExecutionNodeId call)
    {
        var relation = Calls.SingleOrDefault(value =>
            value.Source == caller && value.Call == call)
            ?? throw new ArgumentException(
                $"Execution graph program has no call '{caller}/{call}'.",
                nameof(call));
        return Callers(relation.Target).Count == 1;
    }

    private static void ValidateCallDag(
        ExecutionGraphId root,
        IReadOnlyDictionary<ExecutionGraphId, ExecutionGraph> graphs,
        IReadOnlyList<ExecutionGraphCallRelation> relations)
    {
        var states = new Dictionary<ExecutionGraphId, int>();
        var reachable = new HashSet<ExecutionGraphId>();
        var path = new Stack<ExecutionGraphId>();
        Visit(root);
        if (reachable.Count != graphs.Count)
        {
            var missing = graphs.Keys.Where(id => !reachable.Contains(id));
            throw new InvalidDataException(
                $"Execution graph program contains unreachable graphs: " +
                $"{string.Join(", ", missing)}.");
        }

        void Visit(ExecutionGraphId id)
        {
            if (states.TryGetValue(id, out var state))
            {
                if (state == 1)
                {
                    throw new InvalidDataException(
                        $"Execution graph call cycle detected: " +
                        $"{string.Join(" -> ", path.Reverse())} -> {id}.");
                }

                return;
            }
            states[id] = 1;
            reachable.Add(id);
            path.Push(id);
            foreach (var target in relations.Where(relation => relation.Source == id)
                .Select(relation => relation.Target).Distinct())
            {
                Visit(target);
            }

            path.Pop();
            states[id] = 2;
        }
    }
}

public sealed class ExecutionGraphCompilationContext
{
    public ExecutionGraphCompilationContext(
        string modelType,
        string backend,
        IReadOnlyDictionary<string, string>? modelAttributes = null,
        IReadOnlyDictionary<string, string>? backendAttributes = null,
        IReadOnlyDictionary<string, string>? options = null)
    {
        ModelType = Require(modelType, nameof(modelType));
        Backend = Require(backend, nameof(backend));
        ModelAttributes = Copy(modelAttributes);
        BackendAttributes = Copy(backendAttributes);
        Options = Copy(options);
    }

    public string ModelType { get; }
    public string Backend { get; }
    public IReadOnlyDictionary<string, string> ModelAttributes { get; }
    public IReadOnlyDictionary<string, string> BackendAttributes { get; }
    public IReadOnlyDictionary<string, string> Options { get; }

    private static string Require(string value, string parameter) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException("A nonempty value is required.", parameter)
            : value;

    private static IReadOnlyDictionary<string, string> Copy(
        IReadOnlyDictionary<string, string>? values) =>
        new ReadOnlyDictionary<string, string>(
            new Dictionary<string, string>(
                values ?? new Dictionary<string, string>(),
                StringComparer.Ordinal));
}
