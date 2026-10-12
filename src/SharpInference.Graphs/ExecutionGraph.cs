using System.Collections.ObjectModel;

namespace SharpInference.Graphs;

/// <summary>Identifies one graph in an execution graph program.</summary>
public readonly record struct ExecutionGraphId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Identifies one node within an execution graph.</summary>
public readonly record struct ExecutionNodeId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Architecture-only metadata that processors may use and backends may ignore.</summary>
public sealed record GraphArchitecture(
    string? Name = null,
    string? Role = null,
    string? Description = null,
    IReadOnlyDictionary<string, string>? Attributes = null);

/// <summary>Maps an execution graph back to a logical definition.</summary>
public sealed record ExecutionGraphSource(GraphDefinitionId Definition);

/// <summary>Maps an execution node back to one or more logical nodes and records its transformation.</summary>
public sealed record ExecutionSourceMap(
    IReadOnlyList<GraphDefinitionNodeId> LogicalNodes,
    string? Transformation = null);

/// <summary>One target-independent resource declared by an execution graph.</summary>
public sealed record ExecutionGraphResource(
    ResourceId Id,
    string Name,
    GraphDefinitionResourceKind Kind,
    TensorDescriptor Tensor,
    string? BindingKey = null);

/// <summary>Binds a node port to a graph resource or a direct call instance's internal resource.</summary>
public sealed record ExecutionGraphBinding(
    string Port,
    ResourceId Resource,
    GraphResourceAccess Access,
    GraphBindingOwner Instance = default)
{
    /// <summary>Gets the normalized owner, treating the default value as the containing graph.</summary>
    public GraphBindingOwner ResourceOwner =>
        Instance.IsParent ? GraphBindingOwner.Parent : Instance;
}

/// <summary>Base class for operation and graph-call nodes in an execution graph.</summary>
public abstract record ExecutionGraphNode(
    ExecutionNodeId Id,
    IReadOnlyList<ExecutionGraphBinding> Bindings,
    IReadOnlyList<ExecutionNodeId> Dependencies,
    ExecutionSourceMap Source,
    GraphArchitecture? Architecture);

/// <summary>One concrete operation call in an execution graph.</summary>
public sealed record ExecutionOperationNode(
    ExecutionNodeId Id,
    GraphOperationId Operation,
    IReadOnlyList<ExecutionGraphBinding> Bindings,
    IReadOnlyList<ExecutionNodeId> Dependencies,
    IReadOnlyDictionary<string, string> Attributes,
    PrecisionRequirement Requirements,
    ExecutionSourceMap Source,
    GraphArchitecture? Architecture = null)
    : ExecutionGraphNode(Id, Bindings, Dependencies, Source, Architecture);

/// <summary>One call to another graph in the execution graph program.</summary>
public sealed record ExecutionGraphCallNode(
    ExecutionNodeId Id,
    ExecutionGraphId Target,
    IReadOnlyList<ExecutionGraphBinding> Bindings,
    IReadOnlyList<ExecutionNodeId> Dependencies,
    GraphCompileBoundary CompileBoundary,
    ExecutionSourceMap Source,
    GraphArchitecture? Architecture = null)
    : ExecutionGraphNode(Id, Bindings, Dependencies, Source, Architecture);

/// <summary>One reusable execution graph containing an operation/call DAG.</summary>
public sealed class ExecutionGraph
{
    /// <summary>Creates and validates one execution graph.</summary>
    public ExecutionGraph(
        ExecutionGraphId id,
        IEnumerable<ExecutionGraphResource> resources,
        IEnumerable<ExecutionGraphNode> nodes,
        GraphCompileBoundary defaultCompileBoundary = GraphCompileBoundary.Auto,
        GraphArchitecture? architecture = null,
        ExecutionGraphSource? source = null)
    {
        if (string.IsNullOrWhiteSpace(id.Value))
            throw new ArgumentException("An execution graph identifier is required.", nameof(id));
        if (!Enum.IsDefined(defaultCompileBoundary))
            throw new ArgumentOutOfRangeException(nameof(defaultCompileBoundary));
        Id = id;
        Resources = Array.AsReadOnly(resources?.ToArray()
            ?? throw new ArgumentNullException(nameof(resources)));
        Nodes = Array.AsReadOnly(nodes?.ToArray()
            ?? throw new ArgumentNullException(nameof(nodes)));
        DefaultCompileBoundary = defaultCompileBoundary;
        Architecture = architecture;
        Source = source;
        Validate();
    }

    /// <summary>Gets the graph identifier.</summary>
    public ExecutionGraphId Id { get; }

    /// <summary>Gets resources declared by this graph.</summary>
    public IReadOnlyList<ExecutionGraphResource> Resources { get; }

    /// <summary>Gets operation and graph-call nodes in this graph.</summary>
    public IReadOnlyList<ExecutionGraphNode> Nodes { get; }

    /// <summary>Gets the graph's default compilation boundary.</summary>
    public GraphCompileBoundary DefaultCompileBoundary { get; }

    /// <summary>Gets optional architecture metadata.</summary>
    public GraphArchitecture? Architecture { get; }

    /// <summary>Gets the logical source definition.</summary>
    public ExecutionGraphSource? Source { get; }

    private void Validate()
    {
        var resources = Unique(Resources, resource => resource.Id, "resource");
        var nodes = Unique(Nodes, node => node.Id, "node");
        foreach (var node in Nodes)
        {
            if (node.Dependencies.Any(dependency => !nodes.ContainsKey(dependency)))
                throw new InvalidDataException(
                    $"Node '{Id}/{node.Id}' has an unknown dependency.");
            if (node.Dependencies.Contains(node.Id))
                throw new InvalidDataException(
                    $"Node '{Id}/{node.Id}' depends on itself.");
            foreach (var binding in node.Bindings)
            {
                if (binding.ResourceOwner.IsParent &&
                    !resources.ContainsKey(binding.Resource))
                    throw new InvalidDataException(
                        $"Node '{Id}/{node.Id}' references unknown resource '{binding.Resource}'.");
                if (!binding.ResourceOwner.IsParent)
                {
                    if (!nodes.TryGetValue(
                            new ExecutionNodeId(binding.ResourceOwner.Call!.Value.Value),
                            out var owner) ||
                        owner is not ExecutionGraphCallNode)
                        throw new InvalidDataException(
                            $"Node '{Id}/{node.Id}' references unknown graph-call instance '{binding.ResourceOwner}'.");
                }
            }
        }
        ValidateAcyclic(nodes);
    }

    private static Dictionary<TKey, TValue> Unique<TValue, TKey>(
        IEnumerable<TValue> values,
        Func<TValue, TKey> key,
        string kind)
        where TKey : notnull
    {
        var result = new Dictionary<TKey, TValue>();
        foreach (var value in values)
            if (!result.TryAdd(key(value), value))
                throw new InvalidDataException($"Duplicate execution graph {kind} '{key(value)}'.");
        return result;
    }

    private static void ValidateAcyclic(
        IReadOnlyDictionary<ExecutionNodeId, ExecutionGraphNode> nodes)
    {
        var states = new Dictionary<ExecutionNodeId, int>();
        foreach (var id in nodes.Keys)
            Visit(id);

        void Visit(ExecutionNodeId id)
        {
            if (states.TryGetValue(id, out var state))
            {
                if (state == 1)
                    throw new InvalidDataException($"Execution node cycle detected at '{id}'.");
                return;
            }
            states[id] = 1;
            foreach (var dependency in nodes[id].Dependencies)
                Visit(dependency);
            states[id] = 2;
        }
    }
}

/// <summary>A validated collection of reusable execution graphs with one inferred root.</summary>
public sealed class ExecutionGraphProgram
{
    /// <summary>Creates a program, infers its root, and validates its graph-call DAG.</summary>
    public ExecutionGraphProgram(IEnumerable<ExecutionGraph> graphs)
    {
        var values = graphs?.ToArray() ?? throw new ArgumentNullException(nameof(graphs));
        if (values.Length == 0)
            throw new ArgumentException("An execution graph program requires at least one graph.", nameof(graphs));
        var map = new Dictionary<ExecutionGraphId, ExecutionGraph>();
        foreach (var graph in values)
            if (!map.TryAdd(graph.Id, graph))
                throw new InvalidDataException($"Duplicate execution graph '{graph.Id}'.");

        var referenced = new HashSet<ExecutionGraphId>();
        foreach (var graph in values)
            foreach (var call in graph.Nodes.OfType<ExecutionGraphCallNode>())
            {
                if (!map.ContainsKey(call.Target))
                    throw new InvalidDataException(
                        $"Graph call '{graph.Id}/{call.Id}' targets unknown graph '{call.Target}'.");
                referenced.Add(call.Target);
            }

        var roots = values.Where(graph => !referenced.Contains(graph.Id)).ToArray();
        if (roots.Length != 1)
            throw new InvalidDataException(
                $"An execution graph program requires exactly one root; found {roots.Length}.");
        ValidateCallDag(roots[0].Id, map);
        Graphs = Array.AsReadOnly(values);
        RootGraph = roots[0].Id;
    }

    /// <summary>Gets all reusable execution graphs.</summary>
    public IReadOnlyList<ExecutionGraph> Graphs { get; }

    /// <summary>Gets the single graph not referenced by another graph call.</summary>
    public ExecutionGraphId RootGraph { get; }

    private static void ValidateCallDag(
        ExecutionGraphId root,
        IReadOnlyDictionary<ExecutionGraphId, ExecutionGraph> graphs)
    {
        var states = new Dictionary<ExecutionGraphId, int>();
        var reachable = new HashSet<ExecutionGraphId>();
        var path = new Stack<ExecutionGraphId>();
        Visit(root);
        if (reachable.Count != graphs.Count)
        {
            var missing = graphs.Keys.Where(id => !reachable.Contains(id));
            throw new InvalidDataException(
                $"Execution graph program contains unreachable graphs: {string.Join(", ", missing)}.");
        }

        void Visit(ExecutionGraphId id)
        {
            if (states.TryGetValue(id, out var state))
            {
                if (state == 1)
                    throw new InvalidDataException(
                        $"Execution graph call cycle detected: " +
                        $"{string.Join(" -> ", path.Reverse())} -> {id}.");
                return;
            }
            states[id] = 1;
            reachable.Add(id);
            path.Push(id);
            foreach (var target in graphs[id].Nodes.OfType<ExecutionGraphCallNode>()
                .Select(call => call.Target).Distinct())
                Visit(target);
            path.Pop();
            states[id] = 2;
        }
    }
}

/// <summary>Provides model and backend information to conversion and postprocessing.</summary>
public sealed class ExecutionGraphCompilationContext
{
    /// <summary>Creates a compilation context.</summary>
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
