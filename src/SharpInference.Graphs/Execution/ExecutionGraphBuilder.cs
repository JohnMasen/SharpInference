using SharpInference.Graphs.Core;
using SharpInference.Graphs.Internal;

namespace SharpInference.Graphs.Execution;

/// <summary>Builds one immutable execution graph from typed nodes and edges.</summary>
public sealed class ExecutionGraphBuilder
{
    /// <summary>The identity assigned when the graph is built.</summary>
    private readonly ExecutionGraphId id;
    /// <summary>The compilation-boundary metadata assigned to the built graph.</summary>
    private readonly GraphCompileBoundary boundary;
    /// <summary>The optional architecture metadata assigned to the built graph.</summary>
    private readonly GraphArchitecture? architecture;
    /// <summary>The optional logical provenance assigned to the built graph.</summary>
    private readonly ExecutionGraphSource? source;
    /// <summary>The nodes accumulated in declaration order.</summary>
    private readonly List<ExecutionGraphStructureNode> nodes = [];
    /// <summary>The access and dependency edges accumulated in declaration order.</summary>
    private readonly List<ExecutionGraphEdge> edges = [];

    /// <summary>Creates an execution graph builder.</summary>
    /// <param name="id">The identity assigned to the resulting graph.</param>
    /// <param name="boundary">The boundary metadata, validated when building.</param>
    /// <param name="architecture">Optional architecture metadata.</param>
    /// <param name="source">Optional originating logical definition.</param>
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
    /// <param name="resource">The resource description to add.</param>
    /// <exception cref="ArgumentNullException">The resource is null.</exception>
    public void AddResource(ExecutionGraphResource resource) =>
        AddResource(resource, GraphBindingOwner.Parent);

    /// <summary>Adds one scoped resource node.</summary>
    /// <param name="resource">The resource description to add.</param>
    /// <param name="owner">The parent or graph-call instance owning the resource.</param>
    /// <exception cref="ArgumentNullException">The resource is null.</exception>
    public void AddResource(
        ExecutionGraphResource resource,
        GraphBindingOwner owner)
    {
        ArgumentNullException.ThrowIfNull(resource);
        var resourceId = new ModelResourceNodeId(owner, resource.Id);
        if (nodes.Any(node => node.GraphId == resourceId.GraphId))
            throw new InvalidOperationException($"Resource '{resourceId}' is already declared in '{id}'.");
        nodes.Add(new ExecutionResourceNode(
            resourceId, resource));
    }

    /// <summary>Adds one operation with its complete bindings and dependencies.</summary>
    /// <param name="operation">The operation to append.</param>
    /// <param name="bindings">Bindings expanded into read edges, write edges, or paired edges.</param>
    /// <param name="dependencies">Optional prerequisite execution identities; null means none.</param>
    /// <exception cref="ArgumentNullException">The operation or bindings are null.</exception>
    public void AddOperation(
        ExecutionOperationNode operation,
        IEnumerable<ModelResourceBinding> bindings,
        IEnumerable<ExecutionNodeId>? dependencies = null) =>
        AddNode(operation, bindings, dependencies);

    /// <summary>Adds one graph call with its complete bindings and dependencies.</summary>
    /// <param name="call">The graph-call instance to append.</param>
    /// <param name="bindings">Bindings expanded into read edges, write edges, or paired edges.</param>
    /// <param name="dependencies">Optional prerequisite execution identities; null means none.</param>
    /// <exception cref="ArgumentNullException">The call or bindings are null.</exception>
    public void AddCall(
        ExecutionGraphCallNode call,
        IEnumerable<ModelResourceBinding> bindings,
        IEnumerable<ExecutionNodeId>? dependencies = null) =>
        AddNode(call, bindings, dependencies);

    /// <summary>Creates and validates the immutable graph.</summary>
    /// <returns>A new graph built from the currently accumulated nodes, edges, and metadata.</returns>
    /// <exception cref="ArgumentException">The graph identity is blank or binding identities repeat.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The boundary value is undefined.</exception>
    /// <exception cref="InvalidDataException">Structural identities or endpoints, bindings, or dependency topology are invalid.</exception>
    /// <exception cref="KeyNotFoundException">A scoped resource owner does not identify an existing node.</exception>
    public IReadOnlyList<GraphDiagnostic> Validate()
    {
        var diagnostics = ModelGraphConstruction.Validate(id.Value, nodes, edges,
            node => node is ExecutionGraphCallNode).ToList();
        if (string.IsNullOrWhiteSpace(id.Value))
            diagnostics.Add(new(GraphDiagnosticSeverity.Error, "EG0002",
                "An execution graph identifier is required.", Graph: id.Value));
        if (!Enum.IsDefined(boundary))
            diagnostics.Add(new(GraphDiagnosticSeverity.Error, "EG0003",
                "The execution graph has an invalid compilation boundary.", Graph: id.Value));
        return diagnostics.Count != 0 ? diagnostics : Create(false).Validate();
    }

    public ExecutionGraph Build()
    {
        ModelGraphConstruction.ThrowIfErrors(Validate());
        return Create(true);
    }

    private ExecutionGraph Create(bool rejectErrors) =>
        new(id, new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(nodes, edges),
            boundary, architecture, source, rejectErrors);

    /// <summary>Appends an executable node and expands binding accesses and prerequisites into typed edges.</summary>
    /// <param name="node">The operation or call to append.</param>
    /// <param name="bindings">The bindings whose identities also determine generated edge identities.</param>
    /// <param name="dependencies">Optional local identities of dependency predecessors.</param>
    /// <exception cref="ArgumentNullException">The node or bindings are null.</exception>
    private void AddNode(
        ExecutionGraphNode node,
        IEnumerable<ModelResourceBinding> bindings,
        IEnumerable<ExecutionNodeId>? dependencies)
    {
        ArgumentNullException.ThrowIfNull(node);
        ArgumentNullException.ThrowIfNull(bindings);
        if (nodes.Any(existing => existing.GraphId == node.GraphId))
            throw new InvalidOperationException($"Node '{node.Id}' is already declared in '{id}'.");
        var values = bindings.ToArray();
        if (values.Any(binding => !Enum.IsDefined(binding.Access)))
            throw new ArgumentOutOfRangeException(nameof(bindings));
        nodes.Add(node);
        edges.AddRange(ModelGraphConstruction.ExpandBindings<ExecutionGraphEdge>(
            node.GraphId, values,
            (id, from, to, port, binding) => new ExecutionResourceReadEdge(id, from, to, port, binding),
            (id, from, to, port, binding) => new ExecutionResourceWriteEdge(id, from, to, port, binding)));
        foreach (var dependency in (dependencies ?? []).Distinct())
        {
            edges.Add(new ExecutionDependencyEdge(
                new($"dependency:{dependency.Value}:{node.Id.Value}"),
                ExecutionGraphNode.NodeGraphId(dependency),
                node.GraphId));
        }
    }
}
