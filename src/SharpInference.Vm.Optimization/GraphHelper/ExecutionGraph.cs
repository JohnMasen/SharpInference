using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Internal;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Holds an immutable typed execution structure with validated bindings and dependency ordering.</summary>
/// <remarks>Validation checks local binding ports, scoped resource owners, and dependency cycles.
/// Program-level call targets and reachability are validated by <see cref="ExecutionGraphProgram"/>.</remarks>
public sealed class ExecutionGraph
{
    /// <summary>Creates a graph and validates its bindings and dependency-only topology.</summary>
    /// <param name="id">The nonblank execution graph identity.</param>
    /// <param name="structure">The typed resource, operation, call, and edge structure.</param>
    /// <param name="defaultCompileBoundary">Defined compilation-boundary metadata; no lowering policy is applied here.</param>
    /// <param name="architecture">Optional descriptive architecture metadata.</param>
    /// <param name="source">Optional originating logical definition.</param>
    /// <exception cref="ArgumentException">The graph identity is blank or duplicate binding identities cannot be reconstructed.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The boundary value is undefined.</exception>
    /// <exception cref="ArgumentNullException">The structure is null.</exception>
    /// <exception cref="InvalidDataException">Bindings are inconsistent, ports repeat, a scoped owner is not a call,
    /// or the dependency-only view contains a cycle.</exception>
    /// <exception cref="KeyNotFoundException">A binding refers to a missing scoped call owner.</exception>
    public ExecutionGraph(
        ExecutionGraphId id,
        Graph<ExecutionGraphStructureNode, ExecutionGraphEdge> structure,
        GraphCompileBoundary defaultCompileBoundary = GraphCompileBoundary.Auto,
        GraphArchitecture? architecture = null,
        ExecutionGraphSource? source = null)
        : this(id, structure, defaultCompileBoundary, architecture, source, true)
    {
    }

    internal ExecutionGraph(
        ExecutionGraphId id,
        Graph<ExecutionGraphStructureNode, ExecutionGraphEdge> structure,
        GraphCompileBoundary defaultCompileBoundary,
        GraphArchitecture? architecture,
        ExecutionGraphSource? source,
        bool rejectErrors)
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
        ModelGraphConstruction.ThrowIfErrors(ModelGraphConstruction.Validate(
            id.Value, structure.Nodes, structure.Edges, node => node is ExecutionGraphCallNode));
        if (rejectErrors)
            ModelGraphConstruction.ThrowIfErrors(Validate());
    }

    /// <summary>Gets the execution graph identity.</summary>
    public ExecutionGraphId Id { get; }
    /// <summary>Gets the immutable typed structural graph.</summary>
    public Graph<ExecutionGraphStructureNode, ExecutionGraphEdge> Structure { get; }
    /// <summary>Gets parent-owned resource descriptions in structural node order, excluding call-instance resources.</summary>
    public IReadOnlyList<ExecutionGraphResource> Resources =>
        Structure.NodesOfType<ExecutionResourceNode>()
            .Where(node => node.ResourceNodeId.Owner.IsParent)
            .Select(node => node.Resource).ToArray();
    /// <summary>Gets operation and call nodes in structural node order, excluding resources.</summary>
    public IReadOnlyList<ExecutionGraphNode> Nodes =>
        Structure.Filter(node: node => node is ExecutionGraphNode).Nodes.Cast<ExecutionGraphNode>().ToArray();
    /// <summary>Gets the graph's compilation-boundary metadata.</summary>
    public GraphCompileBoundary DefaultCompileBoundary { get; }
    /// <summary>Gets optional descriptive architecture metadata.</summary>
    public GraphArchitecture? Architecture { get; }
    /// <summary>Gets the optional originating logical definition.</summary>
    public ExecutionGraphSource? Source { get; }

    /// <summary>Reconstructs a node's bindings from its read and write edges.</summary>
    /// <param name="node">The operation or call whose accesses are queried by structural identity.</param>
    /// <returns>Bindings in first-edge occurrence order, merging reads and writes with the same binding identity.</returns>
    /// <exception cref="InvalidDataException">Paired accesses disagree or an access endpoint is not a resource.</exception>
    /// <exception cref="ArgumentException">Read or write edges repeat a binding identity.</exception>
    public IReadOnlyList<ModelResourceBinding> Bindings(ExecutionGraphNode node) =>
        Structure.Bindings(node);

    /// <summary>Gets direct predecessor execution identities from dependency edges only.</summary>
    /// <param name="node">The operation or call whose prerequisites are queried.</param>
    /// <returns>Predecessor identities in incoming-edge order, without deduplication.</returns>
    public IReadOnlyList<ExecutionNodeId> Dependencies(ExecutionGraphNode node) =>
        Structure.Predecessors(node).OfType<ExecutionGraphNode>()
            .Select(value => value.Id).ToArray();

    /// <summary>Checks binding ports, call-instance owners, and the dependency-only directed acyclic graph.</summary>
    /// <exception cref="InvalidDataException">Bindings are inconsistent, ports repeat, scoped owners are not calls,
    /// or dependency edges contain a cycle.</exception>
    /// <exception cref="ArgumentException">Read or write edges repeat a binding identity.</exception>
    /// <exception cref="KeyNotFoundException">A scoped resource owner does not exist in the structure.</exception>
    public IReadOnlyList<GraphDiagnostic> Validate()
    {
        var diagnostics = new List<GraphDiagnostic>();
        foreach (var node in Nodes)
        {
            foreach (var binding in Bindings(node))
            {
                var resource = (ExecutionResourceNode)Structure.GetNode(binding.Resource.GraphId);
                if (resource.Kind is GraphDefinitionResourceKind.Weight or GraphDefinitionResourceKind.Constant &&
                    binding.Access != ResourceAccess.Read)
                    diagnostics.Add(new(GraphDiagnosticSeverity.Error, "EG0004",
                        $"Node '{node.Id}' writes read-only resource '{resource.Resource.Id}'.",
                        node.Id.Value, resource.Resource.Id, binding.Port, Id.Value));
            }
        }
        try
        {
            _ = Structure.DependencyView().TopologicalOrder();
        }
        catch (InvalidDataException exception)
        {
            diagnostics.Add(new(GraphDiagnosticSeverity.Error, "EG0001",
                $"Execution graph '{Id}' contains a dependency cycle: {exception.Message}", Graph: Id.Value));
        }
        return diagnostics;
    }
}
