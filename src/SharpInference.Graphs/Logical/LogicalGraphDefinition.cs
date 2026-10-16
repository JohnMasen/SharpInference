using System.Collections.ObjectModel;
using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Logical;

/// <summary>A validated logical graph with scoped resources, operation dependencies, and completion analysis.</summary>
/// <remarks>Construction rejects error diagnostics; warnings remain available on the completed definition.</remarks>
public sealed class LogicalGraphDefinition
{
    /// <summary>Stores the graph structure and child definitions, analyzes relationships, and rejects errors.</summary>
    /// <param name="id">The definition identity.</param>
    /// <param name="compileBoundary">The requested compilation-boundary policy.</param>
    /// <param name="annotation">Optional descriptive metadata.</param>
    /// <param name="structure">The operation, scoped resource, and relationship structure.</param>
    /// <param name="definitions">The directly referenced child definitions.</param>
    /// <exception cref="ArgumentNullException">The structure or definitions sequence is null.</exception>
    /// <exception cref="InvalidDataException">Binding reconstruction fails or logical analysis reports errors.</exception>
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

    /// <summary>Gets the definition identity.</summary>
    public GraphDefinitionId Id { get; }
    /// <summary>Gets the requested compilation-boundary policy.</summary>
    public GraphCompileBoundary CompileBoundary { get; }
    /// <summary>Gets optional definition metadata.</summary>
    public GraphAnnotation? Annotation { get; }
    /// <summary>Gets the shared graph containing exact nodes and relationship edges.</summary>
    public Graph<LogicalGraphNode, LogicalGraphEdge> Structure { get; }
    /// <summary>Gets declarations from parent-scoped resource nodes, excluding instance-scoped resources.</summary>
    public IReadOnlyList<GraphDefinitionResource> Resources =>
        Structure.NodesOfType<LogicalResourceNode>()
            .Where(node => node.ResourceNodeId.Owner.IsParent)
            .Select(node => node.Resource).ToArray();
    /// <summary>Gets operation and call nodes in structure order.</summary>
    public IReadOnlyList<GraphDefinitionNode> Nodes =>
        Structure.Filter(node: node => node is GraphDefinitionNode).Nodes.Cast<GraphDefinitionNode>().ToArray();
    /// <summary>Gets the supplied directly referenced child definitions.</summary>
    public IReadOnlyList<LogicalGraphDefinition> Definitions { get; }
    /// <summary>Gets operations with no direct dependency successors, independent of resource-access edges.</summary>
    public IReadOnlyList<GraphDefinitionNodeId> CompletionSinks { get; }
    /// <summary>Gets dependency-maximal writers of each parent-scoped output and input/output resource.</summary>
    /// <remarks>Several unordered terminal writers produce a warning rather than an error.</remarks>
    public IReadOnlyDictionary<ResourceId, IReadOnlyList<GraphDefinitionNodeId>> TerminalWriters { get; }
    /// <summary>Gets analysis findings; a successfully constructed definition contains no error findings.</summary>
    public IReadOnlyList<LogicalGraphDiagnostic> Diagnostics { get; }

    /// <summary>Reconstructs resource bindings from an operation's read and write edges.</summary>
    /// <param name="node">The operation or call whose accesses are queried.</param>
    /// <returns>Bindings in first-edge order, pairing read and write edges by binding identity.</returns>
    /// <exception cref="InvalidDataException">Paired accesses disagree about their port or resource, or an endpoint is not a resource node.</exception>
    public IReadOnlyList<ModelResourceBinding> Bindings(GraphDefinitionNode node) =>
        Structure.Bindings(node);

    /// <summary>Gets an operation's direct execution prerequisites, not its data-flow neighbors.</summary>
    /// <param name="node">The operation or call whose incoming dependency edges are queried.</param>
    /// <returns>Prerequisite operation identities in incoming-edge order without additional deduplication.</returns>
    public IReadOnlyList<GraphDefinitionNodeId> Dependencies(GraphDefinitionNode node) =>
        Structure.Predecessors(node).OfType<GraphDefinitionNode>().Select(value => value.Id).ToArray();

    /// <summary>Creates typed public ports, runs a definition builder, and validates the resulting graph.</summary>
    /// <typeparam name="TContract">The self-creating public-port contract.</typeparam>
    /// <param name="id">The nonblank definition identifier.</param>
    /// <param name="build">The callback that adds resources, operations, and calls.</param>
    /// <param name="compileBoundary">The requested compilation-boundary policy.</param>
    /// <param name="annotation">Optional definition metadata.</param>
    /// <returns>The validated definition and its typed definition-level ports.</returns>
    /// <exception cref="ArgumentNullException">The build callback is null.</exception>
    /// <exception cref="ArgumentException">The definition identifier is blank.</exception>
    /// <exception cref="InvalidOperationException">Port or resource declarations conflict during construction.</exception>
    /// <exception cref="InvalidDataException">The constructed graph has invalid structure or error diagnostics.</exception>
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

    /// <summary>Creates a sequential operation chain between exactly one pure input and one pure output.</summary>
    /// <typeparam name="TContract">The public-port contract, which must declare exactly one Input and one Output resource.</typeparam>
    /// <param name="id">The nonblank definition identifier.</param>
    /// <param name="operations">The nonempty operation sequence in execution order.</param>
    /// <param name="compileBoundary">The requested compilation-boundary policy.</param>
    /// <param name="annotation">Optional definition metadata.</param>
    /// <returns>The validated typed graph whose steps bind ports named <c>input</c> and <c>output</c>.</returns>
    /// <remarks>
    /// Steps are named <c>step-{index}</c> and explicitly depend on their immediate predecessor.
    /// Intermediate temporary resources use the public output's tensor descriptor. Other resource
    /// kinds are not included in the one-input/one-output count and remain subject to validation.
    /// </remarks>
    /// <exception cref="ArgumentNullException">The operations sequence is null.</exception>
    /// <exception cref="ArgumentException">The operations sequence is empty or the definition identifier is blank.</exception>
    /// <exception cref="InvalidOperationException">The contract lacks exactly one pure input and output, or declarations conflict.</exception>
    /// <exception cref="InvalidDataException">The resulting graph fails structural or logical validation.</exception>
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

    /// <summary>Requires a nonblank identifier without trimming or otherwise normalizing it.</summary>
    /// <param name="value">The proposed identifier.</param>
    /// <param name="parameterName">The parameter name included in a failure.</param>
    /// <returns>The original identifier string.</returns>
    /// <exception cref="ArgumentException">The value is null, empty, or whitespace.</exception>
    internal static string RequireId(string value, string parameterName) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException("An identifier is required.", parameterName)
            : value;
}
