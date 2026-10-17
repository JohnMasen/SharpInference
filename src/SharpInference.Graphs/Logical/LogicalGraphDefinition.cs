using SharpInference.Instructions;
using System.Collections.ObjectModel;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Internal;

namespace SharpInference.Graphs.Logical;

/// <summary>A logical graph snapshot with named resources, calls, and dependency analysis.</summary>
public sealed class LogicalGraphDefinition
{
    internal LogicalGraphDefinition(
        GraphDefinitionId id, GraphCompileBoundary compileBoundary, GraphAnnotation? annotation,
        Graph<LogicalGraphNode, LogicalGraphEdge> structure,
        IEnumerable<LogicalGraphDefinition> definitions, bool rejectErrors = true)
    {
        Id = id;
        CompileBoundary = compileBoundary;
        Annotation = annotation;
        Structure = structure ?? throw new ArgumentNullException(nameof(structure));
        Definitions = Array.AsReadOnly(definitions.ToArray());
        ModelGraphConstruction.ThrowIfErrors(ModelGraphConstruction.Validate(
            id.Value, structure.Nodes, structure.Edges, node => node is GraphCallNode));
        var analysis = LogicalGraphDefinitionValidator.Analyze(this);
        CompletionSinks = Array.AsReadOnly(analysis.CompletionSinks.ToArray());
        TerminalWriters = new ReadOnlyDictionary<ResourceId, IReadOnlyList<GraphDefinitionNodeId>>(
            analysis.TerminalWriters.ToDictionary(pair => pair.Key,
                pair => (IReadOnlyList<GraphDefinitionNodeId>)Array.AsReadOnly(pair.Value)));
        Diagnostics = Array.AsReadOnly(analysis.Diagnostics.Select(diagnostic =>
            diagnostic with { Graph = id.Value }).ToArray());
        if (rejectErrors)
            ModelGraphConstruction.ThrowIfErrors(Diagnostics);
    }

    public GraphDefinitionId Id { get; }
    public GraphCompileBoundary CompileBoundary { get; }
    public GraphAnnotation? Annotation { get; }
    public Graph<LogicalGraphNode, LogicalGraphEdge> Structure { get; }
    public IReadOnlyList<GraphDefinitionResource> Resources =>
        Structure.NodesOfType<LogicalResourceNode>()
            .Where(node => node.ResourceNodeId.Owner.IsParent).Select(node => node.Resource).ToArray();
    public IReadOnlyList<GraphDefinitionNode> Nodes =>
        Structure.Nodes.OfType<GraphDefinitionNode>().ToArray();
    public IReadOnlyList<LogicalGraphDefinition> Definitions { get; }
    public IReadOnlyList<GraphDefinitionNodeId> CompletionSinks { get; }
    public IReadOnlyDictionary<ResourceId, IReadOnlyList<GraphDefinitionNodeId>> TerminalWriters { get; }
    public IReadOnlyList<GraphDiagnostic> Diagnostics { get; }
    public IReadOnlyList<GraphDiagnostic> Validate() => Diagnostics;

    public IReadOnlyList<ModelResourceBinding> Bindings(GraphDefinitionNode node) => Structure.Bindings(node);
    public IReadOnlyList<GraphDefinitionNodeId> Dependencies(GraphDefinitionNode node) =>
        Structure.Predecessors(node).OfType<GraphDefinitionNode>().Select(value => value.Id).ToArray();

    public static LogicalGraphDefinition Define(string id, Action<GraphDefinitionBuilder> build,
        GraphCompileBoundary compileBoundary = GraphCompileBoundary.Auto, GraphAnnotation? annotation = null)
    {
        ArgumentNullException.ThrowIfNull(build);
        var builder = new GraphDefinitionBuilder(id, compileBoundary, annotation);
        build(builder);
        return builder.Build();
    }

    public static LogicalGraphDefinition CreateListGraph(string id, TensorDescriptor tensor,
        IEnumerable<OperationId> operations, GraphCompileBoundary compileBoundary = GraphCompileBoundary.Auto,
        GraphAnnotation? annotation = null)
    {
        ArgumentNullException.ThrowIfNull(operations);
        var values = operations.ToArray();
        if (values.Length == 0)
            throw new ArgumentException("A list graph requires at least one operation.", nameof(operations));
        return Define(id, builder =>
        {
            var source = builder.AddInput("input", tensor);
            var output = builder.AddOutput("output", tensor);
            for (var index = 0; index < values.Length; index++)
            {
                var target = index == values.Length - 1 ? output :
                    builder.AddResource($"step-{index}-output", $"Step {index} output", tensor);
                builder.AddNode($"step-{index}", values[index], node =>
                {
                    node.AddInput("input", source);
                    node.AddOutput("output", target);
                    if (index != 0)
                        node.DependsOn(new($"step-{index - 1}"));
                });
                source = target;
            }
        }, compileBoundary, annotation);
    }

    internal static string RequireId(string value, string parameterName) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException("A nonblank identifier is required.", parameterName)
            : value;
}
