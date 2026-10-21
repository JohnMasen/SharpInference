using SharpInference.Graphs.Core;
using SharpInference.Instructions;

namespace SharpInference.Graphs;

public static class GraphQueries
{
    public static IEnumerable<GraphAction> Readers(this Graph graph, ResourceRef resource) =>
        graph.Actions.Where(action => action.Bindings.Any(binding =>
            binding.Access != ResourceAccess.Write && SameReference(graph, binding.Target, resource)));

    public static IEnumerable<GraphAction> Writers(this Graph graph, ResourceRef resource) =>
        graph.Actions.Where(action => action.Bindings.Any(binding =>
            binding.Access != ResourceAccess.Read && SameReference(graph, binding.Target, resource)));

    public static IEnumerable<GraphAction> Readers(this Graph graph, Resource resource) =>
        graph.Readers(new ResourceRef(resource));

    public static IEnumerable<GraphAction> Writers(this Graph graph, Resource resource) =>
        graph.Writers(new ResourceRef(resource));

    public static IEnumerable<ActionRef> Dependencies(this Graph graph, GraphAction action) =>
        graph.GraphFlow.Where(flow => flow.Target == new ActionRef(action)).Select(flow => flow.Source);

    public static Core.Graph<ActionNode, FlowEdge> DependencyGraph(this Graph graph) =>
        new(graph.Actions.Select(action => new ActionNode(action)),
            graph.GraphFlow.Select((flow, index) => new FlowEdge(new(index.ToString()),
                new(flow.Source.Action), new(flow.Target.Action))));

    private static bool SameReference(Graph graph, ResourceRef left, ResourceRef right) =>
        left.TargetKind == right.TargetKind && left.TargetProperty == right.TargetProperty &&
        (left.TargetKind == ResourceTarget.Graph
            ? (left.Target ?? graph.Id) == (right.Target ?? graph.Id)
            : left.Target == right.Target);
}

public sealed record ActionNode(GraphAction Action) : IInferenceGraphNode
{
    public GraphNodeId Id { get; } = new(Action.Id);
    public string NodeKind { get; } = Action.Category;
}

public sealed record FlowEdge(GraphEdgeId Id, GraphNodeId Source, GraphNodeId Target) : IInferenceGraphEdge
{
    public string EdgeKind => "flow";
}
