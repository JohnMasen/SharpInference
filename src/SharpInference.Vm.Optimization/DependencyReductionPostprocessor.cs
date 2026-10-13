using SharpInference.Graphs;
using SharpInference.Graphs.Core;

namespace SharpInference.Vm.Optimization;

/// <summary>Removes redundant dependency edges without changing dependency reachability.</summary>
public sealed class DependencyReductionPostprocessor : IExecutionGraphPostprocessor
{
    public string Name => "reduce-execution-dependencies";

    public ExecutionGraphProgram Process(
        ExecutionGraphProgram program, ExecutionGraphPostprocessContext context)
    {
        ArgumentNullException.ThrowIfNull(program);
        ArgumentNullException.ThrowIfNull(context);
        return new(program.Graphs.Select(Reduce));
    }

    private static ExecutionGraph Reduce(ExecutionGraph graph)
    {
        var analysis = new ExecutionDependencyAnalysis(graph);
        var seen = new HashSet<(GraphNodeId, GraphNodeId)>();
        var plan = new GraphRewritePlan<ExecutionGraphStructureNode, ExecutionGraphEdge>();
        var changed = false;
        foreach (var edge in graph.Structure.Edges.OfType<ExecutionDependencyEdge>())
        {
            var source = analysis.Position(edge.Source);
            var target = analysis.Position(edge.Target);
            if (!seen.Add((edge.Source, edge.Target)) ||
                analysis.Successors[source].Any(successor =>
                    successor != target && analysis.Before(successor, target)))
            {
                plan.RemoveEdge(edge.Id);
                changed = true;
            }
        }
        return changed
            ? ExecutionDependencyAnalysis.WithStructure(graph, plan.Apply(graph.Structure))
            : graph;
    }
}
