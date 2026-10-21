using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Execution;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Vm.Optimization;

/// <summary>Removes redundant dependency edges without changing dependency reachability.</summary>
public sealed class DependencyReductionPostprocessor : IExecutionGraphPostprocessor
{
    public string Name => "reduce-execution-dependencies";

    public GraphPackage<ExecutionGraph> Process(GraphPackage<ExecutionGraph> package, ExecutionPassContext context)
    {
        ArgumentNullException.ThrowIfNull(package);
        ArgumentNullException.ThrowIfNull(context);
        GraphValidation.ThrowIfErrors(package.Validate());
        var result = package.Snapshot();
        foreach (var graph in result)
        {
            var structure = graph.DependencyGraph();
            var reach = structure.CreateReachabilityIndex(structure.Nodes, structure.Nodes);
            var unique = graph.GraphFlow.Distinct().ToArray();
            graph.GraphFlow.Clear();
            foreach (var edge in unique)
            {
                var redundant = unique.Any(other => other.Source == edge.Source && other.Target != edge.Target &&
                    reach.IsReachable(new GraphNodeId(other.Target.Action), new GraphNodeId(edge.Target.Action)));
                if (!redundant)
                    graph.GraphFlow.Add(edge);
            }
        }
        return result;
    }

}
