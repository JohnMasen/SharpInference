using SharpInference.Graphs;
using SharpInference.Graphs.Core;

namespace SharpInference.Vm.Optimization;

internal sealed class ExecutionDependencyAnalysis
{
    private readonly Dictionary<GraphNodeId, int> positions;
    private readonly ulong[][] reachable;

    public ExecutionDependencyAnalysis(ExecutionGraph graph)
    {
        Order = graph.Structure.DependencyView().TopologicalOrder()
            .Cast<ExecutionGraphNode>().ToArray();
        positions = Order.Select((node, index) => (node.GraphId, index))
            .ToDictionary(item => item.GraphId, item => item.index);
        Successors = Order.Select(node => graph.Structure.Successors(node)
            .Select(successor => positions[successor.GraphId]).Distinct().ToArray()).ToArray();
        reachable = Order.Select(_ => new ulong[(Order.Count + 63) / 64]).ToArray();
        for (var index = Order.Count - 1; index >= 0; index--)
        {
            foreach (var successor in Successors[index])
            {
                reachable[index][successor / 64] |= 1UL << (successor % 64);
                for (var word = 0; word < reachable[index].Length; word++)
                {
                    reachable[index][word] |= reachable[successor][word];
                }
            }
        }
    }

    public IReadOnlyList<ExecutionGraphNode> Order { get; }
    public int[][] Successors { get; }
    public int Position(GraphNodeId node) => positions[node];
    public bool Before(GraphNodeId first, GraphNodeId second) =>
        Before(positions[first], positions[second]);
    public bool Before(int first, int second) =>
        (reachable[first][second / 64] & (1UL << (second % 64))) != 0;

    public static ExecutionGraph WithStructure(
        ExecutionGraph graph,
        Graph<ExecutionGraphStructureNode, ExecutionGraphEdge> structure) =>
        new(graph.Id, structure, graph.DefaultCompileBoundary, graph.Architecture, graph.Source);
}
