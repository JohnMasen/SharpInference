using SharpInference.Graphs;
using SharpInference.Graphs.Core;

namespace SharpInference.Vm.Optimization;

/// <summary>Combines stable operation positions, direct successors, and indexed dependency reachability.</summary>
internal sealed class ExecutionDependencyAnalysis
{
    private readonly Dictionary<GraphNodeId, int> positions;
    private readonly GraphNodeId[] nodeIds;
    private readonly ReachabilityIndex<ExecutionGraphStructureNode> reachable;

    /// <summary>Analyzes only execution dependencies, excluding resource-access edges.</summary>
    /// <param name="graph">The execution graph whose operation dependencies are analyzed.</param>
    /// <exception cref="InvalidDataException">The dependency view contains a directed cycle.</exception>
    public ExecutionDependencyAnalysis(ExecutionGraph graph)
    {
        var dependencyView = graph.Structure.DependencyView();
        Order = dependencyView.TopologicalOrder()
            .Cast<ExecutionGraphNode>().ToArray();
        nodeIds = Order.Select(node => node.GraphId).ToArray();
        positions = nodeIds.Select((id, index) => (id, index))
            .ToDictionary(item => item.id, item => item.index);
        Successors = Order.Select(node => graph.Structure.Successors(node)
            .Select(successor => positions[successor.GraphId]).Distinct().ToArray()).ToArray();
        reachable = dependencyView.CreateReachabilityIndex(Order, Order);
    }

    /// <summary>Gets operation nodes in stable topological order.</summary>
    public IReadOnlyList<ExecutionGraphNode> Order { get; }
    /// <summary>Gets distinct direct successor positions for each operation position.</summary>
    public int[][] Successors { get; }
    /// <summary>Gets an operation's position in the topological order.</summary>
    /// <param name="node">The structural identity of the operation.</param>
    /// <returns>The zero-based operation position.</returns>
    /// <exception cref="KeyNotFoundException">The identity is not an analyzed operation.</exception>
    public int Position(GraphNodeId node) => positions[node];
    /// <summary>Determines whether one operation strictly precedes another along a dependency path.</summary>
    /// <param name="first">The structural identity of the potential predecessor.</param>
    /// <param name="second">The structural identity of the potential successor.</param>
    /// <returns>Whether a dependency path exists between distinct operations; self-comparison is false.</returns>
    /// <exception cref="KeyNotFoundException">Either identity is not an analyzed operation.</exception>
    public bool Before(GraphNodeId first, GraphNodeId second) =>
        Before(positions[first], positions[second]);
    /// <summary>Tests strict dependency ordering using cached topological positions.</summary>
    /// <param name="first">The zero-based position of the potential predecessor.</param>
    /// <param name="second">The zero-based position of the potential successor.</param>
    /// <returns>Whether a dependency path exists between distinct positions; self-comparison is false.</returns>
    /// <exception cref="IndexOutOfRangeException">Either position is outside the analyzed order.</exception>
    public bool Before(int first, int second)
    {
        var source = nodeIds[first];
        var target = nodeIds[second];
        return first != second && reachable.IsReachable(source, target);
    }

    /// <summary>Creates an execution graph with replacement structure and unchanged graph metadata.</summary>
    /// <param name="graph">The source of graph identity, compilation boundary, architecture, and provenance.</param>
    /// <param name="structure">The graph structure to use in the replacement.</param>
    /// <returns>A new execution graph; the original graph is not modified.</returns>
    public static ExecutionGraph WithStructure(
        ExecutionGraph graph,
        Graph<ExecutionGraphStructureNode, ExecutionGraphEdge> structure) =>
        new(graph.Id, structure, graph.DefaultCompileBoundary, graph.Architecture, graph.Source);
}
