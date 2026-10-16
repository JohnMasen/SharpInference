using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using static SharpInference.Graphs.Core.PatternElements;
using SharpInference.Graphs.Execution;

namespace SharpInference.Vm.Optimization;

/// <summary>Shares compatible temporary resources only when dependencies prove disjoint lifetimes.</summary>
public sealed class TemporaryResourceReusePostprocessor : IExecutionGraphPostprocessor
{
    // Only these version-one contracts establish the entire output without reading its previous contents.
    private static readonly HashSet<string> CompleteOutputOperations = new(StringComparer.Ordinal)
    {
        "core.copy", "core.add", "core.subtract", "core.multiply", "core.divide",
        "core.maximum", "core.exp", "core.tanh", "core.sigmoid", "core.rsqrt",
        "core.square", "core.relu", "core.reduce-sum", "core.reduce-mean",
        "core.mat-vec", "core.matrix-multiply", "core.affine", "core.bias-add",
        "core.gather-row", "core.tensor.fill", "core.tensor.cast-f16-f32",
        "core.tensor.reshape", "core.tensor.slice", "core.tensor.broadcast",
        "core.tensor.batched-mat-vec", "core.tensor.reduce-last-sum",
        "core.tensor.reduce-last-mean", "core.tensor.head-outer",
    };

    public string Name => "reuse-execution-temporaries";

    public ExecutionGraphProgram Process(
        ExecutionGraphProgram program, ExecutionGraphPostprocessContext context)
    {
        ArgumentNullException.ThrowIfNull(program);
        ArgumentNullException.ThrowIfNull(context);
        return new(program.Graphs.Select(graph =>
        {
            var boundPorts = program.Graphs.SelectMany(caller =>
                caller.Structure.Match(Pattern.Create(
                        Node().Where<ExecutionGraphCallNode>(call => call.Target == graph.Id)))
                    .Select(match => (ExecutionGraphCallNode)match.Nodes.Single())
                    .SelectMany(call => caller.Bindings(call).Select(binding => binding.Port)
                        .Concat(caller.Structure.Match(Pattern.Create(Node().Where<ExecutionResourceNode>(resource =>
                                !resource.ResourceNodeId.Owner.IsParent &&
                                resource.ResourceNodeId.Owner.Call!.Value.Value == call.Id.Value)))
                            .Select(match => (ExecutionResourceNode)match.Nodes.Single())
                            .Select(resource => resource.Resource.Id.Value))))
                .ToHashSet(StringComparer.Ordinal);
            return Reuse(graph, boundPorts);
        }));
    }

    private static ExecutionGraph Reuse(ExecutionGraph graph, HashSet<string> boundPorts)
    {
        var analysis = new ExecutionDependencyAnalysis(graph);
        var lifetimes = new List<Lifetime>();
        var candidates = graph.Structure.Match(Pattern.Create(Node().Where<ExecutionResourceNode>(resource =>
            resource.ResourceNodeId.Owner.IsParent &&
            resource.Kind == GraphDefinitionResourceKind.Temporary &&
            resource.Resource.BindingKey is null &&
            !boundPorts.Contains(resource.Resource.Id.Value))));
        foreach (var resource in candidates.Select(match => (ExecutionResourceNode)match.Nodes.Single()))
        {
            var accesses = graph.Structure.Readers(resource)
                .Concat(graph.Structure.Writers(resource))
                .DistinctBy(node => node.GraphId).OfType<ExecutionGraphNode>().ToArray();
            if (accesses.Length == 0 ||
                accesses.Any(node => node is not ExecutionOperationNode))
            {
                continue;
            }

            var first = accesses.OrderBy(node => analysis.Position(node.GraphId)).First();
            var firstBindings = graph.Bindings(first).Where(value =>
                value.Resource == resource.ResourceNodeId).ToArray();
            if (firstBindings.Length != 1 ||
                firstBindings[0].Access != GraphResourceAccess.Write ||
                firstBindings[0].Port != "output" ||
                first is not ExecutionOperationNode operation ||
                operation.Operation.Version != 1 ||
                !CompleteOutputOperations.Contains(operation.Operation.Name) ||
                accesses.Any(node => node != first && !analysis.Before(first.GraphId, node.GraphId)))
            {
                continue;
            }

            var last = accesses.Where(node => !accesses.Any(other =>
                analysis.Before(node.GraphId, other.GraphId))).Select(node => node.GraphId).ToArray();
            lifetimes.Add(new(resource, first.GraphId, last));
        }

        var replacements = new Dictionary<GraphNodeId, GraphNodeId>();
        var slots = new List<Lifetime>();
        foreach (var lifetime in lifetimes.OrderBy(value => analysis.Position(value.First)))
        {
            var slotIndex = slots.FindIndex(slot =>
                SameTensor(slot.Resource.Tensor, lifetime.Resource.Tensor) &&
                slot.Last.All(last => analysis.Before(last, lifetime.First)));
            if (slotIndex < 0)
            {
                slots.Add(lifetime);
                continue;
            }
            var slot = slots[slotIndex];
            replacements.Add(lifetime.Resource.GraphId, slot.Resource.GraphId);
            slots[slotIndex] = lifetime with { Resource = slot.Resource };
        }
        if (replacements.Count == 0)
        {
            return graph;
        }

        GraphNodeId Map(GraphNodeId id) => replacements.GetValueOrDefault(id, id);
        var structure = new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(
            graph.Structure.Nodes.Where(node => !replacements.ContainsKey(node.GraphId)),
            graph.Structure.Edges.Select(edge => edge switch
            {
                ExecutionResourceReadEdge read => read with { Source = Map(read.Source) },
                ExecutionResourceWriteEdge write => write with { Target = Map(write.Target) },
                _ => edge,
            }));
        return ExecutionDependencyAnalysis.WithStructure(graph, structure);
    }

    internal static bool SameTensor(TensorDescriptor first, TensorDescriptor second) =>
        first.ElementType == second.ElementType && first.Layout == second.Layout &&
        first.Dimensions.SequenceEqual(second.Dimensions);

    private sealed record Lifetime(
        ExecutionResourceNode Resource, GraphNodeId First, GraphNodeId[] Last);
}
