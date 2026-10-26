using SharpInference.Instructions;
using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Execution;
using SharpInference.Vm.Optimization.GraphHelper;

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

    public GraphPackage<ExecutionGraph> Process(GraphPackage<ExecutionGraph> package, ExecutionPassContext context)
    {
        ArgumentNullException.ThrowIfNull(package);
        ArgumentNullException.ThrowIfNull(context);
        GraphValidation.ThrowIfErrors(package.Validate());
        var result = package.Snapshot();
        foreach (var graph in result)
        {
            var protectedIds = new HashSet<string>(StringComparer.Ordinal);
            foreach (var caller in result)
            {
                foreach (var binding in caller.Actions.SelectMany(action => action.Bindings))
                {
                    var reference = binding.Target;
                    if (caller.Id != graph.Id && reference.TargetKind == ResourceTarget.Graph && reference.Target == graph.Id)
                    {
                        protectedIds.Add(reference.TargetProperty);
                    }

                    if (reference.TargetKind == ResourceTarget.Call &&
                        caller.Actions.OfType<CallAction>().Any(call => call.Id == reference.Target && call.Target == graph.Id))
                    {
                        protectedIds.Add(reference.TargetProperty);
                    }
                }
            }
            var structure = graph.DependencyGraph();
            var order = structure.TopologicalOrder();
            var positions = order.Select((node, index) => (node.Action.Id, index))
                .ToDictionary(pair => pair.Id, pair => pair.index, StringComparer.Ordinal);
            var reach = structure.CreateReachabilityIndex(order, order);
            bool Before(GraphAction first, GraphAction second) => first != second &&
                reach.IsReachable(new GraphNodeId(first.Id), new GraphNodeId(second.Id));
            var slots = new List<(Resource Resource, GraphAction[] Last, string Slot)>();
            var lifetimes = new List<(Resource Resource, GraphAction First, GraphAction[] Last)>();
            foreach (var resource in graph.Resources.Where(resource =>
                resource.Scope == ResourceScope.Call && resource.GraphIO == GraphIO.Internal &&
                !resource.IsState && resource.Permission == ResourcePermission.ReadWrite &&
                resource.Usage is null or ResourceUsage.Temporary && resource.BindingKey is null &&
                !protectedIds.Contains(resource.Id)))
            {
                var accesses = graph.Readers(resource).Concat(graph.Writers(resource)).Distinct().ToArray();
                if (accesses.Length == 0 || accesses.Any(action => action is not ExecutionOperation))
                {
                    continue;
                }

                var first = accesses.MinBy(action => positions[action.Id])!;
                var initialBindings = first.Bindings.Where(binding =>
                    binding.Target.TargetKind == ResourceTarget.Graph &&
                    (binding.Target.Target is null || binding.Target.Target == graph.Id) &&
                    binding.Target.TargetProperty == resource.Id).ToArray();
                if (first is not ExecutionOperation operation || initialBindings.Length != 1 ||
                    initialBindings[0].Access != ResourceAccess.Write || initialBindings[0].SourceProperty != "output" ||
                    operation.Operation.Version != 1 || !CompleteOutputOperations.Contains(operation.Operation.Name) ||
                    accesses.Any(action => action != first && !Before(first, action)))
                {
                    continue;
                }

                lifetimes.Add((resource, first, accesses.Where(action =>
                    !accesses.Any(other => Before(action, other))).ToArray()));
            }
            foreach (var lifetime in lifetimes.OrderBy(lifetime => positions[lifetime.First.Id]))
            {
                var index = slots.FindIndex(slot =>
                    slot.Resource.Category == lifetime.Resource.Category &&
                    GraphValidation.SameTensor(slot.Resource.Tensor, lifetime.Resource.Tensor) &&
                    slot.Last.All(last => Before(last, lifetime.First)));
                if (index < 0)
                {
                    var storage = graph.Storage.GetValueOrDefault(lifetime.Resource.Id);
                    var slot = storage?.Slot ?? lifetime.Resource.Id;
                    slots.Add((lifetime.Resource, lifetime.Last, slot));
                }
                else
                {
                    var slot = slots[index];
                    graph.Storage[lifetime.Resource.Id] = new(ResourceStorageKind.LocalBuffer, slot.Slot);
                    slots[index] = (lifetime.Resource, lifetime.Last, slot.Slot);
                }
            }
        }
        return result;
    }

}
