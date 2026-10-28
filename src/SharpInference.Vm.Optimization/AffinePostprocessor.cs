using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Execution;
using SharpInference.Instructions;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Vm.Optimization;

/// <summary>Fuses private, nonaliasing FP32 matrix-multiply/bias-add pairs when a registered T1 implementation adapts.</summary>
public sealed class AffinePostprocessor(IEnumerable<IInstructionCollectionProvider> providers) : IExecutionGraphPostprocessor
{
    private readonly InstructionRegistry registry = new(providers);
    private static readonly PrecisionRequirement Float32 = new(ElementType.Float32, ElementType.Float32);

    public string Name => "fuse-execution-affine";

    public ExecutionGraphPackage Process(ExecutionGraphPackage package, ExecutionPassContext context)
    {
        ArgumentNullException.ThrowIfNull(package);
        ArgumentNullException.ThrowIfNull(context);
        GraphValidation.ThrowIfErrors(package.ValidateStructure());
        var target = ExecutionImplementationSelector.Target(context.Compilation.Backend);
        var mappings = registry.QueryGraphInstructionBindings();
        var instructions = registry.QueryInstructionCollection().Any(collection =>
            collection.Id == TierOneAffineOperations.CollectionFloat32 && collection.Architecture == target)
            ? registry.QueryInstruction(TierOneAffineOperations.CollectionFloat32, TierOneAffineOperations.Affine.Name)
                .Where(instruction => instruction.Target == target).ToArray()
            : [];
        var result = package.Snapshot();
        if (instructions.Length == 0)
        {
            return result;
        }

        foreach (var graphId in result.Select(graph => graph.Id).ToArray())
        {
            foreach (var id in result.Single(graph => graph.Id == graphId).Actions
                .OfType<ExecutionOperation>().Where(operation => operation.Operation == PrimitiveOperations.BiasAdd)
                .Select(operation => operation.Id).ToArray())
            {
                var graph = result.Single(graph => graph.Id == graphId);
                var add = graph.Actions.OfType<ExecutionOperation>().Single(operation => operation.Id == id);
                if (!Ports(add, "input", "bias", "output") || add.Attributes.Count != 0)
                {
                    continue;
                }

                var input = Local(graph, add.Bindings.Single(binding => binding.SourceProperty == "input").Target);
                var producers = graph.Actions.OfType<ExecutionOperation>().Where(operation =>
                    operation.Operation == PrimitiveOperations.MatrixMultiply &&
                    operation.Bindings.Any(binding => binding.SourceProperty == "output" &&
                        Local(graph, binding.Target) == input)).ToArray();
                if (input is null || producers.Length != 1)
                {
                    continue;
                }

                var multiply = producers[0];
                if (!Ports(multiply, "left", "right", "output") ||
                    !multiply.Attributes.Keys.ToHashSet(StringComparer.Ordinal)
                        .SetEquals(["transpose_left", "transpose_right"]))
                {
                    continue;
                }

                if (!bool.TryParse(multiply.Attributes["transpose_left"], out var transposeLeft) ||
                    !bool.TryParse(multiply.Attributes["transpose_right"], out var transposeRight))
                {
                    throw new InvalidDataException($"Invalid matrix transpose declaration at '{graph.Id}/{multiply.Id}'.");
                }

                var bindings = multiply.Bindings.Where(binding => binding.SourceProperty != "output")
                    .Concat(add.Bindings.Where(binding => binding.SourceProperty != "input")).ToArray();
                var all = bindings.Select(binding => Local(graph, binding.Target)).Append(input).ToArray();
                if (all.Any(resource => resource is null || resource.Tensor.ElementType != ElementType.Float32 ||
                        resource.Tensor.Layout != "dense") || all.Distinct().Count() != all.Length ||
                    all.Where(resource => resource!.BindingKey is not null).GroupBy(resource => resource!.BindingKey)
                        .Any(group => group.Count() > 1) ||
                    all.Any(resource => resource!.BindingKey is { } key && result.SelectMany(owner => owner.Resources)
                        .Any(alias => alias != resource && alias.Scope == resource.Scope && alias.BindingKey == key)) ||
                    all.Any(resource => graph.Storage.TryGetValue(resource!.Id, out var storage) &&
                        (storage.Slot is not null || storage.Kind == ResourceStorageKind.SharedBinding)) ||
                    graph.Storage.Any(pair => pair.Value.Slot is { } slot && all.Any(resource => resource!.Id == slot)))
                {
                    continue;
                }

                var temporary = input;
                if (temporary.GraphIO != GraphIO.Internal || temporary.Scope != ResourceScope.Call ||
                    temporary.IsState || temporary.BindingKey is not null ||
                    temporary.Usage is not (null or ResourceUsage.Temporary) ||
                    result.Any(owner => owner.Actions.Any(action =>
                        !(owner == graph && (action == multiply || action == add)) &&
                        action.Bindings.Any(binding => References(owner, binding.Target, graph, temporary.Id) ||
                            owner != graph && binding.Access != ResourceAccess.Read &&
                            bindings.Any(boundary => References(owner, binding.Target, graph,
                                Local(graph, boundary.Target)!.Id))))))
                {
                    continue;
                }

                var structure = graph.DependencyGraph();
                bool Before(GraphAction first, GraphAction second) => first != second &&
                    structure.IsReachable(new GraphNodeId(first.Id), new GraphNodeId(second.Id));
                if (!Before(multiply, add) || graph.Actions.Any(action => action != multiply && action != add &&
                    (Before(multiply, action) && Before(action, add) ||
                     action.Bindings.Any(binding => binding.Access != ResourceAccess.Read &&
                         bindings.Any(boundary => Local(graph, boundary.Target) == Local(graph, binding.Target))) &&
                     !Before(action, multiply) && !Before(add, action))))
                {
                    continue;
                }

                var tensors = bindings.ToDictionary(binding => binding.SourceProperty,
                    binding => Local(graph, binding.Target)!.Tensor, StringComparer.Ordinal);
                var left = tensors["left"].Dimensions;
                var right = tensors["right"].Dimensions;
                var output = tensors["output"].Dimensions;
                var bias = tensors["bias"].Dimensions;
                if (left.Count != 2 || right.Count != 2 || output.Count != 2 || bias.Count != 1 ||
                    left[transposeLeft ? 0 : 1] != right[transposeRight ? 1 : 0] ||
                    output[0] != left[transposeLeft ? 1 : 0] || output[1] != right[transposeRight ? 0 : 1] ||
                    bias[0] != output[1] || !GraphValidation.SameTensor(temporary.Tensor, tensors["output"]))
                {
                    continue;
                }

                if (new[] { multiply, add }.Any(operation => operation.SelectedImplementation is { } selected &&
                    (selected.Target != target || selected.Precision != new ExecutionPrecision(ElementType.Float32, ElementType.Float32))))
                {
                    continue;
                }

                var parameters = bindings.Select(binding => (InstructionParameter)new InstructionTensorParameter(
                    binding.SourceProperty, tensors[binding.SourceProperty], binding.Access,
                    Local(graph, binding.Target)!.Id, binding.SourceProperty, "0"))
                    .Concat(multiply.Attributes.Select(pair =>
                        (InstructionParameter)new InstructionAttributeParameter(pair.Key, pair.Value))).ToArray();
                foreach (var instruction in instructions)
                {
                    InstructionSignature signature;
                    try
                    {
                        signature = instruction.Adapt(parameters, Float32);
                    }
                    catch (InstructionAdaptationException)
                    {
                        continue;
                    }

                    if (signature.Precision != new ExecutionPrecision(ElementType.Float32, ElementType.Float32) ||
                        signature.Ports.Count != bindings.Length || !signature.Ports.All(port =>
                            bindings.Any(binding => binding.SourceProperty == port.Name && binding.Access == port.Access &&
                                tensors[port.Name].ElementType == port.ElementType)) ||
                        !signature.Attributes.ToHashSet(StringComparer.Ordinal).SetEquals(multiply.Attributes.Keys))
                    {
                        continue;
                    }

                    var mapping = mappings.SingleOrDefault(mapping => mapping.Target == target &&
                        mapping.Operation == TierOneAffineOperations.Affine &&
                        mapping.CollectionId == instruction.CollectionId && mapping.InstructionName == instruction.Name);
                    var dispatch = target == ExecutionTarget.Direct3D12
                        ? mapping?.Dispatch?.Invoke(new(tensors, multiply.Attributes)) ??
                            new ExecutionDispatch(64, 1, 1, checked((uint)((tensors["output"].Dimensions
                                .Aggregate(1UL, (count, dimension) => checked(count * (ulong)dimension)) + 63) / 64)), 1, 1)
                        : null;
                    result = ExecutionGraphRewriter.FuseNodesToOperation(result, graph.Id, [multiply.Id, add.Id],
                        add.Id, TierOneAffineOperations.Affine, bindings,
                        new(instruction.CollectionId, instruction.Name, target, signature.Precision, dispatch));
                    var fused = result.Single(item => item.Id == graph.Id).Actions.OfType<ExecutionOperation>()
                        .Single(operation => operation.Id == add.Id);
                    fused.Precision = Float32;
                    fused.Category = add.Category;
                    foreach (var pair in multiply.Attributes)
                    {
                        fused.Attributes.Add(pair.Key, pair.Value);
                    }
                    foreach (var pair in multiply.Metadata.Concat(add.Metadata))
                    {
                        fused.Metadata[pair.Key] = pair.Value;
                    }
                    fused.Metadata["transformation"] = Name;
                    break;
                }
            }
        }
        GraphValidation.ThrowIfErrors(result.ValidateStructure());
        return result;
    }

    private static bool Ports(ExecutionOperation operation, params string[] names) =>
        operation.Bindings.Count == names.Length &&
        operation.Bindings.Select(binding => binding.SourceProperty).ToHashSet(StringComparer.Ordinal).SetEquals(names) &&
        operation.Bindings.All(binding => binding.Access ==
            (binding.SourceProperty == "output" ? ResourceAccess.Write : ResourceAccess.Read));

    private static Resource? Local(ExecutionGraph graph, ResourceRef reference) =>
        reference.TargetKind == ResourceTarget.Graph && (reference.Target is null || reference.Target == graph.Id)
            ? graph.Resources.SingleOrDefault(resource => resource.Id == reference.TargetProperty) : null;

    private static bool References(ExecutionGraph owner, ResourceRef reference, ExecutionGraph graph, string id) =>
        reference.TargetProperty == id && (reference.TargetKind == ResourceTarget.Graph
            ? (reference.Target ?? owner.Id) == graph.Id
            : owner.Actions.OfType<ExecutionCall>().Any(call => call.Id == reference.Target && call.Target == graph.Id));
}
