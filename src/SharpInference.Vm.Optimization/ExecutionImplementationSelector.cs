using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Instructions;
using ComputationGraph = SharpInference.Graphs.Graph;

namespace SharpInference.Vm.Optimization;

/// <summary>Resolves graph operations to installed implementations before VM emission.</summary>
public sealed class ExecutionImplementationSelector(IEnumerable<IInstructionCollectionProvider> providers)
{
    private readonly InstructionRegistry registry = new(providers);

    public void Select(GraphPackage<ExecutionGraph> package, string backend) => Resolve(package, backend, select: true);

    public void Validate(GraphPackage<ExecutionGraph> package, string backend) => Resolve(package, backend, select: false);

    private void Resolve(GraphPackage<ExecutionGraph> package, string backend, bool select)
    {
        ArgumentNullException.ThrowIfNull(package);
        ArgumentException.ThrowIfNullOrWhiteSpace(backend);
        var target = Target(backend);
        var index = package.ToDictionary(graph => graph.Id, graph => (ComputationGraph)graph, StringComparer.Ordinal);
        var mappings = registry.QueryGraphInstructionBindings().Where(binding => binding.Target == target)
            .ToDictionary(binding => binding.Operation);
        foreach (var graph in package)
        {
            foreach (var operation in graph.Actions.OfType<ExecutionOperation>())
            {
                var tensors = operation.Bindings.ToDictionary(binding => binding.SourceProperty, binding =>
                    GraphValidation.TryResolve(graph, binding.Target, index, out var resource)
                        ? resource.Tensor : throw new InvalidDataException($"Cannot resolve '{graph.Id}/{operation.Id}/{binding.SourceProperty}'."),
                    StringComparer.Ordinal);
                var access = operation.Bindings.ToDictionary(binding => binding.SourceProperty, binding => binding.Access,
                    StringComparer.Ordinal);
                bool Matches(InstructionSignature signature) =>
                    signature.Ports.Count == tensors.Count &&
                    operation.Attributes.Keys.ToHashSet(StringComparer.Ordinal).SetEquals(signature.Attributes) &&
                    signature.Ports.All(port => tensors.TryGetValue(port.Name, out var tensor) &&
                        tensor.ElementType == port.ElementType &&
                        (access[port.Name] == ResourceAccess.ReadWrite || access[port.Name] == port.Access));
                var parameters = operation.Bindings.Select(binding => (InstructionParameter)new InstructionTensorParameter(
                    binding.SourceProperty, tensors[binding.SourceProperty], binding.Access, binding.SourceProperty,
                    binding.SourceProperty, "0")).Concat(operation.Attributes.Select(pair =>
                        (InstructionParameter)new InstructionAttributeParameter(pair.Key, pair.Value))).ToArray();
                if (operation.SelectedImplementation is { } selected)
                {
                    if (selected.Target != target)
                        throw new InvalidDataException($"Implementation '{graph.Id}/{operation.Id}' targets '{selected.Target}', not '{target}'.");
                    var instruction = registry.Resolve(selected.CollectionId, selected.InstructionName, target);
                    var signature = instruction.Adapt(parameters, operation.Precision ??
                        new(selected.Precision.ArithmeticType, selected.Precision.AccumulatorType));
                    if (signature.Precision != selected.Precision)
                        throw new InvalidDataException($"Implementation precision conflicts with '{graph.Id}/{operation.Id}'.");
                    if (!Matches(signature))
                        throw new InvalidDataException($"Implementation access or parameter contracts conflict with '{graph.Id}/{operation.Id}'.");
                    if (target == InstructionTarget.Direct3D12 && selected.Dispatch is null)
                        throw new InvalidDataException($"GPU operation '{graph.Id}/{operation.Id}' requires a dispatch configuration.");
                    ValidateDispatch(selected.Dispatch, operation.Id);
                    continue;
                }
                if (!select)
                    throw new InvalidDataException($"Action '{graph.Id}/{operation.Id}' has no selected implementation.");
                mappings.TryGetValue(operation.Operation, out var mapping);
                var candidates = mapping is not null
                    ? [registry.Resolve(mapping.CollectionId, mapping.InstructionName, target)]
                    : operation.Operation.Version == 1
                        ? registry.QueryInstructionCollection().Where(collection => collection.Architecture == target)
                            .OrderBy(collection => collection.Tier).ThenBy(collection => collection.Id)
                            .SelectMany(collection => registry.QueryInstruction(collection.Id, operation.Operation.Name))
                            .Where(instruction => instruction.Target == target).ToArray()
                        : [];
                var errors = new List<string>();
                foreach (var instruction in candidates)
                {
                    var matching = instruction.Signatures.Where(Matches).ToArray();
                    if (matching.Length == 0)
                        continue;
                    try
                    {
                        var requirement = operation.Precision ?? new PrecisionRequirement(
                            matching[0].Precision.ArithmeticType, matching[0].Precision.AccumulatorType);
                        var signature = instruction.Adapt(parameters, requirement);
                        var dispatch = target == InstructionTarget.Direct3D12
                            ? mapping?.Dispatch?.Invoke(new(tensors, operation.Attributes)) ?? DefaultDispatch(operation, tensors)
                            : null;
                        ValidateDispatch(dispatch, operation.Id);
                        operation.SelectedImplementation = new(
                            instruction.CollectionId, instruction.Name, target, signature.Precision, dispatch);
                        break;
                    }
                    catch (InstructionAdaptationException error)
                    {
                        errors.Add(error.Message);
                    }
                }
                if (operation.SelectedImplementation is null)
                    throw new NotSupportedException(
                        $"Backend '{target}' cannot implement '{operation.Operation}' at '{graph.Id}/{operation.Id}'. " +
                        string.Join(Environment.NewLine, errors));
            }
        }
        if (select)
            package.SetTarget(backend);
    }

    internal static InstructionTarget Target(string backend) => backend switch
    {
        "cpu-vm" or "cpu" or "cpu.managed" => InstructionTarget.Cpu,
        "d3d12-vm" or "d3d12" or "direct3d12" => InstructionTarget.Direct3D12,
        _ => throw new NotSupportedException($"Unsupported execution backend '{backend}'."),
    };

    private static GraphInstructionDispatch DefaultDispatch(
        ExecutionOperation operation, IReadOnlyDictionary<string, TensorDescriptor> tensors)
    {
        var outputs = operation.Bindings.Where(binding => binding.Access != ResourceAccess.Read)
            .Select(binding => tensors[binding.SourceProperty]).ToArray();
        if (outputs.Length == 0)
            throw new InvalidDataException($"GPU operation '{operation.Id}' requires an explicit dispatch for a read-only operation.");
        var count = outputs.Max(tensor => tensor.Dimensions.Aggregate(1UL, (size, dimension) => checked(size * (ulong)dimension)));
        return new(64, 1, 1, checked((uint)((count + 63) / 64)), 1, 1);
    }

    private static void ValidateDispatch(GraphInstructionDispatch? dispatch, string operation)
    {
        if (dispatch is null)
            return;
        if (dispatch.ThreadsX is 0 or > 1024 || dispatch.ThreadsY is 0 or > 1024 ||
            dispatch.ThreadsZ is 0 or > 64 || (ulong)dispatch.ThreadsX * dispatch.ThreadsY * dispatch.ThreadsZ > 1024 ||
            dispatch.GroupsX is 0 or > 65535 || dispatch.GroupsY is 0 or > 65535 || dispatch.GroupsZ is 0 or > 65535)
            throw new InvalidDataException($"Invalid GPU dispatch for '{operation}'.");
    }
}
