namespace SharpInference.Graphs;

public static class TargetExecutionPlanValidator
{
    public static void Validate(TargetExecutionPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        Name(plan.Name);
        Name(plan.Abi);
        Name(plan.StateSchema);
        if (!Enum.IsDefined(plan.Target) || plan.StateVersion <= 0)
            throw new InvalidDataException("Invalid execution plan target or state version.");
        var slots = Unique(plan.Slots, slot => slot.Id);
        Unique(plan.State, entry => entry.Name);
        if (plan.Bindings is { } metadata)
        {
            GraphValidator.Validate(metadata);
            if (metadata.Nodes.Count != 0 || metadata.Regions.Count != 1 || metadata.Structure is not null ||
                plan.Abi != $"vm:{metadata.Model.StateAbiId}" || plan.StateSchema != metadata.GraphState.Schema.Name)
                throw new InvalidDataException("Binding metadata must be computation-free and match the state ABI.");
            foreach (var id in metadata.Inputs.Concat(metadata.Outputs))
            {
                var resource = metadata.Resources.Single(resource => resource.Id == id);
                var tensor = resource.Tensor;
                var tokenCapacity = metadata.Inputs.Count == 1 && metadata.Inputs[0] == id &&
                    tensor.ElementType == GraphElementType.Int32 && tensor.Dimensions.SequenceEqual([1]) &&
                    plan.Entries.Any(entry => entry.Name == "prefill.1");
                if (!slots.TryGetValue(id.Value, out var slot) || slot.Scope != resource.Scope ||
                    slot.Tensor.ElementType != tensor.ElementType ||
                    !(tokenCapacity ? slot.Tensor.Dimensions.Count == 1 && slot.Tensor.Dimensions[0] >= 1 :
                        slot.Tensor.Dimensions.SequenceEqual(tensor.Dimensions)))
                    throw new InvalidDataException($"Missing or incompatible IO slot '{id}'.");
            }
            if (plan.StateVersion != 1 || plan.State.Count != metadata.GraphState.Count)
                throw new InvalidDataException("Binding metadata and execution plan have different state versions or counts.");
            foreach (var state in metadata.GraphState)
            {
                var binding = plan.State.SingleOrDefault(entry => entry.Name == state.Name);
                var tensor = metadata.Resources.Single(resource => resource.Id == state.Resource).Tensor;
                if (binding is null || !slots.TryGetValue(binding.Slot, out var slot) ||
                    slot.Tensor.ElementType != tensor.ElementType || !slot.Tensor.Dimensions.SequenceEqual(tensor.Dimensions))
                    throw new InvalidDataException($"Incompatible state binding at '{state.Name}'.");
            }
        }
        var definitions = Unique(plan.Definitions, definition => definition.Id);
        var entries = Unique(plan.Entries, entry => entry.Name);
        if (entries.Count == 0) throw new InvalidDataException("An execution plan requires an entry.");
        foreach (var slot in slots.Values)
        {
            Tensor(slot.Tensor);
            Access(slot.Access);
            if (!Enum.IsDefined(slot.Scope)) throw new InvalidDataException($"Invalid scope at slot '{slot.Id}'.");
            if (slot.BindingKey is not null) Name(slot.BindingKey);
        }
        foreach (var binding in Unique(plan.ResourceBindings, binding => binding.Resource).Values)
        {
            Tensor(binding.Tensor);
            if (!slots.TryGetValue(binding.Slot, out var slot) ||
                slot.Tensor.ElementType != binding.Tensor.ElementType ||
                binding.ByteOffset % Size(binding.Tensor.ElementType) != 0 ||
                binding.ByteOffset > Bytes(slot.Tensor) ||
                Bytes(binding.Tensor) > Bytes(slot.Tensor) - binding.ByteOffset)
                throw new InvalidDataException($"Invalid storage mapping at resource '{binding.Resource}'.");
        }
        var stateSlots = new HashSet<string>(StringComparer.Ordinal);
        foreach (var state in plan.State)
            if (!slots.TryGetValue(state.Slot, out var slot) || slot.Scope != GraphResourceScope.Session ||
                !stateSlots.Add(state.Slot))
                throw new InvalidDataException($"Invalid session binding at state '{state.Name}'.");

        foreach (var definition in definitions.Values)
        {
            if (!Enum.IsDefined(definition.Kind))
                throw new InvalidDataException($"Invalid definition kind at '{definition.Id}'.");
            if (definition.Kind == ExecutionPlanDefinitionKind.Kernel)
            {
                if (plan.Target != ExecutionPlanTarget.Direct3D12 || definition.Threads is null)
                    throw new InvalidDataException($"Kernel '{definition.Id}' requires GPU threads.");
                Grid(definition.Threads, true);
            }
            else if (definition.Threads is not null)
                throw new InvalidDataException($"Non-kernel '{definition.Id}' declares threads.");
            var parameters = Unique(definition.Parameters, parameter => parameter.Name);
            foreach (var parameter in parameters.Values)
            {
                Access(parameter.Access);
                Tensor(parameter.Tensor);
            }
            Unique(definition.Nodes, node => node.Id);
            var preceding = new HashSet<string>(StringComparer.Ordinal);
            foreach (var node in definition.Nodes)
            {
                if (node.Dependencies.Distinct(StringComparer.Ordinal).Count() != node.Dependencies.Count ||
                    node.Dependencies.Any(id => !preceding.Contains(id)))
                    throw new InvalidDataException($"Invalid dependency at '{definition.Id}/{node.Id}'.");
                foreach (var source in node.Sources) Name(source);
                switch (node.Instruction)
                {
                    case ExecutionPlanOperator operation:
                        if (definition.Kind == ExecutionPlanDefinitionKind.Orchestration)
                            throw new InvalidDataException($"Orchestration '{definition.Id}' contains an operator.");
                        Name(operation.Name);
                        if (!Enum.IsDefined(operation.Precision.MinimumArithmeticType) ||
                            !Enum.IsDefined(operation.Precision.MinimumAccumulatorType))
                            throw new InvalidDataException($"Invalid precision at '{definition.Id}/{node.Id}'.");
                        if (operation.Implementation is { } implementation &&
                            (implementation.Length is 0 or > 128 ||
                             implementation.Any(c => !char.IsAsciiLetterOrDigit(c) && c is not ('.' or '-' or '_'))))
                            throw new InvalidDataException($"Invalid implementation at '{definition.Id}/{node.Id}'.");
                        if (operation.CooperativeThreads is { } count &&
                            (plan.Target != ExecutionPlanTarget.Direct3D12 || operation.Name != "core.mat-vec" ||
                             count is not (1 or 64)))
                            throw new InvalidDataException($"Unsupported cooperativeThreads at '{definition.Id}/{node.Id}': {operation.CooperativeThreads}.");
                        if (plan.Target == ExecutionPlanTarget.Direct3D12 && operation.Name == "core.mat-vec" &&
                            (operation.CooperativeThreads is null ||
                             operation.Implementation != (operation.CooperativeThreads == 1 ? "serial-matvec" : "cooperative-matvec")))
                            throw new InvalidDataException($"MatVec '{definition.Id}/{node.Id}' requires an explicit matching execution configuration.");
                        if (operation.CooperativeThreads == 64 &&
                            (definition.Kind != ExecutionPlanDefinitionKind.Kernel ||
                             definition.Threads != new ExecutionPlanGrid(64) || definition.Nodes.Count != 1))
                            throw new InvalidDataException($"Cooperative MatVec '{definition.Id}' requires an isolated 64x1x1 kernel.");
                        foreach (var argument in Unique(operation.Arguments, argument => argument.Parameter).Values)
                        {
                            Name(argument.Source);
                            if (!parameters.TryGetValue(argument.Source, out var source) ||
                                argument.ByteOffset != 0)
                                throw new InvalidDataException($"Invalid source at '{definition.Id}/{node.Id}/{argument.Parameter}'.");
                            if (operation.ParameterAccesses.TryGetValue(argument.Parameter, out var access))
                            {
                                Access(access);
                                if (access != GraphResourceAccess.Read && source.Access == GraphResourceAccess.Read)
                                    throw new InvalidDataException($"Read-only write at '{definition.Id}/{node.Id}/{argument.Parameter}'.");
                            }
                        }
                        if (operation.ParameterAccesses.Count != 0 &&
                            (operation.ParameterAccesses.Count != operation.Arguments.Count ||
                             operation.ParameterAccesses.Keys.Any(port => !operation.Arguments.Any(a => a.Parameter == port))))
                            throw new InvalidDataException($"Invalid access contract at '{definition.Id}/{node.Id}'.");
                        if (operation.ResolvedPrecision is { } resolved &&
                            (resolved.Arithmetic != operation.Precision.MinimumArithmeticType ||
                             resolved.Accumulator != operation.Precision.MinimumAccumulatorType))
                            throw new InvalidDataException($"Resolved precision differs from the plan at '{definition.Id}/{node.Id}'.");
                        foreach (var bound in operation.IndexBounds)
                        {
                            var indexArgument = operation.Arguments.SingleOrDefault(argument => argument.Parameter == bound.IndexPort);
                            var tensorArgument = operation.Arguments.SingleOrDefault(argument => argument.Parameter == bound.TensorPort);
                            if (indexArgument is null || tensorArgument is null ||
                                !parameters.TryGetValue(indexArgument.Source, out var indexParameter) ||
                                indexParameter.Tensor.ElementType != GraphElementType.Int32 || indexArgument.ByteOffset != 0 ||
                                !parameters.TryGetValue(tensorArgument.Source, out var tensorParameter) ||
                                bound.Axis < 0 || bound.Axis >= tensorParameter.Tensor.Dimensions.Count || bound.MinimumIndex > 0)
                                throw new InvalidDataException($"Invalid index bound at '{definition.Id}/{node.Id}'.");
                        }
                        foreach (var pair in operation.Attributes)
                        {
                            Name(pair.Key);
                            if (pair.Value is null) throw new InvalidDataException($"Null attribute at '{definition.Id}/{node.Id}'.");
                        }
                        break;
                    case ExecutionPlanCall call:
                        if (!definitions.TryGetValue(call.Definition, out var callee) ||
                            !(definition.Kind == ExecutionPlanDefinitionKind.Orchestration
                                ? callee.Kind == ExecutionPlanDefinitionKind.Orchestration ||
                                  plan.Target == ExecutionPlanTarget.Cpu && callee.Kind == ExecutionPlanDefinitionKind.Function
                                : callee.Kind == ExecutionPlanDefinitionKind.Function))
                            throw new InvalidDataException($"Invalid call at '{definition.Id}/{node.Id}'.");
                        Bind(call.Arguments, callee.Parameters, parameters);
                        break;
                    case ExecutionPlanDispatch dispatch:
                        if (plan.Target != ExecutionPlanTarget.Direct3D12 ||
                            definition.Kind != ExecutionPlanDefinitionKind.Orchestration ||
                            !definitions.TryGetValue(dispatch.Definition, out var kernel) ||
                            kernel.Kind != ExecutionPlanDefinitionKind.Kernel)
                            throw new InvalidDataException($"Invalid dispatch at '{definition.Id}/{node.Id}'.");
                        Grid(dispatch.Groups, false);
                        Bind(dispatch.Arguments, kernel.Parameters, parameters);
                        var totalThreads = checked((ulong)dispatch.Groups.X * dispatch.Groups.Y * dispatch.Groups.Z *
                            kernel.Threads!.X * kernel.Threads.Y * kernel.Threads.Z);
                        if (totalThreads > uint.MaxValue)
                            throw new InvalidDataException($"Dispatch '{definition.Id}/{node.Id}' exceeds 32-bit linear indexing.");
                        foreach (var operation in kernel.Nodes.Select(value => value.Instruction).OfType<ExecutionPlanOperator>())
                            if (operation.CooperativeThreads == 64)
                            {
                                var output = operation.Arguments.SingleOrDefault(argument => argument.Parameter == "output") ??
                                    throw new InvalidDataException($"MatVec '{kernel.Id}' requires an output.");
                                var tensor = kernel.Parameters.Single(parameter => parameter.Name == output.Source).Tensor;
                                var rows = Bytes(tensor) / Size(tensor.ElementType);
                                if (dispatch.Groups != ExecutionPlanGpuContracts.CooperativeMatVecGroups(rows))
                                    throw new InvalidDataException($"Dispatch '{definition.Id}/{node.Id}' has an incompatible cooperative grid.");
                            }
                            else if (operation.Name == "core.mat-vec")
                            {
                                var output = operation.Arguments.SingleOrDefault(argument => argument.Parameter == "output") ??
                                    throw new InvalidDataException($"MatVec '{kernel.Id}' requires an output.");
                                var tensor = kernel.Parameters.Single(parameter => parameter.Name == output.Source).Tensor;
                                if (totalThreads < Bytes(tensor) / Size(tensor.ElementType))
                                    throw new InvalidDataException($"Dispatch '{definition.Id}/{node.Id}' does not cover every MatVec output row.");
                            }
                        break;
                    case ExecutionPlanBarrier barrier:
                        if (plan.Target != ExecutionPlanTarget.Direct3D12 ||
                            definition.Kind != ExecutionPlanDefinitionKind.Orchestration || barrier.Resources.Count == 0 ||
                            barrier.Resources.Distinct(StringComparer.Ordinal).Count() != barrier.Resources.Count ||
                            barrier.Resources.Any(id => !parameters.TryGetValue(id, out var parameter) ||
                                parameter.Access == GraphResourceAccess.Read))
                            throw new InvalidDataException($"Invalid barrier at '{definition.Id}/{node.Id}'.");
                        break;
                    default:
                        throw new InvalidDataException($"Unknown instruction at '{definition.Id}/{node.Id}'.");
                }
                preceding.Add(node.Id);
            }
        }
        var active = new HashSet<string>(StringComparer.Ordinal);
        var visited = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var definition in definitions.Values) Visit(definition.Id);
        int Visit(string id)
        {
            if (visited.TryGetValue(id, out var height)) return height;
            if (active.Count >= 64 || !active.Add(id))
                throw new InvalidDataException($"Recursive or excessively nested definition '{id}'.");
            var children = definitions[id].Nodes.Select(n => n.Instruction switch
            {
                ExecutionPlanCall call => call.Definition,
                ExecutionPlanDispatch dispatch => dispatch.Definition,
                _ => null,
            }).Where(child => child is not null).Select(child => Visit(child!)).ToArray();
            height = children.Length == 0 ? 1 : 1 + children.Max();
            if (height > 64) throw new InvalidDataException($"Definition '{id}' exceeds nesting limit.");
            active.Remove(id);
            visited[id] = height;
            return height;
        }
        var slotParameters = slots.ToDictionary(pair => pair.Key,
            pair => new ExecutionPlanParameter(pair.Key, pair.Value.Access, pair.Value.Tensor), StringComparer.Ordinal);
        foreach (var entry in entries.Values)
        {
            if (!definitions.TryGetValue(entry.Definition, out var definition))
                throw new InvalidDataException($"Unknown entry definition '{entry.Definition}'.");
            Bind(entry.Arguments, definition.Parameters, slotParameters);
            if (plan.Target == ExecutionPlanTarget.Direct3D12)
            {
                if (definition.Kind != ExecutionPlanDefinitionKind.Orchestration)
                    throw new InvalidDataException($"GPU entry '{entry.Name}' requires an explicit orchestration definition.");
                var pendingWrites = new HashSet<string>(StringComparer.Ordinal);
                ValidateGpuSequence(definition, entry.Arguments.ToDictionary(argument => argument.Parameter,
                    argument => (Slot: argument.Source, Offset: argument.ByteOffset), StringComparer.Ordinal), pendingWrites);
            }
            if (definitions.Values.SelectMany(value => value.Nodes).All(node =>
                node.Instruction is not ExecutionPlanOperator operation || operation.ParameterAccesses.Count != 0))
            {
                var initialized = slots.Values.Where(slot => slot.Scope != GraphResourceScope.Local ||
                    plan.Bindings is null && slot.Access == GraphResourceAccess.Read ||
                    plan.Bindings?.Inputs.Any(input => input.Value == slot.Id) == true)
                    .ToDictionary(slot => slot.Id, slot => new List<(ulong Start, ulong End)> { (0, Bytes(slot.Tensor)) },
                        StringComparer.Ordinal);
                ValidateMemorySequence(definition, entry.Arguments.ToDictionary(argument => argument.Parameter,
                    argument => (Slot: argument.Source, Offset: argument.ByteOffset), StringComparer.Ordinal), initialized);
            }
        }
        void ValidateMemorySequence(ExecutionPlanDefinition definition,
            IReadOnlyDictionary<string, (string Slot, ulong Offset)> bindings,
            Dictionary<string, List<(ulong Start, ulong End)>> initialized)
        {
            foreach (var node in definition.Nodes)
            {
                var arguments = node.Instruction switch
                {
                    ExecutionPlanCall call => call.Arguments,
                    ExecutionPlanDispatch dispatch => dispatch.Arguments,
                    _ => null,
                };
                var callee = node.Instruction switch
                {
                    ExecutionPlanCall call => call.Definition,
                    ExecutionPlanDispatch dispatch => dispatch.Definition,
                    _ => null,
                };
                if (arguments is not null && callee is not null)
                {
                    ValidateMemorySequence(definitions[callee], arguments.ToDictionary(argument => argument.Parameter,
                        argument => (bindings[argument.Source].Slot, checked(bindings[argument.Source].Offset + argument.ByteOffset)),
                        StringComparer.Ordinal), initialized);
                    continue;
                }
                if (node.Instruction is not ExecutionPlanOperator operation) continue;
                var views = operation.Arguments.Select(argument =>
                {
                    var source = bindings[argument.Source];
                    var parameter = definition.Parameters.Single(value => value.Name == argument.Source);
                    var start = checked(source.Offset + argument.ByteOffset);
                    return (argument.Parameter, source.Slot, Start: start,
                        End: checked(start + Bytes(parameter.Tensor)),
                        Access: operation.ParameterAccesses[argument.Parameter]);
                }).ToArray();
                for (var i = 0; i < views.Length; i++)
                    for (var j = i + 1; j < views.Length; j++)
                        if (views[i].Slot == views[j].Slot &&
                            (views[i].Access != GraphResourceAccess.Read || views[j].Access != GraphResourceAccess.Read) &&
                            views[i].Start < views[j].End && views[j].Start < views[i].End)
                            throw new InvalidDataException($"Overlapping writable arguments at '{definition.Id}/{node.Id}'.");
                foreach (var view in views.Where(view => view.Access != GraphResourceAccess.Write))
                    if (plan.Bindings is not null &&
                        (!initialized.TryGetValue(view.Slot, out var ranges) ||
                         !ranges.Any(range => range.Start <= view.Start && range.End >= view.End)))
                        throw new InvalidDataException($"Uninitialized local read at '{definition.Id}/{node.Id}/{view.Parameter}'.");
                foreach (var view in views.Where(view => view.Access != GraphResourceAccess.Read))
                {
                    if (!initialized.TryGetValue(view.Slot, out var ranges)) initialized.Add(view.Slot, ranges = []);
                    ranges.Add((view.Start, view.End));
                    ranges.Sort((left, right) => left.Start.CompareTo(right.Start));
                    for (var index = 1; index < ranges.Count;)
                        if (ranges[index - 1].End >= ranges[index].Start)
                        {
                            ranges[index - 1] = (ranges[index - 1].Start, Math.Max(ranges[index - 1].End, ranges[index].End));
                            ranges.RemoveAt(index);
                        }
                        else index++;
                }
            }
        }
        void ValidateGpuSequence(ExecutionPlanDefinition definition,
            IReadOnlyDictionary<string, (string Slot, ulong Offset)> bindings, HashSet<string> pendingWrites)
        {
            Dictionary<string, (string Slot, ulong Offset)> Map(IReadOnlyList<ExecutionPlanArgument> arguments) =>
                arguments.ToDictionary(argument => argument.Parameter, argument =>
                {
                    var source = bindings[argument.Source];
                    return (source.Slot, checked(source.Offset + argument.ByteOffset));
                }, StringComparer.Ordinal);
            foreach (var node in definition.Nodes)
                switch (node.Instruction)
                {
                    case ExecutionPlanCall call:
                        ValidateGpuSequence(definitions[call.Definition], Map(call.Arguments), pendingWrites);
                        break;
                    case ExecutionPlanBarrier barrier:
                        pendingWrites.ExceptWith(barrier.Resources.Select(resource => bindings[resource].Slot));
                        break;
                    case ExecutionPlanDispatch dispatch:
                        var kernel = definitions[dispatch.Definition];
                        var map = Map(dispatch.Arguments);
                        for (var i = 0; i < kernel.Parameters.Count; i++)
                        {
                            var parameter = kernel.Parameters[i];
                            var binding = map[parameter.Name];
                            if (pendingWrites.Contains(binding.Slot))
                                throw new InvalidDataException($"Missing explicit GPU barrier at '{definition.Id}/{node.Id}/{parameter.Name}'.");
                            for (var j = i + 1; j < kernel.Parameters.Count; j++)
                            {
                                var other = kernel.Parameters[j];
                                var otherBinding = map[other.Name];
                                if (binding.Slot == otherBinding.Slot &&
                                    (parameter.Access != GraphResourceAccess.Read || other.Access != GraphResourceAccess.Read) &&
                                    binding.Offset < otherBinding.Offset + Bytes(other.Tensor) &&
                                    otherBinding.Offset < binding.Offset + Bytes(parameter.Tensor))
                                    throw new InvalidDataException($"Overlapping writable GPU arguments at '{definition.Id}/{node.Id}'.");
                            }
                        }
                        foreach (var parameter in kernel.Parameters.Where(parameter => parameter.Access != GraphResourceAccess.Read))
                            pendingWrites.Add(map[parameter.Name].Slot);
                        break;
                }
        }
    }

    private static void Bind(IReadOnlyList<ExecutionPlanArgument> arguments,
        IReadOnlyList<ExecutionPlanParameter> parameters, IReadOnlyDictionary<string, ExecutionPlanParameter> sources)
    {
        var bindings = Unique(arguments, argument => argument.Parameter);
        if (bindings.Count != parameters.Count) throw new InvalidDataException("Call binding count differs from formal slots.");
        foreach (var parameter in parameters)
            if (!bindings.TryGetValue(parameter.Name, out var argument) ||
                !sources.TryGetValue(argument.Source, out var source) ||
                parameter.Access != GraphResourceAccess.Read && source.Access == GraphResourceAccess.Read ||
                source.Tensor.ElementType != GraphElementType.Byte && source.Tensor.ElementType != parameter.Tensor.ElementType ||
                argument.ByteOffset % Size(parameter.Tensor.ElementType) != 0 ||
                argument.ByteOffset > Bytes(source.Tensor) ||
                Bytes(parameter.Tensor) > Bytes(source.Tensor) - argument.ByteOffset)
                throw new InvalidDataException($"Invalid binding for formal slot '{parameter.Name}'.");
    }

    private static Dictionary<string, T> Unique<T>(IEnumerable<T> values, Func<T, string> key)
    {
        var result = new Dictionary<string, T>(StringComparer.Ordinal);
        foreach (var value in values)
        {
            var name = key(value);
            Name(name);
            if (!result.TryAdd(name, value)) throw new InvalidDataException($"Duplicate execution plan ID '{name}'.");
        }
        return result;
    }

    private static void Name(string value)
    {
        if (string.IsNullOrWhiteSpace(value)) throw new InvalidDataException("Execution plan names must be non-empty.");
    }
    private static void Access(GraphResourceAccess access)
    {
        if (!Enum.IsDefined(access)) throw new InvalidDataException("Invalid execution plan access.");
    }
    private static void Tensor(TensorDescriptor tensor)
    {
        if (!Enum.IsDefined(tensor.ElementType) || tensor.Layout != "dense" ||
            tensor.Dimensions.Count == 0 || tensor.Dimensions.Any(size => size <= 0))
            throw new InvalidDataException("Target execution tensors require positive dense dimensions.");
        _ = Bytes(tensor);
    }
    private static ulong Bytes(TensorDescriptor tensor) => GraphTensorView.ByteLengthOf(tensor);
    private static ulong Size(GraphElementType type) => type switch
    {
        GraphElementType.Byte => 1,
        GraphElementType.Float16 => 2,
        _ => 4,
    };
    private static void Grid(ExecutionPlanGrid grid, bool threads)
    {
        if (grid.X == 0 || grid.Y == 0 || grid.Z == 0 ||
            (threads ? grid.X > 1024 || grid.Y > 1024 || grid.Z > 64 ||
                (ulong)grid.X * grid.Y * grid.Z > 1024 :
                grid.X > 65535 || grid.Y > 65535 || grid.Z > 65535))
            throw new InvalidDataException("Invalid execution plan GPU grid.");
    }
}
