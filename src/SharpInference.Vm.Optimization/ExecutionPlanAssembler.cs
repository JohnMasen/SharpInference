using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

public static class ExecutionPlanAssembler
{
    public static VmProgram Assemble(TargetExecutionPlan plan)
    {
        TargetExecutionPlanValidator.Validate(plan);
        var target = plan.Target == ExecutionPlanTarget.Cpu ? VmTarget.Cpu : VmTarget.Direct3D12;
        var definitions = plan.Definitions.Select(definition => new VmDefinition(definition.Id,
            definition.Kind switch
            {
                ExecutionPlanDefinitionKind.Function => VmDefinitionKind.Function,
                ExecutionPlanDefinitionKind.Kernel => VmDefinitionKind.Kernel,
                ExecutionPlanDefinitionKind.Orchestration => VmDefinitionKind.Orchestration,
                _ => throw new InvalidDataException($"Invalid definition '{definition.Id}'."),
            }, definition.Parameters.Select(Parameter), definition.Nodes.Select(node =>
                new VmNode(node.Id, Instruction(node.Instruction, target), node.Dependencies)),
            definition.Threads is null ? null : Grid(definition.Threads)));
        return new(plan.Name, plan.Abi, target,
            plan.Slots.Select(slot => new VmSlot(slot.Id, slot.Scope switch
            {
                GraphResourceScope.Global => VmSlotScope.Global,
                GraphResourceScope.Session => VmSlotScope.Session,
                GraphResourceScope.Local => VmSlotScope.Local,
                _ => throw new InvalidDataException($"Invalid scope at '{slot.Id}'."),
            }, Access(slot.Access), Tensor(slot.Tensor), slot.BindingKey)),
            definitions, plan.Entries.Select(entry => new VmEntry(entry.Name, entry.Definition, entry.Arguments.Select(Argument))),
            new(plan.StateSchema, plan.StateVersion, plan.State.Select(entry => new VmStateEntry(entry.Name, entry.Slot))));
    }

    public static TargetExecutionPlan Bind(TargetExecutionPlan plan, InstructionRegistry registry)
    {
        TargetExecutionPlanValidator.Validate(plan);
        var target = plan.Target == ExecutionPlanTarget.Cpu ? InstructionTarget.Cpu : InstructionTarget.Direct3D12;
        var vmTarget = plan.Target == ExecutionPlanTarget.Cpu ? VmTarget.Cpu : VmTarget.Direct3D12;
        var changed = false;
        ExecutionPlanInstruction BindInstruction(ExecutionPlanDefinition definition, ExecutionPlanNode node)
        {
            if (node.Instruction is not ExecutionPlanOperator operation) return node.Instruction;
            var instruction = registry.Resolve(operation.Collection, operation.Name, target);
            var adapter = (VmOperator)Instruction(operation, vmTarget);
            InstructionSignature signature;
            try
            {
                signature = instruction.Adapt(VmInstructionParameters.Create(adapter,
                    definition.Parameters.Select(Parameter).ToArray(), parameter => (parameter.Name, "0")),
                    operation.Precision, adapter.ExecutionConfiguration);
            }
            catch (InstructionAdaptationException error)
            {
                throw new InstructionAdaptationException(operation.Collection, operation.Name,
                    $"At '{definition.Id}/{node.Id}': {error.Message}", error);
            }
            var precision = new ExecutionPlanPrecision(signature.Precision.ArithmeticType, signature.Precision.AccumulatorType);
            var accesses = signature.Ports.ToDictionary(port => port.Name, port => port.Access, StringComparer.Ordinal);
            var bounds = instruction.IndexBounds.Select(bound =>
                new ExecutionPlanIndexBound(bound.IndexPort, bound.TensorPort, bound.Axis, bound.MinimumIndex)).ToArray();
            if (precision.Arithmetic != operation.Precision.MinimumArithmeticType ||
                precision.Accumulator != operation.Precision.MinimumAccumulatorType ||
                operation.ResolvedPrecision is { } expected && expected != precision ||
                operation.ParameterAccesses.Count != 0 &&
                    (operation.ParameterAccesses.Count != accesses.Count ||
                     operation.ParameterAccesses.Any(pair => !accesses.TryGetValue(pair.Key, out var value) || value != pair.Value)) ||
                operation.IndexBounds.Count != 0 && !operation.IndexBounds.SequenceEqual(bounds))
                throw new InvalidDataException($"Instruction contract differs from the execution plan at '{definition.Id}/{node.Id}'.");
            if (operation.ResolvedPrecision == precision && operation.IndexBounds.SequenceEqual(bounds) &&
                operation.ParameterAccesses.Count == accesses.Count)
                return operation;
            changed = true;
            return new ExecutionPlanOperator(operation.Precision, operation.Collection, operation.Name,
                operation.Arguments, operation.Attributes, operation.Implementation, operation.CooperativeThreads,
                bounds, accesses, precision);
        }
        var result = new TargetExecutionPlan(plan.Name, plan.Abi, plan.Target, plan.Slots, plan.Definitions.Select(definition =>
            new ExecutionPlanDefinition(definition.Id, definition.Kind, definition.Parameters,
                definition.Nodes.Select(node => new ExecutionPlanNode(node.Id,
                    BindInstruction(definition, node),
                    node.Dependencies, node.Sources)), definition.Threads)), plan.Entries,
            plan.StateSchema, plan.StateVersion, plan.State, plan.Bindings, plan.ResourceBindings);
        TargetExecutionPlanValidator.Validate(result);
        return changed ? result : plan;
    }

    public static void ValidateArtifact(TargetExecutionPlan plan, VmProgram program, InstructionRegistry registry)
    {
        var assembled = Assemble(Bind(plan, registry));
        if (VmProgramXml.Serialize(assembled) != VmProgramXml.Serialize(VmInstructionContracts.Bind(program, registry)))
            throw new InvalidDataException("The compiled artifact does not match its target execution plan.");
    }

    public static TargetExecutionPlan Capture(VmProgram program)
    {
        VmProgramValidator.Validate(program);
        return new(program.Name, program.Abi,
            program.Target == VmTarget.Cpu ? ExecutionPlanTarget.Cpu : ExecutionPlanTarget.Direct3D12,
            program.Slots.Select(slot => new ExecutionPlanSlot(slot.Id, slot.Scope switch
            {
                VmSlotScope.Global => GraphResourceScope.Global,
                VmSlotScope.Session => GraphResourceScope.Session,
                VmSlotScope.Local => GraphResourceScope.Local,
                _ => throw new InvalidDataException($"Invalid scope at '{slot.Id}'."),
            }, slot.Access == VmAccess.ReadOnly ? GraphResourceAccess.Read : GraphResourceAccess.ReadWrite,
                Descriptor(slot.Tensor), slot.BindingKey)),
            program.Definitions.Select(definition => new ExecutionPlanDefinition(definition.Id, definition.Kind switch
            {
                VmDefinitionKind.Function => ExecutionPlanDefinitionKind.Function,
                VmDefinitionKind.Kernel => ExecutionPlanDefinitionKind.Kernel,
                VmDefinitionKind.Orchestration => ExecutionPlanDefinitionKind.Orchestration,
                _ => throw new InvalidDataException($"Invalid definition '{definition.Id}'."),
            }, definition.Parameters.Select(parameter => new ExecutionPlanParameter(parameter.Name,
                parameter.Access == VmAccess.ReadOnly ? GraphResourceAccess.Read : GraphResourceAccess.ReadWrite,
                Descriptor(parameter.Tensor))), definition.Nodes.Select(node =>
                    new ExecutionPlanNode(node.Id, CaptureInstruction(node.Instruction), node.Dependencies)),
                definition.Threads is null ? null : new(definition.Threads.X, definition.Threads.Y, definition.Threads.Z))),
            program.Entries.Select(entry => new ExecutionPlanEntry(entry.Name, entry.Definition, entry.Arguments.Select(CaptureArgument))),
            program.State.Schema, program.State.Version, program.State.Entries.Select(entry => new ExecutionPlanStateEntry(entry.Name, entry.Slot)));
    }

    internal static VmParameter Parameter(ExecutionPlanParameter parameter) =>
        new(parameter.Name, Access(parameter.Access), Tensor(parameter.Tensor));

    private static VmAccess Access(GraphResourceAccess access) =>
        access == GraphResourceAccess.Read ? VmAccess.ReadOnly : VmAccess.ReadWrite;

    private static VmTensor Tensor(TensorDescriptor tensor) => new(tensor.ElementType switch
    {
        GraphElementType.Byte => VmElementType.Byte,
        GraphElementType.Int32 => VmElementType.Int32,
        GraphElementType.UInt32 => VmElementType.UInt32,
        GraphElementType.Float16 => VmElementType.Float16,
        GraphElementType.Float32 => VmElementType.Float32,
        _ => throw new InvalidDataException("Unsupported execution plan type."),
    }, tensor.Dimensions);

    private static TensorDescriptor Descriptor(VmTensor tensor) => new(tensor.ElementType switch
    {
        VmElementType.Byte => GraphElementType.Byte,
        VmElementType.Int32 => GraphElementType.Int32,
        VmElementType.UInt32 => GraphElementType.UInt32,
        VmElementType.Float16 => GraphElementType.Float16,
        VmElementType.Float32 => GraphElementType.Float32,
        _ => throw new InvalidDataException("Unsupported VM type."),
    }, tensor.Dimensions);

    private static VmArgument Argument(ExecutionPlanArgument argument) => new(argument.Parameter, argument.Source, argument.ByteOffset);
    private static ExecutionPlanArgument CaptureArgument(VmArgument argument) => new(argument.Parameter, argument.Source, argument.ByteOffset);
    private static VmThreadGroup Grid(ExecutionPlanGrid grid) => new(grid.X, grid.Y, grid.Z);

    private static VmInstruction Instruction(ExecutionPlanInstruction instruction, VmTarget target) => instruction switch
    {
        ExecutionPlanOperator operation => new VmOperator(operation.Precision, operation.Collection, operation.Name,
            operation.Arguments.Select(Argument), operation.Attributes,
            operation.IndexBounds.Select(bound => new InstructionIndexBound(bound.IndexPort, bound.TensorPort, bound.Axis, bound.MinimumIndex)),
            operation.ParameterAccesses,
            operation.ResolvedPrecision is null ? null : new(operation.ResolvedPrecision.Arithmetic, operation.ResolvedPrecision.Accumulator),
            operation.Implementation is null ? null :
                target == VmTarget.Cpu ? new CpuInstructionExecutionConfiguration(new(operation.Implementation)) :
                new D3D12InstructionExecutionConfiguration(new(operation.Implementation))),
        ExecutionPlanCall call => new VmCall(call.Definition, call.Arguments.Select(Argument)),
        ExecutionPlanDispatch dispatch => new VmDispatch(dispatch.Definition, dispatch.Arguments.Select(Argument), Grid(dispatch.Groups)),
        ExecutionPlanBarrier barrier => new VmBarrier(barrier.Resources),
        _ => throw new InvalidDataException("Unknown execution plan instruction."),
    };

    private static ExecutionPlanInstruction CaptureInstruction(VmInstruction instruction) => instruction switch
    {
        VmOperator operation => new ExecutionPlanOperator(operation.Precision, operation.InstructionCollectionId,
            operation.InstructionName, operation.Arguments.Select(CaptureArgument), operation.Attributes,
            operation.ExecutionConfiguration?.Implementation.Value,
            operation.Operation == "core.mat-vec" && operation.ExecutionConfiguration is D3D12InstructionExecutionConfiguration
                ? operation.ExecutionConfiguration == GpuMatVecExecution.Cooperative ? 64u : 1u : null,
            operation.IndexBounds.Select(bound => new ExecutionPlanIndexBound(bound.IndexPort, bound.TensorPort, bound.Axis, bound.MinimumIndex)),
            operation.ParameterAccesses,
            operation.ResolvedPrecision is null ? null : new(operation.ResolvedPrecision.ArithmeticType, operation.ResolvedPrecision.AccumulatorType)),
        VmCall call => new ExecutionPlanCall(call.Definition, call.Arguments.Select(CaptureArgument)),
        VmDispatch dispatch => new ExecutionPlanDispatch(dispatch.Definition, dispatch.Arguments.Select(CaptureArgument),
            new(dispatch.Groups.X, dispatch.Groups.Y, dispatch.Groups.Z)),
        VmBarrier barrier => new ExecutionPlanBarrier(barrier.Resources),
        _ => throw new InvalidDataException("Unknown VM instruction."),
    };
}
