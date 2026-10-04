using SharpInference.Instructions;

namespace SharpInference.Vm;

public static class VmInstructionContracts
{
    public static VmProgram WithoutContracts(VmProgram program) => new(program.Name, program.Abi, program.Target,
        program.Slots, program.Definitions.Select(definition => new VmDefinition(definition.Id, definition.Kind,
            definition.Parameters, definition.Nodes.Select(node => node.Instruction is VmOperator operation
                ? new VmNode(node.Id, new VmOperator(operation.InstructionCollectionId, operation.InstructionName,
                    operation.Arguments, operation.Attributes), node.Dependencies) : node), definition.Threads)),
        program.Entries, program.State);

    public static VmProgram Bind(VmProgram program, InstructionRegistry registry)
    {
        var target = program.Target == VmTarget.Cpu ? InstructionTarget.Cpu : InstructionTarget.Direct3D12;
        var changed = false;
        var definitions = program.Definitions.Select(definition => new VmDefinition(definition.Id, definition.Kind,
            definition.Parameters, definition.Nodes.Select(node =>
            {
                if (node.Instruction is not VmOperator operation) return node;
                if (operation.Version != 1)
                    throw new NotSupportedException("Legacy instruction versions are unsupported; use a distinct IC GUID.");
                var instruction = registry.Resolve(operation.InstructionCollectionId, operation.InstructionName, target);
                var signature = instruction.GetSignature(VmInstructionParameters.Create(operation, definition.Parameters, parameter => (parameter.Name, "0")));
                var accesses = signature.Ports.ToDictionary(port => port.Name, port => port.Access, StringComparer.Ordinal);
                var sameAccess = operation.ParameterAccesses.Count == accesses.Count &&
                    operation.ParameterAccesses.All(pair => accesses.TryGetValue(pair.Key, out var access) && access == pair.Value);
                if (operation.IndexBounds.SequenceEqual(instruction.IndexBounds) && sameAccess) return node;
                if (operation.IndexBounds.Count != 0 && !operation.IndexBounds.SequenceEqual(instruction.IndexBounds) ||
                    operation.ParameterAccesses.Count != 0 && !sameAccess)
                    throw new InvalidDataException($"Index constraints conflict with '{operation.InstructionName}'.");
                changed = true;
                return new VmNode(node.Id, new VmOperator(operation.InstructionCollectionId, operation.InstructionName,
                    operation.Arguments, operation.Attributes, instruction.IndexBounds, accesses), node.Dependencies);
            }), definition.Threads)).ToArray();
        return changed ? new VmProgram(program.Name, program.Abi, program.Target, program.Slots,
            definitions, program.Entries, program.State) : program;
    }
}
