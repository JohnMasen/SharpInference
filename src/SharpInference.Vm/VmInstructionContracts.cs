using SharpInference.Instructions;

namespace SharpInference.Vm;

public static class VmInstructionContracts
{
    public static VmProgram WithoutContracts(VmProgram program) => new(program.Name, program.Abi, program.Target,
        program.Slots, program.Definitions.Select(definition => new VmDefinition(definition.Id, definition.Kind,
            definition.Parameters, definition.Nodes.Select(node => node.Instruction is VmOperator operation
                ? new VmNode(node.Id, new VmOperator(operation.Precision, operation.InstructionCollectionId, operation.InstructionName,
                    operation.Arguments, operation.Attributes, executionConfiguration: operation.ExecutionConfiguration), node.Dependencies) : node), definition.Threads)),
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
                InstructionSignature signature;
                try
                {
                    signature = instruction.Adapt(VmInstructionParameters.Create(operation, definition.Parameters, parameter => (parameter.Name, "0")),
                        operation.Precision, operation.ExecutionConfiguration);
                }
                catch (InstructionAdaptationException error)
                {
                    throw new InstructionAdaptationException(operation.InstructionCollectionId, operation.InstructionName,
                        $"At '{definition.Id}/{node.Id}': {error.Message}", error.InnerException);
                }
                var accesses = signature.Ports.ToDictionary(port => port.Name, port => port.Access, StringComparer.Ordinal);
                var sameAccess = operation.ParameterAccesses.Count == accesses.Count &&
                    operation.ParameterAccesses.All(pair => accesses.TryGetValue(pair.Key, out var access) && access == pair.Value);
                if (operation.ResolvedPrecision is not null && operation.ResolvedPrecision != signature.Precision)
                    throw new InvalidDataException($"Precision contract conflicts with '{definition.Id}/{node.Id}' ({operation.InstructionName}).");
                if (operation.IndexBounds.SequenceEqual(instruction.IndexBounds) && sameAccess &&
                    operation.ResolvedPrecision == signature.Precision) return node;
                if (operation.IndexBounds.Count != 0 && !operation.IndexBounds.SequenceEqual(instruction.IndexBounds) ||
                    operation.ParameterAccesses.Count != 0 && !sameAccess)
                    throw new InvalidDataException($"Index constraints conflict with '{operation.InstructionName}'.");
                changed = true;
                return new VmNode(node.Id, new VmOperator(operation.Precision, operation.InstructionCollectionId, operation.InstructionName,
                    operation.Arguments, operation.Attributes, instruction.IndexBounds, accesses, signature.Precision,
                    operation.ExecutionConfiguration), node.Dependencies);
            }), definition.Threads)).ToArray();
        return changed ? new VmProgram(program.Name, program.Abi, program.Target, program.Slots,
            definitions, program.Entries, program.State) : program;
    }

    public static void ValidateResolved(VmProgram program)
    {
        VmProgramValidator.Validate(program);
        foreach (var definition in program.Definitions)
            foreach (var node in definition.Nodes)
                if (node.Instruction is VmOperator operation &&
                    (operation.ResolvedPrecision is null || operation.ParameterAccesses.Count == 0))
                    throw new InvalidDataException($"Missing resolved instruction contract at '{definition.Id}/{node.Id}'.");
    }
}
