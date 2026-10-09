using SharpInference.Instructions;

namespace SharpInference.Vm;

/// <summary>Resolves, removes, and validates instruction contracts embedded in VM programs.</summary>
public static class VmInstructionContracts
{
    /// <summary>Creates a copy of a program with resolved instruction contract fields removed.</summary>
    public static VmProgram WithoutContracts(VmProgram program) => new(program.Name, program.Abi, program.Target,
        program.Slots, program.Definitions.Select(definition => new VmDefinition(definition.Id, definition.Kind,
            definition.Parameters, definition.Nodes.Select(node => node.Instruction is VmOperator operation
                ? new VmNode(node.Id, new VmOperator(operation.Precision, operation.InstructionCollectionId, operation.InstructionName,
                    operation.Arguments, operation.Attributes, executionConfiguration: operation.ExecutionConfiguration), node.Dependencies) : node), definition.Threads)),
        program.Entries, program.State);

    /// <summary>Resolves instruction implementations and binds their precision, access, and index contracts.</summary>
    /// <param name="program">The program whose operators are to be resolved.</param>
    /// <param name="registry">The instruction registry used for resolution.</param>
    /// <returns>The original program when contracts already match, otherwise a program containing resolved contracts.</returns>
    public static VmProgram Bind(VmProgram program, InstructionRegistry registry)
    {
        var target = program.Target switch
        {
            VmTarget.Cpu => InstructionTarget.Cpu,
            VmTarget.Direct3D12 => InstructionTarget.Direct3D12,
            _ => throw new NotSupportedException($"No instruction target adapter is installed for '{program.Target}'."),
        };
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

    /// <summary>Validates the program and requires every VM operator to contain a resolved contract.</summary>
    /// <param name="program">The program to validate.</param>
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
