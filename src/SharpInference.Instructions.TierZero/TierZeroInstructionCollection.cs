using SharpInference.Graphs;

namespace SharpInference.Instructions;

/// <summary>Builds the supported primitive instruction set using the supplied instruction factory.</summary>
public abstract class TierZeroInstructionCollection : IInstructionCollectionProvider
{
    private readonly IReadOnlyList<InstructionCollectionDescription> collections;
    private readonly IReadOnlyDictionary<string, IReadOnlyList<Instruction>> instructions;

    /// <summary>Builds the supported primitive instruction set using the supplied instruction factory.</summary>
    /// <param name="id">The tier-zero collection identifier selecting Float32 or Float16 storage.</param>
    /// <param name="architecture">The execution target supported by this collection.</param>
    /// <param name="create">The factory that creates an instruction for each supported primitive operation name.</param>
    protected TierZeroInstructionCollection(Guid id, InstructionTarget architecture, Func<string, Instruction> create)
    {
        collections = Array.AsReadOnly(new[] { new InstructionCollectionDescription(id,
            id == InstructionCollectionIds.TierZeroFloat32 ? "T0 FP32" : "T0 FP16", 0, architecture) });
        instructions = TierZeroOperationContracts.Contracts
            .Where(contract => id == InstructionCollectionIds.TierZeroFloat32 ||
                contract.Operation != PortableTensorOperationContracts.CastFp16ToFp32)
            .ToDictionary(contract => contract.Operation.Name,
                contract => (IReadOnlyList<Instruction>)Array.AsReadOnly(new[] { create(contract.Operation.Name) }),
                StringComparer.Ordinal);
    }

    /// <inheritdoc/>
    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() => collections;
    /// <inheritdoc/>
    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == collections[0].Id && instructions.TryGetValue(instructionName, out var values) ? values : [];
}
