using SharpInference.Graphs.Execution;
namespace SharpInference.Instructions;

/// <summary>Builds the supported primitive instruction set using the supplied instruction factory.</summary>
public abstract class TierZeroInstructionCollection : IInstructionCollectionProvider
{
    private readonly IReadOnlyList<InstructionCollectionDescription> collections;
    private readonly IReadOnlyDictionary<string, IReadOnlyList<Instruction>> instructions;

    protected TierZeroInstructionCollection(
        Guid id,
        ExecutionTarget architecture,
        Func<string, Instruction> create)
    {
        collections =
        [
            new(
                id,
                id == InstructionCollectionIds.TierZeroFloat32 ? "T0 FP32" : "T0 FP16",
                0,
                architecture),
        ];
        instructions = TierZeroInstruction.OperationNames
            .Where(name => id == InstructionCollectionIds.TierZeroFloat32 ||
                name != "core.tensor.cast-f16-f32")
            .ToDictionary(
                name => name,
                name => (IReadOnlyList<Instruction>)[create(name)],
                StringComparer.Ordinal);
    }

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() => collections;

    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == collections[0].Id &&
        instructions.TryGetValue(instructionName, out var values)
            ? values
            : [];
}
