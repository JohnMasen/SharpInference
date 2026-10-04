namespace SharpInference.Instructions;

public sealed class TargetInstructionCollection(IInstructionCollectionProvider provider, InstructionTarget target)
    : IInstructionCollectionProvider
{
    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        Array.AsReadOnly(provider.QueryInstructionCollection()
            .Where(collection => collection.Architecture == target).ToArray());
    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        Array.AsReadOnly(provider.QueryInstruction(collectionId, instructionName)
            .Where(instruction => instruction.Target == target).ToArray());
}
