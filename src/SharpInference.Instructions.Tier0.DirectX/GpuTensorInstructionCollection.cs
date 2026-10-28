using SharpInference.Graphs.Execution;

namespace SharpInference.Instructions.D3D12;

/// <summary>Owns the complete portable T0 tensor set for Direct3D12.</summary>
public sealed class GpuTensorInstructionCollection : IInstructionProvider
{
    private readonly IReadOnlyList<Instruction> instructions = Array.AsReadOnly<Instruction>(
        [new GpuArgMaxInstruction(), new GpuEmbeddingGatherBatchInstruction(),
         new GpuCopyEmbeddingRowsInstruction(), new GpuBatchSharedMatVecInstruction()]);

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [TierZeroTensorInstructionSet.Describe(ExecutionTarget.Direct3D12)];

    public IReadOnlyList<Instruction> EnumerateInstructions() => instructions;

    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == TierZeroTensorInstructionSet.Id
            ? instructions.Where(i => i.Name == instructionName).ToArray() : [];
}
