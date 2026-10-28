using SharpInference.Graphs.Execution;

namespace SharpInference.Instructions.D3D12;

/// <summary>Owns the complete portable T1 normalization set for Direct3D12.</summary>
public sealed class GpuNormalizationInstructionCollection : IInstructionProvider
{
    private readonly IReadOnlyList<Instruction> instructions = Array.AsReadOnly<Instruction>(
        [new GpuRmsNormInstruction(), new GpuResidualRmsNormInstruction(),
         new GpuSwiGluInstruction(), new GpuScaledAddInstruction(),
         new GpuBatchRmsNormInstruction(), new GpuBatchResidualRmsNormInstruction(),
         new GpuBatchSwiGluInstruction()]);

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [TierOneNormalizationInstructionSet.Describe(ExecutionTarget.Direct3D12)];

    public IReadOnlyList<Instruction> EnumerateInstructions() => instructions;

    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == TierOneNormalizationInstructionSet.Id
            ? instructions.Where(i => i.Name == instructionName).ToArray() : [];
}
