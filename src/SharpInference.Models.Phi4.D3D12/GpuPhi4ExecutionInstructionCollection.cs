namespace SharpInference.Instructions.D3D12;

/// <summary>Owns the GPU-only Phi-4 position, cache, and active-batch execution protocol.</summary>
public sealed class GpuPhi4ExecutionInstructionCollection : IInstructionProvider
{
    private readonly IReadOnlyList<Instruction> instructions = Array.AsReadOnly<Instruction>(
    [
        new GpuRopeKeyValueWriteInstruction(),
        new GpuGroupedQueryScoresInstruction(),
        new GpuCausalSoftmaxInstruction(),
        new GpuGroupedQueryValuesInstruction(),
        new GpuGroupedQueryAttentionInstruction(),
        new GpuBatchRopeKeyValueWriteInstruction(),
        new GpuBatchGroupedQueryScoresInstruction(),
        new GpuBatchCausalSoftmaxInstruction(),
        new GpuBatchGroupedQueryValuesInstruction(),
        new GpuBatchSelectLastInstruction(),
        new GpuBatchGatherRowsInstruction(),
    ]);

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [Phi4ExecutionInstructionCollectionIds.Describe()];

    public IReadOnlyList<Instruction> EnumerateInstructions() => instructions;

    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == Phi4ExecutionInstructionCollectionIds.Float32
            ? instructions.Where(i => i.Name == instructionName).ToArray() : [];
}
