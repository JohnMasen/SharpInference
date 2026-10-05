namespace SharpInference.Instructions.D3D12;

public sealed class GpuTierOneInstructionCollection : IInstructionCollectionProvider, IInstructionOptimizationProvider
{
    private readonly IReadOnlyDictionary<string, GpuTierOneInstruction> instructions =
        TierOnePointwiseCatalog.Definitions.ToDictionary(definition => definition.Name,
            definition => new GpuTierOneInstruction(definition.Name), StringComparer.Ordinal);

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [new(InstructionCollectionIds.TierOneFloat32, "FP32 optimized pointwise", 1, InstructionTarget.Direct3D12)];

    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == InstructionCollectionIds.TierOneFloat32 && instructions.TryGetValue(instructionName, out var instruction)
            ? [instruction] : [];

    public IReadOnlyList<InstructionOptimizationCapability> QueryOptimizationCapabilities() =>
        Array.AsReadOnly(instructions.Values.Select(instruction => instruction.OptimizationCapability()).ToArray());
}
