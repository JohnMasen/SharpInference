using SharpInference.Graphs.Execution;
namespace SharpInference.Instructions.Cpu;

/// <summary>Provides optimized Float32 CPU pointwise instructions and their graph optimization capabilities.</summary>
public sealed class CpuTierOneInstructionCollection : IInstructionCollectionProvider, IInstructionOptimizationProvider
{
    private readonly IReadOnlyDictionary<string, CpuTierOneInstruction> instructions =
        TierOnePointwiseCatalog.Definitions.ToDictionary(definition => definition.Name,
            definition => new CpuTierOneInstruction(definition.Name), StringComparer.Ordinal);

    /// <inheritdoc/>
    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [new(InstructionCollectionIds.TierOneFloat32, "FP32 optimized pointwise", 1, ExecutionTarget.Cpu)];

    /// <inheritdoc/>
    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == InstructionCollectionIds.TierOneFloat32 && instructions.TryGetValue(instructionName, out var instruction)
            ? [instruction] : [];

    /// <inheritdoc/>
    public IReadOnlyList<InstructionOptimizationCapability> QueryOptimizationCapabilities() =>
        Array.AsReadOnly(instructions.Values.Select(instruction => instruction.OptimizationCapability()).ToArray());
}
