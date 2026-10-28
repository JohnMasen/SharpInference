using SharpInference.Graphs.Execution;
namespace SharpInference.Instructions.D3D12;

/// <summary>Provides optimized Float32 Direct3D12 pointwise instructions and their graph optimization capabilities.</summary>
public sealed class GpuTierOneInstructionCollection : IInstructionProvider, IInstructionOptimizationProvider
{
    private readonly IReadOnlyDictionary<string, GpuTierOneInstruction> instructions =
        TierOnePointwiseCatalog.Definitions.ToDictionary(definition => definition.Name,
            definition => new GpuTierOneInstruction(definition.Name), StringComparer.Ordinal);

    /// <inheritdoc/>
    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [new(TierOneInstructionCollectionIds.Float32, "FP32 optimized pointwise", 1, ExecutionTarget.Direct3D12,
            RequiredInstructions: TierOnePointwiseCatalog.Definitions.Select(definition => definition.Name).ToArray()),
         new(TierOneAffineOperations.CollectionFloat32, "Affine", 1, ExecutionTarget.Direct3D12,
            RequiredInstructions: [TierOneAffineOperations.Affine.Name])];

    private readonly Instruction affine = new GpuAffineInstruction();
    public IReadOnlyList<Instruction> EnumerateInstructions() => instructions.Values.Cast<Instruction>().Append(affine).ToArray();

    /// <inheritdoc/>
    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == TierOneAffineOperations.CollectionFloat32 && instructionName == affine.Name ? [affine] :
        collectionId == TierOneInstructionCollectionIds.Float32 && instructions.TryGetValue(instructionName, out var instruction)
            ? [instruction] : [];

    /// <inheritdoc/>
    public IReadOnlyList<InstructionOptimizationCapability> QueryOptimizationCapabilities() =>
        Array.AsReadOnly(instructions.Values.Select(instruction => instruction.OptimizationCapability()).ToArray());
}
