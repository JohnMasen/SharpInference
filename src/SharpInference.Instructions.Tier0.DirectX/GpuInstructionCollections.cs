using SharpInference.Graphs.Execution;
namespace SharpInference.Instructions.D3D12;
/// <summary>Provides tier-zero Direct3D12 instructions for Float32 tensor storage.</summary>
public sealed class GpuFloat32InstructionCollection() : TierZeroInstructionCollection(TierZeroInstructionCollectionIds.Float32, ExecutionTarget.Direct3D12, name => new GpuTierZeroInstruction(TierZeroInstructionCollectionIds.Float32, name));
/// <summary>Provides tier-zero Direct3D12 instructions for Float16 tensor storage.</summary>
public sealed class GpuFloat16InstructionCollection() : TierZeroInstructionCollection(TierZeroInstructionCollectionIds.Float16, ExecutionTarget.Direct3D12, name => new GpuTierZeroInstruction(TierZeroInstructionCollectionIds.Float16, name));
