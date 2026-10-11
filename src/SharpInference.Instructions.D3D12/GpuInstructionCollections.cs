namespace SharpInference.Instructions.D3D12;
/// <summary>Provides tier-zero Direct3D12 instructions for Float32 tensor storage.</summary>
public sealed class GpuFloat32InstructionCollection() : TierZeroInstructionCollection(InstructionCollectionIds.TierZeroFloat32,InstructionTarget.Direct3D12,name=>new GpuTierZeroInstruction(InstructionCollectionIds.TierZeroFloat32,name));
/// <summary>Provides tier-zero Direct3D12 instructions for Float16 tensor storage.</summary>
public sealed class GpuFloat16InstructionCollection() : TierZeroInstructionCollection(InstructionCollectionIds.TierZeroFloat16,InstructionTarget.Direct3D12,name=>new GpuTierZeroInstruction(InstructionCollectionIds.TierZeroFloat16,name));
