namespace SharpInference.Instructions.Cpu;

/// <summary>Provides tier-zero CPU instructions for Float32 tensor storage.</summary>
public sealed class CpuFloat32InstructionCollection() : TierZeroInstructionCollection(
    InstructionCollectionIds.TierZeroFloat32, InstructionTarget.Cpu,
    name => new CpuTierZeroInstruction(InstructionCollectionIds.TierZeroFloat32, name));

/// <summary>Provides tier-zero CPU instructions for Float16 tensor storage.</summary>
public sealed class CpuFloat16InstructionCollection() : TierZeroInstructionCollection(
    InstructionCollectionIds.TierZeroFloat16, InstructionTarget.Cpu,
    name => new CpuTierZeroInstruction(InstructionCollectionIds.TierZeroFloat16, name));
