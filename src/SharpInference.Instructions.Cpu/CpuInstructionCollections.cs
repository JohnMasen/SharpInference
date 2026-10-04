namespace SharpInference.Instructions.Cpu;

public sealed class CpuFloat32InstructionCollection() : TierZeroInstructionCollection(
    InstructionCollectionIds.TierZeroFloat32, InstructionTarget.Cpu,
    name => new CpuTierZeroInstruction(InstructionCollectionIds.TierZeroFloat32, name));

public sealed class CpuFloat16InstructionCollection() : TierZeroInstructionCollection(
    InstructionCollectionIds.TierZeroFloat16, InstructionTarget.Cpu,
    name => new CpuTierZeroInstruction(InstructionCollectionIds.TierZeroFloat16, name));
