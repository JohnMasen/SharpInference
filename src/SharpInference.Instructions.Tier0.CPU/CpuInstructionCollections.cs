using SharpInference.Graphs.Execution;
namespace SharpInference.Instructions.Cpu;

/// <summary>Provides tier-zero CPU instructions for Float32 tensor storage.</summary>
public sealed class CpuFloat32InstructionCollection() : TierZeroInstructionCollection(
    TierZeroInstructionCollectionIds.Float32, ExecutionTarget.Cpu,
    name => new CpuTierZeroInstruction(TierZeroInstructionCollectionIds.Float32, name));

/// <summary>Provides tier-zero CPU instructions for Float16 tensor storage.</summary>
public sealed class CpuFloat16InstructionCollection() : TierZeroInstructionCollection(
    TierZeroInstructionCollectionIds.Float16, ExecutionTarget.Cpu,
    name => new CpuTierZeroInstruction(TierZeroInstructionCollectionIds.Float16, name));
