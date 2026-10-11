using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;

namespace SharpInference.Runtime.Cpu;

/// <summary>Provides the default CPU VM instruction collections used for compilation and graph generation.</summary>
public static class CpuInstructionCollections
{
    /// <summary>Creates float32, float16, and tier-one CPU instruction collection providers.</summary>
    public static IReadOnlyList<IInstructionCollectionProvider> Create() =>
        [new CpuFloat32InstructionCollection(), new CpuFloat16InstructionCollection(), new CpuTierOneInstructionCollection()];
}
