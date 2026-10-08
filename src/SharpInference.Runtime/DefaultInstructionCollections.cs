using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;

namespace SharpInference.Runtime;

/// <summary>Creates the standard CPU, GPU, and transformer instruction collections.</summary>
public static class DefaultInstructionCollections
{
    /// <summary>Creates the instruction collection providers used by the default runtime factories.</summary>
    /// <returns>The default instruction collection providers.</returns>
    public static IReadOnlyList<IInstructionCollectionProvider> Create() =>
        [new CpuFloat32InstructionCollection(), new CpuFloat16InstructionCollection(),
            new GpuFloat32InstructionCollection(), new GpuFloat16InstructionCollection(),
            new CpuTierOneInstructionCollection(), new GpuTierOneInstructionCollection(),
            new GpuTransformerInstructionCollection()];
}
