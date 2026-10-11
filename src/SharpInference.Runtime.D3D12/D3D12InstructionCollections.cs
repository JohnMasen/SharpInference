using SharpInference.Instructions;
using SharpInference.Instructions.D3D12;

namespace SharpInference.Runtime.D3D12;

/// <summary>Provides the default Direct3D 12 VM instruction collections.</summary>
public static class D3D12InstructionCollections
{
    /// <summary>Creates float32, float16, tier-one, and transformer GPU instruction collection providers.</summary>
    public static IReadOnlyList<IInstructionCollectionProvider> Create() =>
        [new GpuFloat32InstructionCollection(), new GpuFloat16InstructionCollection(),
            new GpuTierOneInstructionCollection(), new GpuTransformerInstructionCollection()];
}
