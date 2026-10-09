using SharpInference.Instructions;
using SharpInference.Instructions.D3D12;

namespace SharpInference.Runtime.D3D12;

public static class D3D12InstructionCollections
{
    public static IReadOnlyList<IInstructionCollectionProvider> Create() =>
        [new GpuFloat32InstructionCollection(), new GpuFloat16InstructionCollection(),
            new GpuTierOneInstructionCollection(), new GpuTransformerInstructionCollection()];
}
