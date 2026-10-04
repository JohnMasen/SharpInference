using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;

namespace SharpInference.Runtime;

public static class DefaultInstructionCollections
{
    public static IReadOnlyList<IInstructionCollectionProvider> Create() =>
        [new CpuFloat32InstructionCollection(), new CpuFloat16InstructionCollection(),
            new GpuFloat32InstructionCollection(), new GpuFloat16InstructionCollection()];
}
