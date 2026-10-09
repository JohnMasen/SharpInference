using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;

namespace SharpInference.Runtime.Cpu;

public static class CpuInstructionCollections
{
    public static IReadOnlyList<IInstructionCollectionProvider> Create() =>
        [new CpuFloat32InstructionCollection(), new CpuFloat16InstructionCollection(), new CpuTierOneInstructionCollection()];
}
