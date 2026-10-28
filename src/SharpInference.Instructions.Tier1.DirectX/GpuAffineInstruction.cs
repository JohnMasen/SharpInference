using SharpInference.Graphs.Execution;

namespace SharpInference.Instructions.D3D12;

internal sealed class GpuAffineInstruction() : TierOneAffineInstruction(ExecutionTarget.Direct3D12)
{
    protected override void ValidateExecutionConfiguration(InstructionExecutionConfiguration? configuration)
    {
        if (configuration is not null && configuration != GpuMatrixMultiplyExecution.Serial &&
            configuration != GpuMatrixMultiplyExecution.Cooperative && configuration != GpuMatrixMultiplyExecution.Tiled)
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Unsupported matrix execution configuration.");
        }
    }

    protected override InstructionRecording Generate(InstructionParameter[] parameters) =>
        GenerateConfigured(parameters, null);

    protected override InstructionRecording GenerateConfigured(InstructionParameter[] parameters,
        InstructionExecutionConfiguration? configuration) =>
        GpuMatrixSource.Generate(parameters, addBias: true,
            cooperative: configuration == GpuMatrixMultiplyExecution.Cooperative,
            tiled: configuration == GpuMatrixMultiplyExecution.Tiled);
}
