using System.Text;

namespace SharpInference.Instructions.D3D12;

internal sealed class GpuTierOneInstruction(string name)
    : TierOnePointwiseInstruction(name, new D3D12InstructionExecutionConfiguration(new("register-pointwise")))
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>().ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);
        var output = tensors["output"];
        var count = output.Tensor.Dimensions.Aggregate(1UL, (value, dimension) => checked(value * (ulong)dimension));
        if (count > uint.MaxValue)
            throw new InstructionAdaptationException(CollectionId, Name, "Element count exceeds the shader index range.");
        var code = new StringBuilder($"{{ if(i<{count}u) {{\n");
        foreach (var (port, index) in Definition.Inputs.Select((port, index) => (port, index)))
        {
            var input = tensors[port];
            code.AppendLine($"precise float s{index}=load32({input.Expression},{input.OffsetExpression},i);");
        }
        string Value(TierOneOperand operand) => operand.IsInput ? $"s{operand.Index}" : $"t{operand.Index}";
        foreach (var (stage, index) in Definition.Stages.Select((stage, index) => (stage, index)))
        {
            var values = stage.Operands.Select(Value).ToArray();
            var expression = stage.Operation.Name switch
            {
                "core.add" => $"{values[0]}+{values[1]}",
                "core.subtract" => $"{values[0]}-{values[1]}",
                "core.multiply" => $"{values[0]}*{values[1]}",
                "core.maximum" => $"maximum32({values[0]},{values[1]})",
                "core.rsqrt" => $"rsqrt({values[0]})",
                "core.sigmoid" => $"1.0f/(1.0f+exp(-({values[0]})))",
                "core.exp" => $"exp({values[0]})",
                "core.relu" => $"maximum32({values[0]},0.0f)",
                "core.square" => $"{values[0]}*{values[0]}",
                _ => throw new NotSupportedException(stage.Operation.Name),
            };
            code.AppendLine($"precise float t{index}={expression};");
        }
        code.AppendLine($"{output.Expression}.Store({output.OffsetExpression}+i*4u,asuint(t{Definition.Stages.Count - 1}));");
        code.AppendLine("} }");
        return new(code.ToString(), [GpuInstructionHelpers.Load32, GpuInstructionHelpers.Maximum32]);
    }
}
