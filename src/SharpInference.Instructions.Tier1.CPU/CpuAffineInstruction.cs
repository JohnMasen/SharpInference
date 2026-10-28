using SharpInference.Graphs.Execution;

namespace SharpInference.Instructions.Cpu;

internal sealed class CpuAffineInstruction() : TierOneAffineInstruction(ExecutionTarget.Cpu)
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>().ToDictionary(parameter => parameter.Name);
        var attributes = parameters.OfType<InstructionAttributeParameter>().ToDictionary(parameter => parameter.Name);
        var left = tensors["left"];
        var right = tensors["right"];
        var bias = tensors["bias"];
        var output = tensors["output"];
        string Span(InstructionTensorParameter tensor) =>
            $"System.Runtime.InteropServices.MemoryMarshal.Cast<byte,{(tensor.Tensor.ElementType == Graphs.ElementType.Float16 ? "System.Half" : "float")}>({tensor.Expression})";
        const string numerics = "SharpInference.Instructions.Cpu.CpuInstructionNumerics";
        return new($$"""
            {
                var affineLeft={{Span(left)}};
                var affineRight={{Span(right)}};
                var affineBias={{Span(bias)}};
                var affineOutput={{Span(output)}};
                {{numerics}}.MatrixMultiply(affineLeft,affineRight,affineOutput,
                    {{left.Tensor.Dimensions[0]}},{{left.Tensor.Dimensions[1]}},
                    {{right.Tensor.Dimensions[0]}},{{right.Tensor.Dimensions[1]}},
                    {{attributes["transpose_left"].Value.ToLowerInvariant()}},
                    {{attributes["transpose_right"].Value.ToLowerInvariant()}});
                {{numerics}}.BiasAdd(affineOutput,affineBias,affineOutput);
            }
            """, references: [typeof(CpuInstructionNumerics).Assembly]);
    }
}
