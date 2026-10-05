using System.Numerics.Tensors;
using System.Text;

namespace SharpInference.Instructions.Cpu;

internal sealed class CpuTierOneInstruction(string name)
    : TierOnePointwiseInstruction(name, new CpuInstructionExecutionConfiguration(new("adaptive-pointwise")))
{
    protected override string NumericalImplementationIdentity =>
        $"{typeof(TensorPrimitives).Assembly.ManifestModule.ModuleVersionId:D}|" +
        $"{typeof(Backends.Cpu.CpuPrimitiveOperatorBackend).Assembly.ManifestModule.ModuleVersionId:D}";

    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>().ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);
        var code = new StringBuilder("{\n");
        foreach (var (port, index) in Definition.Inputs.Select((port, index) => (port, index)))
            code.AppendLine($"System.ReadOnlySpan<float> s{index} = System.Runtime.InteropServices.MemoryMarshal.Cast<byte,float>({tensors[port].Expression});");
        code.AppendLine($"System.Span<float> o = System.Runtime.InteropServices.MemoryMarshal.Cast<byte,float>({tensors["output"].Expression});");
        var method = Name switch
        {
            "add-rsqrt" => "AddRsqrt",
            "maximum-rsqrt" => "MaximumRsqrt",
            "multiply-add" => "MultiplyAdd",
            "add-multiply-add" => "AddMultiplyAdd",
            "multiply-multiply-add" => "MultiplyMultiplyAdd",
            "multiply-add-multiply" => "MultiplyAddMultiply",
            "difference-mix" => "DifferenceMix",
            "multiply-add-subtract" => "MultiplyAddSubtract",
            "multiply-self-sigmoid" => "MultiplySelfSigmoid",
            "add-sigmoid" => "AddSigmoid",
            "exp-subtract-exp" => "ExpSubtractExp",
            "sigmoid-multiply-exp" => "SigmoidMultiplyExp",
            "relu-square" => "ReluSquare",
            _ => throw new NotSupportedException(Name),
        };
        code.AppendLine($"SharpInference.Instructions.Cpu.CpuTierOneNumerics.{method}({string.Join(",", Definition.Inputs.Select((_, index) => $"s{index}").Append("o"))});");
        code.AppendLine("}");
        return new(code.ToString(), references: [typeof(CpuTierOneNumerics).Assembly]);
    }
}
