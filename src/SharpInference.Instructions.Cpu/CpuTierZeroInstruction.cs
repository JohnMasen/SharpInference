using System.Globalization;
using System.Text;
using SharpInference.Graphs;

namespace SharpInference.Instructions.Cpu;

internal sealed class CpuTierZeroInstruction(Guid id, string name) : TierZeroInstruction(id, name, InstructionTarget.Cpu)
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>().ToDictionary(p => p.Name, StringComparer.Ordinal);
        var attributes = parameters.OfType<InstructionAttributeParameter>().ToDictionary(p => p.Name, p => p.Value);
        var inputs = TierZeroOperationContracts.Get(new(Name)).ResolveInputPorts(tensors.Keys).ToArray();
        var code = new StringBuilder("{\n");
        foreach (var port in inputs.Append("output"))
        {
            var tensor = tensors[port];
            var type = tensor.Tensor.ElementType switch
            {
                GraphElementType.Float16 => "System.Half", GraphElementType.Float32 => "float", GraphElementType.Int32 => "int",
                _ => throw new InstructionAdaptationException(CollectionId, Name, "Unsupported type."),
            };
            var variable = port == "output" ? "o" : $"s{Array.IndexOf(inputs, port)}";
            code.AppendLine($"{(port == "output" ? "System.Span" : "System.ReadOnlySpan")}<{type}> {variable} = System.Runtime.InteropServices.MemoryMarshal.Cast<byte,{type}>({tensor.Expression});");
        }
        var output = tensors["output"].Tensor.Dimensions;
        var half = tensors["output"].Tensor.ElementType == GraphElementType.Float16;
        string Result(string expression) => half ? $"(System.Half)({expression})" : expression;
        const string primitive = "SharpInference.Backends.Cpu.CpuPrimitiveOperatorBackend.Instance.";
        int Product(IEnumerable<int> dims) => dims.Aggregate(1, (a, b) => checked(a * b));
        if (!Name.StartsWith("core.tensor.", StringComparison.Ordinal))
        {
            var method = Name switch
            {
                "core.copy" => "Copy", "core.add" => "Add", "core.subtract" => "Subtract", "core.multiply" => "Multiply",
                "core.divide" => "Divide", "core.maximum" => "Maximum", "core.exp" => "Exp", "core.tanh" => "Tanh",
                "core.sigmoid" => "Sigmoid", "core.rsqrt" => "ReciprocalSquareRoot", "core.square" => "Square",
                "core.relu" => "Relu", "core.reduce-sum" => "ReduceSum", "core.reduce-mean" => "ReduceMean",
                "core.mat-vec" => "MatVec", "core.matrix-multiply" => "MatrixMultiply",
                "core.affine" => "Affine",
                "core.bias-add" => "BiasAdd", "core.gather-row" => "GatherRow",
                _ => throw new NotSupportedException(Name),
            };
            var invocation = primitive + method;
            if (Name == "core.tanh" && !half)
                invocation = "SharpInference.Instructions.Cpu.CpuInstructionNumerics.Tanh";
            if (Name is "core.mat-vec" or "core.gather-row" && !half &&
                tensors[inputs[0]].Tensor.ElementType == GraphElementType.Float16)
                invocation = "SharpInference.Instructions.Cpu.CpuInstructionNumerics." + method;
            if (Name is "core.matrix-multiply" or "core.bias-add" or "core.affine")
                invocation = "SharpInference.Instructions.Cpu.CpuInstructionNumerics." + method;
            if (Name is "core.reduce-sum" or "core.reduce-mean")
                code.AppendLine($"o[0] = {invocation}(s0);");
            else if (Name == "core.mat-vec")
                code.AppendLine($"{invocation}(s0,s1,o,{tensors[inputs[0]].Tensor.Dimensions[0]},{tensors[inputs[0]].Tensor.Dimensions[1]});");
            else if (Name == "core.matrix-multiply")
                code.AppendLine($"{invocation}(s0,s1,o,{tensors["left"].Tensor.Dimensions[0]},{tensors["left"].Tensor.Dimensions[1]},{tensors["right"].Tensor.Dimensions[0]},{tensors["right"].Tensor.Dimensions[1]},{attributes["transpose_left"].ToLowerInvariant()},{attributes["transpose_right"].ToLowerInvariant()});");
            else if (Name == "core.affine")
            {
                code.AppendLine($"SharpInference.Instructions.Cpu.CpuInstructionNumerics.MatrixMultiply(s0,s1,o,{tensors["left"].Tensor.Dimensions[0]},{tensors["left"].Tensor.Dimensions[1]},{tensors["right"].Tensor.Dimensions[0]},{tensors["right"].Tensor.Dimensions[1]},{attributes["transpose_left"].ToLowerInvariant()},{attributes["transpose_right"].ToLowerInvariant()});");
                code.AppendLine("SharpInference.Instructions.Cpu.CpuInstructionNumerics.BiasAdd(o,s2,o);");
            }
            else if (Name == "core.bias-add")
                code.AppendLine($"{invocation}(s0,s1,o);");
            else if (Name == "core.gather-row")
                code.AppendLine($"{invocation}(s0,s1[0],{tensors["table"].Tensor.Dimensions[1]},o);");
            else code.AppendLine($"{invocation}({string.Join(",", inputs.Select((_, i) => $"s{i}").Append("o"))});");
        }
        else switch (Name)
        {
            case "core.tensor.fill":
                code.AppendLine($"o.Fill({Result(float.Parse(attributes["value"], CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture) + "f")});");
                break;
            case "core.tensor.cast-f16-f32":
                code.AppendLine("for(int i=0;i<o.Length;i++) o[i]=(float)s0[i];"); break;
            case "core.tensor.reshape": code.AppendLine("s0.CopyTo(o);"); break;
            case "core.tensor.slice":
                var shape = tensors["input"].Tensor.Dimensions;
                var axis = int.Parse(attributes["axis"], CultureInfo.InvariantCulture);
                var start = int.Parse(attributes["start"], CultureInfo.InvariantCulture);
                var length = int.Parse(attributes["length"], CultureInfo.InvariantCulture);
                var inner = Product(shape.Skip(axis + 1));
                code.AppendLine($"for(int i=0;i<{Product(shape.Take(axis))};i++) s0.Slice(i*{shape[axis] * inner}+{start * inner},{length * inner}).CopyTo(o.Slice(i*{length * inner},{length * inner}));");
                break;
            case "core.tensor.broadcast":
                var source = tensors["input"].Tensor.Dimensions;
                var expression = string.Join("+", source.Select((dim, i) => dim == 1 ? "0" :
                    $"((i/{Product(output.Skip(output.Count - source.Count + i + 1))})%{dim})*{Product(source.Skip(i + 1))}"));
                code.AppendLine($"for(int i=0;i<o.Length;i++) o[i]=s0[{expression}];"); break;
            case "core.tensor.batched-mat-vec":
                var matrix = tensors["matrix"].Tensor.Dimensions;
                var matvec = !half && tensors["matrix"].Tensor.ElementType == GraphElementType.Float16
                    ? "SharpInference.Instructions.Cpu.CpuInstructionNumerics.MatVec" : primitive + "MatVec";
                code.AppendLine($"for(int b=0;b<{matrix[0]};b++) {matvec}(s0.Slice(b*{matrix[1] * matrix[2]},{matrix[1] * matrix[2]}),s1.Slice(b*{matrix[2]},{matrix[2]}),o.Slice(b*{matrix[1]},{matrix[1]}),{matrix[1]},{matrix[2]});");
                break;
            case "core.tensor.reduce-last-sum":
            case "core.tensor.reduce-last-mean":
                var width = tensors["input"].Tensor.Dimensions[^1];
                code.AppendLine($"for(int i=0;i<o.Length;i++) o[i]={primitive}{(Name.EndsWith("mean", StringComparison.Ordinal) ? "ReduceMean" : "ReduceSum")}(s0.Slice(i*{width},{width}));"); break;
            case "core.tensor.head-outer":
                code.AppendLine($"for(int h=0;h<{output[0]};h++) for(int i=0;i<{output[1]};i++) for(int j=0;j<{output[2]};j++) o[(h*{output[1]}+i)*{output[2]}+j]={Result($"(float)s0[h*{output[1]}+i]*(float)s1[h*{output[2]}+j]")};"); break;
            default: throw new NotSupportedException(Name);
        }
        code.AppendLine("}");
        return new(code.ToString(), references:
            [typeof(SharpInference.Backends.Cpu.CpuPrimitiveOperatorBackend).Assembly, typeof(CpuInstructionNumerics).Assembly]);
    }
}
