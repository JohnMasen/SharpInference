using System.Globalization;
using System.Text;
using SharpInference.Graphs;
using SharpInference.Vm;
namespace SharpInference.Instructions.D3D12;
internal sealed class GpuTierZeroInstruction(Guid id,string name) : TierZeroInstruction(id,name,InstructionTarget.Direct3D12)
{
    protected override void ValidateExecutionConfiguration(InstructionExecutionConfiguration? configuration)
    {
        if (configuration is null) return;
        if (Name != "core.mat-vec" ||
            configuration != GpuMatVecExecution.Serial && configuration != GpuMatVecExecution.Cooperative)
            throw new InstructionAdaptationException(CollectionId, Name, "Unsupported GPU MatVec execution configuration.");
    }

    protected override InstructionRecording GenerateConfigured(InstructionParameter[] values,
        InstructionExecutionConfiguration? configuration)
    {
        if (configuration != GpuMatVecExecution.Cooperative) return Generate(values);
        var tensors = values.OfType<InstructionTensorParameter>().ToDictionary(parameter => parameter.Name);
        var matrix = tensors.GetValueOrDefault("weight") ?? tensors["matrix"];
        var input = tensors["input"];
        var output = tensors["output"];
        var rows = matrix.Tensor.Dimensions[0];
        var columns = matrix.Tensor.Dimensions[1];
        string Load(InstructionTensorParameter tensor, string index) =>
            $"load{(tensor.Tensor.ElementType == GraphElementType.Float16 ? 16 : 32)}({tensor.Expression},{tensor.OffsetExpression},{index})";
        var store = output.Tensor.ElementType == GraphElementType.Float16
            ? $"store16({output.Expression},{output.OffsetExpression},row,result);"
            : $"{output.Expression}.Store({output.OffsetExpression}+row*4u,asuint(result));";
        var baseline = Generate(values);
        var code = $$"""
            {
                uint row=gpu.groupId.x+gpu.groupId.y*gpu.groupCount.x+gpu.groupId.z*gpu.groupCount.x*gpu.groupCount.y;
                uint lane=gpu.groupIndex;
                if(row<{{rows}}u) {
                    precise float sum=0.0f, correction=0.0f;
                    for(uint j=lane;j<{{columns}}u;j+=64u) {
                        precise float product={{Load(matrix, $"row*{columns}u+j")}}*{{Load(input, "j")}};
                        add32(sum,correction,product);
                    }
                    matvecSum[lane]=sum;
                    matvecCorrection[lane]=correction;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(lane<step) {
                            precise float left=matvecSum[lane], residual=matvecCorrection[lane];
                            add32(left,residual,matvecSum[lane+step]);
                            add32(left,residual,matvecCorrection[lane+step]);
                            matvecSum[lane]=left;
                            matvecCorrection[lane]=residual;
                        }
                        GroupMemoryBarrierWithGroupSync();
                    }
                    if(lane==0u) {
                        precise float result=matvecSum[0]+matvecCorrection[0];
                        {{store}}
                    }
                }
            }
            """;
        return new(code, baseline.Helpers.Append(new("t0.matvec.shared",
            "groupshared float matvecSum[64]; groupshared float matvecCorrection[64];")),
            InstructionSynchronization.GroupMemoryBarrier);
    }

    protected override InstructionRecording Generate(InstructionParameter[] values)
    {
        var tensors=values.OfType<InstructionTensorParameter>().ToArray();
        var parameters=tensors.ToDictionary(p=>p.Name,p=>(p:new VmParameter(p.Name,p.Name=="output"?VmAccess.ReadWrite:VmAccess.ReadOnly,new VmTensor(p.Tensor.ElementType switch
        {
            GraphElementType.Float32 => VmElementType.Float32,
            GraphElementType.Float16 => VmElementType.Float16,
            GraphElementType.Int32 => VmElementType.Int32,
            _ => throw new InstructionAdaptationException(CollectionId, Name, "Unsupported tensor element type."),
        },p.Tensor.Dimensions)),Name:p.Expression));
        var offsets=tensors.ToDictionary(p=>p.Name,p=>p.OffsetExpression);
        var op=new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32),CollectionId,Name,tensors.Select(p=>new VmArgument(p.Name,p.Name)),values.OfType<InstructionAttributeParameter>().ToDictionary(p=>p.Name,p=>p.Value));
        return new InstructionRecording(Operator(op,parameters,offsets),[
            GpuInstructionHelpers.Load32,
            new InstructionHelper("t0.helper.1", "float load16(RWByteAddressBuffer b, uint o, uint i) { uint a=o+i*2; return f16tof32((b.Load(a & ~3u) >> ((a & 2u)*8u)) & 65535u); }"),
            GpuInstructionHelpers.Maximum32,
            new InstructionHelper("t0.helper.3", "void add32(inout float sum, inout float correction, float value) { precise float total=sum+value; if(isfinite(total)) { precise float residual=abs(sum)>=abs(value) ? (sum-total)+value : (value-total)+sum; correction+=residual; } else correction=0.0f; sum=total; }"),
            new InstructionHelper("t0.store16", "void store16(RWByteAddressBuffer b,uint o,uint i,float value) { uint a=o+i*2u; uint shift=(a&2u)*8u; uint mask=65535u<<shift; uint bits=(f32tof16(value)&65535u)<<shift; uint old=b.Load(a&~3u); uint observed; do { uint next=(old&~mask)|bits; b.InterlockedCompareExchange(a&~3u,old,next,observed); if(observed==old) break; old=observed; } while(true); }")]);
    }
    private static string Operator(VmOperator op,
        IReadOnlyDictionary<string, (VmParameter p, string Name)> parameters, IReadOnlyDictionary<string,string> offsets)
    {
        InvalidDataException Error(string reason) => new($"D3D12 VM operator '{op.Operation}@{op.Version}': {reason}");

        var args = op.Arguments.ToDictionary(a => a.Parameter, StringComparer.Ordinal);
        (VmTensor Tensor, string Buffer, string Offset) Get(string port)
        {
            if (!args.TryGetValue(port, out var a)) throw Error($"missing port '{port}'.");
            var (p, name) = parameters[a.Source];
            if (a.ByteOffset != 0)
                throw Error("operator views require a typed helper parameter; use a VmCall view instead.");
            return (p.Tensor, name, offsets[a.Source]);
        }
        string Load(string port, string index)
        {
            var (tensor, buffer, offset) = Get(port);
            if (tensor.ElementType is not (VmElementType.Float32 or VmElementType.Float16))
                throw Error($"'{port}' must be floating point.");
            return $"load{(tensor.ElementType == VmElementType.Float16 ? 16 : 32)}({buffer}, {offset}, {index})";
        }
        var output = Get("output");
        var outputArg = args["output"];
        if (parameters[outputArg.Source].p.Access != VmAccess.ReadWrite ||
            output.Tensor.ElementType is not (VmElementType.Float32 or VmElementType.Float16))
            throw Error("output must be writable FP32.");
        var n = output.Tensor.ElementCount;
        string Store(string expression) => output.Tensor.ElementType==VmElementType.Float16
            ? $"if(i<{n}u) store16({output.Buffer},{output.Offset},i,{expression});"
            : $"if (i < {n}u) {output.Buffer}.Store({output.Offset}+i*4u, asuint({expression}));";
        string[] ports;
        string body;
        var binary = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["core.add"] = "{0}+{1}", ["core.subtract"] = "{0}-{1}",
            ["core.multiply"] = "{0}*{1}", ["core.divide"] = "{0}/{1}",
            ["core.maximum"] = "maximum32({0},{1})",
        };
        var unary = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["core.copy"] = "{0}", ["core.exp"] = "exp({0})", ["core.tanh"] = "({0}==0.0f ? {0} : tanh({0}))",
            ["core.sigmoid"] = "(1.0f/(1.0f+exp(-({0}))))", ["core.rsqrt"] = "rsqrt({0})",
            ["core.square"] = "({0}*{0})", ["core.relu"] = "maximum32({0},0.0f)",
        };
        if (binary.TryGetValue(op.Operation, out var expression))
        {
            ports = ["left", "right"];
            foreach (var p in ports)
                if (!Get(p).Tensor.Dimensions.SequenceEqual(output.Tensor.Dimensions))
                    throw Error("elementwise shapes differ.");
            body = Store(string.Format(CultureInfo.InvariantCulture, expression, Load("left", "i"), Load("right", "i")));
        }
        else if (unary.TryGetValue(op.Operation, out expression))
        {
            ports = ["input"];
            if (!Get("input").Tensor.Dimensions.SequenceEqual(output.Tensor.Dimensions))
                throw Error("elementwise shapes differ.");
            body = Store(string.Format(CultureInfo.InvariantCulture, expression, Load("input", "i")));
        }
        else if (op.Operation is "core.reduce-sum" or "core.reduce-mean")
        {
            ports = ["input"];
            if (n != 1) throw Error("reduction output must contain one element.");
            var count = Get("input").Tensor.ElementCount;
            body = $"if(i == 0u) {{ precise float sum=0.0f, correction=0.0f; for(uint j=0;j<{count}u;j++) add32(sum,correction,{Load("input", "j")}); sum+=correction; " +
                Store(op.Operation == "core.reduce-mean" ? $"sum/{count}.0f" : "sum") + " }";
        }
        else if (op.Operation == "core.mat-vec")
        {
            var matrixPort = args.ContainsKey("weight") ? "weight" : "matrix";
            ports = [matrixPort, "input"];
            var matrix = Get(matrixPort).Tensor.Dimensions;
            if (matrix.Count != 2 || (ulong)matrix[0] != n ||
                Get("input").Tensor.ElementCount != (ulong)matrix[1])
                throw Error("invalid matrix/vector shapes.");
            body = $"if(i<{n}u) {{ precise float sum=0.0f, correction=0.0f; for(uint j=0;j<{matrix[1]}u;j++) add32(sum,correction," +
                $"{Load(matrixPort, $"i*{matrix[1]}u+j")}*{Load("input", "j")}); sum+=correction; {Store("sum")} }}";
        }
        else if (op.Operation == "core.gather-row")
        {
            ports = ["table", "index"];
            var table = Get("table").Tensor.Dimensions;
            var index = Get("index");
            if (table.Count != 2 || (ulong)table[1] != n ||
                index.Tensor.ElementType != VmElementType.Int32 || index.Tensor.ElementCount != 1)
                throw Error("invalid table/index shapes.");
            // Invalid dynamic indices produce a visible NaN, rather than an out-of-bounds load.
            body = $"int row=asint({index.Buffer}.Load({index.Offset})); " +
                Store($"(row>=0 && row<{table[0]} ? {Load("table", $"uint(row)*{n}u+i")} : asfloat(0x7fc00000u))");
        }
        else if (op.Operation.StartsWith("core.tensor.", StringComparison.Ordinal))
        {
            var contract = PortableTensorOperationContracts.Contracts.SingleOrDefault(c =>
                c.Operation.Name == op.Operation && c.Operation.Version == op.Version) ?? throw Error("unsupported tensor operation.");
            ports = contract.InputPorts.ToArray();
            var dims = output.Tensor.Dimensions;
            switch (op.Operation)
            {
                case "core.tensor.fill":
                    body = Store($"asfloat({BitConverter.SingleToUInt32Bits(float.Parse(op.Attributes["value"], CultureInfo.InvariantCulture))}u)");
                    break;
                case "core.tensor.reshape":
                case "core.tensor.cast-f16-f32":
                    body = Store(Load("input", "i"));
                    break;
                case "core.tensor.slice":
                case "core.tensor.broadcast":
                    var inputDims = Get("input").Tensor.Dimensions;
                    var coordinate = new StringBuilder("uint flat=i; ");
                    for (var a = dims.Count - 1; a >= 0; a--)
                        coordinate.Append($"uint c{a}=flat%{dims[a]}u; flat/={dims[a]}u; ");
                    coordinate.Append("uint s=0u; ");
                    var shift = dims.Count - inputDims.Count;
                    for (var a = 0; a < inputDims.Count; a++)
                    {
                        var c = op.Operation == "core.tensor.broadcast" && inputDims[a] == 1 ? "0u" : $"c{a + shift}";
                        if (op.Operation == "core.tensor.slice" && a == int.Parse(op.Attributes["axis"], CultureInfo.InvariantCulture))
                            c += $"+{op.Attributes["start"]}u";
                        coordinate.Append($"s=s*{inputDims[a]}u+({c}); ");
                    }
                    body = $"if(i<{n}u) {{ {coordinate} {Store(Load("input", "s"))} }}";
                    break;
                case "core.tensor.reduce-last-sum":
                case "core.tensor.reduce-last-mean":
                    var columns = Get("input").Tensor.Dimensions[^1];
                    body = $"if(i<{n}u) {{ precise float sum=0.0f, correction=0.0f; for(uint j=0;j<{columns}u;j++) add32(sum,correction,{Load("input", $"i*{columns}u+j")}); sum+=correction; " +
                        Store(op.Operation.EndsWith("mean", StringComparison.Ordinal) ? $"sum/{columns}.0f" : "sum") + " }";
                    break;
                case "core.tensor.batched-mat-vec":
                    var m = Get("matrix").Tensor.Dimensions;
                    body = $"if(i<{n}u) {{ precise float sum=0.0f, correction=0.0f; for(uint j=0;j<{m[2]}u;j++) add32(sum,correction," +
                        $"{Load("matrix", $"i*{m[2]}u+j")}*{Load("vector", $"(i/{m[1]}u)*{m[2]}u+j")}); sum+=correction; {Store("sum")} }}";
                    break;
                case "core.tensor.head-outer":
                    body = Store($"{Load("left", $"(i/({dims[1]}u*{dims[2]}u))*{dims[1]}u+(i/{dims[2]}u)%{dims[1]}u")}*" +
                        Load("right", $"(i/({dims[1]}u*{dims[2]}u))*{dims[2]}u+i%{dims[2]}u"));
                    break;
                default: throw Error("unsupported tensor operation.");
            }
        }
        else throw Error("unsupported operation.");
        if (!ports.Append("output").ToHashSet(StringComparer.Ordinal).SetEquals(args.Keys))
            throw Error("unexpected ports.");
        if (!op.Operation.StartsWith("core.tensor.", StringComparison.Ordinal) && op.Attributes.Count != 0)
            throw Error("unexpected attributes.");
        foreach (var p in ports)
            if (args[p].Source == outputArg.Source)
                throw Error("in-place operator writes are unsupported.");
        return body;
    }

}
