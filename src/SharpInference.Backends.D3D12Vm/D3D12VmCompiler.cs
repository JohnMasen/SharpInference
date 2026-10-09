using System.Globalization;
using System.Text;
using SharpInference.Instructions;
using SharpInference.Vm;
using Vortice.Dxc;

namespace SharpInference.Backends.D3D12Vm;

public sealed record D3D12VmCompilerOptions(string ShaderModel = "cs_6_0", bool EnableDebug = false);

/// <summary>
/// One source and DXIL module per explicit kernel. No graph rewriting is performed.
/// Instruction implementations are provided by the constructor, never discovered here.
/// Kernels support up to 20 raw-UAV parameters with the current root-descriptor ABI.
/// Typed byte-offset views belong on VmCall/VmDispatch arguments; operator arguments use offset zero.
/// Same-thread helper pipelines are legal; cross-thread dependencies require separate dispatches and barriers.
/// </summary>
public sealed class D3D12VmCompiler
{
    private readonly InstructionRegistry registry;
    public D3D12VmCompiler(IEnumerable<IInstructionCollectionProvider> providers) => registry=new(providers);
    public IInstructionCollectionProvider InstructionCollections => registry;

    public D3D12VmArtifact Compile(VmProgram program, D3D12VmCompilerOptions? options = null)
    {
        options ??= new();
        ValidateOptions(options);
        var contracts = VmInstructionContracts.Bind(program, registry);
        var sources = GenerateSources(program);
        var modules = new List<D3D12VmKernel>();
        foreach (var (id, source) in sources)
        {
            using var result = DxcCompiler.Compile(DxcShaderStage.Compute, source, "main",
                new DxcCompilerOptions { ShaderModel = DxcShaderModel.Model6_0, EnableDebugInfo = options.EnableDebug });
            modules.Add(new(id, source, result.GetObjectBytecodeArray()));
        }
        return new(program, options, modules, contracts);
    }

    /// <summary>Validates device operations and orchestration without loading or invoking DXC.</summary>
    public IReadOnlyDictionary<string, string> GenerateSources(VmProgram program)
    {
        VmProgramValidator.Validate(program);
        if (program.Target != VmTarget.Direct3D12)
            throw new NotSupportedException("The D3D12 VM requires a Direct3D12 program.");
        program = VmInstructionContracts.Bind(program, registry);
        foreach (var slot in program.Slots)
            if (slot.Tensor.ByteLength > int.MaxValue)
                throw new NotSupportedException($"Slot '{slot.Id}' exceeds the byte[] control-plane limit.");
        var definitions = program.Definitions.ToDictionary(d => d.Id, StringComparer.Ordinal);
        var names = program.Definitions.Select((d, i) => (d.Id, Name: $"d{i}"))
            .ToDictionary(p => p.Id, p => p.Name, StringComparer.Ordinal);
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        var helpers = new Dictionary<string, InstructionHelper>(StringComparer.Ordinal);
        foreach (var definition in program.Definitions.Where(d => d.Kind != VmDefinitionKind.Orchestration))
        {
            var parameters = definition.Parameters.Select((p, i) => (p, i))
                .ToDictionary(v => v.p.Name, v => (v.p, Name: $"p{v.i}"), StringComparer.Ordinal);
            foreach (var op in definition.Nodes.Select(n => n.Instruction).OfType<VmOperator>())
                Operator(op, parameters, helpers);
        }
        foreach (var kernel in program.Definitions.Where(d => d.Kind == VmDefinitionKind.Kernel))
        {
            if (kernel.Parameters.Count > 20)
                throw new NotSupportedException($"Kernel '{kernel.Id}' exceeds 20 root-buffer parameters.");
            helpers.Clear();
            var source = new StringBuilder();
            var emitted = new HashSet<string>(StringComparer.Ordinal);
            void Emit(VmDefinition definition)
            {
                if (!emitted.Add(definition.Id)) return;
                foreach (var call in definition.Nodes.Select(n => n.Instruction).OfType<VmCall>())
                    Emit(definitions[call.Definition]);
                var parameters = definition.Parameters.Select((p, i) => (p, i))
                    .ToDictionary(v => v.p.Name, v => (v.p, Name: $"p{v.i}"), StringComparer.Ordinal);
                source.Append($"void {names[definition.Id]}(uint i, GpuExecutionContext gpu");
                foreach (var (_, name) in parameters.Values)
                    source.Append($", RWByteAddressBuffer {name}, uint {name}o");
                source.AppendLine(") {");
                foreach (var node in definition.Nodes)
                {
                    source.AppendLine("{");
                    if (node.Instruction is VmCall call)
                    {
                        source.Append($"{names[call.Definition]}(i, gpu");
                        foreach (var p in definitions[call.Definition].Parameters)
                        {
                            var arg = call.Arguments.Single(a => a.Parameter == p.Name);
                            var (_, name) = parameters[arg.Source];
                            source.Append($", {name}, {name}o + {arg.ByteOffset}u");
                        }
                        source.AppendLine(");");
                    }
                    else if (node.Instruction is VmOperator op)
                        source.AppendLine(Operator(op, parameters, helpers));
                    else
                        throw new NotSupportedException($"Device definition '{definition.Id}' contains a host instruction.");
                    source.AppendLine("}");
                }
                source.AppendLine("}");
            }
            ValidateDeviceHazards(kernel, definitions);
            Emit(kernel);
            source.Insert(0,"struct GpuExecutionContext { uint3 groupId; uint3 groupCount; uint groupIndex; };\n" +
                string.Join(Environment.NewLine,helpers.Values.Select(helper=>helper.Source))+Environment.NewLine);
            for (var i = 0; i < kernel.Parameters.Count; i++)
                source.AppendLine($"RWByteAddressBuffer b{i} : register(u{i});");
            source.AppendLine("cbuffer Bindings : register(b0) { uint width; uint height; " +
                string.Join(" ", Enumerable.Range(0, kernel.Parameters.Count).Select(i => $"uint offset{i};")) + " };");
            var t = kernel.Threads!;
            source.AppendLine($"[numthreads({t.X},{t.Y},{t.Z})]");
            source.AppendLine("void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {");
            source.AppendLine($"GpuExecutionContext gpu; gpu.groupId=gid; gpu.groupCount=uint3(width/{t.X}u,height/{t.Y}u,1u); gpu.groupIndex=gi;");
            source.Append($"{names[kernel.Id]}(tid.x + tid.y * width + tid.z * width * height, gpu");
            for (var i = 0; i < kernel.Parameters.Count; i++)
                source.Append($", b{i}, offset{i}");
            source.AppendLine("); }");
            result.Add(kernel.Id, source.ToString());
        }
        D3D12VmSchedule.Create(program);
        return result;
    }

    internal static void ValidateOptions(D3D12VmCompilerOptions options)
    {
        if (options.ShaderModel != "cs_6_0")
            throw new NotSupportedException("This artifact ABI supports only cs_6_0.");
    }

    private string Operator(VmOperator op, IReadOnlyDictionary<string,(VmParameter p,string Name)> parameters,
        Dictionary<string, InstructionHelper> helpers)
    {
        var instruction=registry.Resolve(op.InstructionCollectionId,op.InstructionName,InstructionTarget.Direct3D12);
        var args=VmInstructionParameters.Create(op,parameters.Values.Select(value=>value.p).ToArray(),p=>(parameters[p.Name].Name,parameters[p.Name].Name+"o"));
        var recorder=new SourceInstructionRecorder(InstructionTarget.Direct3D12);
        instruction.Invoke(recorder,args,op.Precision,op.ExecutionConfiguration);
        foreach(var helper in recorder.Helpers)
        {
            if(helpers.TryGetValue(helper.Name,out var existing) && existing.Source!=helper.Source)
                throw new InvalidDataException($"Conflicting helper '{helper.Name}'.");
            helpers.TryAdd(helper.Name,helper);
        }
        return recorder.GetSource();
    }

    private void ValidateDeviceHazards(VmDefinition kernel, IReadOnlyDictionary<string, VmDefinition> definitions)
    {
        var written = new List<(string Resource, ulong Offset, VmTensor Tensor)>();
        var reads = new List<(string Resource, ulong Offset, VmTensor Tensor, bool NonPointwise)>();
        void Walk(VmDefinition d, Dictionary<string, (string Resource, ulong Offset)> bindings)
        {
            foreach (var node in d.Nodes)
                if (node.Instruction is VmCall call)
                    Walk(definitions[call.Definition], call.Arguments.ToDictionary(a => a.Parameter, a =>
                        (bindings[a.Source].Resource, checked(bindings[a.Source].Offset + a.ByteOffset)), StringComparer.Ordinal));
                else if (node.Instruction is VmOperator op)
                {
                    var instruction = registry.Resolve(op.InstructionCollectionId,op.InstructionName,InstructionTarget.Direct3D12);
                    var nonPointwise = instruction.RequiresDispatchIsolation;
                    var signature = instruction.Adapt(VmInstructionParameters.Create(op, d.Parameters, parameter => (parameter.Name, "0")),
                        op.Precision, op.ExecutionConfiguration);
                    var parameters = d.Parameters.ToDictionary(p => p.Name, StringComparer.Ordinal);
                    var priorReadCount = reads.Count;
                    var priorWriteCount = written.Count;
                    foreach (var writePort in signature.Ports.Where(port => port.Access != SharpInference.Graphs.GraphResourceAccess.Read))
                    {
                    var output = op.Arguments.Single(a => a.Parameter == writePort.Name);
                    var destination = bindings[output.Source];
                    var outputTensor = parameters[output.Source].Tensor;
                    bool Overlap((string Resource, ulong Offset) source, VmTensor tensor,
                        (string Resource, ulong Offset) other, VmTensor otherTensor) =>
                        source.Resource == other.Resource && source.Offset < other.Offset + otherTensor.ByteLength &&
                        other.Offset < source.Offset + tensor.ByteLength;
                    foreach (var a in op.Arguments.Where(a => signature.Ports.Any(port =>
                        port.Name == a.Parameter && port.Access != SharpInference.Graphs.GraphResourceAccess.Write)))
                    {
                        var source = bindings[a.Source];
                        var tensor = parameters[a.Source].Tensor;
                        if (a.Parameter != writePort.Name && Overlap(source, tensor, destination, outputTensor))
                            throw new NotSupportedException($"Kernel '{kernel.Id}' aliases operator inputs and output through helper bindings.");
                        foreach (var previous in written.Take(priorWriteCount).Where(w => Overlap(source, tensor, (w.Resource, w.Offset), w.Tensor)))
                            if (nonPointwise || source.Offset != previous.Offset ||
                                tensor.ElementType != previous.Tensor.ElementType)
                                throw new NotSupportedException($"Kernel '{kernel.Id}' has a cross-thread producer/consumer. Use separate kernels and an explicit VmBarrier.");
                        reads.Add((source.Resource, source.Offset, tensor, nonPointwise));
                    }
                    foreach (var previous in written.Where(w => Overlap(destination, outputTensor, (w.Resource, w.Offset), w.Tensor)))
                        if (destination.Offset != previous.Offset)
                            throw new NotSupportedException($"Kernel '{kernel.Id}' has overlapping shifted writes.");
                    foreach (var previous in reads.Take(priorReadCount).Where(r => Overlap(destination, outputTensor, (r.Resource, r.Offset), r.Tensor)))
                        if (previous.NonPointwise || destination.Offset != previous.Offset ||
                            outputTensor.ElementType != previous.Tensor.ElementType)
                            throw new NotSupportedException($"Kernel '{kernel.Id}' has a cross-thread read/write hazard.");
                    written.Add((destination.Resource, destination.Offset, outputTensor));
                    }
                }
        }
        Walk(kernel, kernel.Parameters.ToDictionary(p => p.Name, p => (p.Name, 0UL), StringComparer.Ordinal));
    }
}
