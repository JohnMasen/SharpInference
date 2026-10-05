using System.Text;
using System.Text.Json;
using SharpInference.Instructions;
using SharpInference.Graphs;
using SharpInference.Vm;

namespace SharpInference.Backends.CpuVm;

internal sealed class CpuVmSourceGenerator
{
    private readonly VmProgram program;
    private readonly VmProgram contracts;
    private readonly StringBuilder code = new();
    private readonly Dictionary<string, string> methods;
    private readonly InstructionRegistry registry;
    private readonly SourceInstructionRecorder recorder = new(InstructionTarget.Cpu);
    internal Dictionary<string, List<CpuVmAccess>> Accesses { get; } = new(StringComparer.Ordinal);
    internal Dictionary<string, CpuVmCallFrame> Frames { get; } = new(StringComparer.Ordinal);
    internal IReadOnlyList<System.Reflection.Assembly> References => recorder.References;

    internal CpuVmSourceGenerator(VmProgram program, InstructionRegistry registry, VmProgram? contracts = null)
    {
        VmProgramValidator.Validate(program);
        if (program.Target != VmTarget.Cpu) throw new NotSupportedException("CPU VM requires a CPU program.");
        this.program = program;
        this.contracts = contracts ?? program;
        this.registry = registry;
        methods = program.Definitions.Select((definition, index) => (definition.Id, Name: $"D{index}"))
            .ToDictionary(item => item.Id, item => item.Name);
        foreach (var tensor in program.Slots.Select(slot => slot.Tensor)
                     .Concat(program.Definitions.SelectMany(definition => definition.Parameters.Select(parameter => parameter.Tensor))))
            if (tensor.ByteLength > int.MaxValue) throw new NotSupportedException("CPU spans require Int32-sized tensors.");
    }

    internal string Generate()
    {
        code.AppendLine("using System;\nusing System.Runtime.InteropServices;\nusing SharpInference.Backends.CpuVm;\nnamespace SharpInference.Generated;\npublic sealed class CpuProgram : ICpuVmCode\n{");
        code.AppendLine($"public string ProgramHash => \"{CpuVmCompiledArtifact.ComputeProgramHash(program)}\";");
        foreach (var definition in program.Definitions)
        {
            code.AppendLine($"private static void {methods[definition.Id]}(CpuVmContext context, CpuVmCallFrame frame) {{");
            var callIndex = 0;
            foreach (var node in definition.Nodes)
            {
                code.AppendLine($"#line 1 {Literal($"vm/{definition.Id}/{node.Id}")}");
                if (node.Instruction is VmCall call)
                {
                    code.AppendLine($"{methods[call.Definition]}(context, frame.GetCall({callIndex++}));");
                }
                else if (node.Instruction is VmOperator operation) Emit(definition, node, operation);
                else throw new NotSupportedException($"Unsupported CPU instruction at '{definition.Id}/{node.Id}'.");
                code.AppendLine("#line default");
            }
            code.AppendLine("}");
        }
        code.AppendLine("public void Invoke(string entry, CpuVmContext context) { switch(entry) {");
        foreach (var entry in program.Entries)
        {
            var definition = program.Definitions.Single(item => item.Id == entry.Definition);
            code.AppendLine($"case {Literal(entry.Name)}: {methods[definition.Id]}(context, context.GetFrame({Literal(entry.Name)})); return;");
            var accesses = new List<CpuVmAccess>();
            Frames.Add(entry.Name, Expand(definition, entry.Arguments.ToDictionary(a => a.Parameter,
                a => (Slot: program.Slots.ToList().FindIndex(s => s.Id == a.Source), Offset: checked((int)a.ByteOffset))), accesses));
            Accesses.Add(entry.Name, accesses);
        }
        code.AppendLine("default: throw new ArgumentException(\"Unknown CPU VM entry.\", nameof(entry)); } }");
        foreach (var helper in recorder.Helpers) code.AppendLine(helper.Source);
        code.AppendLine("}");
        return code.ToString();
    }

    private static string Literal(string value) => JsonSerializer.Serialize(value, CpuVmJsonContext.Default.String);

    private static string View(VmDefinition definition, VmArgument argument, VmTensor tensor, bool write) =>
        $"frame.{(write ? "Write" : "Read")}(context, {definition.Parameters.ToList().FindIndex(p => p.Name == argument.Source)}, {argument.ByteOffset}, {tensor.ByteLength})";

    private void Emit(VmDefinition definition, VmNode node, VmOperator operation)
    {
        var instruction=registry.Resolve(operation.InstructionCollectionId,operation.InstructionName,InstructionTarget.Cpu);
        var contract = (VmOperator)contracts.Definitions.Single(value => value.Id == definition.Id)
            .Nodes.Single(value => value.Id == node.Id).Instruction;
        var parameters=VmInstructionParameters.Create(operation,definition.Parameters,p =>
        {
            var index=definition.Parameters.ToList().FindIndex(value=>value.Name==p.Name);
            var method=p.Access==VmAccess.ReadOnly?"Read":"Write";
            return ($"frame.{method}(context, {index}, 0, {p.Tensor.ByteLength})","0");
        });
        parameters = parameters.Select(parameter => parameter is InstructionTensorParameter tensor
            ? tensor with
            {
                Expression = View(definition, new VmArgument(tensor.Name, tensor.Source), definition.Parameters.Single(p => p.Name == tensor.Source).Tensor,
                    contract.ParameterAccesses[tensor.Name] != GraphResourceAccess.Read),
            } : parameter).ToArray();
        var start = recorder.Recordings.Count;
        instruction.Invoke(recorder,parameters,operation.Precision,operation.ExecutionConfiguration);
        foreach (var recording in recorder.Recordings.Skip(start))
            code.AppendLine(recording.Source);
    }

    private CpuVmCallFrame Expand(VmDefinition definition, Dictionary<string, (int Slot, int Offset)> bindings, List<CpuVmAccess> accesses)
    {
        var calls = new List<CpuVmCallFrame>();
        foreach (var node in definition.Nodes)
        {
            if (node.Instruction is VmCall call)
            {
                calls.Add(Expand(program.Definitions.Single(d => d.Id == call.Definition),
                    call.Arguments.ToDictionary(a => a.Parameter,
                        a => (bindings[a.Source].Slot, checked(bindings[a.Source].Offset + (int)a.ByteOffset))), accesses));
            }
            else if (node.Instruction is VmOperator op)
            {
                var contract = (VmOperator)contracts.Definitions.Single(value => value.Id == definition.Id)
                    .Nodes.Single(value => value.Id == node.Id).Instruction;
                var views = op.Arguments.Select(a =>
                {
                    var parameter = definition.Parameters.Single(p => p.Name == a.Source);
                    var tensor = parameter.Tensor;
                    return (Name: a.Parameter, Access: contract.ParameterAccesses[a.Parameter],
                        Slot: bindings[a.Source].Slot,
                        Offset: checked(bindings[a.Source].Offset + (int)a.ByteOffset),
                        Length: checked((int)tensor.ByteLength));
                }).ToArray();
                foreach (var output in views.Where(v => v.Access != GraphResourceAccess.Read))
                    foreach (var input in views.Where(v => v.Access != GraphResourceAccess.Write && v.Name != output.Name))
                        if (input.Slot == output.Slot && input.Offset < output.Offset + output.Length && output.Offset < input.Offset + input.Length)
                            throw new InvalidDataException($"Overlapping input/output at '{definition.Id}/{node.Id}'.");
                accesses.AddRange(views.Where(v => v.Access != GraphResourceAccess.Write)
                    .Select(v => new CpuVmAccess(v.Slot, v.Offset, v.Length, false)));
                accesses.AddRange(views.Where(v => v.Access != GraphResourceAccess.Read)
                    .Select(v => new CpuVmAccess(v.Slot, v.Offset, v.Length, true)));
            }

        }
        return new CpuVmCallFrame(definition.Parameters.Select(p => new CpuVmParameterBinding(
            bindings[p.Name].Slot, bindings[p.Name].Offset, checked((int)p.Tensor.ByteLength), p.Access)).ToArray(), calls.ToArray());
    }

    internal void PrepareBindings()
    {
        foreach (var entry in program.Entries)
        {
            var definition = program.Definitions.Single(item => item.Id == entry.Definition);
            var accesses = new List<CpuVmAccess>();
            Frames.Add(entry.Name, Expand(definition, entry.Arguments.ToDictionary(a => a.Parameter,
                a => (Slot: program.Slots.ToList().FindIndex(s => s.Id == a.Source),
                    Offset: checked((int)a.ByteOffset))), accesses));
            Accesses.Add(entry.Name, accesses);
        }
    }
}
