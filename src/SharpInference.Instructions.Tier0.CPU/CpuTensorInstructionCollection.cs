using System.Text;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;

namespace SharpInference.Instructions.Cpu;

/// <summary>Owns the complete portable T0 tensor set for CPU execution.</summary>
public sealed class CpuTensorInstructionCollection : IInstructionProvider
{
    private readonly IReadOnlyList<Instruction> instructions = Array.AsReadOnly(
        TierZeroTensorInstructionSet.RequiredInstructions.Select(name => (Instruction)new CpuTensorInstruction(name)).ToArray());

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [TierZeroTensorInstructionSet.Describe(ExecutionTarget.Cpu)];
    public IReadOnlyList<Instruction> EnumerateInstructions() => instructions;
    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == TierZeroTensorInstructionSet.Id
            ? instructions.Where(i => i.Name == instructionName).ToArray() : [];
}

internal sealed class CpuTensorInstruction(string name) : Instruction
{
    public override Guid CollectionId => TierZeroTensorInstructionSet.Id;
    public override string Name => name;
    public override string Comment => TierZeroTensorInstructionSet.Comment(Name);
    public override IReadOnlyList<InstructionAttributeDescription> AttributeDescriptions =>
        TierZeroTensorInstructionSet.Attributes(Name);
    public override ExecutionTarget Target => ExecutionTarget.Cpu;
    public override IReadOnlyList<InstructionSignature> Signatures
    {
        get;
    } =
        [TierZeroTensorInstructionSet.Signatures[name]];
    public override IReadOnlyList<InstructionIndexBound> IndexBounds =>
        Name == "transformer.embedding-gather-batch" ? [new("indices", "table", 0)] : [];

    protected override void Validate(InstructionParameter[] parameters) =>
        TierZeroTensorInstructionSet.Validate(Name, parameters);

    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>().ToDictionary(p => p.Name);
        var source = new StringBuilder("{\n");
        foreach (var port in Signatures[0].Ports)
        {
            var type = port.ElementType switch
            {
                ElementType.Float16 => "System.Half",
                ElementType.Int32 => "int",
                _ => "float",
            };
            source.AppendLine($"System.{(port.Access == ResourceAccess.Read ? "ReadOnlySpan" : "Span")}<{type}> {port.Name} = System.Runtime.InteropServices.MemoryMarshal.Cast<byte,{type}>({tensors[port.Name].Expression});");
        }
        var call = Name switch
        {
            "transformer.argmax" => "ArgMax(input, output)",
            "transformer.embedding-gather-batch" =>
                $"Gather(table, indices, output, {tensors["table"].Tensor.Dimensions[1]})",
            "transformer.copy-embedding-rows" =>
                $"CopyRows(input, output, {tensors["input"].Tensor.Dimensions[^1]}, {TierZeroTensorInstructionSet.Integer(parameters, "rows", true)}, {TierZeroTensorInstructionSet.Integer(parameters, "output_start", false)})",
            "transformer.batch-shared-matvec" =>
                $"SharedMatVec(weight, input, control, output, {tensors["input"].Tensor.Dimensions[0]}, {tensors["weight"].Tensor.Dimensions[0]}, {tensors["weight"].Tensor.Dimensions[1]})",
            _ => throw new NotSupportedException(Name),
        };
        source.AppendLine($"SharpInference.Instructions.Cpu.CpuTensorNumerics.{call};");
        source.AppendLine("}");
        return new(source.ToString(), references: [typeof(CpuTensorNumerics).Assembly]);
    }
}
