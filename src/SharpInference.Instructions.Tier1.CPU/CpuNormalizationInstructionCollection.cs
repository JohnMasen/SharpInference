using System.Globalization;
using System.Text;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;

namespace SharpInference.Instructions.Cpu;

/// <summary>Owns the complete portable T1 normalization set for CPU execution.</summary>
public sealed class CpuNormalizationInstructionCollection : IInstructionProvider
{
    private readonly IReadOnlyList<Instruction> instructions = Array.AsReadOnly(
        TierOneNormalizationInstructionSet.RequiredInstructions.Select(name => (Instruction)new CpuNormalizationInstruction(name)).ToArray());

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [TierOneNormalizationInstructionSet.Describe(ExecutionTarget.Cpu)];
    public IReadOnlyList<Instruction> EnumerateInstructions() => instructions;
    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == TierOneNormalizationInstructionSet.Id
            ? instructions.Where(i => i.Name == instructionName).ToArray() : [];
}

internal sealed class CpuNormalizationInstruction(string name) : Instruction
{
    public override Guid CollectionId => TierOneNormalizationInstructionSet.Id;
    public override string Name => name;
    public override string Comment => TierOneNormalizationInstructionSet.Comment(Name);
    public override IReadOnlyList<InstructionAttributeDescription> AttributeDescriptions =>
        TierOneNormalizationInstructionSet.Attributes(Name);
    public override ExecutionTarget Target => ExecutionTarget.Cpu;
    public override IReadOnlyList<InstructionSignature> Signatures { get; } =
        [TierOneNormalizationInstructionSet.Signatures[name]];

    protected override void Validate(InstructionParameter[] parameters) =>
        TierOneNormalizationInstructionSet.Validate(Name, parameters);

    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>().ToDictionary(p => p.Name);
        var source = new StringBuilder("{\n");
        foreach (var port in Signatures[0].Ports)
        {
            var type = port.ElementType == ElementType.Int32 ? "int" : "float";
            source.AppendLine($"System.{(port.Access == ResourceAccess.Read ? "ReadOnlySpan" : "Span")}<{type}> {port.Name} = System.Runtime.InteropServices.MemoryMarshal.Cast<byte,{type}>({tensors[port.Name].Expression});");
        }
        string Attribute(string attribute) =>
            TierOneNormalizationInstructionSet.PositiveFloat(parameters, attribute)
                .ToString("R", CultureInfo.InvariantCulture) + "f";
        var call = Name switch
        {
            "transformer.rms-norm" => $"RmsNorm(input, weight, output, {Attribute("epsilon")})",
            "transformer.residual-rms-norm" =>
                $"ResidualRmsNorm(hidden, residual, weight, updated, output, {Attribute("epsilon")})",
            "transformer.swiglu" => "SwiGlu(gate_up, output)",
            "transformer.scaled-add" => $"ScaledAdd(input, update, output, {Attribute("scale")})",
            "transformer.batch-rms-norm" =>
                $"BatchRmsNorm(input, weight, control, output, {tensors["output"].Tensor.Dimensions[0]}, {tensors["output"].Tensor.Dimensions[1]}, {Attribute("epsilon")})",
            "transformer.batch-residual-rms-norm" =>
                $"BatchResidualRmsNorm(hidden, residual, weight, control, updated, output, {tensors["output"].Tensor.Dimensions[0]}, {tensors["output"].Tensor.Dimensions[1]}, {Attribute("epsilon")})",
            "transformer.batch-swiglu" =>
                $"BatchSwiGlu(gate_up, control, output, {tensors["output"].Tensor.Dimensions[0]}, {tensors["output"].Tensor.Dimensions[1]})",
            _ => throw new NotSupportedException(Name),
        };
        source.AppendLine($"SharpInference.Instructions.Cpu.CpuNormalizationNumerics.{call};");
        source.AppendLine("}");
        return new(source.ToString(), references: [typeof(CpuNormalizationNumerics).Assembly]);
    }
}
