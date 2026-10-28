using SharpInference.Graphs;
using SharpInference.Graphs.Execution;

namespace SharpInference.Instructions;

public static class TierOneAffineOperations
{
    public static readonly OperationId Affine = new("core.affine");
    public static readonly Guid CollectionFloat32 = new("44928a8e-255b-4bdd-8097-597ea5b8f5aa");
    public static IReadOnlyList<string> Decomposition
    {
        get;
    } =
        Array.AsReadOnly(new[] { "core.matrix-multiply", "core.bias-add" });
}

public abstract class TierOneAffineInstruction(ExecutionTarget target) : Instruction
{
    public override Guid CollectionId => TierOneAffineOperations.CollectionFloat32;
    public override string Name => TierOneAffineOperations.Affine.Name;
    public override ExecutionTarget Target => target;
    public override string Comment => "Matrix multiplication followed by column bias addition.";
    public override IReadOnlyList<InstructionAttributeDescription> AttributeDescriptions
    {
        get;
    } =
        [new("transpose_left", "Transpose the left matrix.", "Boolean"),
         new("transpose_right", "Transpose the right matrix.", "Boolean")];
    public override IReadOnlyList<InstructionSignature> Signatures
    {
        get;
    } =
        new[] { ElementType.Float32, ElementType.Float16 }.SelectMany(storage =>
            new[] { ElementType.Float32, ElementType.Float16 }.Select(biasStorage => new InstructionSignature(
            [new("left", ElementType.Float32, ResourceAccess.Read, "Left matrix.", ElementType.Float32, "Rank 2"),
             new("right", storage, ResourceAccess.Read, "Right matrix.", ElementType.Float32, "Rank 2"),
             new("bias", biasStorage, ResourceAccess.Read, "Column bias.", ElementType.Float32, "[N]"),
             new("output", ElementType.Float32, ResourceAccess.Write, "Affine result.", ElementType.Float32, "[M,N]")],
            ["transpose_left", "transpose_right"], new(ElementType.Float32, ElementType.Float32)))).ToArray();

    protected override void Validate(InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>().ToDictionary(parameter => parameter.Name);
        var attributes = parameters.OfType<InstructionAttributeParameter>().ToDictionary(parameter => parameter.Name);
        var left = tensors["left"].Tensor;
        var right = tensors["right"].Tensor;
        var output = tensors["output"].Tensor;
        if (tensors.Values.Any(tensor => tensor.ByteOffset != 0 || tensor.Tensor.Layout != "dense") ||
            tensors.Values.Any(tensor => tensor.Name != "output" && tensor.Source == tensors["output"].Source) ||
            left.Dimensions.Count != 2 || right.Dimensions.Count != 2 || output.Dimensions.Count != 2 ||
            !bool.TryParse(attributes["transpose_left"].Value, out var transposeLeft) ||
            !bool.TryParse(attributes["transpose_right"].Value, out var transposeRight))
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Expected dense nonaliasing rank-two matrices and Boolean transpose attributes.");
        }
        var rows = left.Dimensions[transposeLeft ? 1 : 0];
        var inner = left.Dimensions[transposeLeft ? 0 : 1];
        var columns = right.Dimensions[transposeRight ? 0 : 1];
        if (inner != right.Dimensions[transposeRight ? 1 : 0] ||
            !output.Dimensions.SequenceEqual(new[] { rows, columns }) ||
            !tensors["bias"].Tensor.Dimensions.SequenceEqual(new[] { columns }))
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Incompatible matrix or bias shapes.");
        }
    }
}
