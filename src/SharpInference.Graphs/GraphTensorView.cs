namespace SharpInference.Graphs;

public sealed record GraphTensorView(ulong ByteOffset, TensorDescriptor Tensor)
{
    public ulong ByteLength => ByteLengthOf(Tensor);

    internal static ulong ByteLengthOf(TensorDescriptor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        var size = tensor.ElementType switch
        {
            GraphElementType.Byte => 1UL,
            GraphElementType.Float16 => 2UL,
            GraphElementType.Int32 or GraphElementType.UInt32 or GraphElementType.Float32 => 4UL,
            _ => throw new InvalidDataException("Unsupported tensor element type."),
        };
        try { return tensor.Dimensions.Aggregate(size, (length, dimension) => checked(length * (ulong)dimension)); }
        catch (OverflowException error) { throw new InvalidDataException("Tensor byte length exceeds the supported range.", error); }
    }

    internal void Validate(TensorDescriptor parent)
    {
        ArgumentNullException.ThrowIfNull(Tensor);
        var size = ByteLengthOf(new TensorDescriptor(parent.ElementType, []));
        var parentLength = ByteLengthOf(parent);
        if (parent.Layout != "dense" || Tensor.Layout != "dense" || Tensor.ElementType != parent.ElementType ||
            ByteOffset % size != 0 || ByteOffset > parentLength || ByteLength > parentLength - ByteOffset)
            throw new InvalidDataException("A contiguous tensor view must have a matching type, aligned offset and in-bounds dense storage.");
    }
}
