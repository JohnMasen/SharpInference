namespace SharpInference.Instructions.Cpu;

/// <summary>Scalar reference implementations of portable tensor instructions.</summary>
public static class CpuTensorNumerics
{
    public static void ArgMax(ReadOnlySpan<float> input, Span<int> output)
    {
        ArgumentOutOfRangeException.ThrowIfZero(input.Length);
        if (output.Length != 1) throw new ArgumentException("Expected one output index.", nameof(output));
        var best = float.NegativeInfinity;
        var index = -1;
        for (var i = 0; i < input.Length; i++)
        {
            if (input[i] > best || (index < 0 && input[i] == best))
            {
                best = input[i];
                index = i;
            }
        }
        output[0] = Math.Max(index, 0);
    }

    public static void Gather(ReadOnlySpan<Half> table, ReadOnlySpan<int> indices, Span<float> output, int width)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        if (table.Length % width != 0 || output.Length != checked(indices.Length * width))
            throw new ArgumentException("Invalid gather dimensions.");
        foreach (var index in indices)
            if ((uint)index >= (uint)(table.Length / width))
                throw new ArgumentOutOfRangeException(nameof(indices));
        for (var row = 0; row < indices.Length; row++)
            for (var column = 0; column < width; column++)
                output[row * width + column] = (float)table[indices[row] * width + column];
    }

    public static void CopyRows(ReadOnlySpan<float> input, Span<float> output, int width, int rows, int outputStart)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(outputStart);
        input[..checked(rows * width)].CopyTo(output.Slice(checked(outputStart * width), checked(rows * width)));
    }

    public static void SharedMatVec(ReadOnlySpan<Half> weight, ReadOnlySpan<float> input,
        ReadOnlySpan<int> control, Span<float> output, int batch, int outputWidth, int inputWidth)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batch);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputWidth);
        if (weight.Length != checked(outputWidth * inputWidth) ||
            input.Length != checked(batch * inputWidth) ||
            output.Length != checked(batch * outputWidth) || control.Length != 2)
            throw new ArgumentException("Invalid shared matrix dimensions.");
        var validRows = Math.Clamp(control[1], 0, batch);
        for (var row = 0; row < validRows; row++)
            for (var column = 0; column < outputWidth; column++)
            {
                var sum = 0f;
                for (var k = 0; k < inputWidth; k++)
                    sum += (float)weight[column * inputWidth + k] * input[row * inputWidth + k];
                output[row * outputWidth + column] = sum;
            }
    }
}
