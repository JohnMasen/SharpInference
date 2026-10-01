using System.Buffers;
using System.Numerics.Tensors;

namespace SharpInference.Backends.Cpu;

internal static class CpuHalfMatrixOperators
{
    public static unsafe void MatVec(ReadOnlySpan<Half> matrix, ReadOnlySpan<float> input,
        Span<float> output)
    {
        var columns = input.Length;
        if (output.Length < 64 || (long)output.Length * columns < 65536)
        {
            for (var row = 0; row < output.Length; row++)
            {
                float sum = 0;
                var offset = row * columns;
                for (var column = 0; column < columns; column++)
                    sum += (float)matrix[offset + column] * input[column];
                output[row] = sum;
            }
            return;
        }

        fixed (Half* matrixPointer = matrix)
        fixed (float* inputPointer = input)
        fixed (float* outputPointer = output)
        {
            var pointers = new Pointers(matrixPointer, inputPointer, outputPointer);
            Parallel.For(0, output.Length,
                () => ArrayPool<float>.Shared.Rent(columns),
                (row, _, scratch) =>
                {
                    var converted = scratch.AsSpan(0, columns);
                    TensorPrimitives.ConvertToSingle(
                        new ReadOnlySpan<Half>(pointers.Matrix + (long)row * columns, columns),
                        converted);
                    pointers.Output[row] = TensorPrimitives.Dot(converted,
                        new ReadOnlySpan<float>(pointers.Input, columns));
                    return scratch;
                },
                scratch => ArrayPool<float>.Shared.Return(scratch));
        }
    }

    public static void GatherRow(ReadOnlySpan<Half> table, int rowIndex, Span<float> output)
    {
        if (rowIndex < 0 || rowIndex >= table.Length / output.Length)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));
        TensorPrimitives.ConvertToSingle(
            table.Slice(checked(rowIndex * output.Length), output.Length), output);
    }

    private readonly unsafe struct Pointers(Half* matrix, float* input, float* output)
    {
        public readonly Half* Matrix = matrix;
        public readonly float* Input = input;
        public readonly float* Output = output;
    }
}
