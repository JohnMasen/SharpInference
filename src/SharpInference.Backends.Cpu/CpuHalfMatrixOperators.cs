using System.Buffers;
using System.Numerics.Tensors;

namespace SharpInference.Backends.Cpu;

internal static class CpuHalfMatrixOperators
{
    private const int ParallelRowThreshold = 64;
    private const int ParallelElementThreshold = 65536;

    public static unsafe void MatVec(ReadOnlySpan<Half> matrix, ReadOnlySpan<float> input,
        Span<float> output)
    {
        var columns = input.Length;
        if (output.Length < ParallelRowThreshold ||
            (long)output.Length * columns < ParallelElementThreshold)
        {
            Span<float> converted = stackalloc float[Math.Min(columns, 1024)];
            for (var row = 0; row < output.Length; row++)
                output[row] = Dot(matrix.Slice(row * columns, columns), input, converted);
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

    public static unsafe void MatVec(ReadOnlySpan<Half> matrix, ReadOnlySpan<Half> input,
        Span<Half> output)
    {
        var columns = input.Length;
        // Conversion and Dot doubled the runtime of the measured 3x3 scalar kernel.
        if (columns < 8)
        {
            for (var row = 0; row < output.Length; row++)
            {
                float sum = 0;
                for (var column = 0; column < columns; column++)
                    sum += (float)matrix[row * columns + column] * (float)input[column];
                output[row] = (Half)sum;
            }
            return;
        }
        if (output.Length < ParallelRowThreshold ||
            (long)output.Length * columns < ParallelElementThreshold)
        {
            var rentedInput = columns > 1024 ? ArrayPool<float>.Shared.Rent(columns) : null;
            Span<float> convertedInput = rentedInput is null
                ? stackalloc float[columns] : rentedInput.AsSpan(0, columns);
            try
            {
                TensorPrimitives.ConvertToSingle(input, convertedInput);
                Span<float> convertedMatrix = stackalloc float[Math.Min(columns, 1024)];
                for (var row = 0; row < output.Length; row++)
                    output[row] = (Half)Dot(matrix.Slice(row * columns, columns),
                        convertedInput, convertedMatrix);
            }
            finally
            {
                if (rentedInput is not null)
                    ArrayPool<float>.Shared.Return(rentedInput);
            }
            return;
        }

        var convertedInputArray = ArrayPool<float>.Shared.Rent(columns);
        try
        {
            var convertedInput = convertedInputArray.AsSpan(0, columns);
            TensorPrimitives.ConvertToSingle(input, convertedInput);
            fixed (Half* matrixPointer = matrix)
            fixed (float* inputPointer = convertedInput)
            fixed (Half* outputPointer = output)
            {
                var pointers = new HalfOutputPointers(matrixPointer, inputPointer, outputPointer);
                Parallel.For(0, output.Length,
                    () => ArrayPool<float>.Shared.Rent(columns),
                    (row, _, scratch) =>
                    {
                        var convertedMatrix = scratch.AsSpan(0, columns);
                        TensorPrimitives.ConvertToSingle(
                            new ReadOnlySpan<Half>(
                                pointers.Matrix + (long)row * columns, columns),
                            convertedMatrix);
                        pointers.Output[row] = (Half)TensorPrimitives.Dot(convertedMatrix,
                            new ReadOnlySpan<float>(pointers.Input, columns));
                        return scratch;
                    },
                    scratch => ArrayPool<float>.Shared.Return(scratch));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(convertedInputArray);
        }
    }

    public static void GatherRow(ReadOnlySpan<Half> table, int rowIndex, Span<float> output)
    {
        if (rowIndex < 0 || rowIndex >= table.Length / output.Length)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));
        TensorPrimitives.ConvertToSingle(
            table.Slice(checked(rowIndex * output.Length), output.Length), output);
    }

    private static float Dot(ReadOnlySpan<Half> matrixRow, ReadOnlySpan<float> input,
        Span<float> converted)
    {
        float sum = 0;
        while (!input.IsEmpty)
        {
            var length = Math.Min(input.Length, converted.Length);
            var block = converted[..length];
            TensorPrimitives.ConvertToSingle(matrixRow[..length], block);
            sum += TensorPrimitives.Dot(block, input[..length]);
            matrixRow = matrixRow[length..];
            input = input[length..];
        }
        return sum;
    }

    private readonly unsafe struct Pointers(Half* matrix, float* input, float* output)
    {
        public readonly Half* Matrix = matrix;
        public readonly float* Input = input;
        public readonly float* Output = output;
    }

    private readonly unsafe struct HalfOutputPointers(Half* matrix, float* input, Half* output)
    {
        public readonly Half* Matrix = matrix;
        public readonly float* Input = input;
        public readonly Half* Output = output;
    }
}
