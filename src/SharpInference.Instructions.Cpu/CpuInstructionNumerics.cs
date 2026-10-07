using SharpInference.Backends.Cpu;

namespace SharpInference.Instructions.Cpu;

/// <summary>VM numerical kernels, including explicit FP16-weight/FP32-activation variants.</summary>
public static class CpuInstructionNumerics
{
    public static void Tanh(ReadOnlySpan<float> input, Span<float> output)
    {
        CpuPrimitiveOperatorBackend.Instance.Tanh(input, output);
        // The SIMD approximation can return a nonzero residual at either signed zero.
        for (var i = 0; i < input.Length; i++)
            if (input[i] == 0) output[i] = input[i];
    }

    public static void MatVec(ReadOnlySpan<Half> matrix, ReadOnlySpan<float> input, Span<float> output,
        int rows, int columns)
    {
        if (rows <= 0 || columns <= 0 || matrix.Length != checked(rows * columns) ||
            input.Length != columns || output.Length != rows)
            throw new ArgumentException("Mixed matvec requires [rows,columns] x [columns] -> [rows].");
        CpuHalfMatrixOperators.MatVec(matrix, input, output);
    }

    public static void GatherRow(ReadOnlySpan<Half> table, int index, int rowLength, Span<float> output)
    {
        if (rowLength <= 0 || output.Length != rowLength || table.Length % rowLength != 0)
            throw new ArgumentException("Mixed gather requires a dense table and one complete output row.");
        CpuHalfMatrixOperators.GatherRow(table, index, output);
    }

    public static void MatrixMultiply(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> output,
        int leftRows,
        int leftColumns,
        int rightRows,
        int rightColumns,
        bool transposeLeft,
        bool transposeRight)
    {
        if (!transposeLeft && transposeRight)
        {
            if (leftColumns != rightColumns ||
                output.Length != checked(leftRows * rightRows))
                throw new ArgumentException("Incompatible matrix multiplication dimensions.");
            for (var row = 0; row < leftRows; row++)
                CpuPrimitiveOperatorBackend.Instance.MatVec(
                    right,
                    left.Slice(row * leftColumns, leftColumns),
                    output.Slice(row * rightRows, rightRows),
                    rightRows,
                    rightColumns);
            return;
        }
        MatrixMultiplyCore(left, right, output, leftRows, leftColumns,
            rightRows, rightColumns, transposeLeft, transposeRight);
    }

    public static void MatrixMultiply(
        ReadOnlySpan<float> left,
        ReadOnlySpan<Half> right,
        Span<float> output,
        int leftRows,
        int leftColumns,
        int rightRows,
        int rightColumns,
        bool transposeLeft,
        bool transposeRight)
    {
        if (!transposeLeft && transposeRight)
        {
            if (leftColumns != rightColumns ||
                output.Length != checked(leftRows * rightRows))
                throw new ArgumentException("Incompatible matrix multiplication dimensions.");
            for (var row = 0; row < leftRows; row++)
                CpuHalfMatrixOperators.MatVec(
                    right,
                    left.Slice(row * leftColumns, leftColumns),
                    output.Slice(row * rightRows, rightRows));
            return;
        }
        var rows = transposeLeft ? leftColumns : leftRows;
        var inner = transposeLeft ? leftRows : leftColumns;
        var columns = transposeRight ? rightRows : rightColumns;
        if (inner != (transposeRight ? rightColumns : rightRows) ||
            output.Length != checked(rows * columns))
            throw new ArgumentException("Incompatible matrix multiplication dimensions.");
        for (var row = 0; row < rows; row++)
        for (var column = 0; column < columns; column++)
        {
            var sum = 0f;
            for (var index = 0; index < inner; index++)
            {
                var leftValue = transposeLeft
                    ? left[index * leftColumns + row]
                    : left[row * leftColumns + index];
                var rightValue = transposeRight
                    ? (float)right[column * rightColumns + index]
                    : (float)right[index * rightColumns + column];
                sum += leftValue * rightValue;
            }
            output[row * columns + column] = sum;
        }
    }

    public static void BiasAdd(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> bias,
        Span<float> output)
    {
        if (bias.IsEmpty || input.Length != output.Length || input.Length % bias.Length != 0)
            throw new ArgumentException("Bias must match the final input dimension.");
        for (var index = 0; index < input.Length; index++)
            output[index] = input[index] + bias[index % bias.Length];
    }

    public static void BiasAdd(
        ReadOnlySpan<float> input,
        ReadOnlySpan<Half> bias,
        Span<float> output)
    {
        if (bias.IsEmpty || input.Length != output.Length || input.Length % bias.Length != 0)
            throw new ArgumentException("Bias must match the final input dimension.");
        for (var index = 0; index < input.Length; index++)
            output[index] = input[index] + (float)bias[index % bias.Length];
    }

    private static void MatrixMultiplyCore(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> output,
        int leftRows,
        int leftColumns,
        int rightRows,
        int rightColumns,
        bool transposeLeft,
        bool transposeRight)
    {
        var rows = transposeLeft ? leftColumns : leftRows;
        var inner = transposeLeft ? leftRows : leftColumns;
        var columns = transposeRight ? rightRows : rightColumns;
        if (inner != (transposeRight ? rightColumns : rightRows) ||
            output.Length != checked(rows * columns))
            throw new ArgumentException("Incompatible matrix multiplication dimensions.");
        for (var row = 0; row < rows; row++)
        for (var column = 0; column < columns; column++)
        {
            var sum = 0f;
            for (var index = 0; index < inner; index++)
            {
                var leftValue = transposeLeft
                    ? left[index * leftColumns + row]
                    : left[row * leftColumns + index];
                var rightValue = transposeRight
                    ? right[column * rightColumns + index]
                    : right[index * rightColumns + column];
                sum += leftValue * rightValue;
            }
            output[row * columns + column] = sum;
        }
    }
}
