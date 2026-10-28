using SharpInference.Backends.Cpu;

namespace SharpInference.Instructions.Cpu;

/// <summary>VM numerical kernels, including explicit FP16-weight/FP32-activation variants.</summary>
public static class CpuInstructionNumerics
{
    /// <summary>Computes elementwise hyperbolic tangents while preserving the sign of exact zero inputs.</summary>
    /// <param name="input">The dense input values.</param>
    /// <param name="output">The destination span with the length required by the operation.</param>
    public static void Tanh(ReadOnlySpan<float> input, Span<float> output)
    {
        CpuPrimitiveOperatorBackend.Instance.Tanh(input, output);
        // The SIMD approximation can return a nonzero residual at either signed zero.
        for (var i = 0; i < input.Length; i++)
        {
            if (input[i] == 0)
            {
                output[i] = input[i];
            }
        }
    }

    /// <summary>Multiplies a dense Float16 matrix by a Float32 vector and writes Float32 row results.</summary>
    /// <param name="matrix">The dense row-major Float16 matrix.</param>
    /// <param name="input">The dense input values.</param>
    /// <param name="output">The destination span with the length required by the operation.</param>
    /// <param name="rows">The number of matrix rows and output elements.</param>
    /// <param name="columns">The number of matrix columns and input vector elements.</param>
    public static void MatVec(ReadOnlySpan<Half> matrix, ReadOnlySpan<float> input, Span<float> output,
        int rows, int columns)
    {
        if (rows <= 0 || columns <= 0 || matrix.Length != checked(rows * columns) ||
            input.Length != columns || output.Length != rows)
        {
            throw new ArgumentException("Mixed matvec requires [rows,columns] x [columns] -> [rows].");
        }

        CpuHalfMatrixOperators.MatVec(matrix, input, output);
    }

    /// <summary>Copies a row from a dense Float16 table into a Float32 output vector.</summary>
    /// <param name="table">The dense row-major Float16 lookup table.</param>
    /// <param name="index">The zero-based table row to copy.</param>
    /// <param name="rowLength">The number of elements in each table row and in the output.</param>
    /// <param name="output">The destination span with the length required by the operation.</param>
    public static void GatherRow(ReadOnlySpan<Half> table, int index, int rowLength, Span<float> output)
    {
        if (rowLength <= 0 || output.Length != rowLength || table.Length % rowLength != 0)
        {
            throw new ArgumentException("Mixed gather requires a dense table and one complete output row.");
        }

        CpuHalfMatrixOperators.GatherRow(table, index, output);
    }

    /// <summary>Multiplies dense matrices with optional transposition and Float32 accumulation into a row-major output.</summary>
    /// <param name="left">The dense row-major left matrix before optional transposition.</param>
    /// <param name="right">The dense row-major right matrix before optional transposition.</param>
    /// <param name="output">The destination span with the length required by the operation.</param>
    /// <param name="leftRows">The stored row count of the left matrix.</param>
    /// <param name="leftColumns">The stored column count of the left matrix.</param>
    /// <param name="rightRows">The stored row count of the right matrix.</param>
    /// <param name="rightColumns">The stored column count of the right matrix.</param>
    /// <param name="transposeLeft">Whether to transpose the left matrix logically before multiplication.</param>
    /// <param name="transposeRight">Whether to transpose the right matrix logically before multiplication.</param>
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
            {
                throw new ArgumentException("Incompatible matrix multiplication dimensions.");
            }

            for (var row = 0; row < leftRows; row++)
            {
                CpuPrimitiveOperatorBackend.Instance.MatVec(
                    right,
                    left.Slice(row * leftColumns, leftColumns),
                    output.Slice(row * rightRows, rightRows),
                    rightRows,
                    rightColumns);
            }

            return;
        }
        MatrixMultiplyCore(left, right, output, leftRows, leftColumns,
            rightRows, rightColumns, transposeLeft, transposeRight);
    }

    /// <summary>Multiplies dense matrices with optional transposition and Float32 accumulation into a row-major output.</summary>
    /// <param name="left">The dense row-major left matrix before optional transposition.</param>
    /// <param name="right">The dense row-major right matrix before optional transposition.</param>
    /// <param name="output">The destination span with the length required by the operation.</param>
    /// <param name="leftRows">The stored row count of the left matrix.</param>
    /// <param name="leftColumns">The stored column count of the left matrix.</param>
    /// <param name="rightRows">The stored row count of the right matrix.</param>
    /// <param name="rightColumns">The stored column count of the right matrix.</param>
    /// <param name="transposeLeft">Whether to transpose the left matrix logically before multiplication.</param>
    /// <param name="transposeRight">Whether to transpose the right matrix logically before multiplication.</param>
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
            {
                throw new ArgumentException("Incompatible matrix multiplication dimensions.");
            }

            for (var row = 0; row < leftRows; row++)
            {
                CpuHalfMatrixOperators.MatVec(
                    right,
                    left.Slice(row * leftColumns, leftColumns),
                    output.Slice(row * rightRows, rightRows));
            }

            return;
        }
        var rows = transposeLeft ? leftColumns : leftRows;
        var inner = transposeLeft ? leftRows : leftColumns;
        var columns = transposeRight ? rightRows : rightColumns;
        if (inner != (transposeRight ? rightColumns : rightRows) ||
            output.Length != checked(rows * columns))
        {
            throw new ArgumentException("Incompatible matrix multiplication dimensions.");
        }

        for (var row = 0; row < rows; row++)
        {
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
    }

    /// <summary>Adds a bias vector repeatedly along the final input dimension, writing Float32 output values.</summary>
    /// <param name="input">The dense input values.</param>
    /// <param name="bias">The bias vector matching the final input dimension.</param>
    /// <param name="output">The destination span with the length required by the operation.</param>
    public static void BiasAdd(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> bias,
        Span<float> output)
    {
        if (bias.IsEmpty || input.Length != output.Length || input.Length % bias.Length != 0)
        {
            throw new ArgumentException("Bias must match the final input dimension.");
        }

        for (var index = 0; index < input.Length; index++)
        {
            output[index] = input[index] + bias[index % bias.Length];
        }
    }

    /// <summary>Adds a bias vector repeatedly along the final input dimension, writing Float32 output values.</summary>
    /// <param name="input">The dense input values.</param>
    /// <param name="bias">The bias vector matching the final input dimension.</param>
    /// <param name="output">The destination span with the length required by the operation.</param>
    public static void BiasAdd(
        ReadOnlySpan<float> input,
        ReadOnlySpan<Half> bias,
        Span<float> output)
    {
        if (bias.IsEmpty || input.Length != output.Length || input.Length % bias.Length != 0)
        {
            throw new ArgumentException("Bias must match the final input dimension.");
        }

        for (var index = 0; index < input.Length; index++)
        {
            output[index] = input[index] + (float)bias[index % bias.Length];
        }
    }

    public static void MatrixMultiply(
        ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output,
        int leftRows, int leftColumns, int rightRows, int rightColumns,
        bool transposeLeft, bool transposeRight)
    {
        var rows = transposeLeft ? leftColumns : leftRows;
        var inner = transposeLeft ? leftRows : leftColumns;
        var columns = transposeRight ? rightRows : rightColumns;
        if (leftRows <= 0 || leftColumns <= 0 || rightRows <= 0 || rightColumns <= 0 ||
            left.Length != checked(leftRows * leftColumns) ||
            right.Length != checked(rightRows * rightColumns) ||
            inner != (transposeRight ? rightColumns : rightRows) ||
            output.Length != checked(rows * columns))
        {
            throw new ArgumentException("Incompatible matrix multiplication dimensions.");
        }
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                var sum = 0f;
                for (var index = 0; index < inner; index++)
                {
                    var a = (float)left[transposeLeft ? index * leftColumns + row : row * leftColumns + index];
                    var b = (float)right[transposeRight ? column * rightColumns + index : index * rightColumns + column];
                    sum += a * b;
                }
                output[row * columns + column] = (Half)sum;
            }
        }
    }

    public static void BiasAdd(ReadOnlySpan<Half> input, ReadOnlySpan<Half> bias, Span<Half> output)
    {
        if (bias.IsEmpty || input.Length != output.Length || input.Length % bias.Length != 0)
        {
            throw new ArgumentException("Bias must match the final input dimension.");
        }
        for (var index = 0; index < input.Length; index++)
        {
            output[index] = (Half)((float)input[index] + (float)bias[index % bias.Length]);
        }
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
        {
            throw new ArgumentException("Incompatible matrix multiplication dimensions.");
        }

        for (var row = 0; row < rows; row++)
        {
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
}
