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
}
