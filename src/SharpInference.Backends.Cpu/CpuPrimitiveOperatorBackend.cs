using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using SharpInference.Graphs;

namespace SharpInference.Backends.Cpu;

public sealed class CpuPrimitiveOperatorBackend : IPrimitiveOperatorBackend
{
    public static CpuPrimitiveOperatorBackend Instance { get; } = new();

    private CpuPrimitiveOperatorBackend()
    {
    }

    public IReadOnlyList<PrimitiveOperatorDescription> PrimitiveOperators { get; } =
        PrimitiveGraphOperations.CreateStandardDescriptions();
    public IReadOnlyList<OperatorImplementationDescription> OperatorImplementations { get; } =
        PrimitiveGraphOperations.CreateStandardImplementations(
            "cpu",
            includeFp16: true,
            new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32),
            new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32),
            new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32));

    public void Copy(ReadOnlySpan<float> input, Span<float> output) => CopyCore(input, output);
    public void Add(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        TensorPrimitives.Add(left, right, output);
    }

    public void Subtract(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        TensorPrimitives.Subtract(left, right, output);
    }

    public void Multiply(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        TensorPrimitives.Multiply(left, right, output);
    }

    public void Divide(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        TensorPrimitives.Divide(left, right, output);
    }

    public void Maximum(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        TensorPrimitives.Max(left, right, output);
    }

    public void Exp(ReadOnlySpan<float> input, Span<float> output)
    {
        RequireOutput(input.Length, output.Length);
        TensorPrimitives.Exp(input, output);
    }

    public void Tanh(ReadOnlySpan<float> input, Span<float> output)
    {
        RequireOutput(input.Length, output.Length);
        TensorPrimitives.Tanh(input, output);
    }

    public void Sigmoid(ReadOnlySpan<float> input, Span<float> output)
    {
        RequireOutput(input.Length, output.Length);
        if (!input.IsEmpty) TensorPrimitives.Sigmoid(input, output);
    }

    public void ReciprocalSquareRoot(ReadOnlySpan<float> input, Span<float> output)
    {
        RequireOutput(input.Length, output.Length);
        TensorPrimitives.ReciprocalSqrt(input, output);
    }

    public void Square(ReadOnlySpan<float> input, Span<float> output)
    {
        RequireOutput(input.Length, output.Length);
        TensorPrimitives.Multiply(input, input, output);
    }

    public void Relu(ReadOnlySpan<float> input, Span<float> output)
    {
        RequireOutput(input.Length, output.Length);
        TensorPrimitives.Max(input, 0f, output);
    }

    public float ReduceSum(ReadOnlySpan<float> input) => TensorPrimitives.Sum(input);
    public float ReduceMean(ReadOnlySpan<float> input) => input.IsEmpty ? throw new ArgumentException("The input cannot be empty.", nameof(input)) : TensorPrimitives.Sum(input) / input.Length;
    public void MatVec(ReadOnlySpan<float> matrix, ReadOnlySpan<float> input, Span<float> output, int rows, int columns) => MatVecFloat(matrix, input, output, rows, columns);
    public void GatherRow(ReadOnlySpan<float> table, int rowIndex, int rowLength, Span<float> output) => GatherRowCore(table, rowIndex, rowLength, output);

    public void Copy(ReadOnlySpan<Half> input, Span<Half> output) => CopyCore(input, output);
    public void Add(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        TensorPrimitives.Add(left, right, output);
    }

    public void Subtract(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        TensorPrimitives.Subtract(left, right, output);
    }

    public void Multiply(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        TensorPrimitives.Multiply(left, right, output);
    }

    public void Divide(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        TensorPrimitives.Divide(left, right, output);
    }

    public void Maximum(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        TensorPrimitives.Max(left, right, output);
    }

    public void Exp(ReadOnlySpan<Half> input, Span<Half> output)
    {
        RequireOutput(input.Length, output.Length);
        TensorPrimitives.Exp(input, output);
    }

    public void Tanh(ReadOnlySpan<Half> input, Span<Half> output)
    {
        RequireOutput(input.Length, output.Length);
        TensorPrimitives.Tanh(input, output);
    }

    public void Sigmoid(ReadOnlySpan<Half> input, Span<Half> output)
    {
        RequireOutput(input.Length, output.Length);
        if (Vector128.IsHardwareAccelerated && input.Length >= Vector128<short>.Count)
            TensorPrimitives.Sigmoid(input, output);
        else
            PromotedHalfUnary(input, output, sigmoid: true);
    }

    public void ReciprocalSquareRoot(ReadOnlySpan<Half> input, Span<Half> output)
    {
        RequireOutput(input.Length, output.Length);
        if (Vector128.IsHardwareAccelerated && input.Length >= Vector128<short>.Count)
            TensorPrimitives.ReciprocalSqrt(input, output);
        else
            PromotedHalfUnary(input, output, sigmoid: false);
    }

    public void Square(ReadOnlySpan<Half> input, Span<Half> output)
    {
        RequireOutput(input.Length, output.Length);
        TensorPrimitives.Multiply(input, input, output);
    }

    public void Relu(ReadOnlySpan<Half> input, Span<Half> output)
    {
        RequireOutput(input.Length, output.Length);
        TensorPrimitives.Max(input, (Half)0, output);
    }
    public Half ReduceSum(ReadOnlySpan<Half> input) => (Half)ReduceSumHalf(input);
    public Half ReduceMean(ReadOnlySpan<Half> input) => input.IsEmpty ? throw new ArgumentException("The input cannot be empty.", nameof(input)) : (Half)(ReduceSumHalf(input) / input.Length);
    public void MatVec(ReadOnlySpan<Half> matrix, ReadOnlySpan<Half> input, Span<Half> output, int rows, int columns) => MatVecHalf(matrix, input, output, rows, columns);
    public void GatherRow(ReadOnlySpan<Half> table, int rowIndex, int rowLength, Span<Half> output) => GatherRowCore(table, rowIndex, rowLength, output);

    private static void CopyCore<T>(ReadOnlySpan<T> input, Span<T> output)
    {
        RequireOutput(input.Length, output.Length);
        input.CopyTo(output);
    }

    private static float ReduceSumHalf(ReadOnlySpan<Half> input)
    {
        // Below one SIMD vector, conversion overhead caused a measured >50% regression.
        if (input.Length < 8)
        {
            float scalar = 0;
            foreach (var value in input) scalar += (float)value;
            return scalar;
        }
        const int blockLength = 1024;
        Span<float> converted = stackalloc float[Math.Min(input.Length, blockLength)];
        float result = 0;
        while (!input.IsEmpty)
        {
            var length = Math.Min(input.Length, converted.Length);
            var block = converted[..length];
            TensorPrimitives.ConvertToSingle(input[..length], block);
            result += TensorPrimitives.Sum(block);
            input = input[length..];
        }
        return result;
    }

    private static void PromotedHalfUnary(ReadOnlySpan<Half> input, Span<Half> output, bool sigmoid)
    {
        // The Half scalar composite operators round intermediate results to Half.
        Span<float> converted = stackalloc float[Math.Min(input.Length, 1024)];
        while (!input.IsEmpty)
        {
            var length = Math.Min(input.Length, converted.Length);
            var block = converted[..length];
            TensorPrimitives.ConvertToSingle(input[..length], block);
            if (sigmoid)
                TensorPrimitives.Sigmoid(block, block);
            else
                TensorPrimitives.ReciprocalSqrt(block, block);
            TensorPrimitives.ConvertToHalf(block, output[..length]);
            input = input[length..];
            output = output[length..];
        }
    }

    private static void MatVecFloat(ReadOnlySpan<float> matrix, ReadOnlySpan<float> input, Span<float> output, int rows, int columns)
    {
        ValidateMatVec(matrix.Length, input.Length, output.Length, rows, columns);
        for (var row = 0; row < rows; row++)
            output[row] = TensorPrimitives.Dot(matrix.Slice(row * columns, columns), input);
    }

    private static void MatVecHalf(ReadOnlySpan<Half> matrix, ReadOnlySpan<Half> input, Span<Half> output, int rows, int columns)
    {
        ValidateMatVec(matrix.Length, input.Length, output.Length, rows, columns);
        CpuHalfMatrixOperators.MatVec(matrix, input, output);
    }

    private static void GatherRowCore<T>(ReadOnlySpan<T> table, int rowIndex, int rowLength, Span<T> output)
    {
        if (rowIndex < 0) throw new ArgumentOutOfRangeException(nameof(rowIndex));
        if (rowLength <= 0) throw new ArgumentOutOfRangeException(nameof(rowLength));
        RequireOutput(rowLength, output.Length);
        var offset = checked(rowIndex * rowLength);
        if (offset > table.Length - rowLength) throw new ArgumentOutOfRangeException(nameof(rowIndex));
        CopyCore(table.Slice(offset, rowLength), output);
    }

    private static void RequireOutput(int inputLength, int outputLength)
    {
        if (inputLength != outputLength) throw new ArgumentException("The output length must match the input length.");
    }

    private static void RequireBinary(int leftLength, int rightLength, int outputLength)
    {
        if (leftLength != rightLength || leftLength != outputLength)
            throw new ArgumentException("Binary operator input and output lengths must match.");
    }

    private static void ValidateMatVec(int matrixLength, int inputLength, int outputLength, int rows, int columns)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns <= 0) throw new ArgumentOutOfRangeException(nameof(columns));
        if (matrixLength != checked(rows * columns) || inputLength != columns || outputLength != rows)
            throw new ArgumentException("The MatVec dimensions do not match the supplied buffers.");
    }
}
