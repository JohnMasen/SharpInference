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
    public void Add(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Binary(left, right, output, static (a, b) => a + b);
    public void Subtract(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Binary(left, right, output, static (a, b) => a - b);
    public void Multiply(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Binary(left, right, output, static (a, b) => a * b);
    public void Divide(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Binary(left, right, output, static (a, b) => a / b);
    public void Maximum(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Binary(left, right, output, MathF.Max);
    public void Exp(ReadOnlySpan<float> input, Span<float> output) => Unary(input, output, MathF.Exp);
    public void Tanh(ReadOnlySpan<float> input, Span<float> output) => Unary(input, output, MathF.Tanh);
    public void Sigmoid(ReadOnlySpan<float> input, Span<float> output) => Unary(input, output, static value => 1f / (1f + MathF.Exp(-value)));
    public void ReciprocalSquareRoot(ReadOnlySpan<float> input, Span<float> output) => Unary(input, output, static value => 1f / MathF.Sqrt(value));
    public void Square(ReadOnlySpan<float> input, Span<float> output) => Unary(input, output, static value => value * value);
    public void Relu(ReadOnlySpan<float> input, Span<float> output) => Unary(input, output, static value => MathF.Max(0, value));
    public float ReduceSum(ReadOnlySpan<float> input) => ReduceSumFloat(input);
    public float ReduceMean(ReadOnlySpan<float> input) => input.IsEmpty ? throw new ArgumentException("The input cannot be empty.", nameof(input)) : ReduceSumFloat(input) / input.Length;
    public void MatVec(ReadOnlySpan<float> matrix, ReadOnlySpan<float> input, Span<float> output, int rows, int columns) => MatVecFloat(matrix, input, output, rows, columns);
    public void GatherRow(ReadOnlySpan<float> table, int rowIndex, int rowLength, Span<float> output) => GatherRowCore(table, rowIndex, rowLength, output);

    public void Copy(ReadOnlySpan<Half> input, Span<Half> output) => CopyCore(input, output);
    public void Add(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output) => BinaryHalf(left, right, output, static (a, b) => a + b);
    public void Subtract(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output) => BinaryHalf(left, right, output, static (a, b) => a - b);
    public void Multiply(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output) => BinaryHalf(left, right, output, static (a, b) => a * b);
    public void Divide(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output) => BinaryHalf(left, right, output, static (a, b) => a / b);
    public void Maximum(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output) => BinaryHalf(left, right, output, MathF.Max);
    public void Exp(ReadOnlySpan<Half> input, Span<Half> output) => UnaryHalf(input, output, MathF.Exp);
    public void Tanh(ReadOnlySpan<Half> input, Span<Half> output) => UnaryHalf(input, output, MathF.Tanh);
    public void Sigmoid(ReadOnlySpan<Half> input, Span<Half> output) => UnaryHalf(input, output, static value => 1f / (1f + MathF.Exp(-value)));
    public void ReciprocalSquareRoot(ReadOnlySpan<Half> input, Span<Half> output) => UnaryHalf(input, output, static value => 1f / MathF.Sqrt(value));
    public void Square(ReadOnlySpan<Half> input, Span<Half> output) => UnaryHalf(input, output, static value => value * value);
    public void Relu(ReadOnlySpan<Half> input, Span<Half> output) => UnaryHalf(input, output, static value => MathF.Max(0, value));
    public Half ReduceSum(ReadOnlySpan<Half> input) => (Half)ReduceSumHalf(input);
    public Half ReduceMean(ReadOnlySpan<Half> input) => input.IsEmpty ? throw new ArgumentException("The input cannot be empty.", nameof(input)) : (Half)(ReduceSumHalf(input) / input.Length);
    public void MatVec(ReadOnlySpan<Half> matrix, ReadOnlySpan<Half> input, Span<Half> output, int rows, int columns) => MatVecHalf(matrix, input, output, rows, columns);
    public void GatherRow(ReadOnlySpan<Half> table, int rowIndex, int rowLength, Span<Half> output) => GatherRowCore(table, rowIndex, rowLength, output);

    private static void CopyCore<T>(ReadOnlySpan<T> input, Span<T> output)
    {
        RequireOutput(input.Length, output.Length);
        input.CopyTo(output);
    }

    private static void Unary(ReadOnlySpan<float> input, Span<float> output, Func<float, float> operation)
    {
        RequireOutput(input.Length, output.Length);
        for (var i = 0; i < input.Length; i++) output[i] = operation(input[i]);
    }

    private static void UnaryHalf(ReadOnlySpan<Half> input, Span<Half> output, Func<float, float> operation)
    {
        RequireOutput(input.Length, output.Length);
        for (var i = 0; i < input.Length; i++) output[i] = (Half)operation((float)input[i]);
    }

    private static void Binary(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output, Func<float, float, float> operation)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        for (var i = 0; i < left.Length; i++) output[i] = operation(left[i], right[i]);
    }

    private static void BinaryHalf(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output, Func<float, float, float> operation)
    {
        RequireBinary(left.Length, right.Length, output.Length);
        for (var i = 0; i < left.Length; i++) output[i] = (Half)operation((float)left[i], (float)right[i]);
    }

    private static float ReduceSumFloat(ReadOnlySpan<float> input)
    {
        float result = 0;
        for (var i = 0; i < input.Length; i++) result += input[i];
        return result;
    }

    private static float ReduceSumHalf(ReadOnlySpan<Half> input)
    {
        float result = 0;
        for (var i = 0; i < input.Length; i++) result += (float)input[i];
        return result;
    }

    private static void MatVecFloat(ReadOnlySpan<float> matrix, ReadOnlySpan<float> input, Span<float> output, int rows, int columns)
    {
        ValidateMatVec(matrix.Length, input.Length, output.Length, rows, columns);
        for (var row = 0; row < rows; row++)
        {
            float sum = 0;
            for (var column = 0; column < columns; column++) sum += matrix[row * columns + column] * input[column];
            output[row] = sum;
        }
    }

    private static void MatVecHalf(ReadOnlySpan<Half> matrix, ReadOnlySpan<Half> input, Span<Half> output, int rows, int columns)
    {
        ValidateMatVec(matrix.Length, input.Length, output.Length, rows, columns);
        for (var row = 0; row < rows; row++)
        {
            float sum = 0;
            for (var column = 0; column < columns; column++) sum += (float)matrix[row * columns + column] * (float)input[column];
            output[row] = (Half)sum;
        }
    }

    private static void GatherRowCore<T>(ReadOnlySpan<T> table, int rowIndex, int rowLength, Span<T> output)
    {
        if (rowIndex < 0) throw new ArgumentOutOfRangeException(nameof(rowIndex));
        if (rowLength <= 0) throw new ArgumentOutOfRangeException(nameof(rowLength));
        RequireOutput(rowLength, output.Length);
        var offset = checked(rowIndex * rowLength);
        if (offset > table.Length - rowLength) throw new ArgumentOutOfRangeException(nameof(rowIndex));
        table.Slice(offset, rowLength).CopyTo(output);
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
