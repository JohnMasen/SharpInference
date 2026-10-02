using System.Diagnostics;
using System.Numerics.Tensors;
using SharpInference.Backends.Cpu;

namespace SharpInference.PrefillExperiment;

internal static class CpuTensorPrimitivesBenchmark
{
    public static void Run(int length, int width, int repeats)
    {
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (repeats <= 0) throw new ArgumentOutOfRangeException(nameof(repeats));

        var random = new Random(1701);
        var left = new float[length];
        var right = new float[length];
        for (var index = 0; index < length; index++)
        {
            left[index] = (float)(random.NextDouble() * 2 - 1);
            right[index] = (float)(random.NextDouble() * 2 - 1);
        }

        var leftHalf = left.Select(value => (Half)value).ToArray();
        var rightHalf = right.Select(value => (Half)value).ToArray();
        var positive = left.Select(value => MathF.Abs(value) + 0.01f).ToArray();
        var positiveHalf = positive.Select(value => (Half)value).ToArray();
        var floatOutput = new float[length];
        var halfOutput = new Half[length];
        var backend = CpuPrimitiveOperatorBackend.Instance;
        var iterations = checked(Math.Max(1, 4_194_304 / length) * repeats);

        left.AsSpan().CopyTo(floatOutput);
        backend.Copy(left, floatOutput);
        TensorCopy(left, floatOutput);
        floatOutput.AsSpan().Fill(1f);
        TensorFill(floatOutput, 1f);
        ScalarAdd(left, right, floatOutput);
        backend.Add(left, right, floatOutput);
        TensorPrimitives.Add(left, right, floatOutput);
        ScalarAdd(leftHalf, rightHalf, halfOutput);
        backend.Add(leftHalf, rightHalf, halfOutput);
        TensorPrimitives.Add(leftHalf, rightHalf, halfOutput);
        ScalarExp(left, floatOutput);
        backend.Exp(left, floatOutput);
        TensorPrimitives.Exp(left, floatOutput);
        ScalarExp(leftHalf, halfOutput);
        backend.Exp(leftHalf, halfOutput);
        TensorPrimitives.Exp(leftHalf, halfOutput);
        ScalarSigmoid(left, floatOutput);
        backend.Sigmoid(left, floatOutput);
        TensorPrimitives.Sigmoid(left, floatOutput);
        ScalarSigmoid(leftHalf, halfOutput);
        backend.Sigmoid(leftHalf, halfOutput);
        TensorPrimitives.Sigmoid(leftHalf, halfOutput);
        ScalarTanh(left, floatOutput);
        backend.Tanh(left, floatOutput);
        ScalarTanh(leftHalf, halfOutput);
        backend.Tanh(leftHalf, halfOutput);
        ScalarRsqrt(positive, floatOutput);
        backend.ReciprocalSquareRoot(positive, floatOutput);
        ScalarRsqrt(positiveHalf, halfOutput);
        backend.ReciprocalSquareRoot(positiveHalf, halfOutput);
        Consume(ScalarSum(leftHalf));
        Consume((float)backend.ReduceSum(leftHalf));

        var scalarFloat = Measure(iterations, () => ScalarAdd(left, right, floatOutput));
        var spanCopy = Measure(iterations, () => left.AsSpan().CopyTo(floatOutput));
        var backendCopy = Measure(iterations, () => backend.Copy(left, floatOutput));
        var tensorCopy = Measure(iterations, () => TensorCopy(left, floatOutput));
        var spanFill = Measure(iterations, () => floatOutput.AsSpan().Fill(1f));
        var tensorFill = Measure(iterations, () => TensorFill(floatOutput, 1f));
        var backendFloat = Measure(iterations, () => backend.Add(left, right, floatOutput));
        var tensorFloat = Measure(iterations,
            () => TensorPrimitives.Add(left, right, floatOutput));
        var scalarHalf = Measure(iterations, () => ScalarAdd(leftHalf, rightHalf, halfOutput));
        var backendHalf = Measure(iterations, () => backend.Add(leftHalf, rightHalf, halfOutput));
        var tensorHalf = Measure(iterations,
            () => TensorPrimitives.Add(leftHalf, rightHalf, halfOutput));
        var scalarFloatExp = Measure(iterations, () => ScalarExp(left, floatOutput));
        var backendFloatExp = Measure(iterations, () => backend.Exp(left, floatOutput));
        var tensorFloatExp = Measure(iterations, () => TensorPrimitives.Exp(left, floatOutput));
        var scalarHalfExp = Measure(iterations, () => ScalarExp(leftHalf, halfOutput));
        var backendHalfExp = Measure(iterations, () => backend.Exp(leftHalf, halfOutput));
        var tensorHalfExp = Measure(iterations,
            () => TensorPrimitives.Exp(leftHalf, halfOutput));
        var scalarFloatSigmoid = Measure(iterations, () => ScalarSigmoid(left, floatOutput));
        var backendFloatSigmoid = Measure(iterations, () => backend.Sigmoid(left, floatOutput));
        var tensorFloatSigmoid = Measure(iterations,
            () => TensorPrimitives.Sigmoid(left, floatOutput));
        var scalarHalfSigmoid = Measure(iterations, () => ScalarSigmoid(leftHalf, halfOutput));
        var backendHalfSigmoid = Measure(iterations, () => backend.Sigmoid(leftHalf, halfOutput));
        var tensorHalfSigmoid = Measure(iterations,
            () => TensorPrimitives.Sigmoid(leftHalf, halfOutput));
        var scalarFloatTanh = Measure(iterations, () => ScalarTanh(left, floatOutput));
        var tensorFloatTanh = Measure(iterations, () => backend.Tanh(left, floatOutput));
        var scalarHalfTanh = Measure(iterations, () => ScalarTanh(leftHalf, halfOutput));
        var tensorHalfTanh = Measure(iterations, () => backend.Tanh(leftHalf, halfOutput));
        var scalarFloatRsqrt = Measure(iterations, () => ScalarRsqrt(positive, floatOutput));
        var tensorFloatRsqrt = Measure(iterations,
            () => backend.ReciprocalSquareRoot(positive, floatOutput));
        var scalarHalfRsqrt = Measure(iterations, () => ScalarRsqrt(positiveHalf, halfOutput));
        var tensorHalfRsqrt = Measure(iterations,
            () => backend.ReciprocalSquareRoot(positiveHalf, halfOutput));
        var scalarReduce = Measure(iterations, () => Consume((float)(Half)ScalarSum(leftHalf)));
        var tensorReduce = Measure(iterations, () => Consume((float)backend.ReduceSum(leftHalf)));

        Console.WriteLine($"CPU Tensor migration: length={length} iterations={iterations}");
        Print("FP32 Copy (Span vs Tensor)", spanCopy, backendCopy, tensorCopy);
        Console.WriteLine($"  FP32 Fill: span={spanFill.TotalMilliseconds:F3}ms " +
            $"tensor={tensorFill.TotalMilliseconds:F3}ms " +
            $"tensor-slowdown={tensorFill.TotalMilliseconds / spanFill.TotalMilliseconds:F2}x");
        Print("FP32 Add", scalarFloat, backendFloat, tensorFloat);
        Print("FP16 Add", scalarHalf, backendHalf, tensorHalf);
        Print("FP32 Exp", scalarFloatExp, backendFloatExp, tensorFloatExp);
        Print("FP16 Exp", scalarHalfExp, backendHalfExp, tensorHalfExp);
        Print("FP32 Sigmoid", scalarFloatSigmoid, backendFloatSigmoid, tensorFloatSigmoid);
        Print("FP16 Sigmoid", scalarHalfSigmoid, backendHalfSigmoid, tensorHalfSigmoid);
        Print("FP32 Tanh", scalarFloatTanh, tensorFloatTanh);
        Print("FP16 Tanh", scalarHalfTanh, tensorHalfTanh);
        Print("FP32 Rsqrt", scalarFloatRsqrt, tensorFloatRsqrt);
        Print("FP16 Rsqrt", scalarHalfRsqrt, tensorHalfRsqrt);
        Print("FP16 ReduceSum", scalarReduce, tensorReduce);

        var matrix = new Half[checked(width * width)];
        var vector = new Half[width];
        var output = new Half[width];
        for (var index = 0; index < matrix.Length; index++)
            matrix[index] = (Half)(random.NextDouble() * 0.02 - 0.01);
        for (var index = 0; index < vector.Length; index++)
            vector[index] = (Half)(random.NextDouble() * 0.2 - 0.1);
        var matVecIterations = checked(Math.Max(1, 16_777_216 / matrix.Length) * repeats);
        var scalarMatVec = Measure(matVecIterations,
            () => ScalarMatVec(matrix, vector, output, width));
        var tensorMatVec = Measure(matVecIterations,
            () => backend.MatVec(matrix, vector, output, width, width));
        Print($"FP16 MatVec {width}x{width}", scalarMatVec, tensorMatVec);
        CpuCustomTensorBenchmark.Run(length, repeats);
    }

    private static TimeSpan Measure(int iterations, Action action)
    {
        var start = Stopwatch.GetTimestamp();
        for (var iteration = 0; iteration < iterations; iteration++) action();
        return Stopwatch.GetElapsedTime(start);
    }

    private static void TensorCopy(ReadOnlySpan<float> input, Span<float> output)
    {
        var source = new ReadOnlyTensorSpan<float>(input);
        var destination = new TensorSpan<float>(output);
        source.CopyTo(destination);
    }

    private static void TensorFill(Span<float> output, float value) =>
        new TensorSpan<float>(output).Fill(value);

    private static void Print(string name, TimeSpan scalar, TimeSpan backend)
    {
        Console.WriteLine($"  {name}: scalar={scalar.TotalMilliseconds:F3}ms " +
            $"backend={backend.TotalMilliseconds:F3}ms " +
            $"speedup={scalar.TotalMilliseconds / backend.TotalMilliseconds:F2}x");
    }

    private static void Print(string name, TimeSpan scalar, TimeSpan backend, TimeSpan tensor)
    {
        Console.WriteLine($"  {name}: scalar={scalar.TotalMilliseconds:F3}ms " +
            $"backend={backend.TotalMilliseconds:F3}ms " +
            $"tensor={tensor.TotalMilliseconds:F3}ms " +
            $"tensor-vs-backend={backend.TotalMilliseconds / tensor.TotalMilliseconds:F3}x");
    }

    private static void ScalarAdd(ReadOnlySpan<float> left, ReadOnlySpan<float> right,
        Span<float> output)
    {
        for (var index = 0; index < left.Length; index++)
            output[index] = left[index] + right[index];
    }

    private static void ScalarAdd(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right,
        Span<Half> output)
    {
        for (var index = 0; index < left.Length; index++)
            output[index] = (Half)((float)left[index] + (float)right[index]);
    }

    private static float ScalarSum(ReadOnlySpan<Half> input)
    {
        float result = 0;
        for (var index = 0; index < input.Length; index++) result += (float)input[index];
        return result;
    }

    private static void ScalarExp(ReadOnlySpan<float> input, Span<float> output)
    {
        for (var index = 0; index < input.Length; index++) output[index] = MathF.Exp(input[index]);
    }

    private static void ScalarExp(ReadOnlySpan<Half> input, Span<Half> output)
    {
        for (var index = 0; index < input.Length; index++)
            output[index] = (Half)MathF.Exp((float)input[index]);
    }

    private static void ScalarSigmoid(ReadOnlySpan<float> input, Span<float> output)
    {
        for (var index = 0; index < input.Length; index++)
            output[index] = 1f / (1f + MathF.Exp(-input[index]));
    }

    private static void ScalarSigmoid(ReadOnlySpan<Half> input, Span<Half> output)
    {
        for (var index = 0; index < input.Length; index++)
            output[index] = (Half)(1f / (1f + MathF.Exp(-(float)input[index])));
    }

    private static void ScalarTanh(ReadOnlySpan<float> input, Span<float> output)
    {
        for (var index = 0; index < input.Length; index++) output[index] = MathF.Tanh(input[index]);
    }

    private static void ScalarTanh(ReadOnlySpan<Half> input, Span<Half> output)
    {
        for (var index = 0; index < input.Length; index++)
            output[index] = (Half)MathF.Tanh((float)input[index]);
    }

    private static void ScalarRsqrt(ReadOnlySpan<float> input, Span<float> output)
    {
        for (var index = 0; index < input.Length; index++)
            output[index] = 1f / MathF.Sqrt(input[index]);
    }

    private static void ScalarRsqrt(ReadOnlySpan<Half> input, Span<Half> output)
    {
        for (var index = 0; index < input.Length; index++)
            output[index] = (Half)(1f / MathF.Sqrt((float)input[index]));
    }

    private static void ScalarMatVec(ReadOnlySpan<Half> matrix, ReadOnlySpan<Half> input,
        Span<Half> output, int width)
    {
        for (var row = 0; row < width; row++)
        {
            float sum = 0;
            for (var column = 0; column < width; column++)
                sum += (float)matrix[row * width + column] * (float)input[column];
            output[row] = (Half)sum;
        }
    }

    private static float consumed;

    private static void Consume(float value)
    {
        consumed = value;
    }
}
