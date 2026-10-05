using System.Numerics;
using System.Numerics.Tensors;

namespace SharpInference.Instructions.Cpu;

public static class CpuTierOneNumerics
{
    private const int BlockLength = 256;

    public static void MultiplyAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> c, Span<float> output)
    {
        Validate(a, output);
        Validate(b, output);
        Validate(c, output);
        TensorPrimitives.MultiplyAdd(a, b, c, output);
    }
    public static void DifferenceMix(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> c, Span<float> output) =>
        Ternary<DifferenceMixOperator>(a, b, c, output);
    public static void MultiplyAddSubtract(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> c, Span<float> output) =>
        Ternary<MultiplyAddSubtractOperator>(a, b, c, output);
    public static void AddMultiplyAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> c, ReadOnlySpan<float> d, Span<float> output) =>
        Quaternary<AddMultiplyAddOperator>(a, b, c, d, output);
    public static void MultiplyMultiplyAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> c, ReadOnlySpan<float> d, Span<float> output) =>
        Quaternary<MultiplyMultiplyAddOperator>(a, b, c, d, output);
    public static void MultiplyAddMultiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> c, ReadOnlySpan<float> d, Span<float> output) =>
        Quaternary<MultiplyAddMultiplyOperator>(a, b, c, d, output);

    public static void ReluSquare(ReadOnlySpan<float> a, Span<float> output)
    {
        Validate(a, output);
        var index = 0;
        if (Vector.IsHardwareAccelerated)
            for (; index <= output.Length - Vector<float>.Count; index += Vector<float>.Count)
            {
                var value = new Vector<float>(a[index..]);
                var positive = Vector.ConditionalSelect(Vector.GreaterThan(value, Vector<float>.Zero), value, Vector<float>.Zero);
                var relu = Vector.ConditionalSelect(Vector.Equals(value, value), positive, value);
                (relu * relu).CopyTo(output[index..]);
            }
        for (; index < output.Length; index++)
        {
            var relu = float.Max(a[index], 0f);
            output[index] = relu * relu;
        }
    }

    public static void AddRsqrt(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> output)
    {
        Validate(a, output);
        Validate(b, output);
        for (var index = 0; index < output.Length; index += BlockLength)
        {
            var count = Math.Min(BlockLength, output.Length - index);
            var destination = output.Slice(index, count);
            TensorPrimitives.Add(a.Slice(index, count), b.Slice(index, count), destination);
            TensorPrimitives.ReciprocalSqrt(destination, destination);
        }
    }

    public static void MaximumRsqrt(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> output)
    {
        Validate(a, output);
        Validate(b, output);
        for (var index = 0; index < output.Length; index += BlockLength)
        {
            var count = Math.Min(BlockLength, output.Length - index);
            var destination = output.Slice(index, count);
            TensorPrimitives.Max(a.Slice(index, count), b.Slice(index, count), destination);
            TensorPrimitives.ReciprocalSqrt(destination, destination);
        }
    }

    public static void MultiplySelfSigmoid(ReadOnlySpan<float> a, Span<float> output)
    {
        Validate(a, output);
        for (var index = 0; index < output.Length; index += BlockLength)
        {
            var count = Math.Min(BlockLength, output.Length - index);
            var input = a.Slice(index, count);
            var destination = output.Slice(index, count);
            TensorPrimitives.Sigmoid(input, destination);
            TensorPrimitives.Multiply(input, destination, destination);
        }
    }

    public static void AddSigmoid(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> output)
    {
        Validate(a, output);
        Validate(b, output);
        for (var index = 0; index < output.Length; index += BlockLength)
        {
            var count = Math.Min(BlockLength, output.Length - index);
            var destination = output.Slice(index, count);
            TensorPrimitives.Add(a.Slice(index, count), b.Slice(index, count), destination);
            TensorPrimitives.Sigmoid(destination, destination);
        }
    }

    public static void ExpSubtractExp(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> output)
    {
        Validate(a, output);
        Validate(b, output);
        for (var index = 0; index < output.Length; index += BlockLength)
        {
            var count = Math.Min(BlockLength, output.Length - index);
            var destination = output.Slice(index, count);
            TensorPrimitives.Exp(a.Slice(index, count), destination);
            TensorPrimitives.Subtract(b.Slice(index, count), destination, destination);
            TensorPrimitives.Exp(destination, destination);
        }
    }

    public static void SigmoidMultiplyExp(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> output)
    {
        Validate(a, output);
        Validate(b, output);
        for (var index = 0; index < output.Length; index += BlockLength)
        {
            var count = Math.Min(BlockLength, output.Length - index);
            var destination = output.Slice(index, count);
            TensorPrimitives.Sigmoid(a.Slice(index, count), destination);
            TensorPrimitives.Multiply(b.Slice(index, count), destination, destination);
            TensorPrimitives.Exp(destination, destination);
        }
    }

    private static void Validate(ReadOnlySpan<float> input, Span<float> output)
    {
        if (output.IsEmpty || input.Length != output.Length)
            throw new ArgumentException("T1 pointwise spans must have identical positive lengths.");
        if (input.Overlaps(output))
            throw new ArgumentException("T1 pointwise output must not overlap its inputs.");
    }

    private interface ITernaryOperator
    {
        static abstract float Invoke(float a, float b, float c);
        static abstract Vector<float> Invoke(Vector<float> a, Vector<float> b, Vector<float> c);
    }

    private interface IQuaternaryOperator
    {
        static abstract float Invoke(float a, float b, float c, float d);
        static abstract Vector<float> Invoke(Vector<float> a, Vector<float> b, Vector<float> c, Vector<float> d);
    }

    private static void Ternary<TOperator>(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> c, Span<float> output)
        where TOperator : struct, ITernaryOperator
    {
        Validate(a, output);
        Validate(b, output);
        Validate(c, output);
        var index = 0;
        if (Vector.IsHardwareAccelerated)
            for (; index <= output.Length - Vector<float>.Count; index += Vector<float>.Count)
                TOperator.Invoke(new Vector<float>(a[index..]), new Vector<float>(b[index..]), new Vector<float>(c[index..])).CopyTo(output[index..]);
        for (; index < output.Length; index++)
            output[index] = TOperator.Invoke(a[index], b[index], c[index]);
    }

    private static void Quaternary<TOperator>(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> c,
        ReadOnlySpan<float> d, Span<float> output) where TOperator : struct, IQuaternaryOperator
    {
        Validate(a, output);
        Validate(b, output);
        Validate(c, output);
        Validate(d, output);
        var index = 0;
        if (Vector.IsHardwareAccelerated)
            for (; index <= output.Length - Vector<float>.Count; index += Vector<float>.Count)
                TOperator.Invoke(new Vector<float>(a[index..]), new Vector<float>(b[index..]), new Vector<float>(c[index..]), new Vector<float>(d[index..])).CopyTo(output[index..]);
        for (; index < output.Length; index++)
            output[index] = TOperator.Invoke(a[index], b[index], c[index], d[index]);
    }

    private readonly struct DifferenceMixOperator : ITernaryOperator
    {
        public static float Invoke(float a, float b, float c) { var difference = a - b; var product = difference * c; return b + product; }
        public static Vector<float> Invoke(Vector<float> a, Vector<float> b, Vector<float> c) { var difference = a - b; var product = difference * c; return b + product; }
    }

    private readonly struct MultiplyAddSubtractOperator : ITernaryOperator
    {
        public static float Invoke(float a, float b, float c) { var product = a * b; var sum = product + c; return sum - b; }
        public static Vector<float> Invoke(Vector<float> a, Vector<float> b, Vector<float> c) { var product = a * b; var sum = product + c; return sum - b; }
    }

    private readonly struct AddMultiplyAddOperator : IQuaternaryOperator
    {
        public static float Invoke(float a, float b, float c, float d) { var sum = a + b; var product = sum * c; return product + d; }
        public static Vector<float> Invoke(Vector<float> a, Vector<float> b, Vector<float> c, Vector<float> d) { var sum = a + b; var product = sum * c; return product + d; }
    }

    private readonly struct MultiplyMultiplyAddOperator : IQuaternaryOperator
    {
        public static float Invoke(float a, float b, float c, float d) { var first = a * b; var second = first * c; return second + d; }
        public static Vector<float> Invoke(Vector<float> a, Vector<float> b, Vector<float> c, Vector<float> d) { var first = a * b; var second = first * c; return second + d; }
    }

    private readonly struct MultiplyAddMultiplyOperator : IQuaternaryOperator
    {
        public static float Invoke(float a, float b, float c, float d) { var product = a * b; var sum = product + c; return sum * d; }
        public static Vector<float> Invoke(Vector<float> a, Vector<float> b, Vector<float> c, Vector<float> d) { var product = a * b; var sum = product + c; return sum * d; }
    }
}
