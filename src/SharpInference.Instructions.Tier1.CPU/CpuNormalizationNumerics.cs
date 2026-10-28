namespace SharpInference.Instructions.Cpu;

/// <summary>Scalar reference implementations of portable normalization fusions.</summary>
public static class CpuNormalizationNumerics
{
    public static void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, Span<float> output, float epsilon)
    {
        ValidateLengths(input.Length, weight.Length, output.Length);
        Positive(epsilon);
        var sum = 0f;
        foreach (var value in input)
        {
            sum += value * value;
        }

        var scale = 1f / MathF.Sqrt(sum / input.Length + epsilon);
        for (var i = 0; i < output.Length; i++)
        {
            output[i] = input[i] * scale * weight[i];
        }
    }

    public static void ResidualRmsNorm(ReadOnlySpan<float> hidden, ReadOnlySpan<float> residual,
        ReadOnlySpan<float> weight, Span<float> updated, Span<float> output, float epsilon)
    {
        ValidateLengths(hidden.Length, residual.Length, updated.Length);
        ValidateLengths(hidden.Length, weight.Length, output.Length);
        Positive(epsilon);
        for (var i = 0; i < updated.Length; i++)
        {
            updated[i] = hidden[i] + residual[i];
        }

        RmsNorm(updated, weight, output, epsilon);
    }

    public static void SwiGlu(ReadOnlySpan<float> gateUp, Span<float> output)
    {
        if (output.Length == 0 || gateUp.Length != checked(output.Length * 2))
        {
            throw new ArgumentException("Expected two contiguous gate/up halves.");
        }

        for (var i = 0; i < output.Length; i++)
        {
            var gate = gateUp[i];
            output[i] = gate / (1f + MathF.Exp(-gate)) * gateUp[i + output.Length];
        }
    }

    public static void ScaledAdd(ReadOnlySpan<float> input, ReadOnlySpan<float> update, Span<float> output, float scale)
    {
        ValidateLengths(input.Length, update.Length, output.Length);
        Positive(scale);
        for (var i = 0; i < output.Length; i++)
        {
            output[i] = input[i] + scale * update[i];
        }
    }

    public static void BatchRmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weight,
        ReadOnlySpan<int> control, Span<float> output, int batch, int width, float epsilon)
    {
        var validRows = ValidateBatch(control, batch, width, input.Length, output.Length);
        if (weight.Length != width)
        {
            throw new ArgumentException("Weight must have shape [H].");
        }

        Positive(epsilon);
        for (var row = 0; row < validRows; row++)
        {
            RmsNorm(input.Slice(row * width, width), weight, output.Slice(row * width, width), epsilon);
        }
    }

    public static void BatchResidualRmsNorm(ReadOnlySpan<float> hidden, ReadOnlySpan<float> residual,
        ReadOnlySpan<float> weight, ReadOnlySpan<int> control, Span<float> updated,
        Span<float> output, int batch, int width, float epsilon)
    {
        var validRows = ValidateBatch(control, batch, width, hidden.Length, output.Length);
        ValidateLengths(hidden.Length, residual.Length, updated.Length);
        if (weight.Length != width)
        {
            throw new ArgumentException("Weight must have shape [H].");
        }

        Positive(epsilon);
        for (var row = 0; row < validRows; row++)
        {
            ResidualRmsNorm(hidden.Slice(row * width, width), residual.Slice(row * width, width),
                weight, updated.Slice(row * width, width), output.Slice(row * width, width), epsilon);
        }
    }

    public static void BatchSwiGlu(ReadOnlySpan<float> gateUp, ReadOnlySpan<int> control,
        Span<float> output, int batch, int width)
    {
        if (gateUp.Length != checked(output.Length * 2))
        {
            throw new ArgumentException("Gate/up must have shape [B,2H].");
        }

        var validRows = ValidateBatch(control, batch, width, output.Length, output.Length);
        for (var row = 0; row < validRows; row++)
        {
            SwiGlu(gateUp.Slice(checked(row * width * 2), checked(width * 2)),
                output.Slice(row * width, width));
        }
    }

    private static int ValidateBatch(ReadOnlySpan<int> control, int batch, int width, int input, int output)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batch);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        if (control.Length != 2 || input != checked(batch * width) || output != input)
        {
            throw new ArgumentException("Expected tensors [B,H] and control [2].");
        }

        return Math.Clamp(control[1], 0, batch);
    }

    private static void ValidateLengths(int input, int weight, int output)
    {
        if (input == 0 || weight != input || output != input)
        {
            throw new ArgumentException("Tensor element counts must match and be positive.");
        }
    }

    private static void Positive(float value)
    {
        if (!float.IsFinite(value) || value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(value));
        }
    }
}
