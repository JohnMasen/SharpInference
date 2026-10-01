using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;

namespace SharpInference.PrefillExperiment;

internal static class CpuProjectionBenchmark
{
    public static (TimeSpan Sequential, TimeSpan Batched) Measure(int tokens, int width, int batch, int repeats)
    {
        if (tokens <= 0 || width <= 0 || batch <= 0 || repeats <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokens));
        var random = new Random(1701);
        var weights = new float[checked(width * width)];
        var input = new float[checked(tokens * width)];
        for (var i = 0; i < weights.Length; i++)
            weights[i] = (float)(random.NextDouble() * 0.02 - 0.01);
        for (var i = 0; i < input.Length; i++)
            input[i] = (float)(random.NextDouble() * 0.2 - 0.1);

        var baseline = Sequential(weights, input, tokens, width);
        CpuWkv6Benchmark.AssertClose(baseline, Batched(weights, input, tokens, width, batch),
            "CPU projected activations");
        var sequential = TimeSpan.Zero;
        var batched = TimeSpan.Zero;
        for (var run = 0; run < repeats; run++)
        {
            var watch = Stopwatch.StartNew();
            _ = Sequential(weights, input, tokens, width);
            sequential += watch.Elapsed;
            watch.Restart();
            _ = Batched(weights, input, tokens, width, batch);
            batched += watch.Elapsed;
        }
        return (sequential / repeats, batched / repeats);
    }

    public static (TimeSpan Sequential, TimeSpan Batched) MeasureVectorized(
        int tokens, int width, int batch, int repeats)
    {
        if (tokens <= 0 || width <= 0 || batch <= 0 || repeats <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokens));
        var random = new Random(1701);
        var weights = new float[checked(width * width)];
        var input = new float[checked(tokens * width)];
        for (var i = 0; i < weights.Length; i++)
            weights[i] = (float)(random.NextDouble() * 0.02 - 0.01);
        for (var i = 0; i < input.Length; i++)
            input[i] = (float)(random.NextDouble() * 0.2 - 0.1);

        var reference = Sequential(weights, input, tokens, width);
        CpuWkv6Benchmark.AssertClose(reference, Vectorized(weights, input, tokens, width, batch),
            "CPU SIMD projected activations");
        var sequential = TimeSpan.Zero;
        var vectorized = TimeSpan.Zero;
        for (var run = 0; run < repeats; run++)
        {
            var clock = Stopwatch.StartNew();
            _ = SequentialDot(weights, input, tokens, width);
            sequential += clock.Elapsed;
            clock.Restart();
            _ = Vectorized(weights, input, tokens, width, batch);
            vectorized += clock.Elapsed;
        }
        return (sequential / repeats, vectorized / repeats);
    }

    private static float[] SequentialDot(float[] weights, float[] input, int tokens, int width)
    {
        var output = new float[checked(tokens * width)];
        for (var token = 0; token < tokens; token++)
        for (var row = 0; row < width; row++)
            output[token * width + row] = TensorPrimitives.Dot(
                weights.AsSpan(row * width, width),
                input.AsSpan(token * width, width));
        return output;
    }

    private static float[] Sequential(float[] weights, float[] input, int tokens, int width)
    {
        var output = new float[checked(tokens * width)];
        for (var token = 0; token < tokens; token++)
        for (var row = 0; row < width; row++)
        {
            var sum = 0f;
            for (var col = 0; col < width; col++)
                sum += weights[row * width + col] * input[token * width + col];
            output[token * width + row] = sum;
        }
        return output;
    }

    private static float[] Batched(float[] weights, float[] input, int tokens, int width, int batch)
    {
        var output = new float[checked(tokens * width)];
        var sums = new float[Math.Min(batch, tokens)];
        for (var start = 0; start < tokens; start += batch)
        {
            var count = Math.Min(batch, tokens - start);
            for (var row = 0; row < width; row++)
            {
                Array.Clear(sums, 0, count);
                for (var col = 0; col < width; col++)
                {
                    var weight = weights[row * width + col];
                    for (var t = 0; t < count; t++)
                        sums[t] += weight * input[(start + t) * width + col];
                }
                for (var t = 0; t < count; t++)
                    output[(start + t) * width + row] = sums[t];
            }
        }
        return output;
    }

    private static float[] Vectorized(float[] weights, float[] input, int tokens, int width, int batch)
    {
        var stride = checked((batch + Vector<float>.Count - 1) / Vector<float>.Count * Vector<float>.Count);
        var blocks = (tokens - 1) / batch + 1;
        var transposed = new float[checked(blocks * width * stride)];
        for (var token = 0; token < tokens; token++)
        for (var col = 0; col < width; col++)
        {
            var block = token / batch;
            transposed[(block * width + col) * stride + token % batch] = input[token * width + col];
        }

        var output = new float[checked(tokens * width)];
        var accumulators = new Vector<float>[stride / Vector<float>.Count];
        var partial = new float[stride];
        for (var block = 0; block < blocks; block++)
        {
            var start = block * batch;
            var count = Math.Min(batch, tokens - start);
            for (var row = 0; row < width; row++)
            {
                Array.Clear(accumulators);
                for (var col = 0; col < width; col++)
                {
                    var weight = new Vector<float>(weights[row * width + col]);
                    for (var lane = 0; lane < accumulators.Length; lane++)
                        accumulators[lane] += weight *
                            new Vector<float>(transposed, (block * width + col) * stride +
                                lane * Vector<float>.Count);
                }
                for (var lane = 0; lane < accumulators.Length; lane++)
                    accumulators[lane].CopyTo(partial, lane * Vector<float>.Count);
                for (var token = 0; token < count; token++)
                    output[(start + token) * width + row] = partial[token];
            }
        }
        return output;
    }
}
