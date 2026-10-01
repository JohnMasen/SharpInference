using System.Diagnostics;

namespace SharpInference.PrefillExperiment;

internal sealed class Wkv6Inputs
{
    public int HeadCount { get; }
    public int HeadSize { get; }
    public int TokenCount { get; }
    public float[] Keys { get; }
    public float[] Values { get; }
    public float[] Decays { get; }
    public float[] InitialState { get; }
    public int EmbeddingSize => checked(HeadCount * HeadSize);
    public int StateSize => checked(EmbeddingSize * HeadSize);

    public Wkv6Inputs(int headCount, int headSize, int tokenCount, bool initialize = true)
    {
        if (headCount <= 0 || headSize <= 0 || tokenCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "All dimensions must be positive.");
        HeadCount = headCount;
        HeadSize = headSize;
        TokenCount = tokenCount;
        Keys = new float[checked(tokenCount * EmbeddingSize)];
        Values = new float[Keys.Length];
        Decays = new float[Keys.Length];
        InitialState = new float[StateSize];
        if (!initialize)
            return;
        var random = new Random(1701);
        for (var i = 0; i < Keys.Length; i++)
        {
            Keys[i] = (float)(random.NextDouble() * 0.1 - 0.05);
            Values[i] = (float)(random.NextDouble() * 0.1 - 0.05);
            Decays[i] = (float)(0.85 + random.NextDouble() * 0.14);
        }
        for (var i = 0; i < InitialState.Length; i++)
            InitialState[i] = (float)(random.NextDouble() * 0.02 - 0.01);
    }
}

internal static class CpuWkv6Benchmark
{
    public static float[] Sequential(Wkv6Inputs inputs, int tokenCount)
    {
        var state = (float[])inputs.InitialState.Clone();
        for (var token = 0; token < tokenCount; token++)
        {
            for (var head = 0; head < inputs.HeadCount; head++)
            for (var row = 0; row < inputs.HeadSize; row++)
            {
                var vector = token * inputs.EmbeddingSize + head * inputs.HeadSize;
                var stateRow = (head * inputs.HeadSize + row) * inputs.HeadSize;
                var key = inputs.Keys[vector + row];
                var decay = inputs.Decays[vector + row];
                for (var col = 0; col < inputs.HeadSize; col++)
                    state[stateRow + col] = state[stateRow + col] * decay +
                        key * inputs.Values[vector + col];
            }
        }
        return state;
    }

    public static float[] Chunked(Wkv6Inputs inputs, int chunkSize, int tokenCount)
    {
        if (chunkSize <= 0 || tokenCount < 0 || tokenCount > inputs.TokenCount)
            throw new ArgumentOutOfRangeException(nameof(chunkSize));
        var chunks = (tokenCount + chunkSize - 1) / chunkSize;
        var stateSize = inputs.StateSize;
        var multipliers = new float[checked(chunks * stateSize)];
        var additions = new float[multipliers.Length];

        Parallel.For(0, chunks, chunk =>
        {
            var start = chunk * chunkSize;
            var end = Math.Min(start + chunkSize, tokenCount);
            for (var head = 0; head < inputs.HeadCount; head++)
            for (var row = 0; row < inputs.HeadSize; row++)
            for (var col = 0; col < inputs.HeadSize; col++)
            {
                var index = (head * inputs.HeadSize + row) * inputs.HeadSize + col;
                var a = 1f;
                var b = 0f;
                for (var token = start; token < end; token++)
                {
                    var vector = token * inputs.EmbeddingSize + head * inputs.HeadSize;
                    var decay = inputs.Decays[vector + row];
                    b = b * decay + inputs.Keys[vector + row] * inputs.Values[vector + col];
                    a *= decay;
                }
                multipliers[chunk * stateSize + index] = a;
                additions[chunk * stateSize + index] = b;
            }
        });

        var state = new float[stateSize];
        Parallel.For(0, stateSize, index =>
        {
            var value = inputs.InitialState[index];
            for (var chunk = 0; chunk < chunks; chunk++)
            {
                var offset = chunk * stateSize + index;
                value = value * multipliers[offset] + additions[offset];
            }
            state[index] = value;
        });
        return state;
    }

    public static void Validate(Wkv6Inputs inputs, int chunkSize)
    {
        var baseline = Sequential(inputs, inputs.TokenCount);
        AssertClose(baseline, Chunked(inputs, chunkSize, inputs.TokenCount), "CPU final state");

        var previous = Chunked(inputs, chunkSize, inputs.TokenCount - 1);
        var serialPrevious = Sequential(inputs, inputs.TokenCount - 1);
        var last = inputs.TokenCount - 1;
        var vector = last * inputs.EmbeddingSize;
        var serialOutput = new float[inputs.EmbeddingSize];
        var chunkOutput = new float[inputs.EmbeddingSize];
        for (var head = 0; head < inputs.HeadCount; head++)
        for (var row = 0; row < inputs.HeadSize; row++)
        for (var col = 0; col < inputs.HeadSize; col++)
        {
            var index = (head * inputs.HeadSize + row) * inputs.HeadSize + col;
            var input = vector + head * inputs.HeadSize;
            var keyValue = inputs.Keys[input + row] * inputs.Values[input + col];
            var receptance = 0.7f + row * 0.001f;
            var timeFirst = 0.1f + row * 0.001f;
            serialOutput[head * inputs.HeadSize + col] +=
                (serialPrevious[index] + keyValue * timeFirst) * receptance;
            chunkOutput[head * inputs.HeadSize + col] +=
                (previous[index] + keyValue * timeFirst) * receptance;
            previous[index] = previous[index] * inputs.Decays[input + row] +
                keyValue;
        }
        AssertClose(serialOutput, chunkOutput, "CPU last-token attention output");
        AssertClose(baseline, previous, "CPU prefix state + last token");
    }

    public static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string label)
    {
        if (expected.Length != actual.Length)
            throw new InvalidOperationException($"{label}: state length mismatch.");
        for (var i = 0; i < expected.Length; i++)
            if (!float.IsFinite(actual[i]) ||
                MathF.Abs(expected[i] - actual[i]) > 0.00002f + MathF.Abs(expected[i]) * 0.0001f)
                throw new InvalidOperationException(
                    $"{label}: mismatch at {i}: sequential={expected[i]:G9}, chunked={actual[i]:G9}.");
    }

    public static (TimeSpan Sequential, TimeSpan Chunked) Measure(Wkv6Inputs inputs, int chunkSize, int repeats)
    {
        _ = Sequential(inputs, inputs.TokenCount);
        _ = Chunked(inputs, chunkSize, inputs.TokenCount);
        var sequential = TimeSpan.Zero;
        var chunked = TimeSpan.Zero;
        for (var run = 0; run < repeats; run++)
        {
            var clock = Stopwatch.StartNew();
            _ = Sequential(inputs, inputs.TokenCount);
            sequential += clock.Elapsed;
            clock.Restart();
            _ = Chunked(inputs, chunkSize, inputs.TokenCount);
            chunked += clock.Elapsed;
        }
        return (sequential / repeats, chunked / repeats);
    }
}
