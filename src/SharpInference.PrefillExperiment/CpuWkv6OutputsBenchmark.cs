using System.Diagnostics;

namespace SharpInference.PrefillExperiment;

internal sealed class Wkv6Sequence(Wkv6Inputs inputs)
{
    public Wkv6Inputs Inputs { get; } = inputs;
    public float[] Receptances { get; } = Create(inputs);
    public float[] TimeFirst { get; } = Enumerable.Range(0, inputs.EmbeddingSize)
        .Select(i => 0.1f + i % inputs.HeadSize * 0.001f).ToArray();

    private static float[] Create(Wkv6Inputs inputs)
    {
        var values = new float[inputs.Keys.Length];
        for (var i = 0; i < values.Length; i++)
            values[i] = 0.6f + i % inputs.HeadSize * 0.001f;
        return values;
    }
}

internal sealed record Wkv6SequenceResult(float[] Outputs, float[] FinalState);

internal static class CpuWkv6OutputsBenchmark
{
    public static Wkv6SequenceResult Sequential(Wkv6Sequence sequence)
    {
        var inputs = sequence.Inputs;
        var state = (float[])inputs.InitialState.Clone();
        var outputs = new float[inputs.Keys.Length];
        for (var token = 0; token < inputs.TokenCount; token++)
        {
            for (var head = 0; head < inputs.HeadCount; head++)
            for (var row = 0; row < inputs.HeadSize; row++)
            for (var col = 0; col < inputs.HeadSize; col++)
            {
                var channel = head * inputs.HeadSize + row;
                var valueChannel = head * inputs.HeadSize + col;
                var vector = token * inputs.EmbeddingSize;
                var index = channel * inputs.HeadSize + col;
                var keyValue = inputs.Keys[vector + channel] * inputs.Values[vector + valueChannel];
                var previous = state[index];
                outputs[vector + valueChannel] +=
                    (previous + keyValue * sequence.TimeFirst[channel]) *
                    sequence.Receptances[vector + channel];
                state[index] = previous * inputs.Decays[vector + channel] + keyValue;
            }
        }
        return new Wkv6SequenceResult(outputs, state);
    }

    public static Wkv6SequenceResult Chunked(Wkv6Sequence sequence, int chunkSize)
    {
        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSize));
        var inputs = sequence.Inputs;
        var chunks = (inputs.TokenCount - 1) / chunkSize + 1;
        var stateSize = inputs.StateSize;
        var summaryA = new float[checked(chunks * stateSize)];
        var summaryB = new float[summaryA.Length];
        Parallel.For(0, chunks, chunk =>
        {
            var start = chunk * chunkSize;
            var end = Math.Min(inputs.TokenCount, start + chunkSize);
            for (var head = 0; head < inputs.HeadCount; head++)
            for (var row = 0; row < inputs.HeadSize; row++)
            for (var col = 0; col < inputs.HeadSize; col++)
            {
                var index = (head * inputs.HeadSize + row) * inputs.HeadSize + col;
                var a = 1f;
                var b = 0f;
                for (var token = start; token < end; token++)
                {
                    var channel = token * inputs.EmbeddingSize + head * inputs.HeadSize;
                    var decay = inputs.Decays[channel + row];
                    b = b * decay + inputs.Keys[channel + row] * inputs.Values[channel + col];
                    a *= decay;
                }
                summaryA[chunk * stateSize + index] = a;
                summaryB[chunk * stateSize + index] = b;
            }
        });

        var chunkStates = new float[checked(chunks * stateSize)];
        Parallel.For(0, stateSize, index =>
        {
            var current = inputs.InitialState[index];
            for (var chunk = 0; chunk < chunks; chunk++)
            {
                var offset = chunk * stateSize + index;
                chunkStates[offset] = current;
                current = current * summaryA[offset] + summaryB[offset];
            }
        });

        var outputs = new float[inputs.Keys.Length];
        Parallel.For(0, chunks, chunk =>
        {
            var start = chunk * chunkSize;
            var end = Math.Min(inputs.TokenCount, start + chunkSize);
            var stateStart = chunk * stateSize;
            for (var token = start; token < end; token++)
            for (var head = 0; head < inputs.HeadCount; head++)
            for (var row = 0; row < inputs.HeadSize; row++)
            for (var col = 0; col < inputs.HeadSize; col++)
            {
                var channel = head * inputs.HeadSize + row;
                var valueChannel = head * inputs.HeadSize + col;
                var vector = token * inputs.EmbeddingSize;
                var index = stateStart + channel * inputs.HeadSize + col;
                var keyValue = inputs.Keys[vector + channel] * inputs.Values[vector + valueChannel];
                var previous = chunkStates[index];
                outputs[vector + valueChannel] +=
                    (previous + keyValue * sequence.TimeFirst[channel]) *
                    sequence.Receptances[vector + channel];
                chunkStates[index] = previous * inputs.Decays[vector + channel] + keyValue;
            }
        });
        return new Wkv6SequenceResult(outputs, chunkStates.AsSpan((chunks - 1) * stateSize, stateSize).ToArray());
    }

    public static (TimeSpan Sequential, TimeSpan Chunked) Measure(
        Wkv6Sequence sequence, int chunkSize, int repeats)
    {
        var expected = Sequential(sequence);
        var actual = Chunked(sequence, chunkSize);
        CpuWkv6Benchmark.AssertClose(expected.Outputs, actual.Outputs, "CPU all WKV outputs");
        CpuWkv6Benchmark.AssertClose(expected.FinalState, actual.FinalState, "CPU all-output final state");

        var sequential = TimeSpan.Zero;
        var chunked = TimeSpan.Zero;
        for (var run = 0; run < repeats; run++)
        {
            var clock = Stopwatch.StartNew();
            _ = Sequential(sequence);
            sequential += clock.Elapsed;
            clock.Restart();
            _ = Chunked(sequence, chunkSize);
            chunked += clock.Elapsed;
        }
        return (sequential / repeats, chunked / repeats);
    }
}
