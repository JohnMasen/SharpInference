using System.Diagnostics;

namespace SharpInference.Backends.Cpu;

/// <summary>Opt-in per-thread CPU graph operator timing; inactive executions have no timer calls.</summary>
public sealed class CpuPrimitiveGraphProfiler : IDisposable
{
    [ThreadStatic]
    private static CpuPrimitiveGraphProfiler? active;

    private readonly CpuPrimitiveGraphProfiler? previous;
    private readonly Dictionary<string, (long Ticks, int Calls)> timings = [];
    private bool disposed;

    private CpuPrimitiveGraphProfiler()
    {
        previous = active;
        active = this;
    }

    public static CpuPrimitiveGraphProfiler Start() => new();

    internal static CpuPrimitiveGraphProfiler? Active => active;

    internal void Record(string operation, long started)
    {
        var elapsed = Stopwatch.GetTimestamp() - started;
        var current = timings.GetValueOrDefault(operation);
        timings[operation] = (current.Ticks + elapsed, current.Calls + 1);
    }

    public IReadOnlyList<CpuGraphOperationTiming> Snapshot() => timings
        .Select(entry => new CpuGraphOperationTiming(entry.Key, entry.Value.Calls,
            entry.Value.Ticks * 1000.0 / Stopwatch.Frequency))
        .OrderByDescending(entry => entry.Milliseconds).ToArray();

    public void Dispose()
    {
        if (disposed) return;
        if (!ReferenceEquals(active, this))
            throw new InvalidOperationException("CPU graph profilers must be disposed on their originating thread.");
        active = previous;
        disposed = true;
    }
}

public sealed record CpuGraphOperationTiming(string Operation, int Calls, double Milliseconds);
