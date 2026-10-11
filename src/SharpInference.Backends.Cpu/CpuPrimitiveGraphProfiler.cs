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

    /// <summary>Starts profiling graph operations on the current thread, preserving any enclosing profiler.</summary>
    public static CpuPrimitiveGraphProfiler Start() => new();

    internal static CpuPrimitiveGraphProfiler? Active => active;

    internal void Record(string operation, long started)
    {
        var elapsed = Stopwatch.GetTimestamp() - started;
        var current = timings.GetValueOrDefault(operation);
        timings[operation] = (current.Ticks + elapsed, current.Calls + 1);
    }

    /// <summary>Returns recorded operation timings ordered by decreasing elapsed time.</summary>
    public IReadOnlyList<CpuGraphOperationTiming> Snapshot() => timings
        .Select(entry => new CpuGraphOperationTiming(entry.Key, entry.Value.Calls,
            entry.Value.Ticks * 1000.0 / Stopwatch.Frequency))
        .OrderByDescending(entry => entry.Milliseconds).ToArray();

    /// <inheritdoc/>
    public void Dispose()
    {
        if (disposed) return;
        if (!ReferenceEquals(active, this))
            throw new InvalidOperationException("CPU graph profilers must be disposed on their originating thread.");
        active = previous;
        disposed = true;
    }
}

/// <summary>Reports cumulative call count and elapsed CPU time for one graph operation.</summary>
/// <param name="Operation">The graph operation name.</param>
/// <param name="Calls">The number of recorded invocations.</param>
/// <param name="Milliseconds">The accumulated CPU elapsed time in milliseconds.</param>
public sealed record CpuGraphOperationTiming(string Operation, int Calls, double Milliseconds);
