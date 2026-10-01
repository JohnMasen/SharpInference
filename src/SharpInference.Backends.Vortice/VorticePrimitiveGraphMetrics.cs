namespace SharpInference.Backends.Vortice;

/// <summary>Recorded GPU work and host/device traffic for one completed token submission.</summary>
public sealed record VorticePrimitiveGraphTokenMetrics(
    int CommandSubmissions,
    int NodeDispatches,
    ulong ActivationUploadBytes,
    ulong OutputReadbackBytes);

/// <summary>CPU command-recording cost per kernel, not GPU timestamp-query execution time.</summary>
public sealed record VorticePrimitiveGraphKernelRecording(
    string Kernel, int Dispatches, double CpuRecordingMilliseconds);

/// <summary>Optional per-token CPU recording profile and total submit/fence-wait wall time.</summary>
public sealed record VorticePrimitiveGraphExecutionProfile(
    double CpuRecordingMilliseconds,
    double SubmissionAndFenceMilliseconds,
    IReadOnlyList<VorticePrimitiveGraphKernelRecording> Kernels);

/// <summary>
/// Cumulative GPU command submissions and host/device transfer bytes for one graph executor.
/// Readback counts include only graph outputs and explicitly requested session-state snapshots.
/// </summary>
public sealed record VorticePrimitiveGraphMetrics(
    ulong TokenCount,
    ulong TokenCommandSubmissions,
    ulong WeightCommandSubmissions,
    ulong StateUploadCommandSubmissions,
    ulong StateReadbackCommandSubmissions,
    ulong WeightUploadBytes,
    ulong ActivationUploadBytes,
    ulong StateUploadBytes,
    ulong OutputReadbackBytes,
    ulong StateReadbackBytes)
{
    public ulong ReplayListRecordings { get; init; }
    public ulong ReplayedTokens { get; init; }
    public ulong ScalarUploadBytes { get; init; }
}

/// <summary>
/// Committed default-heap buffer sizes. Excludes transient upload/readback heaps, driver
/// overhead and residency decisions; this is not a DXGI physical-VRAM usage measurement.
/// </summary>
public sealed record VorticePrimitiveGraphMemoryMetrics(
    ulong ModelBufferBytes,
    ulong ActiveSessionBufferBytes,
    ulong PeakBufferBytes,
    ulong PooledLocalBufferBytesPerSession,
    ulong LogicalCastOutputBytes);
