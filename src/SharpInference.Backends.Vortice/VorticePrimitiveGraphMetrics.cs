namespace SharpInference.Backends.Vortice;

/// <summary>Recorded GPU work and host/device traffic for one completed token submission.</summary>
/// <param name="CommandSubmissions">The number of command lists submitted for the token.</param>
/// <param name="NodeDispatches">The number of graph node dispatches executed for the token.</param>
/// <param name="ActivationUploadBytes">The Float32 activation bytes uploaded from the host.</param>
/// <param name="OutputReadbackBytes">The graph output bytes read back to the host.</param>
public sealed record VorticePrimitiveGraphTokenMetrics(
    int CommandSubmissions,
    int NodeDispatches,
    ulong ActivationUploadBytes,
    ulong OutputReadbackBytes);

/// <summary>CPU command-recording cost per kernel, not GPU timestamp-query execution time.</summary>
/// <param name="Kernel">The GPU kernel name selected for this dispatch.</param>
/// <param name="Dispatches">The number of dispatches recorded for this kernel.</param>
/// <param name="CpuRecordingMilliseconds">The CPU time spent recording commands in milliseconds.</param>
public sealed record VorticePrimitiveGraphKernelRecording(
    string Kernel, int Dispatches, double CpuRecordingMilliseconds);

/// <summary>Optional per-token CPU recording profile and total submit/fence-wait wall time.</summary>
/// <param name="CpuRecordingMilliseconds">The CPU time spent recording commands in milliseconds.</param>
/// <param name="SubmissionAndFenceMilliseconds">The wall-clock submission and fence-wait time in milliseconds.</param>
/// <param name="Kernels">The per-kernel CPU recording measurements.</param>
public sealed record VorticePrimitiveGraphExecutionProfile(
    double CpuRecordingMilliseconds,
    double SubmissionAndFenceMilliseconds,
    IReadOnlyList<VorticePrimitiveGraphKernelRecording> Kernels);

/// <summary>
/// Cumulative GPU command submissions and host/device transfer bytes for one graph executor.
/// Readback counts include only graph outputs and explicitly requested session-state snapshots.
/// </summary>
/// <param name="TokenCount">The cumulative number of completed tokens.</param>
/// <param name="TokenCommandSubmissions">The cumulative command submissions used to execute tokens.</param>
/// <param name="WeightCommandSubmissions">The cumulative command submissions used to upload weights.</param>
/// <param name="StateUploadCommandSubmissions">The cumulative command submissions used to upload session state.</param>
/// <param name="StateReadbackCommandSubmissions">The cumulative command submissions used to read session state.</param>
/// <param name="WeightUploadBytes">The cumulative weight bytes uploaded to GPU model buffers.</param>
/// <param name="ActivationUploadBytes">The Float32 activation bytes uploaded from the host.</param>
/// <param name="StateUploadBytes">The cumulative session-state bytes uploaded to the GPU.</param>
/// <param name="OutputReadbackBytes">The graph output bytes read back to the host.</param>
/// <param name="StateReadbackBytes">The cumulative session-state bytes read back to the host.</param>
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
    /// <summary>Gets the cumulative number of per-session closed replay lists recorded.</summary>
    public ulong ReplayListRecordings { get; init; }
    /// <summary>Gets the cumulative number of tokens executed using closed-list replay.</summary>
    public ulong ReplayedTokens { get; init; }
    /// <summary>Gets the cumulative Int32 scalar input bytes uploaded for replayed tokens.</summary>
    public ulong ScalarUploadBytes { get; init; }
}

/// <summary>
/// Committed default-heap buffer sizes. Excludes transient upload/readback heaps, driver
/// overhead and residency decisions; this is not a DXGI physical-VRAM usage measurement.
/// </summary>
/// <param name="ModelBufferBytes">The committed default-heap buffer bytes for model resources.</param>
/// <param name="ActiveSessionBufferBytes">The committed default-heap buffer bytes for active sessions.</param>
/// <param name="PeakBufferBytes">The peak combined model and session default-heap buffer bytes.</param>
/// <param name="PooledLocalBufferBytesPerSession">The reusable token-local buffer capacity allocated per session.</param>
/// <param name="LogicalCastOutputBytes">The logical output bytes for cast nodes before buffer elision.</param>
public sealed record VorticePrimitiveGraphMemoryMetrics(
    ulong ModelBufferBytes,
    ulong ActiveSessionBufferBytes,
    ulong PeakBufferBytes,
    ulong PooledLocalBufferBytesPerSession,
    ulong LogicalCastOutputBytes);
