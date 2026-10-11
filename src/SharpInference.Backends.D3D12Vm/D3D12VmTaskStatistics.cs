namespace SharpInference.Backends.D3D12Vm;

/// <summary>Captures cumulative completed-task counts, transfer volumes, and execution phase timings for a resource pool.</summary>
/// <param name="CompletedTasks">The cumulative number of successfully completed tasks.</param>
/// <param name="ComputeRecordings">The cumulative number of compute entry command lists recorded for tasks.</param>
/// <param name="InputUploadBytes">The cumulative bytes uploaded for task input tensors.</param>
/// <param name="InitialStateUploadBytes">The cumulative bytes uploaded to initialize GPU session state.</param>
/// <param name="InitialLocalUploadBytes">The cumulative bytes uploaded to initialize GPU local storage.</param>
/// <param name="StateLoadBytes">The cumulative bytes copied to load task session state on the GPU.</param>
/// <param name="StateCommitBytes">The cumulative bytes copied to commit task session state on the GPU.</param>
/// <param name="CpuReadbackBytes">The cumulative GPU bytes read back to CPU storage through the pool.</param>
/// <param name="PreparationMilliseconds">The accumulated task preparation wall-clock time in milliseconds.</param>
/// <param name="ComputeMilliseconds">The accumulated task compute phase wall-clock time in milliseconds.</param>
/// <param name="CommitMilliseconds">The accumulated task commit phase wall-clock time in milliseconds.</param>
public sealed record D3D12VmTaskStatistics(
    ulong CompletedTasks, ulong ComputeRecordings, ulong InputUploadBytes,
    ulong InitialStateUploadBytes, ulong InitialLocalUploadBytes, ulong StateLoadBytes, ulong StateCommitBytes,
    ulong CpuReadbackBytes, double PreparationMilliseconds, double ComputeMilliseconds,
    double CommitMilliseconds);
