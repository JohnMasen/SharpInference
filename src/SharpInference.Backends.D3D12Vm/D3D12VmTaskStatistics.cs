namespace SharpInference.Backends.D3D12Vm;

public sealed record D3D12VmTaskStatistics(
    ulong CompletedTasks, ulong ComputeRecordings, ulong InputUploadBytes,
    ulong InitialStateUploadBytes, ulong InitialLocalUploadBytes, ulong StateLoadBytes, ulong StateCommitBytes,
    ulong CpuReadbackBytes, double PreparationMilliseconds, double ComputeMilliseconds,
    double CommitMilliseconds);
