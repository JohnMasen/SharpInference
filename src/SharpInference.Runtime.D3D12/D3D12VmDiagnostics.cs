using SharpInference.Backends.D3D12Vm;

namespace SharpInference.Runtime.D3D12;

public sealed class D3D12VmDiagnostics(Func<D3D12VmTaskStatistics> readStatistics) : IVmBackendDiagnostics
{
    public D3D12VmTaskStatistics TaskStatistics => readStatistics();

    public IReadOnlyDictionary<string, double> ReadMetrics()
    {
        var value = TaskStatistics;
        return new Dictionary<string, double>(StringComparer.Ordinal)
        {
            ["d3d12.completed-tasks"] = value.CompletedTasks,
            ["d3d12.compute-recordings"] = value.ComputeRecordings,
            ["d3d12.input-upload-bytes"] = value.InputUploadBytes,
            ["d3d12.initial-state-upload-bytes"] = value.InitialStateUploadBytes,
            ["d3d12.initial-local-upload-bytes"] = value.InitialLocalUploadBytes,
            ["d3d12.state-load-bytes"] = value.StateLoadBytes,
            ["d3d12.state-commit-bytes"] = value.StateCommitBytes,
            ["d3d12.cpu-readback-bytes"] = value.CpuReadbackBytes,
            ["d3d12.preparation-ms"] = value.PreparationMilliseconds,
            ["d3d12.compute-ms"] = value.ComputeMilliseconds,
            ["d3d12.commit-ms"] = value.CommitMilliseconds,
        };
    }
}
