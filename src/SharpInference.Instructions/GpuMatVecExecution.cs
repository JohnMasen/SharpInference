using System.Security.Cryptography;
using System.Text;

namespace SharpInference.Instructions;

public static class GpuMatVecExecution
{
    public const uint Threads = SharpInference.Graphs.ExecutionPlanGpuContracts.CooperativeMatVecThreads;
    public static D3D12InstructionExecutionConfiguration Serial { get; } = new(new("serial-matvec"));
    public static D3D12InstructionExecutionConfiguration Cooperative { get; } = new(new("cooperative-matvec"));
    public static string ImplementationFingerprint { get; } = Convert.ToHexString(
        SHA256.HashData(Encoding.UTF8.GetBytes(
            $"{typeof(GpuMatVecExecution).Assembly.ManifestModule.ModuleVersionId}|{Cooperative.Implementation}|{Threads}")));

    public static (uint X, uint Y) Groups(ulong rows)
    {
        var grid = SharpInference.Graphs.ExecutionPlanGpuContracts.CooperativeMatVecGroups(rows);
        return (grid.X, grid.Y);
    }
}
