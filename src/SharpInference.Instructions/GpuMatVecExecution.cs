using System.Security.Cryptography;
using System.Text;

namespace SharpInference.Instructions;

/// <summary>Defines GPU matrix-vector implementation identities and cooperative launch dimensions.</summary>
public static class GpuMatVecExecution
{
    /// <summary>Identifies the 64-thread cooperative matrix-vector group size.</summary>
    public const uint Threads = SharpInference.Graphs.ExecutionPlanGpuContracts.CooperativeMatVecThreads;
    /// <summary>Gets the implementation assigning each output's reduction to one GPU thread.</summary>
    public static D3D12InstructionExecutionConfiguration Serial { get; } = new(new("serial-matvec"));
    /// <summary>Gets the implementation using a thread group to cooperate on each output reduction.</summary>
    public static D3D12InstructionExecutionConfiguration Cooperative { get; } = new(new("cooperative-matvec"));
    /// <summary>Gets the implementation identity used to match trusted measurements.</summary>
    public static string ImplementationFingerprint { get; } = Convert.ToHexString(
        SHA256.HashData(Encoding.UTF8.GetBytes(
            $"{typeof(GpuMatVecExecution).Assembly.ManifestModule.ModuleVersionId}|{Cooperative.Implementation}|{Threads}")));

    /// <summary>Computes a two-dimensional cooperative dispatch grid within GPU indexing limits.</summary>
    /// <param name="rows">The positive number of output rows requiring cooperative reduction groups.</param>
    public static (uint X, uint Y) Groups(ulong rows)
    {
        var grid = SharpInference.Graphs.ExecutionPlanGpuContracts.CooperativeMatVecGroups(rows);
        return (grid.X, grid.Y);
    }
}
