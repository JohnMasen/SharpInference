using System.Security.Cryptography;
using System.Text;

namespace SharpInference.Instructions;

/// <summary>Defines GPU matrix-vector implementation identities and cooperative launch dimensions.</summary>
public static class GpuMatVecExecution
{
    /// <summary>Identifies the 64-thread cooperative matrix-vector group size.</summary>
    public const uint Threads = 64;
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
        if (rows == 0 || rows > uint.MaxValue / Threads)
        {
            throw new ArgumentOutOfRangeException(nameof(rows), "Cooperative MatVec exceeds 32-bit thread indexing.");
        }

        var y = checked((uint)((rows + 65534) / 65535));
        var x = checked((uint)((rows + y - 1) / y));
        if ((ulong)x * y * Threads > uint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(rows), "Padded cooperative grid exceeds 32-bit thread indexing.");
        }

        return (x, y);
    }
}
