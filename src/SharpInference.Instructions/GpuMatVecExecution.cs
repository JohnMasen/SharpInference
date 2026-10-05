using System.Security.Cryptography;
using System.Text;

namespace SharpInference.Instructions;

public static class GpuMatVecExecution
{
    public const uint Threads = 64;
    public static D3D12InstructionExecutionConfiguration Serial { get; } = new(new("serial-matvec"));
    public static D3D12InstructionExecutionConfiguration Cooperative { get; } = new(new("cooperative-matvec"));
    public static string ImplementationFingerprint { get; } = Convert.ToHexString(
        SHA256.HashData(Encoding.UTF8.GetBytes(
            $"{typeof(GpuMatVecExecution).Assembly.ManifestModule.ModuleVersionId}|{Cooperative.Implementation}|{Threads}")));

    public static (uint X, uint Y) Groups(ulong rows)
    {
        if (rows == 0 || rows > uint.MaxValue / Threads)
            throw new ArgumentOutOfRangeException(nameof(rows), "Cooperative MatVec exceeds 32-bit thread indexing.");
        var y = checked((uint)((rows + 65534) / 65535));
        var x = checked((uint)((rows + y - 1) / y));
        if ((ulong)x * y * Threads > uint.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(rows), "Padded cooperative grid exceeds 32-bit thread indexing.");
        return (x, y);
    }
}
