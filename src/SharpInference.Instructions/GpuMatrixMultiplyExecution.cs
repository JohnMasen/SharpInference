namespace SharpInference.Instructions;

public static class GpuMatrixMultiplyExecution
{
    public const uint Threads = 64;
    public const uint TileSize = 8;

    public static D3D12InstructionExecutionConfiguration Serial { get; } =
        new(new("serial-matrix-multiply"));

    public static D3D12InstructionExecutionConfiguration Cooperative { get; } =
        new(new("cooperative-matrix-multiply"));

    public static D3D12InstructionExecutionConfiguration Tiled { get; } =
        new(new("tiled-matrix-multiply-8x8"));

    public static (uint X, uint Y) Groups(ulong outputElements)
    {
        if (outputElements == 0 || outputElements > uint.MaxValue / Threads)
            throw new ArgumentOutOfRangeException(
                nameof(outputElements),
                "Cooperative matrix multiplication exceeds 32-bit thread indexing.");
        var y = checked((uint)((outputElements + 65534) / 65535));
        var x = checked((uint)((outputElements + y - 1) / y));
        if ((ulong)x * y * Threads > uint.MaxValue)
            throw new ArgumentOutOfRangeException(
                nameof(outputElements),
                "Padded cooperative matrix grid exceeds 32-bit thread indexing.");
        return (x, y);
    }

    public static (uint X, uint Y) TileGroups(int rows, int columns)
    {
        if (rows <= 0 || columns <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows));
        return (
            checked((uint)((columns + TileSize - 1) / TileSize)),
            checked((uint)((rows + TileSize - 1) / TileSize)));
    }
}
