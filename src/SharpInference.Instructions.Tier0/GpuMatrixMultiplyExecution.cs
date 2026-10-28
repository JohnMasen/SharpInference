namespace SharpInference.Instructions;

/// <summary>Defines GPU matrix-multiplication implementation identities and launch dimensions.</summary>
public static class GpuMatrixMultiplyExecution
{
    /// <summary>Identifies the 64-thread cooperative matrix-multiplication group size.</summary>
    public const uint Threads = 64;
    /// <summary>Identifies the row and column extent of the square GPU tile.</summary>
    public const uint TileSize = 8;

    /// <summary>Gets the implementation assigning each output's reduction to one GPU thread.</summary>
    public static D3D12InstructionExecutionConfiguration Serial { get; } =
        new(new("serial-matrix-multiply"));

    /// <summary>Gets the implementation using a thread group to cooperate on each output reduction.</summary>
    public static D3D12InstructionExecutionConfiguration Cooperative { get; } =
        new(new("cooperative-matrix-multiply"));

    /// <summary>Gets the implementation using an 8-by-8 thread group and shared-memory tiles.</summary>
    public static D3D12InstructionExecutionConfiguration Tiled { get; } =
        new(new("tiled-matrix-multiply-8x8"));

    /// <summary>Computes a two-dimensional cooperative dispatch grid within GPU indexing limits.</summary>
    /// <param name="outputElements">The positive number of output elements requiring cooperative reduction groups.</param>
    public static (uint X, uint Y) Groups(ulong outputElements)
    {
        if (outputElements == 0 || outputElements > uint.MaxValue / Threads)
        {
            throw new ArgumentOutOfRangeException(
                nameof(outputElements),
                "Cooperative matrix multiplication exceeds 32-bit thread indexing.");
        }

        var y = checked((uint)((outputElements + 65534) / 65535));
        var x = checked((uint)((outputElements + y - 1) / y));
        if ((ulong)x * y * Threads > uint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(outputElements),
                "Padded cooperative matrix grid exceeds 32-bit thread indexing.");
        }

        return (x, y);
    }

    /// <summary>Computes the dispatch grid covering the matrix with 8-by-8 output tiles.</summary>
    /// <param name="rows">The positive number of matrix rows.</param>
    /// <param name="columns">The positive number of matrix columns.</param>
    public static (uint X, uint Y) TileGroups(int rows, int columns)
    {
        if (rows <= 0 || columns <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rows));
        }

        return (
            checked((uint)((columns + TileSize - 1) / TileSize)),
            checked((uint)((rows + TileSize - 1) / TileSize)));
    }
}
