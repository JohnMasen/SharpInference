namespace SharpInference;

/// <summary>Controls token sampling and stopping behavior during RWKV text generation.</summary>
public sealed record RwkvGenerationOptions
{
    /// <summary>Gets the maximum number of tokens to generate.</summary>
    public int MaxTokens { get; init; } = 256;

    /// <summary>Gets the sampling temperature.</summary>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>Gets the cumulative probability threshold used by nucleus sampling.</summary>
    public float TopP { get; init; } = 0.8f;

    /// <summary>Gets the maximum candidate count used by top-k sampling; zero disables that limit.</summary>
    public int TopK { get; init; }

    /// <summary>Gets the optional random seed.</summary>
    public int? Seed { get; init; }

    /// <summary>Gets token identifiers that stop generation before their bytes are emitted.</summary>
    public ISet<int> StopTokenIds { get; init; } = new HashSet<int>();

    /// <summary>Gets non-empty strings that stop generation when matched in decoded output.</summary>
    public IReadOnlyList<string> StopStrings { get; init; } = Array.Empty<string>();

    /// <summary>Validates the numeric ranges and stop-string values used for generation.</summary>
    public void Validate()
    {
        if (MaxTokens <= 0) throw new ArgumentOutOfRangeException(nameof(MaxTokens));
        if (!float.IsFinite(Temperature) || Temperature < 0) throw new ArgumentOutOfRangeException(nameof(Temperature));
        if (!float.IsFinite(TopP) || TopP is <= 0 or > 1) throw new ArgumentOutOfRangeException(nameof(TopP));
        if (TopK < 0) throw new ArgumentOutOfRangeException(nameof(TopK));
        ArgumentNullException.ThrowIfNull(StopStrings);
        if (StopStrings.Any(string.IsNullOrEmpty))
        {
            throw new ArgumentException("Stop strings must be non-empty.", nameof(StopStrings));
        }
    }
}
