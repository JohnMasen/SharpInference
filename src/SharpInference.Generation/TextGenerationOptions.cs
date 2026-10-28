namespace SharpInference;

/// <summary>Controls sampling and stopping for token-based text generation.</summary>
public record TextGenerationOptions
{
    /// <summary>Gets the maximum number of sampled output tokens.</summary>
    public int MaxTokens { get; init; } = 256;
    /// <summary>Gets the sampling temperature; zero selects greedy sampling.</summary>
    public float Temperature { get; init; } = 1.0f;
    /// <summary>Gets the cumulative probability threshold for nucleus sampling.</summary>
    public float TopP { get; init; } = 0.8f;
    /// <summary>Gets the maximum candidate count; zero disables top-k filtering.</summary>
    public int TopK
    {
        get; init;
    }
    /// <summary>Gets the optional seed for reproducible sampling.</summary>
    public int? Seed
    {
        get; init;
    }
    /// <summary>Gets token identifiers that stop generation before their text is emitted.</summary>
    public ISet<int> StopTokenIds { get; init; } = new HashSet<int>();
    /// <summary>Gets decoded strings that terminate generation and are omitted from output.</summary>
    public IReadOnlyList<string> StopStrings { get; init; } = Array.Empty<string>();

    /// <summary>Rejects invalid sampling limits or null and empty stopping rules.</summary>
    public void Validate()
    {
        if (MaxTokens <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxTokens));
        }

        if (!float.IsFinite(Temperature) || Temperature < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(Temperature));
        }

        if (!float.IsFinite(TopP) || TopP is <= 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(TopP));
        }

        if (TopK < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(TopK));
        }

        ArgumentNullException.ThrowIfNull(StopTokenIds);
        ArgumentNullException.ThrowIfNull(StopStrings);
        if (StopStrings.Any(string.IsNullOrEmpty))
        {
            throw new ArgumentException("Stop strings must be non-empty.", nameof(StopStrings));
        }
    }
}
