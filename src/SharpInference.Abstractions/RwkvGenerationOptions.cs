namespace SharpInference;

public sealed record RwkvGenerationOptions
{
    public int MaxTokens { get; init; } = 256;
    public float Temperature { get; init; } = 1.0f;
    public float TopP { get; init; } = 0.8f;
    public int TopK { get; init; }
    public int? Seed { get; init; }
    public ISet<int> StopTokenIds { get; init; } = new HashSet<int>();
    public IReadOnlyList<string> StopStrings { get; init; } = Array.Empty<string>();

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
