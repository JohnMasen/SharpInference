namespace SharpInference.Graphs;

/// <summary>Identifies a tensor resource within a graph's resource namespace.</summary>
/// <param name="Value">The resource identity; this value type does not validate it.</param>
public readonly record struct ResourceId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}
