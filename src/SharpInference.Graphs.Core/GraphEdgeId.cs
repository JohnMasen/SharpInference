namespace SharpInference.Graphs.Core;

/// <summary>Identifies one edge within a graph.</summary>
/// <param name="Value">The identifier that distinguishes this edge, including parallel edges.</param>
public readonly record struct GraphEdgeId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}
