namespace SharpInference.Graphs.Core;

/// <summary>Identifies one node within a graph.</summary>
/// <param name="Value">The structural identifier, distinct from a node's display name.</param>
public readonly record struct GraphNodeId(string Value)
{
    /// <summary>Converts a string to a node identifier without normalizing or validating it.</summary>
    /// <param name="value">The identifier string to preserve.</param>
    /// <returns>A node identifier containing the supplied string.</returns>
    public static implicit operator GraphNodeId(string value) => new(value);

    /// <inheritdoc/>
    public override string ToString() => Value;
}
