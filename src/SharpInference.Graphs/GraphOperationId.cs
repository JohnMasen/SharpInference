namespace SharpInference.Graphs;

/// <summary>Identifies an operation contract by name and version.</summary>
/// <param name="Name">The operation contract name; this value type does not validate it.</param>
/// <param name="Version">The contract version, defaulting to one.</param>
public readonly record struct GraphOperationId(string Name, int Version = 1)
{
    /// <inheritdoc/>
    public override string ToString() => $"{Name}@{Version}";
}
