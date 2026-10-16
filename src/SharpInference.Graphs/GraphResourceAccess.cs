namespace SharpInference.Graphs;

/// <summary>Specifies whether a node reads, writes, or updates a resource.</summary>
public enum GraphResourceAccess
{
    /// <summary>Reads existing resource contents.</summary>
    Read,
    /// <summary>Writes resource contents without requiring previous values.</summary>
    Write,
    /// <summary>Reads and updates resource contents.</summary>
    ReadWrite,
}
