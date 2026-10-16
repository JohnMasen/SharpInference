
namespace SharpInference.Graphs;

/// <summary>Classifies conflicting accesses to one mutable resource.</summary>
[Flags]
public enum ModelResourceConflictKind
{
    /// <summary>No conflicting access.</summary>
    None = 0,
    /// <summary>The first operation writes a value read by the second.</summary>
    ReadAfterWrite = 1,
    /// <summary>The first operation reads a value overwritten by the second.</summary>
    WriteAfterRead = 2,
    /// <summary>Both operations write the same resource.</summary>
    WriteAfterWrite = 4,
}
