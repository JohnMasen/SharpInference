namespace SharpInference.Graphs;

/// <summary>Specifies the scalar element representation of a tensor.</summary>
public enum ElementType
{
    /// <summary>An unsigned eight-bit integer element.</summary>
    Byte,
    /// <summary>A signed 32-bit integer element.</summary>
    Int32,
    /// <summary>An unsigned 32-bit integer element.</summary>
    UInt32,
    /// <summary>A 16-bit floating-point element.</summary>
    Float16,
    /// <summary>A 32-bit floating-point element.</summary>
    Float32,
}
