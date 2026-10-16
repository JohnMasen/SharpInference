
namespace SharpInference.Graphs.Logical;

/// <summary>A logical graph definition identity stored without constructor validation.</summary>
/// <param name="Value">The underlying definition identifier.</param>
public readonly record struct GraphDefinitionId(string Value)
{
    /// <summary>Returns the underlying identifier.</summary>
    /// <returns>The stored identifier string.</returns>
    public override string ToString() => Value;
}
