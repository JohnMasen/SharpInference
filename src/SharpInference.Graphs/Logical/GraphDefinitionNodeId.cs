
namespace SharpInference.Graphs.Logical;

/// <summary>An operation identity local to a logical graph definition.</summary>
/// <param name="Value">The underlying operation identifier, stored without validation.</param>
public readonly record struct GraphDefinitionNodeId(string Value)
{
    /// <summary>Returns the underlying identifier.</summary>
    /// <returns>The stored identifier string.</returns>
    public override string ToString() => Value;
}
