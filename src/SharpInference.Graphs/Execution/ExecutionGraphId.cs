
namespace SharpInference.Graphs.Execution;

/// <summary>Identifies an execution graph in a program using value equality.</summary>
/// <param name="Value">The identifier text; validated when constructing an execution graph.</param>
public readonly record struct ExecutionGraphId(string Value)
{
    /// <summary>Returns the stored identifier text.</summary>
    /// <returns>The unmodified identifier value.</returns>
    public override string ToString() => Value;
}
