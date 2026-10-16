
namespace SharpInference.Graphs.Execution;

/// <summary>Identifies an operation or call within an execution graph using value equality.</summary>
/// <param name="Value">The local identifier text.</param>
public readonly record struct ExecutionNodeId(string Value)
{
    /// <summary>Returns the stored identifier text.</summary>
    /// <returns>The unmodified identifier value.</returns>
    public override string ToString() => Value;
}
