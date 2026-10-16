
namespace SharpInference.Graphs.Logical;

/// <summary>Determines whether a logical graph finding prevents definition construction.</summary>
public enum LogicalGraphDiagnosticSeverity
{
    /// <summary>A finding retained on an otherwise valid definition.</summary>
    Warning,
    /// <summary>A finding that causes definition construction to fail.</summary>
    Error,
}
