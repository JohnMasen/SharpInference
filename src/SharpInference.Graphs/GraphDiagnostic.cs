namespace SharpInference.Graphs;

public enum GraphDiagnosticSeverity
{
    Warning,
    Error,
}

/// <summary>A validation finding associated with graph declarations.</summary>
public sealed record GraphDiagnostic(
    GraphDiagnosticSeverity Severity,
    string Code,
    string Message,
    string? Node = null,
    string? Resource = null,
    string? Port = null,
    string? Graph = null);
