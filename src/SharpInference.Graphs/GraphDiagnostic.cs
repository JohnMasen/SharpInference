namespace SharpInference.Graphs;

public enum GraphDiagnosticSeverity
{
    Warning,
    Error,
}

/// <summary>A validation finding in a logical or execution graph.</summary>
public sealed record GraphDiagnostic(
    GraphDiagnosticSeverity Severity,
    string Code,
    string Message,
    string? Node = null,
    ResourceId? Resource = null,
    string? Port = null,
    string? Graph = null);
