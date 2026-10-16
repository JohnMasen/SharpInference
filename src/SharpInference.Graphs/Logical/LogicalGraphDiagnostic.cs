
namespace SharpInference.Graphs.Logical;

/// <summary>A finding produced while validating and analyzing a logical definition.</summary>
/// <param name="Severity">Whether the finding is a warning or a construction-blocking error.</param>
/// <param name="Code">The diagnostic code.</param>
/// <param name="Message">The human-readable finding.</param>
/// <param name="Node">The affected operation identity, if applicable.</param>
/// <param name="Resource">The affected resource identity, if applicable.</param>
public sealed record LogicalGraphDiagnostic(
    LogicalGraphDiagnosticSeverity Severity,
    string Code,
    string Message,
    GraphDefinitionNodeId? Node = null,
    ResourceId? Resource = null);
