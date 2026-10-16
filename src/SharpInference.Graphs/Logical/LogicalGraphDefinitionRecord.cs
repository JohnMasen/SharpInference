using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Logical;

/// <summary>Stores one parsed definition until its referenced definitions have been reconstructed.</summary>
/// <param name="Id">The serialized definition identity.</param>
/// <param name="Boundary">The compilation boundary of the definition.</param>
/// <param name="Annotation">The optional logical annotation.</param>
/// <param name="Structure">The exact nodes and edges parsed from XML.</param>
internal sealed record LogicalGraphDefinitionRecord(
    string Id,
    GraphCompileBoundary Boundary,
    GraphAnnotation? Annotation,
    Graph<LogicalGraphNode, LogicalGraphEdge> Structure);
