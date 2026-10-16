
namespace SharpInference.Graphs.Logical;

/// <summary>The completion information and diagnostics derived from one logical graph.</summary>
/// <param name="CompletionSinks">Operations with no direct dependency successors.</param>
/// <param name="TerminalWriters">Dependency-maximal writer identities for each output or in/out resource.</param>
/// <param name="Diagnostics">Validation findings collected during analysis.</param>
internal sealed record LogicalGraphAnalysis(
    IReadOnlyList<GraphDefinitionNodeId> CompletionSinks,
    IReadOnlyDictionary<ResourceId, GraphDefinitionNodeId[]> TerminalWriters,
    IReadOnlyList<GraphDiagnostic> Diagnostics);
