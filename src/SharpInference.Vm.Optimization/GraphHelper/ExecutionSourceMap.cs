using SharpInference.Graphs.Logical;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Associates an execution node with its logical origins and optional rewrite provenance.</summary>
/// <param name="LogicalNodes">The logical node identities represented by this execution node.</param>
/// <param name="Transformation">An optional label describing the transformation that produced the node.</param>
public sealed record ExecutionSourceMap(
    IReadOnlyList<GraphDefinitionNodeId> LogicalNodes,
    string? Transformation = null);
