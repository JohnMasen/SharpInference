using SharpInference.Graphs.Logical;

namespace SharpInference.Graphs.Execution;

/// <summary>Records the logical definition from which an execution graph originated.</summary>
/// <param name="Definition">The source logical graph definition identity.</param>
public sealed record ExecutionGraphSource(GraphDefinitionId Definition);
