using SharpInference.Graphs.Logical;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Records the logical definition from which an execution graph originated.</summary>
/// <param name="Definition">The source logical graph definition identity.</param>
public sealed record ExecutionGraphSource(GraphDefinitionId Definition);
