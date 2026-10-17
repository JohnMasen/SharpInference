using SharpInference.Graphs;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Represents one call instance targeting a separately stored execution graph.</summary>
/// <param name="Id">The call identity within its containing graph.</param>
/// <param name="Target">The identity of the called graph.</param>
/// <param name="CompileBoundary">The compilation-boundary metadata carried by this call.</param>
/// <param name="Source">The logical origins and optional transformation label.</param>
/// <param name="Architecture">Optional descriptive architecture metadata.</param>
public sealed record ExecutionGraphCallNode(
    ExecutionNodeId Id,
    ExecutionGraphId Target,
    GraphCompileBoundary CompileBoundary,
    ExecutionSourceMap Source,
    GraphArchitecture? Architecture = null)
    : ExecutionGraphNode(Id, Source, Architecture);
