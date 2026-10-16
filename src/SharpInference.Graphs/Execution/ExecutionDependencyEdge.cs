using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Execution;

/// <summary>Orders the source operation or graph call before the target independently of resource accesses.</summary>
/// <param name="Id">The edge identity within the graph.</param>
/// <param name="Source">The structural identity of the prerequisite node.</param>
/// <param name="Target">The structural identity of the dependent node.</param>
public sealed record ExecutionDependencyEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target)
    : ExecutionGraphEdge(Id, Source, Target), IDependencyEdge;
