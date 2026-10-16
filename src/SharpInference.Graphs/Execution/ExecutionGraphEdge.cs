using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Execution;

/// <summary>Provides the identity and directed endpoints shared by execution graph edges.</summary>
/// <param name="Id">The edge identity within its graph.</param>
/// <param name="Source">The source node's structural identity.</param>
/// <param name="Target">The target node's structural identity.</param>
public abstract record ExecutionGraphEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target) : IModelGraphEdge;
