using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Execution;

/// <summary>Connects an operation or call to a resource that it writes through a named binding.</summary>
/// <param name="Id">The edge identity within its graph.</param>
/// <param name="Source">The writing operation or call's structural identity.</param>
/// <param name="Target">The resource node's structural identity.</param>
/// <param name="Port">The writer's binding port.</param>
/// <param name="BindingId">The binding identity shared with a paired read for read-write access.</param>
public sealed record ExecutionResourceWriteEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target,
    string Port,
    string BindingId)
    : ExecutionGraphEdge(Id, Source, Target), IResourceWriteEdge;
