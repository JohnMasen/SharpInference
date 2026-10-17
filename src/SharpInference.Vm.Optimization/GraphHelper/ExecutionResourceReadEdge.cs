using SharpInference.Graphs;
using SharpInference.Graphs.Core;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Connects a resource to an operation or call that reads it through a named binding.</summary>
/// <param name="Id">The edge identity within its graph.</param>
/// <param name="Source">The resource node's structural identity.</param>
/// <param name="Target">The reading operation or call's structural identity.</param>
/// <param name="Port">The reader's binding port.</param>
/// <param name="BindingId">The binding identity shared with a paired write for read-write access.</param>
public sealed record ExecutionResourceReadEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target,
    string Port,
    string BindingId)
    : ExecutionGraphEdge(Id, Source, Target), IResourceReadEdge;
