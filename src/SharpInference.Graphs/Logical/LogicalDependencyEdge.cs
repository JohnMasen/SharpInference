using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Logical;

/// <summary>A directed execution-order dependency between logical operations.</summary>
/// <param name="Id">The edge identity.</param>
/// <param name="Source">The prerequisite operation's graph identity.</param>
/// <param name="Target">The dependent operation's graph identity.</param>
public sealed record LogicalDependencyEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target) : LogicalGraphEdge(Id, Source, Target), IDependencyEdge;
