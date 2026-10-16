using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Logical;

/// <summary>An operation-to-resource write access, optionally paired with a read by binding identity.</summary>
/// <param name="Id">The edge identity.</param>
/// <param name="Source">The writing operation's graph identity.</param>
/// <param name="Target">The scoped resource's graph identity.</param>
/// <param name="Port">The operation port name.</param>
/// <param name="BindingId">The identity used to combine read and write edges into one binding.</param>
public sealed record LogicalResourceWriteEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target,
    string Port,
    string BindingId) : LogicalGraphEdge(Id, Source, Target), IResourceWriteEdge;
