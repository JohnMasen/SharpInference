using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Logical;

/// <summary>A resource-to-operation read access, optionally paired with a write by binding identity.</summary>
/// <param name="Id">The edge identity.</param>
/// <param name="Source">The scoped resource's graph identity.</param>
/// <param name="Target">The reading operation's graph identity.</param>
/// <param name="Port">The operation port name.</param>
/// <param name="BindingId">The identity used to combine read and write edges into one binding.</param>
public sealed record LogicalResourceReadEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target,
    string Port,
    string BindingId) : LogicalGraphEdge(Id, Source, Target), IResourceReadEdge;
