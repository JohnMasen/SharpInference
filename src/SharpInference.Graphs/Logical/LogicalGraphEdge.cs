using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Logical;

/// <summary>The identity and endpoints shared by logical dependency and resource-access edges.</summary>
/// <param name="Id">The edge identity.</param>
/// <param name="Source">The source graph-node identity.</param>
/// <param name="Target">The target graph-node identity.</param>
public abstract record LogicalGraphEdge(
    GraphEdgeId Id,
    GraphNodeId Source,
    GraphNodeId Target) : IModelGraphEdge;
