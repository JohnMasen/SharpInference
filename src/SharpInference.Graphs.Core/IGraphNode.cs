namespace SharpInference.Graphs.Core;

/// <summary>Provides the structural identity required by a graph node.</summary>
public interface IGraphNode
{
    /// <summary>Gets the structural identity of this node within its graph.</summary>
    GraphNodeId Id { get; }
}
