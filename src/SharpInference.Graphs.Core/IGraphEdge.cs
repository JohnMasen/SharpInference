namespace SharpInference.Graphs.Core;

/// <summary>Provides the structural identity and endpoints required by a directed graph edge.</summary>
public interface IGraphEdge
{
    /// <summary>Gets the identity of this edge within its graph.</summary>
    GraphEdgeId Id
    {
        get;
    }
    /// <summary>Gets the identity of the node from which this edge originates.</summary>
    GraphNodeId Source
    {
        get;
    }
    /// <summary>Gets the identity of the node to which this edge points.</summary>
    GraphNodeId Target
    {
        get;
    }
}
