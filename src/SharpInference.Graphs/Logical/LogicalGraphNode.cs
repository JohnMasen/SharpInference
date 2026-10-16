using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Logical;

/// <summary>The common base of logical operation and scoped resource nodes.</summary>
public abstract record LogicalGraphNode : IModelGraphNode
{
    /// <summary>Gets the identity used by the shared graph structure.</summary>
    public abstract GraphNodeId GraphId { get; }
    /// <summary>Gets the shared graph identity exposed through the node contract.</summary>
    GraphNodeId IGraphNode.Id => GraphId;
}
