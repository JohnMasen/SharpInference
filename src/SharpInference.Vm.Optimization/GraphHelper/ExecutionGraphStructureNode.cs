using SharpInference.Graphs;
using SharpInference.Graphs.Core;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Provides the structural identity shared by resources, operations, and calls.</summary>
public abstract record ExecutionGraphStructureNode : IModelGraphNode
{
    /// <summary>Gets the identity used by structural edges and graph queries.</summary>
    public abstract GraphNodeId GraphId { get; }
    /// <summary>Gets the structural identity exposed through the generic graph-node contract.</summary>
    GraphNodeId IGraphNode.Id => GraphId;
}
