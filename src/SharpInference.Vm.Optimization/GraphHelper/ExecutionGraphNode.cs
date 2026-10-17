using SharpInference.Graphs;
using SharpInference.Graphs.Core;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Provides the identity, provenance, and metadata shared by executable operation and call nodes.</summary>
/// <param name="Id">The node identity within its execution graph.</param>
/// <param name="Source">The logical origins and optional transformation label.</param>
/// <param name="Architecture">Optional descriptive architecture metadata.</param>
public abstract record ExecutionGraphNode(
    ExecutionNodeId Id,
    ExecutionSourceMap Source,
    GraphArchitecture? Architecture) : ExecutionGraphStructureNode, IModelOperationNode
{
    /// <summary>Gets the structural identity formed by prefixing the local identity with <c>operation:</c>.</summary>
    public override GraphNodeId GraphId => NodeGraphId(Id);

    /// <summary>Maps a local operation or call identity to the shared structural node namespace.</summary>
    /// <param name="id">The local execution node identity.</param>
    /// <returns>The identity prefixed with <c>operation:</c>.</returns>
    internal static GraphNodeId NodeGraphId(ExecutionNodeId id) =>
        new($"operation:{id.Value}");
}
