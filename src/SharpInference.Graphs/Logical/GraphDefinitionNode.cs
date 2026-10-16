using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Logical;

/// <summary>A logical operation node, including calls to child graph definitions.</summary>
/// <param name="Id">The operation identity local to the containing definition.</param>
/// <param name="Annotation">Optional descriptive metadata.</param>
public abstract record GraphDefinitionNode(
    GraphDefinitionNodeId Id,
    GraphAnnotation? Annotation) : LogicalGraphNode, IModelOperationNode
{
    /// <summary>Gets the graph identity formed by prefixing the operation identifier with <c>operation:</c>.</summary>
    public override GraphNodeId GraphId => NodeGraphId(Id);

    /// <summary>Maps a definition-local operation identity into the graph node namespace.</summary>
    /// <param name="id">The operation identity.</param>
    /// <returns>The identifier prefixed with <c>operation:</c>.</returns>
    internal static GraphNodeId NodeGraphId(GraphDefinitionNodeId id) =>
        new($"operation:{id.Value}");
}
