
namespace SharpInference.Graphs.Logical;

/// <summary>An operation that invokes another logical graph definition.</summary>
/// <param name="Id">The operation identity within the containing graph.</param>
/// <param name="Definition">The identity of the invoked definition.</param>
/// <param name="Annotation">Optional descriptive metadata.</param>
public sealed record GraphCallNode(
    GraphDefinitionNodeId Id,
    GraphDefinitionId Definition,
    GraphAnnotation? Annotation = null)
    : GraphDefinitionNode(Id, Annotation);
