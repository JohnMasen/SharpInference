using SharpInference.Instructions;

namespace SharpInference.Graphs.Logical;

/// <summary>A logical operation with attributes and minimum precision requirements.</summary>
/// <param name="Id">The operation identity within its definition.</param>
/// <param name="Operation">The operation name and version.</param>
/// <param name="Attributes">Operation-specific string attributes.</param>
/// <param name="Requirements">Minimum arithmetic and accumulator precision.</param>
/// <param name="Annotation">Optional descriptive metadata.</param>
public sealed record GraphOperationNode(
    GraphDefinitionNodeId Id,
    OperationId Operation,
    IReadOnlyDictionary<string, string> Attributes,
    PrecisionRequirement Requirements,
    GraphAnnotation? Annotation = null)
    : GraphDefinitionNode(Id, Annotation);
