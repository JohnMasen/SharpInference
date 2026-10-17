using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Describes an execution operation with attributes, minimum precision requirements, and provenance.</summary>
/// <param name="Id">The operation identity within its graph.</param>
/// <param name="Operation">The operation name and version.</param>
/// <param name="Attributes">The operation's named string attributes.</param>
/// <param name="Requirements">The minimum arithmetic and accumulator element types.</param>
/// <param name="Source">The logical origins and optional transformation label.</param>
/// <param name="Architecture">Optional descriptive architecture metadata.</param>
public sealed record ExecutionOperationNode(
    ExecutionNodeId Id,
    OperationId Operation,
    IReadOnlyDictionary<string, string> Attributes,
    PrecisionRequirement Requirements,
    ExecutionSourceMap Source,
    GraphArchitecture? Architecture = null)
    : ExecutionGraphNode(Id, Source, Architecture);
