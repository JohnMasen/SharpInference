using SharpInference.Graphs;
using SharpInference.Graphs.Execution;

namespace SharpInference.Instructions;

public enum InstructionCategory
{
    Tier0, Tier1, Specialized
}

public sealed record InstructionAttributeDescription(
    string Name, string Meaning, string DataType, bool Required = true, string? Constraints = null);

public sealed record InstructionCatalogRow(
    ExecutionTarget Architecture, Guid InstructionSet, string Instruction,
    InstructionCategory Category, string Domain, string? Comment);

public sealed record InstructionParameterRow(
    ExecutionTarget Architecture, Guid InstructionSet, string Instruction, int Signature,
    string Name, string? Meaning, string Kind, ResourceAccess? Access,
    string DataType, ElementType? InternalPrecision, string? Constraints);

public sealed record InstructionSignatureRow(
    ExecutionTarget Architecture, Guid InstructionSet, string Instruction, int Signature,
    ElementType ArithmeticPrecision, ElementType AccumulatorPrecision);

/// <summary>Marks concrete instruction-set owners for DI discovery, not catalogs or decorators.</summary>
public interface IInstructionProvider : IInstructionCollectionProvider;

public interface IInstructionCatalog : IInstructionCollectionProvider
{
    IReadOnlyList<InstructionCatalogRow> Rows
    {
        get;
    }
    IReadOnlyList<InstructionParameterRow> Parameters
    {
        get;
    }
    IReadOnlyList<InstructionSignatureRow> Signatures
    {
        get;
    }
    Instruction Resolve(Guid collectionId, string name, ExecutionTarget target);
}
