using SharpInference.Graphs.Execution;
using SharpInference.Graphs;

namespace SharpInference.Instructions;

/// <summary>Compares instruction precision capabilities with graph requirements.</summary>
public static class InstructionPrecisionCompatibility
{
    public static bool Satisfies(
        ExecutionPrecision capability,
        PrecisionRequirement requirement) =>
        Rank(capability.ArithmeticType) >= Rank(requirement.MinimumArithmeticType) &&
        Rank(capability.AccumulatorType) >= Rank(requirement.MinimumAccumulatorType);

    private static int Rank(ElementType type) => type switch
    {
        ElementType.Float16 => 1,
        ElementType.Float32 => 2,
        _ => 0,
    };
}
