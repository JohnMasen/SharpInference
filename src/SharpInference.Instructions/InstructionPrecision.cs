using SharpInference.Graphs;

namespace SharpInference.Instructions;

/// <summary>Describes an instruction's floating-point arithmetic and accumulation precision.</summary>
public sealed record KernelPrecisionProfile
{
    public KernelPrecisionProfile(
        GraphElementType arithmeticType,
        GraphElementType accumulatorType)
    {
        if (!NumericTypeCompatibility.IsFloatingPoint(arithmeticType))
            throw new ArgumentOutOfRangeException(nameof(arithmeticType));
        if (!NumericTypeCompatibility.IsFloatingPoint(accumulatorType))
            throw new ArgumentOutOfRangeException(nameof(accumulatorType));
        ArithmeticType = arithmeticType;
        AccumulatorType = accumulatorType;
    }

    public GraphElementType ArithmeticType { get; }
    public GraphElementType AccumulatorType { get; }
}

/// <summary>Compares instruction precision capabilities with graph requirements.</summary>
public static class InstructionPrecisionCompatibility
{
    public static bool Satisfies(
        KernelPrecisionProfile capability,
        PrecisionRequirement requirement) =>
        Rank(capability.ArithmeticType) >= Rank(requirement.MinimumArithmeticType) &&
        Rank(capability.AccumulatorType) >= Rank(requirement.MinimumAccumulatorType);

    private static int Rank(GraphElementType type) => type switch
    {
        GraphElementType.Float16 => 1,
        GraphElementType.Float32 => 2,
        _ => 0,
    };
}
