
namespace SharpInference.Instructions;

/// <summary>Describes an instruction's floating-point arithmetic and accumulation precision.</summary>
public sealed record KernelPrecisionProfile
{
    public KernelPrecisionProfile(
        ElementType arithmeticType,
        ElementType accumulatorType)
    {
        if (!NumericTypeCompatibility.IsFloatingPoint(arithmeticType))
        {
            throw new ArgumentOutOfRangeException(nameof(arithmeticType));
        }

        if (!NumericTypeCompatibility.IsFloatingPoint(accumulatorType))
        {
            throw new ArgumentOutOfRangeException(nameof(accumulatorType));
        }

        ArithmeticType = arithmeticType;
        AccumulatorType = accumulatorType;
    }

    public ElementType ArithmeticType { get; }
    public ElementType AccumulatorType { get; }
}

/// <summary>Compares instruction precision capabilities with graph requirements.</summary>
public static class InstructionPrecisionCompatibility
{
    public static bool Satisfies(
        KernelPrecisionProfile capability,
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
