namespace SharpInference.Graphs;

/// <summary>Specifies minimum arithmetic and accumulator precision for an operation.</summary>
public sealed record PrecisionRequirement
{
    /// <summary>Creates a precision requirement.</summary>
    /// <param name="minimumArithmeticType">The minimum operation arithmetic precision, Float16 or Float32.</param>
    /// <param name="minimumAccumulatorType">The minimum accumulation precision, Float16 or Float32.</param>
    /// <remarks>The two values are validated independently; no ordering between them is imposed.</remarks>
    /// <exception cref="ArgumentOutOfRangeException">Either representation is not floating-point.</exception>
    public PrecisionRequirement(
        ElementType minimumArithmeticType,
        ElementType minimumAccumulatorType)
    {
        NumericTypeCompatibility.RequireFloatingPoint(minimumArithmeticType, nameof(minimumArithmeticType));
        NumericTypeCompatibility.RequireFloatingPoint(minimumAccumulatorType, nameof(minimumAccumulatorType));
        MinimumArithmeticType = minimumArithmeticType;
        MinimumAccumulatorType = minimumAccumulatorType;
    }

    /// <summary>Gets the minimum arithmetic type.</summary>
    public ElementType MinimumArithmeticType
    {
        get;
    }
    /// <summary>Gets the minimum accumulator type.</summary>
    public ElementType MinimumAccumulatorType
    {
        get;
    }
}
