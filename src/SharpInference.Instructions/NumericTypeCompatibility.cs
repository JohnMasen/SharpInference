namespace SharpInference.Instructions;

/// <summary>Provides numeric precision helpers for operation contracts.</summary>
public static class NumericTypeCompatibility
{
    /// <summary>Returns whether the type is a supported floating-point type.</summary>
    /// <param name="type">The scalar representation to inspect.</param>
    /// <returns>True only for Float16 and Float32.</returns>
    public static bool IsFloatingPoint(ElementType type) =>
        type is ElementType.Float16 or ElementType.Float32;

    /// <summary>Returns the highest precision in a nonempty sequence.</summary>
    /// <param name="types">The scalar representations, consumed once and required to be floating-point.</param>
    /// <returns>Float32 if present; otherwise Float16.</returns>
    /// <exception cref="ArgumentNullException">The sequence is null.</exception>
    /// <exception cref="ArgumentException">The sequence is empty.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An element is neither Float16 nor Float32.</exception>
    public static ElementType Maximum(IEnumerable<ElementType> types)
    {
        var values = types?.ToArray() ?? throw new ArgumentNullException(nameof(types));
        if (values.Length == 0)
        {
            throw new ArgumentException("At least one numeric type is required.", nameof(types));
        }

        foreach (var value in values)
        {
            RequireFloatingPoint(value, nameof(types));
        }

        return values.Contains(ElementType.Float32)
            ? ElementType.Float32
            : ElementType.Float16;
    }

    /// <summary>Rejects scalar representations other than the two supported floating-point types.</summary>
    /// <param name="type">The representation to validate.</param>
    /// <param name="parameterName">The argument name reported if validation fails.</param>
    /// <exception cref="ArgumentOutOfRangeException">The type is neither Float16 nor Float32.</exception>
    internal static void RequireFloatingPoint(ElementType type, string parameterName)
    {
        if (!IsFloatingPoint(type))
        {
            throw new ArgumentOutOfRangeException(parameterName, "A floating-point element type is required.");
        }
    }
}
