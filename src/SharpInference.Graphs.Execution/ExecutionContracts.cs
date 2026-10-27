namespace SharpInference.Graphs.Execution;

/// <summary>Identifies the architecture on which an instruction implementation executes.</summary>
public readonly record struct ExecutionTarget
{
    /// <summary>Creates a target identity from a nonempty architecture name.</summary>
    /// <param name="architecture">The architecture identifier used to select implementations.</param>
    public ExecutionTarget(string architecture)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(architecture);
        Architecture = architecture;
    }
    /// <summary>Gets the architecture identifier used to select implementations.</summary>
    public string Architecture { get; }
    /// <summary>Identifies the managed CPU instruction architecture.</summary>
    public static readonly ExecutionTarget Cpu = new("cpu.managed");
    /// <summary>Identifies the Direct3D12 instruction architecture.</summary>
    public static readonly ExecutionTarget Direct3D12 = new("direct3d12");
    /// <inheritdoc/>
    public override string ToString() => Architecture;
}
/// <summary>Describes an instruction's floating-point arithmetic and accumulation precision.</summary>
public sealed record ExecutionPrecision
{
    public ExecutionPrecision(
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

/// <summary>Describes GPU thread-group dimensions and dispatch-group counts.</summary>
/// <param name="ThreadsX">The number of threads per group along the X axis.</param>
/// <param name="ThreadsY">The number of threads per group along the Y axis.</param>
/// <param name="ThreadsZ">The number of threads per group along the Z axis.</param>
/// <param name="GroupsX">The number of dispatch groups along the X axis.</param>
/// <param name="GroupsY">The number of dispatch groups along the Y axis.</param>
/// <param name="GroupsZ">The number of dispatch groups along the Z axis.</param>
public sealed record ExecutionDispatch(
    uint ThreadsX, uint ThreadsY, uint ThreadsZ, uint GroupsX, uint GroupsY, uint GroupsZ);
