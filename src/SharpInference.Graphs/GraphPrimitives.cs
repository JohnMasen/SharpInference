namespace SharpInference.Graphs;

/// <summary>Identifies a logical tensor resource.</summary>
public readonly record struct ResourceId(string Value)
{
    /// <inheritdoc/>
    public override string ToString() => Value;
}

/// <summary>Identifies an operation contract by name and version.</summary>
public readonly record struct GraphOperationId(string Name, int Version = 1)
{
    /// <inheritdoc/>
    public override string ToString() => $"{Name}@{Version}";
}

/// <summary>Specifies the scalar element representation of a graph tensor.</summary>
public enum GraphElementType
{
    /// <summary>An unsigned eight-bit integer element.</summary>
    Byte,
    /// <summary>A signed 32-bit integer element.</summary>
    Int32,
    /// <summary>An unsigned 32-bit integer element.</summary>
    UInt32,
    /// <summary>A 16-bit floating-point element.</summary>
    Float16,
    /// <summary>A 32-bit floating-point element.</summary>
    Float32,
}

/// <summary>Specifies whether a node reads, writes, or updates a resource.</summary>
public enum GraphResourceAccess
{
    /// <summary>Reads existing resource contents.</summary>
    Read,
    /// <summary>Writes resource contents without requiring previous values.</summary>
    Write,
    /// <summary>Reads and updates resource contents.</summary>
    ReadWrite,
}

/// <summary>Describes a tensor's element type, positive dimensions, and logical layout.</summary>
public sealed record TensorDescriptor
{
    /// <summary>Creates a tensor descriptor.</summary>
    public TensorDescriptor(GraphElementType elementType, IEnumerable<int> dimensions, string layout = "dense")
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        var values = dimensions.ToArray();
        if (values.Length == 0 || values.Any(value => value <= 0))
            throw new ArgumentOutOfRangeException(nameof(dimensions), "Tensor dimensions must be positive.");
        ElementType = elementType;
        Dimensions = Array.AsReadOnly(values);
        Layout = string.IsNullOrWhiteSpace(layout)
            ? throw new ArgumentException("A tensor layout is required.", nameof(layout))
            : layout;
    }

    /// <summary>Gets the scalar element type.</summary>
    public GraphElementType ElementType { get; }
    /// <summary>Gets tensor dimensions in axis order.</summary>
    public IReadOnlyList<int> Dimensions { get; }
    /// <summary>Gets the logical tensor layout.</summary>
    public string Layout { get; }
}

/// <summary>Specifies minimum arithmetic and accumulator precision for an operation.</summary>
public sealed record PrecisionRequirement
{
    /// <summary>Creates a precision requirement.</summary>
    public PrecisionRequirement(
        GraphElementType minimumArithmeticType,
        GraphElementType minimumAccumulatorType)
    {
        NumericTypeCompatibility.RequireFloatingPoint(minimumArithmeticType, nameof(minimumArithmeticType));
        NumericTypeCompatibility.RequireFloatingPoint(minimumAccumulatorType, nameof(minimumAccumulatorType));
        MinimumArithmeticType = minimumArithmeticType;
        MinimumAccumulatorType = minimumAccumulatorType;
    }

    /// <summary>Gets the minimum arithmetic type.</summary>
    public GraphElementType MinimumArithmeticType { get; }
    /// <summary>Gets the minimum accumulator type.</summary>
    public GraphElementType MinimumAccumulatorType { get; }
}

/// <summary>Provides numeric precision helpers used while defining logical operations.</summary>
public static class NumericTypeCompatibility
{
    /// <summary>Returns whether the type is a supported floating-point type.</summary>
    public static bool IsFloatingPoint(GraphElementType type) =>
        type is GraphElementType.Float16 or GraphElementType.Float32;

    /// <summary>Returns the highest precision in a nonempty sequence.</summary>
    public static GraphElementType Maximum(IEnumerable<GraphElementType> types)
    {
        var values = types?.ToArray() ?? throw new ArgumentNullException(nameof(types));
        if (values.Length == 0)
            throw new ArgumentException("At least one numeric type is required.", nameof(types));
        foreach (var value in values) RequireFloatingPoint(value, nameof(types));
        return values.Contains(GraphElementType.Float32)
            ? GraphElementType.Float32
            : GraphElementType.Float16;
    }

    internal static void RequireFloatingPoint(GraphElementType type, string parameterName)
    {
        if (!IsFloatingPoint(type))
            throw new ArgumentOutOfRangeException(parameterName, "A floating-point element type is required.");
    }
}

/// <summary>Standard backend-neutral primitive operation identifiers.</summary>
public static class PrimitiveGraphOperations
{
    public static readonly GraphOperationId Copy = new("core.copy");
    public static readonly GraphOperationId Add = new("core.add");
    public static readonly GraphOperationId Subtract = new("core.subtract");
    public static readonly GraphOperationId Multiply = new("core.multiply");
    public static readonly GraphOperationId Divide = new("core.divide");
    public static readonly GraphOperationId Maximum = new("core.maximum");
    public static readonly GraphOperationId Exp = new("core.exp");
    public static readonly GraphOperationId Tanh = new("core.tanh");
    public static readonly GraphOperationId Sigmoid = new("core.sigmoid");
    public static readonly GraphOperationId ReciprocalSquareRoot = new("core.rsqrt");
    public static readonly GraphOperationId Square = new("core.square");
    public static readonly GraphOperationId Relu = new("core.relu");
    public static readonly GraphOperationId ReduceSum = new("core.reduce-sum");
    public static readonly GraphOperationId ReduceMean = new("core.reduce-mean");
    public static readonly GraphOperationId MatVec = new("core.mat-vec");
    public static readonly GraphOperationId MatrixMultiply = new("core.matrix-multiply");
    public static readonly GraphOperationId Affine = new("core.affine");
    public static readonly GraphOperationId BiasAdd = new("core.bias-add");
    public static readonly GraphOperationId GatherRow = new("core.gather-row");
}
