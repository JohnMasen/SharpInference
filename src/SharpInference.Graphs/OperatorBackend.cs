namespace SharpInference.Graphs;

/// <summary>Describes ordered input and output element types for an operator.</summary>
public sealed record OperatorSignature
{
    /// <summary>Creates an operator signature with ordered input types and at least one output type.</summary>
    /// <param name="inputTypes">The input element types in port order.</param>
    /// <param name="outputTypes">The output element types in port order.</param>
    public OperatorSignature(
        IEnumerable<GraphElementType> inputTypes,
        IEnumerable<GraphElementType> outputTypes)
    {
        InputTypes = Array.AsReadOnly(inputTypes?.ToArray() ?? throw new ArgumentNullException(nameof(inputTypes)));
        OutputTypes = Array.AsReadOnly(outputTypes?.ToArray() ?? throw new ArgumentNullException(nameof(outputTypes)));
        if (OutputTypes.Count == 0)
        {
            throw new ArgumentException("An operator signature requires at least one output.", nameof(outputTypes));
        }
    }

    /// <summary>Gets the input element types in port order.</summary>
    public IReadOnlyList<GraphElementType> InputTypes { get; }
    /// <summary>Gets the output element types in port order.</summary>
    public IReadOnlyList<GraphElementType> OutputTypes { get; }

    /// <summary>Determines whether ordered input and output element types exactly match another signature.</summary>
    /// <param name="other">The value to compare.</param>
    public bool Matches(OperatorSignature other) =>
        other is not null &&
        InputTypes.SequenceEqual(other.InputTypes) &&
        OutputTypes.SequenceEqual(other.OutputTypes);
}

/// <summary>Describes a kernel&apos;s floating-point arithmetic and accumulation precision.</summary>
public sealed record KernelPrecisionProfile
{
    /// <summary>Creates a kernel profile with supported floating-point arithmetic and accumulation types.</summary>
    /// <param name="arithmeticType">The floating-point arithmetic type.</param>
    /// <param name="accumulatorType">The floating-point accumulation type.</param>
    public KernelPrecisionProfile(GraphElementType arithmeticType, GraphElementType accumulatorType)
    {
        NumericTypeCompatibility.RequireFloatingPoint(arithmeticType, nameof(arithmeticType));
        NumericTypeCompatibility.RequireFloatingPoint(accumulatorType, nameof(accumulatorType));
        ArithmeticType = arithmeticType;
        AccumulatorType = accumulatorType;
    }

    /// <summary>Gets the kernel&apos;s floating-point arithmetic type.</summary>
    public GraphElementType ArithmeticType { get; }
    /// <summary>Gets the kernel&apos;s floating-point accumulation type.</summary>
    public GraphElementType AccumulatorType { get; }
}

/// <summary>Specifies minimum floating-point arithmetic and accumulation precision for an operation.</summary>
public sealed record PrecisionRequirement
{
    /// <summary>Creates minimum arithmetic and accumulation requirements using supported floating-point types.</summary>
    /// <param name="minimumArithmeticType">The minimum required floating-point arithmetic type.</param>
    /// <param name="minimumAccumulatorType">The minimum required floating-point accumulation type.</param>
    public PrecisionRequirement(GraphElementType minimumArithmeticType, GraphElementType minimumAccumulatorType)
    {
        NumericTypeCompatibility.RequireFloatingPoint(minimumArithmeticType, nameof(minimumArithmeticType));
        NumericTypeCompatibility.RequireFloatingPoint(minimumAccumulatorType, nameof(minimumAccumulatorType));
        MinimumArithmeticType = minimumArithmeticType;
        MinimumAccumulatorType = minimumAccumulatorType;
    }

    /// <summary>Gets the minimum required floating-point arithmetic type.</summary>
    public GraphElementType MinimumArithmeticType { get; }
    /// <summary>Gets the minimum required floating-point accumulation type.</summary>
    public GraphElementType MinimumAccumulatorType { get; }

    /// <summary>Combines precision requirements by taking the highest arithmetic and accumulation types.</summary>
    /// <param name="requirements">The precision requirements to combine.</param>
    public static PrecisionRequirement Merge(IEnumerable<PrecisionRequirement> requirements)
    {
        var values = requirements?.ToArray() ?? throw new ArgumentNullException(nameof(requirements));
        if (values.Length == 0)
        {
            throw new ArgumentException("At least one precision requirement is required.", nameof(requirements));
        }

        return new(
            NumericTypeCompatibility.Maximum(values.Select(value => value.MinimumArithmeticType)),
            NumericTypeCompatibility.Maximum(values.Select(value => value.MinimumAccumulatorType)));
    }
}

/// <summary>Describes a backend implementation, its signature, precision, and selection preference.</summary>
public sealed record OperatorImplementationDescription
{
    /// <summary>Creates a named backend implementation description with signature, precision, and selection preference.</summary>
    /// <param name="implementationId">The backend implementation identifier.</param>
    /// <param name="operation">The operation contract identifier.</param>
    /// <param name="signature">The ordered input and output element-type signature.</param>
    /// <param name="precision">The implementation&apos;s arithmetic and accumulation precision profile.</param>
    /// <param name="preference">The nonnegative selection preference; lower values are preferred.</param>
    public OperatorImplementationDescription(
        string implementationId,
        GraphOperationId operation,
        OperatorSignature signature,
        KernelPrecisionProfile precision,
        int preference = 0)
    {
        ImplementationId = string.IsNullOrWhiteSpace(implementationId)
            ? throw new ArgumentException("An implementation identifier is required.", nameof(implementationId))
            : implementationId;
        if (string.IsNullOrWhiteSpace(operation.Name))
        {
            throw new ArgumentException("An operation identifier is required.", nameof(operation));
        }
        if (preference < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(preference), "Preference cannot be negative.");
        }

        Operation = operation;
        Signature = signature ?? throw new ArgumentNullException(nameof(signature));
        Precision = precision ?? throw new ArgumentNullException(nameof(precision));
        Preference = preference;
    }

    /// <summary>Gets the backend implementation identifier.</summary>
    public string ImplementationId { get; }
    /// <summary>Gets the versioned operation contract identifier.</summary>
    public GraphOperationId Operation { get; }
    /// <summary>Gets the ordered input and output element-type signature.</summary>
    public OperatorSignature Signature { get; }
    /// <summary>Gets the operation&apos;s arithmetic and accumulation precision contract.</summary>
    public KernelPrecisionProfile Precision { get; }
    /// <summary>Gets the nonnegative selection preference; lower values are preferred.</summary>
    public int Preference { get; }
}

/// <summary>Compares supported floating-point types and kernel precision requirements.</summary>
public static class NumericTypeCompatibility
{
    /// <summary>Determines whether an element type is supported FP16 or FP32 floating point.</summary>
    /// <param name="type">The graph region type or numeric element type.</param>
    public static bool IsFloatingPoint(GraphElementType type) =>
        type is GraphElementType.Float16 or GraphElementType.Float32;

    /// <summary>Determines whether the actual floating-point precision meets the required precision.</summary>
    /// <param name="actual">The actual floating-point type or precision profile.</param>
    /// <param name="required">The minimum floating-point type or precision requirement.</param>
    public static bool Satisfies(GraphElementType actual, GraphElementType required)
    {
        RequireFloatingPoint(actual, nameof(actual));
        RequireFloatingPoint(required, nameof(required));
        return actual == required ||
               actual == GraphElementType.Float32 && required == GraphElementType.Float16;
    }

    /// <summary>Determines whether the actual floating-point precision meets the required precision.</summary>
    /// <param name="actual">The actual floating-point type or precision profile.</param>
    /// <param name="required">The minimum floating-point type or precision requirement.</param>
    public static bool Satisfies(KernelPrecisionProfile actual, PrecisionRequirement required)
    {
        ArgumentNullException.ThrowIfNull(actual);
        ArgumentNullException.ThrowIfNull(required);
        return Satisfies(actual.ArithmeticType, required.MinimumArithmeticType) &&
            Satisfies(actual.AccumulatorType, required.MinimumAccumulatorType);
    }

    /// <summary>Returns the highest precision among a nonempty sequence of supported floating-point types.</summary>
    /// <param name="types">The floating-point element types to compare.</param>
    public static GraphElementType Maximum(IEnumerable<GraphElementType> types)
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

        return values.Contains(GraphElementType.Float32)
            ? GraphElementType.Float32
            : GraphElementType.Float16;
    }

    /// <summary>Determines whether neither component of the left precision profile exceeds the right profile.</summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    public static bool IsNoMorePrecise(KernelPrecisionProfile left, KernelPrecisionProfile right) =>
        Satisfies(right.ArithmeticType, left.ArithmeticType) &&
        Satisfies(right.AccumulatorType, left.AccumulatorType);

    /// <summary>Rejects element types other than supported FP16 and FP32 floating point.</summary>
    /// <param name="type">The graph region type or numeric element type.</param>
    /// <param name="parameterName">The argument name used when reporting invalid precision.</param>
    public static void RequireFloatingPoint(GraphElementType type, string parameterName)
    {
        if (!IsFloatingPoint(type))
        {
            throw new ArgumentException($"'{type}' is not a supported floating-point precision.", parameterName);
        }
    }
}

/// <summary>Selects a compatible implementation on the least-precision frontier, then by preference and identifier.</summary>
public static class OperatorImplementationSelector
{
    /// <summary>Selects a matching least-precision implementation, preferring lower preference values and then ordinal identifiers.</summary>
    /// <param name="operation">The operation contract identifier.</param>
    /// <param name="signature">The ordered input and output element-type signature.</param>
    /// <param name="requirement">The minimum operator precision requirement.</param>
    /// <param name="implementations">The candidate backend implementations.</param>
    public static OperatorImplementationDescription? Select(
        GraphOperationId operation,
        OperatorSignature signature,
        PrecisionRequirement requirement,
        IEnumerable<OperatorImplementationDescription> implementations)
    {
        ArgumentNullException.ThrowIfNull(signature);
        ArgumentNullException.ThrowIfNull(requirement);
        var candidates = implementations?
            .Where(candidate =>
                candidate.Operation == operation &&
                candidate.Signature.Matches(signature) &&
                NumericTypeCompatibility.Satisfies(
                    candidate.Precision.ArithmeticType,
                    requirement.MinimumArithmeticType) &&
                NumericTypeCompatibility.Satisfies(
                    candidate.Precision.AccumulatorType,
                    requirement.MinimumAccumulatorType))
            .ToArray() ?? throw new ArgumentNullException(nameof(implementations));

        var frontier = candidates
            .Where(candidate => !candidates.Any(other =>
                !ReferenceEquals(candidate, other) &&
                NumericTypeCompatibility.IsNoMorePrecise(other.Precision, candidate.Precision) &&
                !NumericTypeCompatibility.IsNoMorePrecise(candidate.Precision, other.Precision)))
            .OrderBy(candidate => candidate.Preference)
            .ThenBy(candidate => candidate.ImplementationId, StringComparer.Ordinal)
            .ToArray();
        return frontier.FirstOrDefault();
    }
}

/// <summary>Describes a primitive operation and its supported same-type floating-point signatures.</summary>
public sealed record PrimitiveOperatorDescription
{
    /// <summary>Creates a primitive description with one or more same-type FP16 or FP32 signatures.</summary>
    /// <param name="operation">The operation contract identifier.</param>
    /// <param name="signatures">The supported operator signatures.</param>
    public PrimitiveOperatorDescription(
        GraphOperationId operation,
        IEnumerable<OperatorSignature> signatures)
    {
        if (string.IsNullOrWhiteSpace(operation.Name))
        {
            throw new ArgumentException("A primitive operation identifier is required.", nameof(operation));
        }

        var values = signatures?.ToArray() ?? throw new ArgumentNullException(nameof(signatures));
        if (values.Length == 0)
        {
            throw new ArgumentException("A primitive implementation requires at least one signature.", nameof(signatures));
        }
        if (values.Any(signature => signature is null))
        {
            throw new ArgumentException("Primitive signatures cannot contain null entries.", nameof(signatures));
        }

        foreach (var signature in values)
        {
            var types = signature.InputTypes.Concat(signature.OutputTypes).Distinct().ToArray();
            if (types.Length != 1 || types[0] is not GraphElementType.Float16 and not GraphElementType.Float32)
            {
                throw new ArgumentException(
                    $"Primitive operation '{operation}' only supports same-type FP16 or FP32 signatures.",
                    nameof(signatures));
            }
        }

        Operation = operation;
        Signatures = values;
    }

    /// <summary>Gets the versioned operation contract identifier.</summary>
    public GraphOperationId Operation { get; }
    /// <summary>Gets the supported operator signatures.</summary>
    public IReadOnlyList<OperatorSignature> Signatures { get; }
}

/// <summary>Associates a backend fused operator identifier with its sequence fusion rule.</summary>
public sealed record FusedOperatorDescription
{
    /// <summary>Creates a named fused operator description from a sequence fusion rule.</summary>
    /// <param name="id">The identifier of the graph element or resource.</param>
    /// <param name="rule">The sequence fusion rule describing the fused operator.</param>
    public FusedOperatorDescription(string id, SequenceFusionRule rule)
    {
        Id = string.IsNullOrWhiteSpace(id)
            ? throw new ArgumentException("A fused operator identifier is required.", nameof(id))
            : id;
        Rule = rule ?? throw new ArgumentNullException(nameof(rule));
    }

    /// <summary>Gets the identifier within the containing graph or plan.</summary>
    public string Id { get; }
    /// <summary>Gets the sequence fusion rule defining this fused operator.</summary>
    public SequenceFusionRule Rule { get; }
}

/// <summary>Executes tensor primitives over unmanaged element spans.</summary>
/// <typeparam name="T">The unmanaged tensor element type.</typeparam>
public interface ITensorPrimitiveBackend<T>
    where T : unmanaged
{
    /// <summary>Copies input elements unchanged.</summary>
    /// <param name="input">The input tensor values.</param>
    /// <param name="output">The output tensor buffer.</param>
    void Copy(ReadOnlySpan<T> input, Span<T> output);
    /// <summary>Adds corresponding input elements.</summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="output">The output tensor buffer.</param>
    void Add(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output);
    /// <summary>Subtracts right-hand elements from left-hand elements.</summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="output">The output tensor buffer.</param>
    void Subtract(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output);
    /// <summary>Multiplies corresponding input elements.</summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="output">The output tensor buffer.</param>
    void Multiply(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output);
    /// <summary>Divides left-hand elements by right-hand elements.</summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="output">The output tensor buffer.</param>
    void Divide(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output);
    /// <summary>Selects the greater corresponding input element.</summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="output">The output tensor buffer.</param>
    void Maximum(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output);
    /// <summary>Computes the exponential of each input element.</summary>
    /// <param name="input">The input tensor values.</param>
    /// <param name="output">The output tensor buffer.</param>
    void Exp(ReadOnlySpan<T> input, Span<T> output);
    /// <summary>Computes the hyperbolic tangent of each input element.</summary>
    /// <param name="input">The input tensor values.</param>
    /// <param name="output">The output tensor buffer.</param>
    void Tanh(ReadOnlySpan<T> input, Span<T> output);
    /// <summary>Computes the logistic sigmoid of each input element.</summary>
    /// <param name="input">The input tensor values.</param>
    /// <param name="output">The output tensor buffer.</param>
    void Sigmoid(ReadOnlySpan<T> input, Span<T> output);
    /// <summary>Computes the reciprocal square root of each input element.</summary>
    /// <param name="input">The input tensor values.</param>
    /// <param name="output">The output tensor buffer.</param>
    void ReciprocalSquareRoot(ReadOnlySpan<T> input, Span<T> output);
    /// <summary>Squares each input element.</summary>
    /// <param name="input">The input tensor values.</param>
    /// <param name="output">The output tensor buffer.</param>
    void Square(ReadOnlySpan<T> input, Span<T> output);
    /// <summary>Clamps each input element to a minimum of zero.</summary>
    /// <param name="input">The input tensor values.</param>
    /// <param name="output">The output tensor buffer.</param>
    void Relu(ReadOnlySpan<T> input, Span<T> output);
    /// <summary>Sums all input elements.</summary>
    /// <param name="input">The input tensor values.</param>
    T ReduceSum(ReadOnlySpan<T> input);
    /// <summary>Computes the mean of all input elements.</summary>
    /// <param name="input">The input tensor values.</param>
    T ReduceMean(ReadOnlySpan<T> input);
    /// <summary>Multiplies a matrix by an input vector.</summary>
    /// <param name="matrix">The row-major matrix values.</param>
    /// <param name="input">The input tensor values.</param>
    /// <param name="output">The output tensor buffer.</param>
    /// <param name="rows">The number of matrix rows.</param>
    /// <param name="columns">The number of matrix columns.</param>
    void MatVec(ReadOnlySpan<T> matrix, ReadOnlySpan<T> input, Span<T> output, int rows, int columns);
    /// <summary>Copies a selected table row.</summary>
    /// <param name="table">The row-major table values.</param>
    /// <param name="rowIndex">The zero-based table row to gather.</param>
    /// <param name="rowLength">The number of elements in each table row.</param>
    /// <param name="output">The output tensor buffer.</param>
    void GatherRow(ReadOnlySpan<T> table, int rowIndex, int rowLength, Span<T> output);
}

/// <summary>Provides FP32 and FP16 tensor primitives and their implementation descriptions.</summary>
public interface IPrimitiveOperatorBackend :
    ITensorPrimitiveBackend<float>,
    ITensorPrimitiveBackend<Half>
{
    /// <summary>Gets the primitive operation descriptions supported by this backend.</summary>
    IReadOnlyList<PrimitiveOperatorDescription> PrimitiveOperators { get; }
    /// <summary>Gets the backend primitive implementation descriptions.</summary>
    IReadOnlyList<OperatorImplementationDescription> OperatorImplementations { get; }
}

/// <summary>Describes a custom primitive operation and its supported signatures.</summary>
public sealed record CustomPrimitiveOperatorDescription
{
    /// <summary>Creates a custom primitive description with one or more supported signatures.</summary>
    /// <param name="operation">The operation contract identifier.</param>
    /// <param name="signatures">The supported operator signatures.</param>
    public CustomPrimitiveOperatorDescription(
        GraphOperationId operation,
        IEnumerable<OperatorSignature> signatures)
    {
        if (string.IsNullOrWhiteSpace(operation.Name))
        {
            throw new ArgumentException("A custom primitive operation identifier is required.", nameof(operation));
        }

        var values = signatures?.ToArray() ?? throw new ArgumentNullException(nameof(signatures));
        if (values.Length == 0)
        {
            throw new ArgumentException("A custom primitive implementation requires at least one signature.", nameof(signatures));
        }
        if (values.Any(signature => signature is null))
        {
            throw new ArgumentException("Custom primitive signatures cannot contain null entries.", nameof(signatures));
        }

        Operation = operation;
        Signatures = values;
    }

    /// <summary>Gets the versioned operation contract identifier.</summary>
    public GraphOperationId Operation { get; }
    /// <summary>Gets the supported operator signatures.</summary>
    public IReadOnlyList<OperatorSignature> Signatures { get; }
}

/// <summary>Exposes descriptions of backend-defined primitive operations.</summary>
public interface ICustomPrimitiveOperatorProvider
{
    /// <summary>Gets the backend-defined primitive operation descriptions.</summary>
    IReadOnlyList<CustomPrimitiveOperatorDescription> CustomPrimitiveOperators { get; }
}

/// <summary>Exposes backend-supported fused operators and their matching rules.</summary>
public interface IFusedOperatorProvider
{
    /// <summary>Gets the backend-supported fused operator descriptions.</summary>
    IReadOnlyList<FusedOperatorDescription> FusedOperators { get; }
}

/// <summary>Defines standard primitive graph operation identifiers and floating-point implementation descriptions.</summary>
public static class PrimitiveGraphOperations
{
    /// <summary>The operation identifier for a primitive that copies input elements unchanged.</summary>
    public static readonly GraphOperationId Copy = new("core.copy");
    /// <summary>The operation identifier for a primitive that adds corresponding input elements.</summary>
    public static readonly GraphOperationId Add = new("core.add");
    /// <summary>The operation identifier for a primitive that subtracts right-hand elements from left-hand elements.</summary>
    public static readonly GraphOperationId Subtract = new("core.subtract");
    /// <summary>The operation identifier for a primitive that multiplies corresponding input elements.</summary>
    public static readonly GraphOperationId Multiply = new("core.multiply");
    /// <summary>The operation identifier for a primitive that divides left-hand elements by right-hand elements.</summary>
    public static readonly GraphOperationId Divide = new("core.divide");
    /// <summary>The operation identifier for a primitive that selects the greater corresponding input element.</summary>
    public static readonly GraphOperationId Maximum = new("core.maximum");
    /// <summary>The operation identifier for a primitive that computes the exponential of each input element.</summary>
    public static readonly GraphOperationId Exp = new("core.exp");
    /// <summary>The operation identifier for a primitive that computes the hyperbolic tangent of each input element.</summary>
    public static readonly GraphOperationId Tanh = new("core.tanh");
    /// <summary>The operation identifier for a primitive that computes the logistic sigmoid of each input element.</summary>
    public static readonly GraphOperationId Sigmoid = new("core.sigmoid");
    /// <summary>The operation identifier for a primitive that computes the reciprocal square root of each input element.</summary>
    public static readonly GraphOperationId ReciprocalSquareRoot = new("core.rsqrt");
    /// <summary>The operation identifier for a primitive that squares each input element.</summary>
    public static readonly GraphOperationId Square = new("core.square");
    /// <summary>The operation identifier for a primitive that clamps each input element to a minimum of zero.</summary>
    public static readonly GraphOperationId Relu = new("core.relu");
    /// <summary>The operation identifier for a primitive that sums all input elements.</summary>
    public static readonly GraphOperationId ReduceSum = new("core.reduce-sum");
    /// <summary>The operation identifier for a primitive that computes the mean of all input elements.</summary>
    public static readonly GraphOperationId ReduceMean = new("core.reduce-mean");
    /// <summary>The operation identifier for a primitive that multiplies a matrix by an input vector.</summary>
    public static readonly GraphOperationId MatVec = new("core.mat-vec");
    /// <summary>The operation identifier for a primitive that multiplies matrices with optional input transposes.</summary>
    public static readonly GraphOperationId MatrixMultiply = new("core.matrix-multiply");
    /// <summary>The operation identifier for a primitive that multiplies matrices with optional transposes and adds a bias.</summary>
    public static readonly GraphOperationId Affine = new("core.affine");
    /// <summary>The operation identifier for a primitive that adds a bias along the final tensor axis.</summary>
    public static readonly GraphOperationId BiasAdd = new("core.bias-add");
    /// <summary>The operation identifier for a primitive that copies a selected table row.</summary>
    public static readonly GraphOperationId GatherRow = new("core.gather-row");

    /// <summary>Creates descriptions for the standard primitive operations with FP32 and optional FP16 signatures.</summary>
    /// <param name="includeFp16">Whether to include same-type FP16 primitive signatures.</param>
    public static IReadOnlyList<PrimitiveOperatorDescription> CreateStandardDescriptions(bool includeFp16 = true) =>
    [
        UnaryOrCopy(Copy, includeFp16),
        Binary(Add, includeFp16),
        Binary(Subtract, includeFp16),
        Binary(Multiply, includeFp16),
        Binary(Divide, includeFp16),
        Binary(Maximum, includeFp16),
        UnaryOrCopy(Exp, includeFp16),
        UnaryOrCopy(Tanh, includeFp16),
        UnaryOrCopy(Sigmoid, includeFp16),
        UnaryOrCopy(ReciprocalSquareRoot, includeFp16),
        UnaryOrCopy(Square, includeFp16),
        UnaryOrCopy(Relu, includeFp16),
        Reduction(ReduceSum, includeFp16),
        Reduction(ReduceMean, includeFp16),
        new(MatVec, includeFp16 ? [
            SameTypeSignature(2, 1, GraphElementType.Float32),
            SameTypeSignature(2, 1, GraphElementType.Float16),
        ] : [SameTypeSignature(2, 1, GraphElementType.Float32)]),
        new(GatherRow, includeFp16 ? [
            SameTypeSignature(1, 1, GraphElementType.Float32),
            SameTypeSignature(1, 1, GraphElementType.Float16),
        ] : [SameTypeSignature(1, 1, GraphElementType.Float32)]),
    ];

    /// <summary>Creates standard implementation descriptions with the requested FP16 capabilities.</summary>
    /// <param name="backendId">The backend identifier.</param>
    /// <param name="includeFp16">Whether to include same-type FP16 primitive signatures.</param>
    /// <param name="fp16Elementwise">Whether to include FP16 elementwise implementations.</param>
    /// <param name="fp16Reduction">Whether to include FP16 reduction implementations.</param>
    /// <param name="fp16MatVec">Whether to include FP16 matrix-vector implementations.</param>
    public static IReadOnlyList<OperatorImplementationDescription> CreateStandardImplementations(
        string backendId,
        bool includeFp16,
        KernelPrecisionProfile fp16Elementwise,
        KernelPrecisionProfile fp16Reduction,
        KernelPrecisionProfile fp16MatVec)
    {
        if (string.IsNullOrWhiteSpace(backendId))
        {
            throw new ArgumentException("A backend identifier is required.", nameof(backendId));
        }

        var descriptions = CreateStandardDescriptions(includeFp16);
        return descriptions
            .SelectMany(description => description.Signatures.Select(signature =>
            {
                var elementType = signature.OutputTypes[0];
                var precision = elementType == GraphElementType.Float32
                    ? new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32)
                    : description.Operation == MatVec
                        ? fp16MatVec
                        : description.Operation is var operation &&
                          (operation == ReduceSum || operation == ReduceMean)
                            ? fp16Reduction
                            : fp16Elementwise;
                return new OperatorImplementationDescription(
                    $"{backendId}.{description.Operation.Name}.{elementType.ToString().ToLowerInvariant()}",
                    description.Operation,
                    signature,
                    precision);
            }))
            .ToArray();
    }

    private static PrimitiveOperatorDescription UnaryOrCopy(GraphOperationId operation, bool includeFp16) =>
        new(operation, includeFp16
            ? [SameTypeSignature(1, 1, GraphElementType.Float32), SameTypeSignature(1, 1, GraphElementType.Float16)]
            : [SameTypeSignature(1, 1, GraphElementType.Float32)]);

    private static PrimitiveOperatorDescription Binary(GraphOperationId operation, bool includeFp16) =>
        new(operation, includeFp16
            ? [SameTypeSignature(2, 1, GraphElementType.Float32), SameTypeSignature(2, 1, GraphElementType.Float16)]
            : [SameTypeSignature(2, 1, GraphElementType.Float32)]);

    private static PrimitiveOperatorDescription Reduction(GraphOperationId operation, bool includeFp16) =>
        new(operation, includeFp16
            ? [SameTypeSignature(1, 1, GraphElementType.Float32), SameTypeSignature(1, 1, GraphElementType.Float16)]
            : [SameTypeSignature(1, 1, GraphElementType.Float32)]);

    private static OperatorSignature SameTypeSignature(
        int inputCount,
        int outputCount = 1,
        GraphElementType elementType = GraphElementType.Float32) =>
        new(
            Enumerable.Repeat(elementType, inputCount),
            Enumerable.Repeat(elementType, outputCount));
}
