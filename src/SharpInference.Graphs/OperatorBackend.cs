namespace SharpInference.Graphs;

public sealed record OperatorSignature
{
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

    public IReadOnlyList<GraphElementType> InputTypes { get; }
    public IReadOnlyList<GraphElementType> OutputTypes { get; }

    public bool Matches(OperatorSignature other) =>
        other is not null &&
        InputTypes.SequenceEqual(other.InputTypes) &&
        OutputTypes.SequenceEqual(other.OutputTypes);
}

public sealed record KernelPrecisionProfile
{
    public KernelPrecisionProfile(GraphElementType arithmeticType, GraphElementType accumulatorType)
    {
        NumericTypeCompatibility.RequireFloatingPoint(arithmeticType, nameof(arithmeticType));
        NumericTypeCompatibility.RequireFloatingPoint(accumulatorType, nameof(accumulatorType));
        ArithmeticType = arithmeticType;
        AccumulatorType = accumulatorType;
    }

    public GraphElementType ArithmeticType { get; }
    public GraphElementType AccumulatorType { get; }
}

public sealed record PrecisionRequirement
{
    public PrecisionRequirement(GraphElementType minimumArithmeticType, GraphElementType minimumAccumulatorType)
    {
        NumericTypeCompatibility.RequireFloatingPoint(minimumArithmeticType, nameof(minimumArithmeticType));
        NumericTypeCompatibility.RequireFloatingPoint(minimumAccumulatorType, nameof(minimumAccumulatorType));
        MinimumArithmeticType = minimumArithmeticType;
        MinimumAccumulatorType = minimumAccumulatorType;
    }

    public GraphElementType MinimumArithmeticType { get; }
    public GraphElementType MinimumAccumulatorType { get; }

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

public sealed record OperatorImplementationDescription
{
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

    public string ImplementationId { get; }
    public GraphOperationId Operation { get; }
    public OperatorSignature Signature { get; }
    public KernelPrecisionProfile Precision { get; }
    public int Preference { get; }
}

public static class NumericTypeCompatibility
{
    public static bool IsFloatingPoint(GraphElementType type) =>
        type is GraphElementType.Float16 or GraphElementType.Float32;

    public static bool Satisfies(GraphElementType actual, GraphElementType required)
    {
        RequireFloatingPoint(actual, nameof(actual));
        RequireFloatingPoint(required, nameof(required));
        return actual == required ||
               actual == GraphElementType.Float32 && required == GraphElementType.Float16;
    }

    public static bool Satisfies(KernelPrecisionProfile actual, PrecisionRequirement required)
    {
        ArgumentNullException.ThrowIfNull(actual);
        ArgumentNullException.ThrowIfNull(required);
        return Satisfies(actual.ArithmeticType, required.MinimumArithmeticType) &&
            Satisfies(actual.AccumulatorType, required.MinimumAccumulatorType);
    }

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

    public static bool IsNoMorePrecise(KernelPrecisionProfile left, KernelPrecisionProfile right) =>
        Satisfies(right.ArithmeticType, left.ArithmeticType) &&
        Satisfies(right.AccumulatorType, left.AccumulatorType);

    public static void RequireFloatingPoint(GraphElementType type, string parameterName)
    {
        if (!IsFloatingPoint(type))
        {
            throw new ArgumentException($"'{type}' is not a supported floating-point precision.", parameterName);
        }
    }
}

public static class OperatorImplementationSelector
{
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

public sealed record PrimitiveOperatorDescription
{
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

    public GraphOperationId Operation { get; }
    public IReadOnlyList<OperatorSignature> Signatures { get; }
}

public sealed record FusedOperatorDescription
{
    public FusedOperatorDescription(string id, SequenceFusionRule rule)
    {
        Id = string.IsNullOrWhiteSpace(id)
            ? throw new ArgumentException("A fused operator identifier is required.", nameof(id))
            : id;
        Rule = rule ?? throw new ArgumentNullException(nameof(rule));
    }

    public string Id { get; }
    public SequenceFusionRule Rule { get; }
}

public interface ITensorPrimitiveBackend<T>
    where T : unmanaged
{
    void Copy(ReadOnlySpan<T> input, Span<T> output);
    void Add(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output);
    void Subtract(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output);
    void Multiply(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output);
    void Divide(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output);
    void Maximum(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output);
    void Exp(ReadOnlySpan<T> input, Span<T> output);
    void Tanh(ReadOnlySpan<T> input, Span<T> output);
    void Sigmoid(ReadOnlySpan<T> input, Span<T> output);
    void ReciprocalSquareRoot(ReadOnlySpan<T> input, Span<T> output);
    void Square(ReadOnlySpan<T> input, Span<T> output);
    void Relu(ReadOnlySpan<T> input, Span<T> output);
    T ReduceSum(ReadOnlySpan<T> input);
    T ReduceMean(ReadOnlySpan<T> input);
    void MatVec(ReadOnlySpan<T> matrix, ReadOnlySpan<T> input, Span<T> output, int rows, int columns);
    void GatherRow(ReadOnlySpan<T> table, int rowIndex, int rowLength, Span<T> output);
}

public interface IPrimitiveOperatorBackend :
    ITensorPrimitiveBackend<float>,
    ITensorPrimitiveBackend<Half>
{
    IReadOnlyList<PrimitiveOperatorDescription> PrimitiveOperators { get; }
    IReadOnlyList<OperatorImplementationDescription> OperatorImplementations { get; }
}

public sealed record CustomPrimitiveOperatorDescription
{
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

    public GraphOperationId Operation { get; }
    public IReadOnlyList<OperatorSignature> Signatures { get; }
}

public interface ICustomPrimitiveOperatorProvider
{
    IReadOnlyList<CustomPrimitiveOperatorDescription> CustomPrimitiveOperators { get; }
}

public interface IFusedOperatorProvider
{
    IReadOnlyList<FusedOperatorDescription> FusedOperators { get; }
}

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
    public static readonly GraphOperationId GatherRow = new("core.gather-row");

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
