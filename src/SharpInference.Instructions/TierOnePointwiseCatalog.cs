using SharpInference.Graphs;

namespace SharpInference.Instructions;

/// <summary>Refers to a fusion input or the result of an earlier pointwise stage.</summary>
public readonly record struct TierOneOperand
{
    private TierOneOperand(bool input, int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        IsInput = input;
        Index = index;
    }

    /// <summary>Gets whether the operand refers to a fusion input rather than a prior stage.</summary>
    public bool IsInput { get; }
    /// <summary>Gets the zero-based input or earlier-stage index.</summary>
    public int Index { get; }
    /// <summary>Creates an operand referring to the external fusion input at the specified index.</summary>
    /// <param name="index">The nonnegative zero-based input or prior-stage index.</param>
    public static TierOneOperand Input(int index) => new(true, index);
    /// <summary>Creates an operand referring to an earlier pointwise stage's result.</summary>
    /// <param name="index">The nonnegative zero-based input or prior-stage index.</param>
    public static TierOneOperand Step(int index) => new(false, index);
}

/// <summary>Describes one primitive operation and its operands within a pointwise fusion.</summary>
public sealed class TierOnePointwiseStage
{
    internal TierOnePointwiseStage(GraphOperationId operation, params TierOneOperand[] operands)
    {
        Operation = operation;
        Operands = Array.AsReadOnly(operands.ToArray());
    }

    /// <summary>Gets the primitive graph operation evaluated by this stage.</summary>
    public GraphOperationId Operation { get; }
    /// <summary>Gets the ordered input and earlier-stage operands.</summary>
    public IReadOnlyList<TierOneOperand> Operands { get; }
}

/// <summary>Describes a supported FP32 pointwise fusion and validates operand compatibility.</summary>
public sealed class TierOnePointwiseDefinition
{
    internal TierOnePointwiseDefinition(string name, string[] inputs, params TierOnePointwiseStage[] stages)
    {
        Name = name;
        Inputs = Array.AsReadOnly(inputs.ToArray());
        Stages = Array.AsReadOnly(stages.ToArray());
    }

    /// <summary>Gets the registered name of the pointwise fusion.</summary>
    public string Name { get; }
    /// <summary>Gets the ordered external inputs to the fusion.</summary>
    public IReadOnlyList<string> Inputs { get; }
    /// <summary>Gets the primitive operations in fusion evaluation order.</summary>
    public IReadOnlyList<TierOnePointwiseStage> Stages { get; }
    /// <summary>Gets the number of intermediate executions eliminated by fusion.</summary>
    public int EliminatedStages => Stages.Count - 1;

    /// <summary>Checks whether operand types, shapes, access, aliases, and precision permit this pointwise fusion.</summary>
    /// <param name="operands">The named tensor operands, including the writable output.</param>
    /// <param name="requirement">The minimum arithmetic and accumulator precision required by the caller.</param>
    /// <param name="diagnostic">Receives an empty string on success or the incompatibility reason on failure.</param>
    /// <returns><see langword="true"/> for compatible dense, shape-matched FP32 operands with a nonaliasing output; otherwise, <see langword="false"/>.</returns>
    public bool TryAdapt(IReadOnlyList<InstructionOperandDescription> operands, PrecisionRequirement requirement,
        out string diagnostic)
    {
        ArgumentNullException.ThrowIfNull(operands);
        ArgumentNullException.ThrowIfNull(requirement);
        diagnostic = "";
        if (operands.Any(operand => operand is null || string.IsNullOrWhiteSpace(operand.Name) ||
                string.IsNullOrWhiteSpace(operand.Source) || operand.Tensor is null || !Enum.IsDefined(operand.Access)) ||
            operands.Select(operand => operand.Name).Distinct(StringComparer.Ordinal).Count() != operands.Count)
            throw new ArgumentException("Invalid or duplicate operand descriptions.", nameof(operands));
        if (operands.Count != Inputs.Count + 1 || operands.Any(operand =>
                operand.Name != "output" && !Inputs.Contains(operand.Name, StringComparer.Ordinal)) ||
            operands.All(operand => operand.Name != "output"))
            diagnostic = "Unsupported ports.";
        else if (!NumericTypeCompatibility.Satisfies(TierOnePointwiseCatalog.Precision, requirement))
            diagnostic = "Insufficient internal precision.";
        else
        {
            var output = operands.Single(operand => operand.Name == "output");
            if (output.Access == GraphResourceAccess.Read ||
                operands.Any(operand => operand.Name != "output" && operand.Access == GraphResourceAccess.Write))
                diagnostic = "Unsupported operand access.";
            else if (operands.Any(operand => operand.Tensor.ElementType != GraphElementType.Float32 ||
                         operand.Tensor.Layout != "dense" || operand.Tensor.Dimensions.Count == 0 ||
                         !operand.Tensor.Dimensions.SequenceEqual(output.Tensor.Dimensions)))
                diagnostic = "FP32 dense operands must have identical nonempty shapes.";
            else if (operands.Any(operand => operand.Name != "output" && operand.Source == output.Source))
                diagnostic = "Output must not alias an input.";
        }
        return diagnostic.Length == 0;
    }
}

/// <summary>Provides the supported FP32 pointwise fusion definitions and their precision contract.</summary>
public static class TierOnePointwiseCatalog
{
    /// <summary>Gets the internal arithmetic and accumulator precision contract.</summary>
    public static KernelPrecisionProfile Precision { get; } = new(GraphElementType.Float32, GraphElementType.Float32);
    /// <summary>Gets the complete set of supported FP32 pointwise fusion definitions.</summary>
    public static IReadOnlyList<TierOnePointwiseDefinition> Definitions { get; } = Array.AsReadOnly(Create());
    private static readonly IReadOnlyDictionary<string, TierOnePointwiseDefinition> ByName =
        Definitions.ToDictionary(definition => definition.Name, StringComparer.Ordinal);

    /// <summary>Returns the supported FP32 pointwise fusion definition with the specified name.</summary>
    /// <param name="name">The registered name of the FP32 pointwise fusion to retrieve.</param>
    public static TierOnePointwiseDefinition Get(string name) =>
        ByName.TryGetValue(name, out var definition) ? definition :
            throw new NotSupportedException($"Unknown FP32 T1 pointwise operation '{name}'.");

    private static TierOnePointwiseDefinition[] Create()
    {
        var a = TierOneOperand.Input(0);
        var b = TierOneOperand.Input(1);
        var c = TierOneOperand.Input(2);
        var d = TierOneOperand.Input(3);
        var t0 = TierOneOperand.Step(0);
        var t1 = TierOneOperand.Step(1);
        return
        [
            new("add-rsqrt", ["a", "b"], new(PrimitiveGraphOperations.Add, a, b),
                new(PrimitiveGraphOperations.ReciprocalSquareRoot, t0)),
            new("maximum-rsqrt", ["a", "b"], new(PrimitiveGraphOperations.Maximum, a, b),
                new(PrimitiveGraphOperations.ReciprocalSquareRoot, t0)),
            new("multiply-add", ["a", "b", "c"], new(PrimitiveGraphOperations.Multiply, a, b),
                new(PrimitiveGraphOperations.Add, t0, c)),
            new("add-multiply-add", ["a", "b", "c", "d"], new(PrimitiveGraphOperations.Add, a, b),
                new(PrimitiveGraphOperations.Multiply, t0, c), new(PrimitiveGraphOperations.Add, t1, d)),
            new("multiply-multiply-add", ["a", "b", "c", "d"], new(PrimitiveGraphOperations.Multiply, a, b),
                new(PrimitiveGraphOperations.Multiply, t0, c), new(PrimitiveGraphOperations.Add, t1, d)),
            new("multiply-add-multiply", ["a", "b", "c", "d"], new(PrimitiveGraphOperations.Multiply, a, b),
                new(PrimitiveGraphOperations.Add, t0, c), new(PrimitiveGraphOperations.Multiply, t1, d)),
            new("difference-mix", ["a", "b", "c"], new(PrimitiveGraphOperations.Subtract, a, b),
                new(PrimitiveGraphOperations.Multiply, t0, c), new(PrimitiveGraphOperations.Add, b, t1)),
            new("multiply-add-subtract", ["a", "b", "c"], new(PrimitiveGraphOperations.Multiply, a, b),
                new(PrimitiveGraphOperations.Add, t0, c), new(PrimitiveGraphOperations.Subtract, t1, b)),
            new("multiply-self-sigmoid", ["a"], new(PrimitiveGraphOperations.Sigmoid, a),
                new(PrimitiveGraphOperations.Multiply, a, t0)),
            new("add-sigmoid", ["a", "b"], new(PrimitiveGraphOperations.Add, a, b),
                new(PrimitiveGraphOperations.Sigmoid, t0)),
            new("exp-subtract-exp", ["a", "b"], new(PrimitiveGraphOperations.Exp, a),
                new(PrimitiveGraphOperations.Subtract, b, t0), new(PrimitiveGraphOperations.Exp, t1)),
            new("sigmoid-multiply-exp", ["a", "b"], new(PrimitiveGraphOperations.Sigmoid, a),
                new(PrimitiveGraphOperations.Multiply, b, t0), new(PrimitiveGraphOperations.Exp, t1)),
            new("relu-square", ["a"], new(PrimitiveGraphOperations.Relu, a),
                new(PrimitiveGraphOperations.Square, t0)),
        ];
    }
}
