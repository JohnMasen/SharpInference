using SharpInference.Graphs;

namespace SharpInference.Instructions;

public readonly record struct TierOneOperand
{
    private TierOneOperand(bool input, int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        IsInput = input;
        Index = index;
    }

    public bool IsInput { get; }
    public int Index { get; }
    public static TierOneOperand Input(int index) => new(true, index);
    public static TierOneOperand Step(int index) => new(false, index);
}

public sealed class TierOnePointwiseStage
{
    internal TierOnePointwiseStage(GraphOperationId operation, params TierOneOperand[] operands)
    {
        Operation = operation;
        Operands = Array.AsReadOnly(operands.ToArray());
    }

    public GraphOperationId Operation { get; }
    public IReadOnlyList<TierOneOperand> Operands { get; }
}

public sealed class TierOnePointwiseDefinition
{
    internal TierOnePointwiseDefinition(string name, string[] inputs, params TierOnePointwiseStage[] stages)
    {
        Name = name;
        Inputs = Array.AsReadOnly(inputs.ToArray());
        Stages = Array.AsReadOnly(stages.ToArray());
    }

    public string Name { get; }
    public IReadOnlyList<string> Inputs { get; }
    public IReadOnlyList<TierOnePointwiseStage> Stages { get; }
    public int EliminatedStages => Stages.Count - 1;

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

public static class TierOnePointwiseCatalog
{
    public static KernelPrecisionProfile Precision { get; } = new(GraphElementType.Float32, GraphElementType.Float32);
    public static IReadOnlyList<TierOnePointwiseDefinition> Definitions { get; } = Array.AsReadOnly(Create());
    private static readonly IReadOnlyDictionary<string, TierOnePointwiseDefinition> ByName =
        Definitions.ToDictionary(definition => definition.Name, StringComparer.Ordinal);

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
