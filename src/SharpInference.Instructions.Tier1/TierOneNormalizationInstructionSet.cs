using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;

namespace SharpInference.Instructions;

/// <summary>
/// Defines portable normalization, gated activation, and scaled residual fusions.
/// Batch operations preserve legacy control [2] signatures but only read control[1] as an active-row count.
/// They neither read positions nor access model caches. Inactive rows are left untouched.
/// </summary>
public static class TierOneNormalizationInstructionSet
{
    public static readonly Guid Id = new("7dd90891-006b-43ea-8179-64c06746e8b1");

    public static IReadOnlyDictionary<string, InstructionSignature> Signatures { get; } =
        new System.Collections.ObjectModel.ReadOnlyDictionary<string, InstructionSignature>(
            new Dictionary<string, InstructionSignature>(StringComparer.Ordinal)
            {
                ["transformer.rms-norm"] = Signature(
                    [Read("input"), Read("weight"), Write("output")], ["epsilon"]),
                ["transformer.residual-rms-norm"] = Signature(
                    [Read("hidden"), Read("residual"), Read("weight"), Write("updated"), Write("output")],
                    ["epsilon"]),
                ["transformer.swiglu"] = Signature([Read("gate_up"), Write("output")]),
                ["transformer.scaled-add"] = Signature(
                    [Read("input"), Read("update"), Write("output")], ["scale"]),
                ["transformer.batch-rms-norm"] = Signature(
                    [Read("input"), Read("weight"), Control(), Write("output")], ["epsilon"]),
                ["transformer.batch-residual-rms-norm"] = Signature(
                    [Read("hidden"), Read("residual"), Read("weight"), Control(), Write("updated"), Write("output")],
                    ["epsilon"]),
                ["transformer.batch-swiglu"] = Signature([Read("gate_up"), Control(), Write("output")]),
            });

    public static IReadOnlyList<string> RequiredInstructions { get; } =
        Array.AsReadOnly(Signatures.Keys.ToArray());

    public static InstructionCollectionDescription Describe(ExecutionTarget target) =>
        new(Id, "T1 normalization and gated activation", 1, target,
            Category: InstructionCategory.Tier1, RequiredInstructions: RequiredInstructions);

    public static string Comment(string name) =>
        name.StartsWith("transformer.batch-", StringComparison.Ordinal)
            ? "Row-wise FP32 fusion with shared weights where applicable; only control[1] limits active rows. control[0] is ignored, with no position/cache protocol."
            : name switch
            {
                "transformer.rms-norm" => "Flat FP32 weighted root-mean-square normalization.",
                "transformer.residual-rms-norm" => "Add the residual, publish the updated tensor, and apply weighted RMS normalization.",
                "transformer.swiglu" => "FP32 gate/(1+exp(-gate))*up with contiguous gate and up halves.",
                "transformer.scaled-add" => "FP32 input + positive scale * update.",
                _ => throw new ArgumentOutOfRangeException(nameof(name)),
            };

    public static IReadOnlyList<InstructionAttributeDescription> Attributes(string name) =>
        Signatures[name].Attributes.Select(attribute => new InstructionAttributeDescription(attribute,
            attribute == "epsilon" ? "Positive stabilizer added to the mean squared value." : "Positive update multiplier.",
            "Float32", Constraints: "finite, > 0")).ToArray();

    public static void Validate(string name, InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>()
            .ToDictionary(p => p.Name, StringComparer.Ordinal);
        if (name.StartsWith("transformer.batch-", StringComparison.Ordinal))
        {
            ValidateBatch(name, tensors);
            foreach (var attribute in parameters.OfType<InstructionAttributeParameter>())
                PositiveFloat(parameters, attribute.Name);
            return;
        }
        var count = tensors["output"].Tensor.Dimensions.Aggregate(1, (a, b) => checked(a * b));
        foreach (var (port, tensor) in tensors)
        {
            var actual = tensor.Tensor.Dimensions.Aggregate(1, (a, b) => checked(a * b));
            if (actual != checked(count * (port == "gate_up" ? 2 : 1)))
                throw new InstructionAdaptationException(Id, name,
                    "All ports must contain the same number of elements, except gate_up which contains two halves.");
        }
        foreach (var attribute in parameters.OfType<InstructionAttributeParameter>())
            PositiveFloat(parameters, attribute.Name);
    }

    public static float PositiveFloat(InstructionParameter[] parameters, string name)
    {
        var text = parameters.OfType<InstructionAttributeParameter>().Single(p => p.Name == name).Value;
        if (!float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) ||
            !float.IsFinite(value) || value <= 0)
            throw new InstructionAdaptationException(Id, name, $"'{name}' must be finite and positive.");
        return value;
    }

    private static InstructionSignature Signature(InstructionPort[] ports, string[]? attributes = null) =>
        new(ports, attributes ?? [], new(ElementType.Float32, ElementType.Float32));
    private static InstructionPort Control() => new("control", ElementType.Int32, ResourceAccess.Read,
        "Only control[1] is the valid row count; control[0] is ignored.", ShapeConstraint: "[2]");
    private static InstructionPort Read(string name) => new(name, ElementType.Float32, ResourceAccess.Read);
    private static InstructionPort Write(string name) => new(name, ElementType.Float32, ResourceAccess.Write);

    private static void ValidateBatch(string name, IReadOnlyDictionary<string, InstructionTensorParameter> tensors)
    {
        void Require(bool condition, string message)
        {
            if (!condition) throw new InstructionAdaptationException(Id, name, message);
        }
        var output = tensors["output"].Tensor.Dimensions;
        Require(output.Count == 2, "Batched output must have shape [B,H].");
        Require(tensors["control"].Tensor.Dimensions.SequenceEqual([2]), "Control must have shape [2].");
        foreach (var (port, tensor) in tensors)
        {
            var dimensions = tensor.Tensor.Dimensions;
            if (port == "control") continue;
            if (port == "weight")
                Require(dimensions.Count == 1 && dimensions[0] == output[1], "Weight must have shape [H].");
            else if (port == "gate_up")
                Require(dimensions.Count == 2 && dimensions[0] == output[0] &&
                    dimensions[1] == checked(output[1] * 2), "Gate/up must have shape [B,2H].");
            else
                Require(dimensions.SequenceEqual(output), "Batched tensor shapes must match [B,H].");
        }
    }
}
