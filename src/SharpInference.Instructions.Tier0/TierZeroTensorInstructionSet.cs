using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;

namespace SharpInference.Instructions;

/// <summary>
/// Defines portable indexing, copying, reduction, and shared-matrix instructions.
/// Historical transformer-prefixed names are retained for invocation compatibility, not domain classification.
/// </summary>
public static class TierZeroTensorInstructionSet
{
    public static readonly Guid Id = new("03e8bac1-92e4-4a1f-83e9-388cdb08ca10");

    public static IReadOnlyDictionary<string, InstructionSignature> Signatures { get; } =
        new System.Collections.ObjectModel.ReadOnlyDictionary<string, InstructionSignature>(
            new Dictionary<string, InstructionSignature>(StringComparer.Ordinal)
            {
                ["transformer.argmax"] = Signature(
                    [Read("input"), new("output", ElementType.Int32, ResourceAccess.Write)]),
                ["transformer.embedding-gather-batch"] = Signature(
                    [new("table", ElementType.Float16, ResourceAccess.Read),
                     new("indices", ElementType.Int32, ResourceAccess.Read), Write("output")]),
                ["transformer.copy-embedding-rows"] = Signature(
                    [Read("input"), Write("output")], ["rows", "output_start"]),
                ["transformer.batch-shared-matvec"] = Signature(
                    [new("weight", ElementType.Float16, ResourceAccess.Read), Read("input"),
                     new("control", ElementType.Int32, ResourceAccess.Read,
                         "Only control[1] is used as the valid row count; control[0] is ignored.",
                         ShapeConstraint: "[2]"),
                     Write("output")]),
            });

    public static IReadOnlyList<string> RequiredInstructions { get; } =
        Array.AsReadOnly(Signatures.Keys.ToArray());

    public static InstructionCollectionDescription Describe(ExecutionTarget target) =>
        new(Id, "T0 indexing and shared matrix operations", 0, target,
            Category: InstructionCategory.Tier0, RequiredInstructions: RequiredInstructions);

    public static string Comment(string name) => name switch
    {
        "transformer.argmax" => "Flat FP32 argmax, first maximum index; NaNs are ignored and all-NaN input returns zero.",
        "transformer.embedding-gather-batch" => "Gather FP16 table rows by Int32 indices and convert to FP32.",
        "transformer.copy-embedding-rows" => "Copy a prefix of input rows into an explicit destination row range.",
        "transformer.batch-shared-matvec" =>
            "Shared FP16 weight [O,I] times FP32 input [B,I]; control[1] limits active rows, control[0] is ignored. No position or cache semantics; inactive output rows are untouched.",
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    public static IReadOnlyList<InstructionAttributeDescription> Attributes(string name) =>
        name == "transformer.copy-embedding-rows" ?
        [
            new("rows", "Number of prefix input rows to copy.", "Int32", Constraints: "> 0"),
            new("output_start", "Zero-based first destination row.", "Int32", Constraints: ">= 0"),
        ] : [];

    public static void Validate(string name, InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>()
            .ToDictionary(p => p.Name, StringComparer.Ordinal);
        var input = tensors.GetValueOrDefault("input");
        var output = tensors["output"];
        int Count(InstructionTensorParameter tensor) =>
            tensor.Tensor.Dimensions.Aggregate(1, (a, b) => checked(a * b));
        void Require(bool condition, string message)
        {
            if (!condition)
            {
                throw new InstructionAdaptationException(Id, name, message);
            }
        }

        switch (name)
        {
            case "transformer.argmax":
                Require(Count(output) == 1, "Argmax output must contain one index.");
                break;
            case "transformer.embedding-gather-batch":
                var table = tensors["table"];
                Require(table.Tensor.Dimensions.Count == 2, "Table must have shape [N,H].");
                Require(output.Tensor.Dimensions[^1] == table.Tensor.Dimensions[1] &&
                    Count(output) == checked(Count(tensors["indices"]) * table.Tensor.Dimensions[1]),
                    "Output must contain one table row per index.");
                break;
            case "transformer.copy-embedding-rows":
                var rows = Integer(parameters, "rows", positive: true);
                var start = Integer(parameters, "output_start", positive: false);
                var width = input!.Tensor.Dimensions[^1];
                Require(output.Tensor.Dimensions[^1] == width &&
                    checked((long)rows * width) <= Count(input) &&
                    checked(((long)start + rows) * width) <= Count(output),
                    "Copy rows must fit the input and destination with matching widths.");
                break;
            case "transformer.batch-shared-matvec":
                var weight = tensors["weight"].Tensor.Dimensions;
                var x = input!.Tensor.Dimensions;
                var y = output.Tensor.Dimensions;
                Require(weight.Count == 2 && x.Count == 2 && y.Count == 2,
                    "Expected weight [O,I], input [B,I], output [B,O].");
                Require(weight[1] == x[1] && x[0] == y[0] && weight[0] == y[1],
                    "Shared matrix dimensions must match.");
                Require(tensors["control"].Tensor.Dimensions.SequenceEqual([2]),
                    "Control must have shape [2].");
                break;
        }
    }

    public static int Integer(InstructionParameter[] parameters, string name, bool positive)
    {
        var text = parameters.OfType<InstructionAttributeParameter>().Single(p => p.Name == name).Value;
        if (!int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var value) ||
            value < (positive ? 1 : 0))
        {
            throw new InstructionAdaptationException(Id, name,
                $"'{name}' must be a {(positive ? "positive" : "nonnegative")} integer.");
        }

        return value;
    }

    private static InstructionSignature Signature(InstructionPort[] ports, string[]? attributes = null) =>
        new(ports, attributes ?? [], new(ElementType.Float32, ElementType.Float32));
    private static InstructionPort Read(string name) => new(name, ElementType.Float32, ResourceAccess.Read);
    private static InstructionPort Write(string name) => new(name, ElementType.Float32, ResourceAccess.Write);
}
