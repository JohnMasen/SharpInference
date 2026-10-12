using SharpInference.Graphs;

namespace SharpInference.Instructions;

/// <summary>Defines primitive instruction signatures without depending on a logical graph IR.</summary>
public abstract class TierZeroInstruction(Guid collectionId, string name, InstructionTarget target) : Instruction
{
    private static readonly IReadOnlyDictionary<string, string[]> Ports =
        new Dictionary<string, string[]>(StringComparer.Ordinal)
        {
            ["core.copy"] = ["input"],
            ["core.add"] = ["left", "right"],
            ["core.subtract"] = ["left", "right"],
            ["core.multiply"] = ["left", "right"],
            ["core.divide"] = ["left", "right"],
            ["core.maximum"] = ["left", "right"],
            ["core.exp"] = ["input"],
            ["core.tanh"] = ["input"],
            ["core.sigmoid"] = ["input"],
            ["core.rsqrt"] = ["input"],
            ["core.square"] = ["input"],
            ["core.relu"] = ["input"],
            ["core.reduce-sum"] = ["input"],
            ["core.reduce-mean"] = ["input"],
            ["core.mat-vec"] = ["matrix", "input"],
            ["core.matrix-multiply"] = ["left", "right"],
            ["core.affine"] = ["left", "right", "bias"],
            ["core.bias-add"] = ["input", "bias"],
            ["core.gather-row"] = ["table", "index"],
            ["core.tensor.fill"] = [],
            ["core.tensor.cast-f16-f32"] = ["input"],
            ["core.tensor.reshape"] = ["input"],
            ["core.tensor.slice"] = ["input"],
            ["core.tensor.broadcast"] = ["input"],
            ["core.tensor.batched-mat-vec"] = ["matrix", "vector"],
            ["core.tensor.reduce-last-sum"] = ["input"],
            ["core.tensor.reduce-last-mean"] = ["input"],
            ["core.tensor.head-outer"] = ["left", "right"],
        };

    internal static IReadOnlyList<string> OperationNames => Ports.Keys.ToArray();

    public static IReadOnlyList<string> InputPorts(string operation) =>
        Ports.TryGetValue(operation, out var ports)
            ? ports
            : throw new NotSupportedException($"Unknown tier-zero operation '{operation}'.");

    public override Guid CollectionId { get; } = collectionId;
    public override string Name { get; } = name;
    public override InstructionTarget Target { get; } = target;
    public override bool RequiresDispatchIsolation => Name is not (
        "core.copy" or "core.add" or "core.subtract" or "core.multiply" or "core.divide" or
        "core.maximum" or "core.exp" or "core.tanh" or "core.sigmoid" or "core.rsqrt" or
        "core.square" or "core.relu" or "core.tensor.fill" or
        "core.tensor.cast-f16-f32" or "core.tensor.reshape");
    public override IReadOnlyList<InstructionSignature> Signatures { get; } =
        CreateSignatures(collectionId, name);
    public override IReadOnlyList<InstructionIndexBound> IndexBounds { get; } =
        name == "core.gather-row" ? [new("index", "table", 0)] : [];

    protected sealed override void Validate(InstructionParameter[] parameters)
    {
        try
        {
            var tensors = parameters.OfType<InstructionTensorParameter>().ToArray();
            if (tensors.Any(parameter => parameter.ByteOffset != 0))
                throw new InvalidDataException(
                    "Put typed offsets on calls, not instruction parameters.");
            if (tensors.Any(parameter =>
                parameter.Name == "output" && parameter.Access == GraphResourceAccess.Read))
                throw new InvalidDataException("Output is read-only.");
            if (tensors.GroupBy(parameter => parameter.Source).Any(group =>
                group.Select(parameter => (
                    parameter.Tensor.ElementType,
                    Shape: string.Join(",", parameter.Tensor.Dimensions)))
                .Distinct().Count() != 1))
                throw new InvalidDataException(
                    "Aliased source parameters have conflicting tensor descriptors.");
        }
        catch (Exception exception) when (
            exception is InvalidDataException or NotSupportedException or ArgumentException)
        {
            throw new InstructionAdaptationException(
                CollectionId,
                Name,
                exception.Message + " Actual: " + string.Join(", ", parameters.Select(parameter =>
                    parameter is InstructionTensorParameter tensor
                        ? $"{tensor.Name}:{tensor.Tensor.ElementType}" +
                          $"[{string.Join(",", tensor.Tensor.Dimensions)}]/{tensor.Access}"
                        : parameter.Name)),
                exception);
        }
    }

    private static IReadOnlyList<InstructionSignature> CreateSignatures(
        Guid collectionId,
        string name)
    {
        if (collectionId != InstructionCollectionIds.TierZeroFloat32 &&
            collectionId != InstructionCollectionIds.TierZeroFloat16)
            throw new ArgumentException("Unknown T0 collection.", nameof(collectionId));
        var inputs = InputPorts(name);
        if (collectionId == InstructionCollectionIds.TierZeroFloat16 &&
            name == "core.tensor.cast-f16-f32")
            return [];
        var storage = collectionId == InstructionCollectionIds.TierZeroFloat16
            ? GraphElementType.Float16
            : GraphElementType.Float32;
        var output = name == "core.tensor.cast-f16-f32"
            ? GraphElementType.Float32
            : storage;
        var attributes = name switch
        {
            "core.tensor.fill" => new[] { "value" },
            "core.tensor.slice" => ["axis", "start", "length"],
            "core.matrix-multiply" or "core.affine" =>
                ["transpose_left", "transpose_right"],
            _ => [],
        };
        InstructionSignature Signature(
            GraphElementType inputStorage,
            string? matrixPort = null,
            GraphElementType? matrixStorage = null) =>
            new(
                inputs.Select(port => new InstructionPort(
                        port,
                        port == "index"
                            ? GraphElementType.Int32
                            : port == matrixPort
                                ? matrixStorage!.Value
                                : inputStorage,
                        GraphResourceAccess.Read))
                    .Append(new("output", output, GraphResourceAccess.Write))
                    .ToArray(),
                attributes,
                new(GraphElementType.Float32, GraphElementType.Float32));
        var result = new List<InstructionSignature>
        {
            Signature(name == "core.tensor.cast-f16-f32"
                ? GraphElementType.Float16
                : storage),
        };
        if (collectionId == InstructionCollectionIds.TierZeroFloat32 &&
            name is "core.mat-vec" or "core.tensor.batched-mat-vec")
            result.Add(Signature(storage, "matrix", GraphElementType.Float16));
        if (collectionId == InstructionCollectionIds.TierZeroFloat32 &&
            name == "core.gather-row")
            result.Add(Signature(storage, "table", GraphElementType.Float16));
        if (name == "core.mat-vec")
            result.AddRange(result.ToArray().Select(signature => new InstructionSignature(
                signature.Ports.Select(port =>
                    port.Name == "matrix" ? port with { Name = "weight" } : port).ToArray(),
                signature.Attributes,
                signature.Precision)));
        return result;
    }
}
