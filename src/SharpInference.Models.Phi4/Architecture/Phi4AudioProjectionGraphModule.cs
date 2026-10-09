using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Architectures.Phi4;

public enum Phi4AudioProjector
{
    Speech,
    Vision,
}

public sealed class Phi4AudioProjectionGraphModule : IModelGraphModule, IGraphOperationValidator
{
    private readonly int tokens;
    private readonly string prefix;

    public Phi4AudioProjectionGraphModule(Phi4AudioProjector projector, int tokens)
    {
        if (tokens <= 0) throw new ArgumentOutOfRangeException(nameof(tokens));
        prefix = projector switch
        {
            Phi4AudioProjector.Speech => "mm.a.mlp",
            Phi4AudioProjector.Vision => "mm.a.vis",
            _ => throw new ArgumentOutOfRangeException(nameof(projector)),
        };
        this.tokens = tokens;
        ArchitectureId = projector == Phi4AudioProjector.Speech
            ? "phi4.audio.projection.speech" : "phi4.audio.projection.vision";
    }

    public string ArchitectureId { get; }
    public IGraphOperationValidator OperationValidator => this;

    public bool CanLoad(IModelTensorCatalog tensors) =>
        new[] { ".0.weight", ".0.bias", ".2.weight", ".2.bias" }
            .All(suffix => tensors.TryGet(prefix + suffix, out _));

    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var first = tensors.GetRequired(prefix + ".0.weight");
        var second = tensors.GetRequired(prefix + ".2.weight");
        if (first.Dimensions.Count != 2 || second.Dimensions.Count != 2 ||
            first.Dimensions.Any(size => size <= 0) || second.Dimensions.Any(size => size <= 0) ||
            first.Dimensions[0] != second.Dimensions[1])
            throw new InvalidDataException("Phi4 audio projection requires compatible positive matrix dimensions.");
        Require(first);
        Require(second);
        Require(tensors.GetRequired(prefix + ".0.bias"), [first.Dimensions[0]]);
        Require(tensors.GetRequired(prefix + ".2.bias"), [second.Dimensions[0]]);
        return new(ArchitectureId, new Dictionary<string, long>
        {
            ["tokens"] = tokens, ["audio_width"] = first.Dimensions[1],
            ["hidden_width"] = first.Dimensions[0], ["embedding_width"] = second.Dimensions[0],
        });

        static void Require(IModelTensor tensor, int[]? dimensions = null)
        {
            if (tensor.DataType is not (TensorDataType.Float16 or TensorDataType.Float32) ||
                dimensions is not null && !tensor.Dimensions.SequenceEqual(dimensions))
                throw new InvalidDataException($"Phi4 audio projection tensor '{tensor.Name}' has an incompatible type or shape.");
        }
    }

    public LogicalGraph Build(IModelTensorCatalog tensors)
    {
        var metadata = ReadMetadata(tensors);
        var dimensions = metadata.Dimensions.ToDictionary(value => value.Key,
            value => checked((int)value.Value), StringComparer.Ordinal);
        var inputWidth = dimensions["audio_width"];
        var hiddenWidth = dimensions["hidden_width"];
        var outputWidth = dimensions["embedding_width"];
        var builder = new LogicalGraphBuilder(new(ArchitectureId, 2, "forward"),
                new GraphModelSignature(ArchitectureId, ArchitectureId + ".empty@1", dimensions))
            .AddRegion("root", GraphRegionTypes.Graph, "Audio projection")
            .AddResource("audio_hidden", "Conformer output", GraphResourceKind.Input, GraphResourceLifetime.External,
                F(tokens, inputWidth), graphInput: true)
            .AddResource("output", "Projected audio embeddings", GraphResourceKind.Output, GraphResourceLifetime.External,
                F(tokens, outputWidth), graphOutput: true);
        string? previous = null;
        Temporary("projector_linear", tokens, hiddenWidth);
        Temporary("projector_hidden", tokens, hiddenWidth);
        var firstWeight = Weight(prefix + ".0.weight");
        var firstBias = Weight(prefix + ".0.bias");
        var secondWeight = Weight(prefix + ".2.weight");
        var secondBias = Weight(prefix + ".2.bias");
        var transpose = new Dictionary<string, string> { ["transpose_left"] = "false", ["transpose_right"] = "true" };
        Emit("projector_up", PrimitiveGraphOperations.MatrixMultiply,
            [GraphBindings.Read("left", "audio_hidden"), GraphBindings.Read("right", firstWeight),
             GraphBindings.Write("output", "projector_linear")], transpose);
        Emit("projector_gelu", Phi4AudioGraphOperations.BiasActivation,
            [new("input", new("projector_linear"), GraphResourceAccess.Read, View: new(0, F(checked(tokens * hiddenWidth)))),
             GraphBindings.Read("bias", firstBias),
             new("output", new("projector_hidden"), GraphResourceAccess.Write, View: new(0, F(checked(tokens * hiddenWidth))))],
            new Dictionary<string, string>
            {
                ["count"] = checked(tokens * hiddenWidth).ToString(CultureInfo.InvariantCulture),
                ["width"] = hiddenWidth.ToString(CultureInfo.InvariantCulture), ["activation"] = "gelu",
            });
        Emit("projector_down", PrimitiveGraphOperations.Affine,
            [GraphBindings.Read("left", "projector_hidden"), GraphBindings.Read("right", secondWeight),
             GraphBindings.Read("bias", secondBias), GraphBindings.Write("output", "output")], transpose);
        var graph = builder.Build();
        GraphValidator.Validate(graph, this);
        return graph;

        void Temporary(string id, params int[] shape) => builder.AddResource(id, id,
            GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, F(shape));
        string Weight(string name)
        {
            var tensor = tensors.GetRequired(name);
            var id = "weight." + name;
            builder.AddResource(id, name, GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(tensor.DataType == TensorDataType.Float16 ? GraphElementType.Float16 : GraphElementType.Float32,
                    tensor.Dimensions), name);
            if (tensor.DataType == TensorDataType.Float32) return id;
            var converted = "converted." + name;
            Temporary(converted, tensor.Dimensions.ToArray());
            Emit("cast." + name, PortableTensorOperationContracts.CastFp16ToFp32,
                [GraphBindings.Read("input", id), GraphBindings.Write("output", converted)], new Dictionary<string, string>());
            return converted;
        }
        void Emit(string id, GraphOperationId operation, NodeResourceBinding[] bindings, IReadOnlyDictionary<string, string> attributes)
        {
            builder.AddNode(id, operation, "root", bindings, previous is null ? [] : [previous], attributes);
            previous = id;
        }
    }

    public void Validate(GraphOperationValidationContext context)
    {
        if (!context.Operation.Name.StartsWith("phi4.audio.", StringComparison.Ordinal)) return;
        if (context.Operation != Phi4AudioGraphOperations.BiasActivation)
            throw new NotSupportedException($"No Phi4 audio projection contract is registered for '{context.Operation}'.");
        if (!NumericTypeCompatibility.Satisfies(new(GraphElementType.Float32, GraphElementType.Float32), context.Requirements) ||
            !new[] { "count", "width", "activation" }.ToHashSet(StringComparer.Ordinal).SetEquals(context.Attributes.Keys) ||
            !int.TryParse(context.Attributes["count"], NumberStyles.None, CultureInfo.InvariantCulture, out var count) || count <= 0 ||
            !int.TryParse(context.Attributes["width"], NumberStyles.None, CultureInfo.InvariantCulture, out var width) || width <= 0 ||
            count % width != 0 || context.Attributes["activation"] != "gelu" ||
            context.Bindings.Count != 3 ||
            !new[] { "input", "bias", "output" }.ToHashSet(StringComparer.Ordinal).SetEquals(context.Bindings.Select(binding => binding.Port)) ||
            context.Bindings.Any(binding => binding.Access != (binding.Port == "output" ? GraphResourceAccess.Write : GraphResourceAccess.Read)))
            throw new InvalidDataException($"Phi4 audio projection node '{context.NodeId}' has incompatible precision, ports or attributes.");
        foreach (var port in new[] { "input", "bias", "output" })
        {
            var tensor = context.Tensor(port);
            if (tensor.ElementType != GraphElementType.Float32 || tensor.Layout != "dense" ||
                !tensor.Dimensions.SequenceEqual([port == "bias" ? width : count]))
                throw new InvalidDataException($"Phi4 audio projection node '{context.NodeId}' has an incompatible '{port}' tensor.");
        }
    }

    private static TensorDescriptor F(params int[] dimensions) => new(GraphElementType.Float32, dimensions);
}
