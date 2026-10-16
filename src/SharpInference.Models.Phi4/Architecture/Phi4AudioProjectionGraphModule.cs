using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;
using SharpInference.Graphs.Logical;

namespace SharpInference.Architectures.Phi4;

/// <summary>Selects the audio projection weights used to map encoder output into text embeddings.</summary>
public enum Phi4AudioProjector
{
    /// <summary>Projects audio using the speech-mode weights.</summary>
    Speech,
    /// <summary>Projects audio using the vision-mode weights.</summary>
    Vision,
}

/// <summary>Builds and validates the speech or vision audio projection graph.</summary>
public sealed class Phi4AudioProjectionGraphModule : IPhi4GraphModule
{
    private readonly int tokens;
    private readonly string prefix;

    /// <summary>Initializes an audio projection graph for the selected projection weights and token count.</summary>
    public Phi4AudioProjectionGraphModule(Phi4AudioProjector projector, int tokens)
    {
        if (tokens <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokens));
        }

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

    /// <inheritdoc/>
    public string ArchitectureId { get; }
    /// <inheritdoc/>

    /// <inheritdoc/>
    public bool CanLoad(IModelTensorCatalog tensors) =>
        new[] { ".0.weight", ".0.bias", ".2.weight", ".2.bias" }
            .All(suffix => tensors.TryGet(prefix + suffix, out _));

    /// <inheritdoc/>
    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var first = tensors.GetRequired(prefix + ".0.weight");
        var second = tensors.GetRequired(prefix + ".2.weight");
        if (first.Dimensions.Count != 2 || second.Dimensions.Count != 2 ||
            first.Dimensions.Any(size => size <= 0) || second.Dimensions.Any(size => size <= 0) ||
            first.Dimensions[0] != second.Dimensions[1])
        {
            throw new InvalidDataException("Phi4 audio projection requires compatible positive matrix dimensions.");
        }

        Require(first);
        Require(second);
        Require(tensors.GetRequired(prefix + ".0.bias"), [first.Dimensions[0]]);
        Require(tensors.GetRequired(prefix + ".2.bias"), [second.Dimensions[0]]);
        return new(ArchitectureId, new Dictionary<string, long>
        {
            ["tokens"] = tokens,
            ["audio_width"] = first.Dimensions[1],
            ["hidden_width"] = first.Dimensions[0],
            ["embedding_width"] = second.Dimensions[0],
        });

        static void Require(IModelTensor tensor, int[]? dimensions = null)
        {
            if (tensor.DataType is not (TensorDataType.Float16 or TensorDataType.Float32) ||
                dimensions is not null && !tensor.Dimensions.SequenceEqual(dimensions))
            {
                throw new InvalidDataException($"Phi4 audio projection tensor '{tensor.Name}' has an incompatible type or shape.");
            }
        }
    }

    /// <inheritdoc/>
    public LogicalGraphDefinition Build(IModelTensorCatalog tensors)
    {
        var metadata = ReadMetadata(tensors);
        var dimensions = metadata.Dimensions.ToDictionary(value => value.Key,
            value => checked((int)value.Value), StringComparer.Ordinal);
        var inputWidth = dimensions["audio_width"];
        var hiddenWidth = dimensions["hidden_width"];
        var outputWidth = dimensions["embedding_width"];
        var builder = new Phi4GraphDefinitionBuilder(new(ArchitectureId, 2, "forward"),
                new GraphModelSignature(ArchitectureId, ArchitectureId + ".empty@1", dimensions))
            .Root("root", GraphSectionTypes.Graph, "Audio projection", architecture: new(
                Description: "Two-layer audio-to-decoder projector with biased GELU.",
                DefaultView: GraphArchitectureView.Architecture))
            .AddResource("audio_hidden", "Conformer output", GraphResourceKind.Input, GraphResourceLifetime.External,
                F(tokens, inputWidth), graphInput: true)
            .AddResource("output", "Projected audio embeddings", GraphResourceKind.Output, GraphResourceLifetime.External,
                F(tokens, outputWidth), graphOutput: true);
        Temporary("projector_up_output", tokens, hiddenWidth);
        Temporary("projector_gelu_output", tokens, hiddenWidth);
        var (firstWeight, firstBias, secondWeight, secondBias) = builder.SectionValue(
            "weights", GraphSectionTypes.Architecture, "Weight Preparation", scope =>
            {
                var upWeight = Weight(scope, prefix + ".0.weight");
                var upBias = Weight(scope, prefix + ".0.bias");
                var downWeight = Weight(scope, prefix + ".2.weight");
                var downBias = Weight(scope, prefix + ".2.bias");
                return (upWeight, upBias, downWeight, downBias);
            }, architecture: new(Description: "Converts projector weights to FP32 when required."));
        builder.SectionValue("up", GraphSectionTypes.Architecture, "Up Projection",
            scoped => scoped.MatrixMultiply("projector_up", "audio_hidden", firstWeight, "projector_up_output", transposeRight: true),
            architecture: new(Description: "Projects Conformer features to the projector hidden width."));
        builder.Section("activation", GraphSectionTypes.Architecture, "Bias / GELU", scope =>
        {
            Emit(scope, "projector_gelu", Phi4AudioGraphOperations.BiasActivation,
                [new("input", new("projector_up_output"), GraphResourceAccess.Read, View: new(0, F(checked(tokens * hiddenWidth)))),
                 GraphBindings.Read("bias", firstBias),
                 new("output", new("projector_gelu_output"), GraphResourceAccess.Write, View: new(0, F(checked(tokens * hiddenWidth))))],
                new Dictionary<string, string>
                {
                    ["count"] = checked(tokens * hiddenWidth).ToString(CultureInfo.InvariantCulture),
                    ["width"] = hiddenWidth.ToString(CultureInfo.InvariantCulture),
                    ["activation"] = "gelu",
                });
        }, architecture: new(Description: "Adds the hidden bias and applies GELU.", Formula: "h = gelu(x W_up^T + b_up)"));
        builder.SectionValue("down", GraphSectionTypes.Architecture, "Down Projection",
            scoped => scoped.Affine("projector_down", "projector_gelu_output", secondWeight, secondBias, "output", transposeRight: true),
            architecture: new(Description: "Projects activated features to decoder embeddings.", Formula: "output = h W_down^T + b_down"));
        return builder.Build();

        void Temporary(string id, params int[] shape) => builder.AddResource(id, id,
            GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, F(shape));
        string Weight(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string name)
        {
            var tensor = tensors.GetRequired(name);
            var id = "weight." + name;
            builder.AddResource(id, name, GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(tensor.DataType == TensorDataType.Float16 ? GraphElementType.Float16 : GraphElementType.Float32,
                    tensor.Dimensions), name);
            if (tensor.DataType == TensorDataType.Float32)
            {
                return id;
            }

            var converted = "cast." + name + "_output";
            Temporary(converted, tensor.Dimensions.ToArray());
            Emit(scope, "cast." + name, PortableTensorOperationContracts.CastFp16ToFp32,
                [GraphBindings.Read("input", id), GraphBindings.Write("output", converted)], new Dictionary<string, string>());
            return converted;
        }
        static void Emit(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string id, GraphOperationId operation, NodeResourceBinding[] bindings, IReadOnlyDictionary<string, string> attributes)
        {
            scope.AddNode(id, operation, bindings, null, attributes);
        }
    }

    private static TensorDescriptor F(params int[] dimensions) => new(GraphElementType.Float32, dimensions);
}
