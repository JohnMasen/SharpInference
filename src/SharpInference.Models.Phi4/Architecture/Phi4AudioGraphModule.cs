using SharpInference.Instructions;
using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;
using SharpInference.Graphs.Logical;

namespace SharpInference.Architectures.Phi4;

/// <summary>Builds the Phi-4 audio graph package with subsampling, Conformer layers and projection; callers explicitly validate it.</summary>
public sealed class Phi4AudioGraphModule : IPhi4GraphModule
{
    private readonly int layers;
    private readonly Phi4AudioSubsamplingGraphModule subsampling;
    private readonly Phi4AudioProjectionGraphModule projection;

    /// <summary>Initializes an audio encoder graph with the specified projector, frame capacity, and conformer layer count.</summary>
    public Phi4AudioGraphModule(Phi4AudioProjector projector = Phi4AudioProjector.Speech, int frames = 352, int layers = 24)
    {
        if (layers is <= 0 or > 24)
        {
            throw new ArgumentOutOfRangeException(nameof(layers));
        }

        this.layers = layers;
        subsampling = new(frames);
        projection = new(projector, frames / 8 + (frames % 8 == 0 ? 0 : 1));
        ArchitectureId = projector == Phi4AudioProjector.Speech ? "phi4.audio.speech" : "phi4.audio.vision";
    }

    /// <inheritdoc/>
    public string ArchitectureId { get; }
    /// <inheritdoc/>

    /// <inheritdoc/>
    public bool CanLoad(IModelTensorCatalog tensors) => subsampling.CanLoad(tensors) && projection.CanLoad(tensors) &&
        tensors.TryGet("a.conv1d.out.weight", out _) && tensors.TryGet("a.rel_attn_bias", out _) &&
        tensors.TryGet($"a.blk.{layers - 1}.ln.weight", out _);

    /// <inheritdoc/>
    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        var sub = subsampling.ReadMetadata(tensors);
        var projected = projection.ReadMetadata(tensors);
        var width = checked((int)sub.Dimensions["channels"]);
        var frequency = checked((int)sub.Dimensions["features"]);
        for (var index = 0; index < 3; index++)
        {
            frequency = frequency / 2 + frequency % 2;
        }

        if (projected.Dimensions["audio_width"] != width)
        {
            throw new InvalidDataException("Phi4 audio projector input width does not match the Conformer width.");
        }

        var relative = tensors.GetRequired("a.rel_attn_bias");
        if (relative.Dimensions.Count != 2 || relative.Dimensions[0] != 1000 || relative.Dimensions[1] <= 0 ||
            width % relative.Dimensions[1] != 0)
        {
            throw new InvalidDataException("Phi4 audio relative bias requires [1000, heads] and width divisible by heads.");
        }

        Require(relative.Name, relative.Dimensions.ToArray());
        var up = tensors.GetRequired("a.blk.0.ffn_in.up.weight");
        if (up.Dimensions.Count != 2 || up.Dimensions[0] <= 0 || up.Dimensions[0] % 2 != 0)
        {
            throw new InvalidDataException("Phi4 audio feed-forward expansion must have positive, paired channels.");
        }

        var intermediate = up.Dimensions[0] / 2;
        Linear("a.conv1d.out", checked(width * frequency), width);
        for (var layer = 0; layer < layers; layer++)
        {
            var prefix = $"a.blk.{layer}.";
            foreach (var part in new[] { "ffn_in", "ffn_out" })
            {
                Norm(prefix + part + ".ln");
                Linear(prefix + part + ".up", width, checked(intermediate * 2));
                Linear(prefix + part + ".down", intermediate, width);
            }
            foreach (var part in new[] { "ln_att", "conv.ln", "ln" })
            {
                Norm(prefix + part);
            }

            foreach (var part in new[] { "attn_q", "attn_k", "attn_v", "attn_out" })
            {
                Linear(prefix + part, width, width);
            }

            Linear(prefix + "conv.glu.pw", width, checked(width * 2));
            Require(prefix + "conv.glu.b1", [width]);
            Require(prefix + "conv.glu.b2", [width]);
            Require(prefix + "conv.dw.weight", [width, 1, 3]);
            Require(prefix + "conv.dw.bias", [width]);
            foreach (var part in new[] { "conv.pw_mid", "conv.pw_ext" })
            {
                Require(prefix + part + ".weight", [width, width, 1]);
                Require(prefix + part + ".bias", [width]);
            }
        }
        var dimensions = sub.Dimensions.ToDictionary(value => value.Key, value => value.Value, StringComparer.Ordinal);
        dimensions["tokens"] = projected.Dimensions["tokens"];
        dimensions["embedding_width"] = projected.Dimensions["embedding_width"];
        dimensions["layers"] = layers;
        dimensions["heads"] = relative.Dimensions[1];
        dimensions["intermediate"] = intermediate;
        return new(ArchitectureId, dimensions);

        void Require(string name, int[] shape)
        {
            var tensor = tensors.GetRequired(name);
            if (tensor.DataType is not (TensorDataType.Float16 or TensorDataType.Float32) || !tensor.Dimensions.SequenceEqual(shape))
            {
                throw new InvalidDataException($"Phi4 audio tensor '{name}' has an incompatible type or shape.");
            }
        }
        void Norm(string prefix)
        {
            Require(prefix + ".weight", [width]);
            Require(prefix + ".bias", [width]);
        }
        void Linear(string prefix, int input, int output)
        {
            Require(prefix + ".weight", [output, input]);
            Require(prefix + ".bias", [output]);
        }
    }

    /// <inheritdoc/>
    public LogicalGraphPackage Build(IModelTensorCatalog tensors)
    {
        var metadata = ReadMetadata(tensors);
        var dimensions = metadata.Dimensions.ToDictionary(value => value.Key, value => checked((int)value.Value), StringComparer.Ordinal);
        var tokens = dimensions["tokens"];
        var width = dimensions["channels"];
        var intermediate = dimensions["intermediate"];
        var builder = new Phi4GraphComposer(new(ArchitectureId, 2, "forward"),
                new GraphModelSignature(ArchitectureId, ArchitectureId + ".empty@1", dimensions))
            .Root("root", GraphSectionTypes.Graph, "Audio encoder and projection", architecture: new(
                Description: "Audio subsampling, Conformer Blocks and decoder-width projection.",
                DefaultView: GraphArchitectureView.Architecture));
        var resources = new Dictionary<string, TensorDescriptor>(StringComparer.Ordinal);
        var weights = new Dictionary<string, string>(StringComparer.Ordinal);
        var zeros = new Dictionary<int, string>();
        var index = 0;
        string hidden = null!;
        string relative = null!;
        builder.Section("embedding", GraphSectionTypes.Stage, "Acoustic Frontend", embeddingScope =>
        {
            Append(embeddingScope, subsampling.Build(tensors), "subsampling", new Dictionary<string, string> { ["output"] = "subsampling.flatten_output" });
            (hidden, relative) = embeddingScope.SectionValue("embedding.projection", GraphSectionTypes.Architecture, "Frame Projection", scope =>
            {
                var projected = Linear(scope, "subsampling.flatten_output", "a.conv1d.out", width);
                var bias = Weight(scope, "a.rel_attn_bias");
                return (projected, bias);
            }, architecture: new(Description: "Projects subsampled frames to Conformer width and prepares relative attention bias."));
        }, architecture: new(Description: "Normalizes and subsamples audio features, then projects frames and prepares relative attention bias."));
        builder.Section("blocks", GraphSectionTypes.Architecture, "Conformer Blocks", blocks =>
        {
            for (var layer = 0; layer < layers; layer++)
            {
                var layerRegion = $"layer.{layer}";
                blocks.Section(layerRegion, GraphSectionTypes.Layer, "Conformer Block", layerScope =>
                {
                    var prefix = $"a.blk.{layer}.";
                    var inputFfn = layerRegion + ".ffn-in";
                    layerScope.Section(inputFfn, GraphSectionTypes.Stage, "Input Feed Forward", inputFfnScope =>
                    {
                        var inputUpdate = FeedForward(inputFfnScope, hidden, prefix + "ffn_in", inputFfn);
                        inputFfnScope.Section(inputFfn + ".residual", GraphSectionTypes.Architecture, "Half-Step Residual", scope =>
                        {
                            hidden = Residual(scope, hidden, inputUpdate, 0.5f);
                        }, architecture: new("residual-add", "Adds half of the FFN output.", "x <- x + 0.5 * ffn_output"));
                    }, architecture: new(Description: "Pre-normalized SwiGLU FFN with a half-scaled residual."));
                    var attentionRegion = layerRegion + ".attention";
                    layerScope.Section(attentionRegion, GraphSectionTypes.Stage, "Attention", attentionScope =>
                    {
                        var normalized = attentionScope.SectionValue(attentionRegion + ".norm", GraphSectionTypes.Architecture, "LayerNorm",
                            scope => Norm(scope, hidden, prefix + "ln_att"),
                            architecture: new(null, "Normalizes the attention input.", null));
                        var projections = attentionRegion + ".qkv";
                        string query = null!;
                        string key = null!;
                        string value = null!;
                        attentionScope.Section(projections, GraphSectionTypes.Architecture, "QKV Projections", projectionsScope =>
                        {
                            query = projectionsScope.SectionValue(projections + ".q", GraphSectionTypes.Architecture, "Query", scope => Linear(scope, normalized, prefix + "attn_q", width), architecture: new(null, "Projects query features.", null));
                            key = projectionsScope.SectionValue(projections + ".k", GraphSectionTypes.Architecture, "Key", scope => Linear(scope, normalized, prefix + "attn_k", width), architecture: new(null, "Projects key features.", null));
                            value = projectionsScope.SectionValue(projections + ".v", GraphSectionTypes.Architecture, "Value", scope => Linear(scope, normalized, prefix + "attn_v", width), architecture: new(null, "Projects value features.", null));
                        }, architecture: new(null, "Projects separate query, key and value branches.", null));
                        var attention = attentionScope.SectionValue(attentionRegion + ".attention", GraphSectionTypes.Architecture, "Relative Attention", scope =>
                        {
                            var attention = Temporary(tokens, width);
                            Emit(scope, Phi4AudioGraphOperations.RelativeAttention, [GraphBindings.Read("query", query), GraphBindings.Read("key", key), GraphBindings.Read("value", value), GraphBindings.Read("relative_bias", relative), GraphBindings.Read("frame_count", "frame_count"), GraphBindings.Write("output", attention)], Attr(("tokens", tokens), ("width", width), ("heads", dimensions["heads"]), ("subsampling", 8)));
                            return attention;
                        }, architecture: new(null, "Uses relative-position bias and valid frame counts to attend over frames.", null));
                        var attentionOutput = attentionScope.SectionValue(attentionRegion + ".output", GraphSectionTypes.Architecture, "Output Projection", scope => Linear(scope, attention, prefix + "attn_out", width), architecture: new(null, "Projects attended features to Conformer width.", null));
                        attentionScope.Section(attentionRegion + ".residual", GraphSectionTypes.Architecture, "Attention Residual", scope =>
                        {
                            hidden = Residual(scope, hidden, attentionOutput, 1f);
                        }, architecture: new("residual-add", "Adds the attention output.", "x <- x + attention_output"));
                    }, architecture: new(Description: "Masked relative-position self-attention over subsampled frames."));
                    var convolution = layerRegion + ".convolution";
                    string convolutionInput = null!;
                    layerScope.Section(convolution, GraphSectionTypes.Stage, "Convolution", convolutionScope =>
                    {
                        convolutionScope.Section(convolution + ".norm", GraphSectionTypes.Architecture, "LayerNorm", scope =>
                        {
                            convolutionInput = Norm(scope, hidden, prefix + "conv.ln");
                        }, architecture: new(null, "Normalizes the convolution input.", null));
                        var wide = convolutionScope.SectionValue(convolution + ".expand", GraphSectionTypes.Architecture, "Pointwise Expansion", scope => Linear(scope, convolutionInput, prefix + "conv.glu.pw", checked(width * 2)), architecture: new(null, "Projects paired gating and value channels.", null));
                        var gated = convolutionScope.SectionValue(convolution + ".gate", GraphSectionTypes.Architecture, "Gated Activation", scope => SwiGlu(scope, wide, width, Weight(scope, prefix + "conv.glu.b1"), Weight(scope, prefix + "conv.glu.b2")), architecture: new(null, "Applies the model's biased SwiGLU gate.", null));
                        var depthwise = convolutionScope.SectionValue(convolution + ".depthwise", GraphSectionTypes.Architecture, "Depthwise Convolution", scope => Conv1D(scope, gated, prefix + "conv.dw", 3, 2, width), architecture: new(null, "Filters each feature channel along the frame axis.", null));
                        var middle = convolutionScope.SectionValue(convolution + ".middle", GraphSectionTypes.Architecture, "Pointwise / Swish", scope =>
                        {
                            var middle = Conv1D(scope, depthwise, prefix + "conv.pw_mid", 1, 0, 1);
                            middle = Activate(scope, middle, width, "swish");
                            return middle;
                        }, architecture: new(null, "Mixes feature channels and applies Swish.", null));
                        var convolutionOutput = convolutionScope.SectionValue(convolution + ".output", GraphSectionTypes.Architecture, "Pointwise Output", scope => Conv1D(scope, middle, prefix + "conv.pw_ext", 1, 0, 1), architecture: new(null, "Projects the convolution output.", null));
                        convolutionScope.Section(convolution + ".residual", GraphSectionTypes.Architecture, "Convolution Residual", scope =>
                        {
                            hidden = Residual(scope, hidden, convolutionOutput, 1f);
                        }, architecture: new("residual-add", "Adds the convolution output.", "x <- x + convolution_output"));
                    }, architecture: new(Description: "Pre-normalized pointwise expansion, gated depthwise convolution and pointwise output."));
                    var outputFfn = layerRegion + ".ffn-out";
                    layerScope.Section(outputFfn, GraphSectionTypes.Stage, "Output Feed Forward", outputFfnScope =>
                    {
                        var outputUpdate = FeedForward(outputFfnScope, hidden, prefix + "ffn_out", outputFfn);
                        outputFfnScope.Section(outputFfn + ".residual", GraphSectionTypes.Architecture, "Half-Step Residual", scope =>
                        {
                            hidden = Residual(scope, hidden, outputUpdate, 0.5f);
                        }, architecture: new("residual-add", "Adds half of the FFN output.", "x <- x + 0.5 * ffn_output"));
                        outputFfnScope.Section(outputFfn + ".final-norm", GraphSectionTypes.Architecture, "Final LayerNorm", scope =>
                        {
                            hidden = Norm(scope, hidden, prefix + "ln");
                        }, architecture: new(null, "Normalizes the completed Conformer Block.", null));
                    }, architecture: new(Description: "Half-scaled output FFN and final Block normalization."));
                }, architecture: new(Description: "Half-step input FFN, relative attention, convolution, half-step output FFN and final normalization."));
            }
        }, architecture: new(Description: "Ordered Conformer Block instances.", DefaultCollapsed: true, RepeatGroup: true));
        builder.Section("output", GraphSectionTypes.Stage, "Audio Projector", outputScope =>
        {
            Append(outputScope, projection.Build(tensors), "projection", new Dictionary<string, string> { ["audio_hidden"] = hidden });
        }, architecture: new(Description: "Projects Conformer features to decoder embeddings."));
        return builder.Build();

        void Append(Phi4GraphComposer.Scope parentScope, LogicalGraphPackage component, string tag, IReadOnlyDictionary<string, string> aliases) =>
            parentScope.AppendDefinition(component, tag, aliases);
        string Temporary(params int[] shape)
        {
            var id = "conformer.node." + index++ + "_output";
            resources.Add(id, F(shape));
            builder.Resource(id, resources[id]);
            return id;
        }
        void Emit(Phi4GraphComposer.Scope scope, OperationId operation, NodeResourceBinding[] bindings, IReadOnlyDictionary<string, string>? attributes = null)
        {
            var output = bindings.FirstOrDefault(binding => binding.Access == ResourceAccess.Write &&
                binding.Resource.EndsWith("_output", StringComparison.Ordinal));
            var id = output is null ? "conformer.node." + index++ : output.Resource[..^"_output".Length];
            scope.AddNode(id, operation, bindings, null, attributes);
        }
        string Weight(Phi4GraphComposer.Scope scope, string name)
        {
            if (weights.TryGetValue(name, out var found))
            {
                return found;
            }

            var tensor = tensors.GetRequired(name);
            var id = "weight." + name;
            var descriptor = new TensorDescriptor(tensor.DataType == TensorDataType.Float16 ? ElementType.Float16 : ElementType.Float32,
                tensor.Dimensions);
            builder.Resource(id, descriptor, scope: ResourceScope.Global, usage: ResourceUsage.Weight, permission: ResourcePermission.ReadOnly, bindingKey: name, name: name);
            resources.Add(id, descriptor);
            if (tensor.DataType == TensorDataType.Float16)
            {
                var converted = Temporary(tensor.Dimensions.ToArray());
                Emit(scope, PortableTensorOperationContracts.CastFp16ToFp32,
                    [GraphBindings.Read("input", id), GraphBindings.Write("output", converted)]);
                id = converted;
            }
            weights.Add(name, id);
            return id;
        }
        string Zero(Phi4GraphComposer.Scope scope, int size)
        {
            if (zeros.TryGetValue(size, out var found))
            {
                return found;
            }

            var id = Temporary(size);
            Emit(scope, PortableTensorOperationContracts.Fill, [GraphBindings.Write("output", id)], new TensorFillValue(0).ToAttributes());
            zeros.Add(size, id);
            return id;
        }
        string Linear(Phi4GraphComposer.Scope scope, string source, string prefix, int outputWidth)
        {
            var output = Temporary(tokens, outputWidth);
            var weight = Weight(scope, prefix + ".weight");
            var bias = Weight(scope, prefix + ".bias");
            scope.Affine(output[..^"_output".Length], source, weight, bias, output, transposeRight: true);
            return output;
        }
        string Norm(Phi4GraphComposer.Scope scope, string source, string prefix)
        {
            var output = Temporary(tokens, width);
            var weight = Weight(scope, prefix + ".weight");
            var bias = Weight(scope, prefix + ".bias");
            Emit(scope, Phi4AudioGraphOperations.LayerNorm,
                [GraphBindings.Read("input", source), GraphBindings.Read("weight", weight), GraphBindings.Read("bias", bias),
                 GraphBindings.Write("output", output)], Attr(("rows", tokens), ("width", width), ("epsilon", 1e-5f)));
            return output;
        }
        string SwiGlu(Phi4GraphComposer.Scope scope, string source, int size, string first, string second)
        {
            var output = Temporary(tokens, size);
            Emit(scope, Phi4AudioGraphOperations.SwiGlu,
                [GraphBindings.Read("input", source), GraphBindings.Read("bias_first", first), GraphBindings.Read("bias_second", second),
                 GraphBindings.Write("output", output)], Attr(("rows", tokens), ("width", size)));
            return output;
        }
        string FeedForward(Phi4GraphComposer.Scope parentScope, string source, string prefix, string region)
        {
            var normalized = parentScope.SectionValue(region + ".norm", GraphSectionTypes.Architecture, "LayerNorm",
                scope => Norm(scope, source, prefix + ".ln"),
                architecture: new(Description: "Normalizes the feed-forward input."));
            var wide = parentScope.SectionValue(region + ".up", GraphSectionTypes.Architecture, "Gate / Up Projection",
                scope => Linear(scope, normalized, prefix + ".up", checked(intermediate * 2)),
                architecture: new(Description: "Projects paired gate and value channels."));
            var gated = parentScope.SectionValue(region + ".activation", GraphSectionTypes.Architecture, "SwiGLU",
                scope => SwiGlu(scope, wide, intermediate, Zero(scope, intermediate), Zero(scope, intermediate)),
                architecture: new(Description: "Applies the gated activation.", Formula: "h = silu(gate) .* up"));
            return parentScope.SectionValue(region + ".down", GraphSectionTypes.Architecture, "Down Projection",
                scope => Linear(scope, gated, prefix + ".down", width),
                architecture: new(Description: "Projects back to Conformer width."));
        }
        string Residual(Phi4GraphComposer.Scope scope, string source, string update, float scale)
        {
            var output = Temporary(tokens, width);
            var count = checked(tokens * width);
            Emit(scope, Phi4AudioGraphOperations.Residual,
                [Flat("hidden", source, ResourceAccess.Read, count), Flat("update", update, ResourceAccess.Read, count),
                 Flat("output", output, ResourceAccess.Write, count)], Attr(("count", count), ("scale", scale)));
            return output;
        }
        string Conv1D(Phi4GraphComposer.Scope scope, string source, string prefix, int kernel, int padding, int groups)
        {
            var outputLength = checked(tokens + 2 * padding - kernel + 1);
            var output = Temporary(outputLength, width);
            var weight = Weight(scope, prefix + ".weight");
            var bias = Weight(scope, prefix + ".bias");
            Emit(scope, Phi4AudioGraphOperations.Conv1D,
                [new("input", source, ResourceAccess.Read, View: new(0, F(tokens, width))),
                 GraphBindings.Read("weight", weight), GraphBindings.Read("bias", bias), GraphBindings.Write("output", output)],
                Attr(("input_length", tokens), ("input_channels", width), ("output_length", outputLength), ("output_channels", width),
                    ("kernel", kernel), ("padding", padding), ("stride", 1), ("groups", groups), ("activation", "none")));
            return output;
        }
        string Activate(Phi4GraphComposer.Scope scope, string source, int size, string activation)
        {
            var output = Temporary(tokens, size);
            var zero = Zero(scope, size);
            var count = checked(tokens * size);
            Emit(scope, Phi4AudioGraphOperations.BiasActivation,
                [Flat("input", source, ResourceAccess.Read, count), GraphBindings.Read("bias", zero),
                 Flat("output", output, ResourceAccess.Write, count)], Attr(("count", count), ("width", size), ("activation", activation)));
            return output;
        }
    }

    /// <inheritdoc/>

    private static NodeResourceBinding Flat(string port, string id, ResourceAccess access, int count) =>
        new(port, id, access, View: new(0, F(count)));
    private static TensorDescriptor F(params int[] dimensions) => new(ElementType.Float32, dimensions);
    private static IReadOnlyDictionary<string, string> Attr(params (string Name, object Value)[] values) =>
        values.ToDictionary(value => value.Name, value => Convert.ToString(value.Value, CultureInfo.InvariantCulture)!, StringComparer.Ordinal);
}
