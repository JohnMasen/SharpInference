using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Architectures.Phi4;

public sealed class Phi4AudioGraphModule : IModelGraphModule, IGraphOperationValidator
{
    private readonly int layers;
    private readonly Phi4AudioSubsamplingGraphModule subsampling;
    private readonly Phi4AudioProjectionGraphModule projection;

    public Phi4AudioGraphModule(Phi4AudioProjector projector = Phi4AudioProjector.Speech, int frames = 352, int layers = 24)
    {
        if (layers is <= 0 or > 24) throw new ArgumentOutOfRangeException(nameof(layers));
        this.layers = layers;
        subsampling = new(frames);
        projection = new(projector, frames / 8 + (frames % 8 == 0 ? 0 : 1));
        ArchitectureId = projector == Phi4AudioProjector.Speech ? "phi4.audio.speech" : "phi4.audio.vision";
    }

    public string ArchitectureId { get; }
    public IGraphOperationValidator OperationValidator => this;

    public bool CanLoad(IModelTensorCatalog tensors) => subsampling.CanLoad(tensors) && projection.CanLoad(tensors) &&
        tensors.TryGet("a.conv1d.out.weight", out _) && tensors.TryGet("a.rel_attn_bias", out _) &&
        tensors.TryGet($"a.blk.{layers - 1}.ln.weight", out _);

    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        var sub = subsampling.ReadMetadata(tensors);
        var projected = projection.ReadMetadata(tensors);
        var width = checked((int)sub.Dimensions["channels"]);
        var frequency = checked((int)sub.Dimensions["features"]);
        for (var index = 0; index < 3; index++) frequency = frequency / 2 + frequency % 2;
        if (projected.Dimensions["audio_width"] != width)
            throw new InvalidDataException("Phi4 audio projector input width does not match the Conformer width.");
        var relative = tensors.GetRequired("a.rel_attn_bias");
        if (relative.Dimensions.Count != 2 || relative.Dimensions[0] != 1000 || relative.Dimensions[1] <= 0 ||
            width % relative.Dimensions[1] != 0)
            throw new InvalidDataException("Phi4 audio relative bias requires [1000, heads] and width divisible by heads.");
        Require(relative.Name, relative.Dimensions.ToArray());
        var up = tensors.GetRequired("a.blk.0.ffn_in.up.weight");
        if (up.Dimensions.Count != 2 || up.Dimensions[0] <= 0 || up.Dimensions[0] % 2 != 0)
            throw new InvalidDataException("Phi4 audio feed-forward expansion must have positive, paired channels.");
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
            foreach (var part in new[] { "ln_att", "conv.ln", "ln" }) Norm(prefix + part);
            foreach (var part in new[] { "attn_q", "attn_k", "attn_v", "attn_out" }) Linear(prefix + part, width, width);
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
                throw new InvalidDataException($"Phi4 audio tensor '{name}' has an incompatible type or shape.");
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

    public LogicalGraph Build(IModelTensorCatalog tensors)
    {
        var metadata = ReadMetadata(tensors);
        var dimensions = metadata.Dimensions.ToDictionary(value => value.Key, value => checked((int)value.Value), StringComparer.Ordinal);
        var tokens = dimensions["tokens"];
        var width = dimensions["channels"];
        var intermediate = dimensions["intermediate"];
        var builder = new LogicalGraphBuilder(new(ArchitectureId, 2, "forward"),
                new GraphModelSignature(ArchitectureId, ArchitectureId + ".empty@1", dimensions))
            .AddRegion("root", GraphRegionTypes.Graph, "Audio encoder and projection", architecture: new(
                Description: "Audio subsampling, Conformer Blocks and decoder-width projection.",
                DefaultView: GraphArchitectureView.Architecture));
        var resources = new Dictionary<string, TensorDescriptor>(StringComparer.Ordinal);
        var weights = new Dictionary<string, string>(StringComparer.Ordinal);
        var zeros = new Dictionary<int, string>();
        var index = 0;
        builder.AddRegion("blocks", GraphRegionTypes.Architecture, "Conformer Blocks", "root",
            architecture: new(Description: "Ordered Conformer Block instances.", DefaultCollapsed: true, RepeatGroup: true));
        var currentRegion = "embedding";
        builder.AddRegion(currentRegion, GraphRegionTypes.Stage, "Acoustic Frontend", "root",
            architecture: new(Description: "Normalizes and subsamples audio features, then projects frames and prepares relative attention bias."));
        Append(subsampling.Build(tensors), "subsampling", new Dictionary<string, string> { ["output"] = "subsampling.flatten_output" });
        var (hidden, relative) = builder.WithRegion("embedding.projection", GraphRegionTypes.Architecture, "Frame Projection", _ =>
        {
            var projected = Linear("subsampling.flatten_output", "a.conv1d.out", width);
            var bias = Weight("a.rel_attn_bias");
            return (projected, bias);
        }, parentId: "embedding", architecture: new(Description: "Projects subsampled frames to Conformer width and prepares relative attention bias."));
        for (var layer = 0; layer < layers; layer++)
        {
            var layerRegion = $"layer.{layer}";
            builder.AddRegion(layerRegion, GraphRegionTypes.Layer, "Conformer Block", "blocks",
                architecture: new(Description: "Half-step input FFN, relative attention, convolution, half-step output FFN and final normalization."));
            currentRegion = layerRegion + ".ffn-in";
            builder.AddRegion(currentRegion, GraphRegionTypes.Stage, "Input Feed Forward", layerRegion,
                architecture: new(Description: "Pre-normalized SwiGLU FFN with a half-scaled residual."));
            var inputFfn = currentRegion;
            var prefix = $"a.blk.{layer}.";
            var inputUpdate = FeedForward(hidden, prefix + "ffn_in", inputFfn);
            builder.WithRegion(inputFfn + ".residual", GraphRegionTypes.Architecture, "Half-Step Residual", _ =>
            {
                hidden = Residual(hidden, inputUpdate, 0.5f);
            }, parentId: inputFfn, architecture: new("residual-add", "Adds half of the FFN output.", "x <- x + 0.5 * ffn_output"));
            currentRegion = layerRegion + ".attention";
            builder.AddRegion(currentRegion, GraphRegionTypes.Stage, "Attention", layerRegion,
                architecture: new(Description: "Masked relative-position self-attention over subsampled frames."));
            var attentionRegion = currentRegion;
            var (normalized, projections) = builder.WithRegion(attentionRegion + ".norm", GraphRegionTypes.Architecture, "LayerNorm", _ =>
            {
                var normalized = Norm(hidden, prefix + "ln_att");
                var projections = attentionRegion + ".qkv";
                return (normalized, projections);
            }, parentId: attentionRegion, architecture: new(null, "Normalizes the attention input.", null));
            var (query, key, value) = builder.WithRegion(projections, GraphRegionTypes.Architecture, "QKV Projections", _ =>
            {
                var query = builder.WithRegion(projections + ".q", GraphRegionTypes.Architecture, "Query", _ => Linear(normalized, prefix + "attn_q", width), architecture: new(null, "Projects query features.", null));
                var key = builder.WithRegion(projections + ".k", GraphRegionTypes.Architecture, "Key", _ => Linear(normalized, prefix + "attn_k", width), architecture: new(null, "Projects key features.", null));
                var value = builder.WithRegion(projections + ".v", GraphRegionTypes.Architecture, "Value", _ => Linear(normalized, prefix + "attn_v", width), architecture: new(null, "Projects value features.", null));
                return (query, key, value);
            }, parentId: attentionRegion, architecture: new(null, "Projects separate query, key and value branches.", null));
            var attention = builder.WithRegion(attentionRegion + ".attention", GraphRegionTypes.Architecture, "Relative Attention", _ =>
            {
                var attention = Temporary(tokens, width);
                Emit(Phi4AudioGraphOperations.RelativeAttention, [GraphBindings.Read("query", query), GraphBindings.Read("key", key), GraphBindings.Read("value", value), GraphBindings.Read("relative_bias", relative), GraphBindings.Read("frame_count", "frame_count"), GraphBindings.Write("output", attention)], Attr(("tokens", tokens), ("width", width), ("heads", dimensions["heads"]), ("subsampling", 8)));
                return attention;
            }, parentId: attentionRegion, architecture: new(null, "Uses relative-position bias and valid frame counts to attend over frames.", null));
            var attentionOutput = builder.WithRegion(attentionRegion + ".output", GraphRegionTypes.Architecture, "Output Projection", _ => Linear(attention, prefix + "attn_out", width), parentId: attentionRegion, architecture: new(null, "Projects attended features to Conformer width.", null));
            builder.WithRegion(attentionRegion + ".residual", GraphRegionTypes.Architecture, "Attention Residual", _ =>
            {
                hidden = Residual(hidden, attentionOutput, 1f);
            }, parentId: attentionRegion, architecture: new("residual-add", "Adds the attention output.", "x <- x + attention_output"));
            currentRegion = layerRegion + ".convolution";
            builder.AddRegion(currentRegion, GraphRegionTypes.Stage, "Convolution", layerRegion,
                architecture: new(Description: "Pre-normalized pointwise expansion, gated depthwise convolution and pointwise output."));
            var convolution = currentRegion;
            builder.WithRegion(convolution + ".norm", GraphRegionTypes.Architecture, "LayerNorm", _ =>
            {
                normalized = Norm(hidden, prefix + "conv.ln");
            }, parentId: convolution, architecture: new(null, "Normalizes the convolution input.", null));
            var wide = builder.WithRegion(convolution + ".expand", GraphRegionTypes.Architecture, "Pointwise Expansion", _ => Linear(normalized, prefix + "conv.glu.pw", checked(width * 2)), parentId: convolution, architecture: new(null, "Projects paired gating and value channels.", null));
            var gated = builder.WithRegion(convolution + ".gate", GraphRegionTypes.Architecture, "Gated Activation", _ => SwiGlu(wide, width, Weight(prefix + "conv.glu.b1"), Weight(prefix + "conv.glu.b2")), parentId: convolution, architecture: new(null, "Applies the model's biased SwiGLU gate.", null));
            var depthwise = builder.WithRegion(convolution + ".depthwise", GraphRegionTypes.Architecture, "Depthwise Convolution", _ => Conv1D(gated, prefix + "conv.dw", 3, 2, width), parentId: convolution, architecture: new(null, "Filters each feature channel along the frame axis.", null));
            var middle = builder.WithRegion(convolution + ".middle", GraphRegionTypes.Architecture, "Pointwise / Swish", _ =>
            {
                var middle = Conv1D(depthwise, prefix + "conv.pw_mid", 1, 0, 1);
                middle = Activate(middle, width, "swish");
                return middle;
            }, parentId: convolution, architecture: new(null, "Mixes feature channels and applies Swish.", null));
            var convolutionOutput = builder.WithRegion(convolution + ".output", GraphRegionTypes.Architecture, "Pointwise Output", _ => Conv1D(middle, prefix + "conv.pw_ext", 1, 0, 1), parentId: convolution, architecture: new(null, "Projects the convolution output.", null));
            builder.WithRegion(convolution + ".residual", GraphRegionTypes.Architecture, "Convolution Residual", _ =>
            {
                hidden = Residual(hidden, convolutionOutput, 1f);
            }, parentId: convolution, architecture: new("residual-add", "Adds the convolution output.", "x <- x + convolution_output"));
            currentRegion = layerRegion + ".ffn-out";
            builder.AddRegion(currentRegion, GraphRegionTypes.Stage, "Output Feed Forward", layerRegion,
                architecture: new(Description: "Half-scaled output FFN and final Block normalization."));
            var outputFfn = currentRegion;
            var outputUpdate = FeedForward(hidden, prefix + "ffn_out", outputFfn);
            builder.WithRegion(outputFfn + ".residual", GraphRegionTypes.Architecture, "Half-Step Residual", _ =>
            {
                hidden = Residual(hidden, outputUpdate, 0.5f);
            }, parentId: outputFfn, architecture: new("residual-add", "Adds half of the FFN output.", "x <- x + 0.5 * ffn_output"));
            builder.WithRegion(outputFfn + ".final-norm", GraphRegionTypes.Architecture, "Final LayerNorm", _ =>
            {
                hidden = Norm(hidden, prefix + "ln");
            }, parentId: outputFfn, architecture: new(null, "Normalizes the completed Conformer Block.", null));
        }
        currentRegion = "output";
        builder.AddRegion(currentRegion, GraphRegionTypes.Stage, "Audio Projector", "root",
            architecture: new(Description: "Projects Conformer features to decoder embeddings."));
        Append(projection.Build(tensors), "projection", new Dictionary<string, string> { ["audio_hidden"] = hidden });
        var graph = GraphLayerReuse.Extract(builder.BuildSequential(), region =>
            region.Type == GraphRegionTypes.Layer ? "Phi4.Audio.Layer" : null);
        GraphValidator.Validate(graph, this);
        return graph;

        void Append(LogicalGraph component, string tag, IReadOnlyDictionary<string, string> aliases)
        {
            var temporaries = component.Resources.Where(resource => resource.Kind == GraphResourceKind.Temporary)
                .Select(resource => resource.Id).ToHashSet();
            string Map(ResourceId id) => aliases.TryGetValue(id.Value, out var alias) ? alias :
                temporaries.Contains(id) ? tag + "." + id.Value : id.Value;
            foreach (var resource in component.Resources)
            {
                var id = Map(resource.Id);
                if (resources.TryGetValue(id, out var existing))
                {
                    if (existing.ElementType != resource.Tensor.ElementType || existing.Layout != resource.Tensor.Layout ||
                        !existing.Dimensions.SequenceEqual(resource.Tensor.Dimensions))
                        throw new InvalidDataException($"Phi4 audio component '{tag}' has an incompatible port '{resource.Id}'.");
                    continue;
                }
                var internalOutput = aliases.ContainsKey(resource.Id.Value) && component.Outputs.Contains(resource.Id);
                builder.AddResource(id, resource.Name, internalOutput ? GraphResourceKind.Temporary : resource.Kind,
                    internalOutput ? GraphResourceLifetime.Invocation : resource.Lifetime, resource.Tensor, resource.BindingKey,
                    component.Inputs.Contains(resource.Id), !internalOutput && component.Outputs.Contains(resource.Id));
                resources.Add(id, resource.Tensor);
            }
            foreach (var region in component.Regions)
                builder.AddRegion(tag + "." + region.Id.Value, region.Type == GraphRegionTypes.Graph ? GraphRegionTypes.Stage : region.Type,
                    region.Name, region.ParentId is { } parent ? tag + "." + parent.Value : currentRegion, region.Role, region.Attributes,
                    region.Architecture?.MapResources(id => new ResourceId(Map(id))));
            foreach (var node in component.Nodes)
            {
                var id = tag + "." + node.Id.Value;
                builder.AddNode(id, node.Operation, tag + "." + node.Region.Value, node.Resources.Select(binding => binding with { Resource = new(Map(binding.Resource)) }),
                    null, node.Attributes, node.Requirements);
            }
        }
        string Temporary(params int[] shape)
        {
            var id = "conformer.node." + index++ + "_output";
            resources.Add(id, F(shape));
            builder.AddResource(id, id, GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, resources[id]);
            return id;
        }
        void Emit(GraphOperationId operation, NodeResourceBinding[] bindings, IReadOnlyDictionary<string, string>? attributes = null)
        {
            var output = bindings.FirstOrDefault(binding => binding.Access == GraphResourceAccess.Write &&
                binding.Resource.Value.EndsWith("_output", StringComparison.Ordinal));
            var id = output is null ? "conformer.node." + index++ : output.Resource.Value[..^"_output".Length];
            builder.AddNode(id, operation, bindings, null, attributes);
        }
        string Weight(string name)
        {
            if (weights.TryGetValue(name, out var found)) return found;
            var tensor = tensors.GetRequired(name);
            var id = "weight." + name;
            var descriptor = new TensorDescriptor(tensor.DataType == TensorDataType.Float16 ? GraphElementType.Float16 : GraphElementType.Float32,
                tensor.Dimensions);
            builder.AddResource(id, name, GraphResourceKind.Weight, GraphResourceLifetime.Model, descriptor, name);
            resources.Add(id, descriptor);
            if (tensor.DataType == TensorDataType.Float16)
            {
                var converted = Temporary(tensor.Dimensions.ToArray());
                Emit(PortableTensorOperationContracts.CastFp16ToFp32,
                    [GraphBindings.Read("input", id), GraphBindings.Write("output", converted)]);
                id = converted;
            }
            weights.Add(name, id);
            return id;
        }
        string Zero(int size)
        {
            if (zeros.TryGetValue(size, out var found)) return found;
            var id = Temporary(size);
            Emit(PortableTensorOperationContracts.Fill, [GraphBindings.Write("output", id)], new TensorFillValue(0).ToAttributes());
            zeros.Add(size, id);
            return id;
        }
        string Linear(string source, string prefix, int outputWidth)
        {
            var output = Temporary(tokens, outputWidth);
            var weight = Weight(prefix + ".weight");
            var bias = Weight(prefix + ".bias");
            builder.Affine(output[..^"_output".Length], source, weight, bias, output, transposeRight: true);
            return output;
        }
        string Norm(string source, string prefix)
        {
            var output = Temporary(tokens, width);
            var weight = Weight(prefix + ".weight");
            var bias = Weight(prefix + ".bias");
            Emit(Phi4AudioGraphOperations.LayerNorm,
                [GraphBindings.Read("input", source), GraphBindings.Read("weight", weight), GraphBindings.Read("bias", bias),
                 GraphBindings.Write("output", output)], Attr(("rows", tokens), ("width", width), ("epsilon", 1e-5f)));
            return output;
        }
        string SwiGlu(string source, int size, string first, string second)
        {
            var output = Temporary(tokens, size);
            Emit(Phi4AudioGraphOperations.SwiGlu,
                [GraphBindings.Read("input", source), GraphBindings.Read("bias_first", first), GraphBindings.Read("bias_second", second),
                 GraphBindings.Write("output", output)], Attr(("rows", tokens), ("width", size)));
            return output;
        }
        string FeedForward(string source, string prefix, string region)
        {
            var normalized = builder.WithRegion(region + ".norm", GraphRegionTypes.Architecture, "LayerNorm",
                _ => Norm(source, prefix + ".ln"), parentId: region,
                architecture: new(Description: "Normalizes the feed-forward input."));
            var wide = builder.WithRegion(region + ".up", GraphRegionTypes.Architecture, "Gate / Up Projection",
                _ => Linear(normalized, prefix + ".up", checked(intermediate * 2)), parentId: region,
                architecture: new(Description: "Projects paired gate and value channels."));
            var gated = builder.WithRegion(region + ".activation", GraphRegionTypes.Architecture, "SwiGLU",
                _ => SwiGlu(wide, intermediate, Zero(intermediate), Zero(intermediate)), parentId: region,
                architecture: new(Description: "Applies the gated activation.", Formula: "h = silu(gate) .* up"));
            return builder.WithRegion(region + ".down", GraphRegionTypes.Architecture, "Down Projection",
                _ => Linear(gated, prefix + ".down", width), parentId: region,
                architecture: new(Description: "Projects back to Conformer width."));
        }
        string Residual(string source, string update, float scale)
        {
            var output = Temporary(tokens, width);
            var count = checked(tokens * width);
            Emit(Phi4AudioGraphOperations.Residual,
                [Flat("hidden", source, GraphResourceAccess.Read, count), Flat("update", update, GraphResourceAccess.Read, count),
                 Flat("output", output, GraphResourceAccess.Write, count)], Attr(("count", count), ("scale", scale)));
            return output;
        }
        string Conv1D(string source, string prefix, int kernel, int padding, int groups)
        {
            var outputLength = checked(tokens + 2 * padding - kernel + 1);
            var output = Temporary(outputLength, width);
            var weight = Weight(prefix + ".weight");
            var bias = Weight(prefix + ".bias");
            Emit(Phi4AudioGraphOperations.Conv1D,
                [new("input", new(source), GraphResourceAccess.Read, View: new(0, F(tokens, width))),
                 GraphBindings.Read("weight", weight), GraphBindings.Read("bias", bias), GraphBindings.Write("output", output)],
                Attr(("input_length", tokens), ("input_channels", width), ("output_length", outputLength), ("output_channels", width),
                    ("kernel", kernel), ("padding", padding), ("stride", 1), ("groups", groups), ("activation", "none")));
            return output;
        }
        string Activate(string source, int size, string activation)
        {
            var output = Temporary(tokens, size);
            var zero = Zero(size);
            var count = checked(tokens * size);
            Emit(Phi4AudioGraphOperations.BiasActivation,
                [Flat("input", source, GraphResourceAccess.Read, count), GraphBindings.Read("bias", zero),
                 Flat("output", output, GraphResourceAccess.Write, count)], Attr(("count", count), ("width", size), ("activation", activation)));
            return output;
        }
    }

    public void Validate(GraphOperationValidationContext context) => Phi4AudioOperationValidation.Validate(context);

    private static NodeResourceBinding Flat(string port, string id, GraphResourceAccess access, int count) =>
        new(port, new(id), access, View: new(0, F(count)));
    private static TensorDescriptor F(params int[] dimensions) => new(GraphElementType.Float32, dimensions);
    private static IReadOnlyDictionary<string, string> Attr(params (string Name, object Value)[] values) =>
        values.ToDictionary(value => value.Name, value => Convert.ToString(value.Value, CultureInfo.InvariantCulture)!, StringComparer.Ordinal);
}
