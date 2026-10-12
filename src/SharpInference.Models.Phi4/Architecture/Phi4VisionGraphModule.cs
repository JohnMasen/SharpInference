using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Architectures.Phi4;

/// <summary>Builds and validates the Phi-4 vision transformer and high-definition embedding projection graph.</summary>
public sealed class Phi4VisionGraphModule : IPhi4GraphModule
{
    private readonly int crops;
    private readonly int imageTokens;
    private readonly int heads;
    private readonly int layers;

    /// <summary>Initializes a vision graph for the specified crop count, projected token count, attention heads, and layers.</summary>
    public Phi4VisionGraphModule(int crops, int imageTokens, int heads = 16, int layers = 26)
    {
        if (crops <= 0) throw new ArgumentOutOfRangeException(nameof(crops));
        if (imageTokens <= 0) throw new ArgumentOutOfRangeException(nameof(imageTokens));
        if (heads <= 0) throw new ArgumentOutOfRangeException(nameof(heads));
        if (layers is <= 0 or > 26) throw new ArgumentOutOfRangeException(nameof(layers));
        this.crops = crops;
        this.imageTokens = imageTokens;
        this.heads = heads;
        this.layers = layers;
    }

    /// <inheritdoc/>
    public string ArchitectureId => "phi4.vision";
    /// <inheritdoc/>
    /// <inheritdoc/>
    public bool CanLoad(IModelTensorCatalog tensors) => tensors.TryGet("v.patch_embd.weight", out _) &&
        tensors.TryGet("v.position_embd.weight", out _) && tensors.TryGet($"v.blk.{layers - 1}.ln2.weight", out _) &&
        tensors.TryGet("mm.2.weight", out _);

    /// <inheritdoc/>
    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var patch = tensors.GetRequired("v.patch_embd.weight");
        var position = tensors.GetRequired("v.position_embd.weight");
        if (patch.Dimensions.Count != 4 || patch.Dimensions[0] <= 0 || patch.Dimensions[1] != 3 ||
            patch.Dimensions[2] <= 0 || patch.Dimensions[2] != patch.Dimensions[3] ||
            position.Dimensions.Count != 2 || position.Dimensions[0] <= 0)
            throw new InvalidDataException("Phi4 vision requires square RGB patch weights and a positional matrix.");
        var width = patch.Dimensions[0];
        var patchSize = patch.Dimensions[2];
        var grid = (int)Math.Sqrt(position.Dimensions[0]);
        if (checked(grid * grid) != position.Dimensions[0] || grid % 2 != 0 || width % heads != 0)
            throw new InvalidDataException("Phi4 vision requires an even square patch grid and width divisible by heads.");
        Require("v.patch_embd.weight", [width, 3, patchSize, patchSize], half: true);
        Require("v.patch_embd.bias", [width], half: true);
        Require("v.position_embd.weight", [checked(grid * grid), width], half: true);
        Require("v.sub_GN", [width], half: true);
        Require("v.glb_GN", [width], half: true);
        var up = tensors.GetRequired("v.blk.0.ffn_up.weight");
        var projected = tensors.GetRequired("mm.0.weight");
        if (up.Dimensions.Count != 2 || up.Dimensions[0] <= 0 || projected.Dimensions.Count != 2 || projected.Dimensions[0] <= 0)
            throw new InvalidDataException("Phi4 vision requires positive feed-forward and projector matrix dimensions.");
        var intermediate = up.Dimensions[0];
        var textWidth = projected.Dimensions[0];
        for (var layer = 0; layer < layers; layer++)
        {
            var prefix = $"v.blk.{layer}.";
            foreach (var part in new[] { "ln1", "ln2" })
            {
                Require(prefix + part + ".weight", [width], half: true);
                Require(prefix + part + ".bias", [width], half: true);
            }
            foreach (var part in new[] { "attn_q", "attn_k", "attn_v", "attn_out" }) Linear(prefix + part, width, width);
            Linear(prefix + "ffn_up", width, intermediate);
            Linear(prefix + "ffn_down", intermediate, width);
        }
        Linear("mm.0", width, textWidth);
        Linear("mm.2", textWidth, textWidth);
        return new(ArchitectureId, new Dictionary<string, long>
        {
            ["crops"] = crops,
            ["image_tokens"] = imageTokens,
            ["patch_size"] = patchSize,
            ["grid"] = grid,
            ["crop_size"] = checked(grid * patchSize),
            ["width"] = width,
            ["intermediate"] = intermediate,
            ["embedding_width"] = textWidth,
            ["heads"] = heads,
            ["layers"] = layers,
        });

        void Require(string name, int[] shape, bool half = false)
        {
            var tensor = tensors.GetRequired(name);
            if (!tensor.Dimensions.SequenceEqual(shape) ||
                (half ? tensor.DataType != TensorDataType.Float16 : tensor.DataType is not (TensorDataType.Float16 or TensorDataType.Float32)))
                throw new InvalidDataException($"Phi4 vision tensor '{name}' has an incompatible type or shape.");
        }
        void Linear(string prefix, int input, int output)
        {
            Require(prefix + ".weight", [output, input]);
            Require(prefix + ".bias", [output]);
        }
    }

    /// <inheritdoc/>
    public LogicalGraphDefinition Build(IModelTensorCatalog tensors)
    {
        var metadata = ReadMetadata(tensors);
        var dimensions = metadata.Dimensions.ToDictionary(value => value.Key, value => checked((int)value.Value), StringComparer.Ordinal);
        var width = dimensions["width"];
        var grid = dimensions["grid"];
        var cropSize = dimensions["crop_size"];
        var builder = new Phi4GraphDefinitionBuilder(new(ArchitectureId, 2, "forward"),
                new GraphModelSignature(ArchitectureId, "phi4.vision.empty@1", dimensions))
            .Root("root", GraphSectionTypes.Graph, "Vision encoder and HD projection", architecture: new(
                Description: "Patch embedding, vision transformer Blocks, HD token gathering and multimodal projection.",
                DefaultView: GraphArchitectureView.Architecture))
            .AddResource("pixels", "Pixels", GraphResourceKind.Input, GraphResourceLifetime.External, F(crops, 3, cropSize, cropSize), graphInput: true)
            .AddResource("mask", "Patch mask", GraphResourceKind.Input, GraphResourceLifetime.External, F(crops, grid, grid), graphInput: true)
            .AddResource("mapping", "HD token mapping", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [imageTokens]), graphInput: true)
            .AddResource("output", "Projected image embeddings", GraphResourceKind.Output, GraphResourceLifetime.External,
                F(imageTokens, dimensions["embedding_width"]), graphOutput: true);
        var descriptors = new Dictionary<string, TensorDescriptor>(StringComparer.Ordinal);
        var weights = new Dictionary<string, string>(StringComparer.Ordinal);
        var converted = new Dictionary<string, string>(StringComparer.Ordinal);
        var index = 0;
        var hidden = builder.SectionValue("embedding", GraphSectionTypes.Stage, "Patch Embedding", scope =>
        {
            var embedded = Temporary(crops, grid, grid, width);
            var patch = Weight(scope, "v.patch_embd.weight");
            var bias = Weight(scope, "v.patch_embd.bias");
            var position = Weight(scope, "v.position_embd.weight");
            Emit(scope, Phi4VisionGraphOperations.PatchEmbedding,
                [GraphBindings.Read("pixels", "pixels"), GraphBindings.Read("mask", "mask"), GraphBindings.Read("weight", patch),
                 GraphBindings.Read("bias", bias), GraphBindings.Read("position", position), GraphBindings.Write("output", embedded)],
                Attr(("crop_size", cropSize), ("patch_size", dimensions["patch_size"]), ("width", width)));
            return embedded;
        }, architecture: new(Description: "Projects RGB patches, applies patch masks and adds positional embeddings."));
        builder.Section("blocks", GraphSectionTypes.Architecture, "Vision Blocks", blocks =>
        {
            for (var layer = 0; layer < layers; layer++)
            {
                var layerRegion = $"layer.{layer}";
                blocks.Section(layerRegion, GraphSectionTypes.Layer, "Vision Block", layerScope =>
                {
                    var attentionRegion = layerRegion + ".attention";
                    var prefix = $"v.blk.{layer}.";
                    layerScope.Section(attentionRegion, GraphSectionTypes.Stage, "Attention", attentionScope =>
                    {
            var normalized = attentionScope.SectionValue(attentionRegion + ".norm", GraphSectionTypes.Architecture, "LayerNorm",
                scope => Norm(scope, hidden, prefix + "ln1"),
                architecture: new(null, "Normalizes patch features before attention.", null));
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
            var attention = attentionScope.SectionValue(attentionRegion + ".attention", GraphSectionTypes.Architecture, "Masked Attention", scope =>
            {
                var attention = Temporary(crops, grid, grid, width);
                var tokens = checked(grid * grid);
                Emit(scope, Phi4VisionGraphOperations.Attention, [View("query", query, GraphResourceAccess.Read, crops, tokens, width), View("key", key, GraphResourceAccess.Read, crops, tokens, width), View("value", value, GraphResourceAccess.Read, crops, tokens, width), View("mask", "mask", GraphResourceAccess.Read, crops, tokens), View("output", attention, GraphResourceAccess.Write, crops, tokens, width)], Attr(("heads", heads)));
                return attention;
            }, architecture: new(null, "Computes masked multi-head attention.", "attention = softmax(mask(Q K^T / sqrt(head_size))) V"));
            var attentionOutput = attentionScope.SectionValue(attentionRegion + ".output", GraphSectionTypes.Architecture, "Output Projection", scope => Linear(scope, attention, prefix + "attn_out", width), architecture: new(null, "Projects attended features back to vision width.", null));
                        attentionScope.Section(attentionRegion + ".residual", GraphSectionTypes.Architecture, "Attention Residual", scope =>
                        {
                            hidden = Add(scope, hidden, attentionOutput);
                        }, architecture: new("residual-add", "Adds the attention output.", "x <- x + attention_output"));
                    }, role: "attention", architecture: new(Description: "Masked multi-head attention over patch tokens."));
                    var ffnRegion = layerRegion + ".ffn";
                    layerScope.Section(ffnRegion, GraphSectionTypes.Stage, "Feed forward", ffnScope =>
                    {
            var ffnInput = ffnScope.SectionValue(ffnRegion + ".norm", GraphSectionTypes.Architecture, "LayerNorm", scope => Norm(scope, hidden, prefix + "ln2"), architecture: new(null, "Normalizes the feed-forward input.", null));
            var up = ffnScope.SectionValue(ffnRegion + ".up", GraphSectionTypes.Architecture, "Up Projection", scope => Linear(scope, ffnInput, prefix + "ffn_up", dimensions["intermediate"]), architecture: new(null, "Expands to the feed-forward width.", null));
            var activated = ffnScope.SectionValue(ffnRegion + ".activation", GraphSectionTypes.Architecture, "GELU", scope => Gelu(scope, up), architecture: new(null, "Applies GELU.", null));
            var ffnOutput = ffnScope.SectionValue(ffnRegion + ".down", GraphSectionTypes.Architecture, "Down Projection", scope => Linear(scope, activated, prefix + "ffn_down", width), architecture: new(null, "Projects back to vision width.", null));
                        ffnScope.Section(ffnRegion + ".residual", GraphSectionTypes.Architecture, "FFN Residual", scope =>
                        {
                            hidden = Add(scope, hidden, ffnOutput);
                        }, architecture: new("residual-add", "Adds the feed-forward output.", "x <- x + ffn_output"));
                    }, role: "ffn", architecture: new(Description: "Pre-normalized GELU feed forward."));
                }, architecture: new(Description: "Pre-normalized masked self-attention and GELU feed forward with residual connections."));
            }
        }, architecture: new(Description: "Ordered vision transformer Block instances.", DefaultCollapsed: true, RepeatGroup: true));
        var outputRegion = "output";
        builder.Section(outputRegion, GraphSectionTypes.Stage, "HD Projection", outputScope =>
        {
            var compressed = outputScope.SectionValue(outputRegion + ".pool", GraphSectionTypes.Architecture, "2x2 Pooling", scope =>
            {
                var compressed = Temporary(crops, grid / 2, grid / 2, width);
                Emit(scope, Phi4VisionGraphOperations.Pool2x2, [GraphBindings.Read("input", hidden), GraphBindings.Write("output", compressed)]);
                return compressed;
            }, architecture: new(null, "Compresses spatial patch tokens.", null));
            var gathered = outputScope.SectionValue(outputRegion + ".gather", GraphSectionTypes.Architecture, "HD Token Gathering", scope =>
            {
                var gathered = Temporary(imageTokens, width);
                var subSeparator = Weight(scope, "v.sub_GN");
                var globalSeparator = Weight(scope, "v.glb_GN");
                Emit(scope, Phi4VisionGraphOperations.HdGather, [View("input", compressed, GraphResourceAccess.Read, checked(crops * (grid / 2) * (grid / 2)), width), GraphBindings.Read("mapping", "mapping"), GraphBindings.Read("sub_separator", subSeparator), GraphBindings.Read("global_separator", globalSeparator), GraphBindings.Write("output", gathered)]);
                return gathered;
            }, architecture: new(null, "Assembles local/global image tokens and separator embeddings.", null));
            outputScope.Section(outputRegion + ".projector", GraphSectionTypes.Architecture, "Multimodal Projector", scope =>
            {
                var projected = Linear(scope, Gelu(scope, Linear(scope, gathered, "mm.0", dimensions["embedding_width"])), "mm.2", dimensions["embedding_width"]);
                Emit(scope, PrimitiveGraphOperations.Copy, [GraphBindings.Read("input", projected), GraphBindings.Write("output", "output")]);
            }, architecture: new(null, "Two linear projections with GELU map image tokens to decoder embeddings.", null));
        }, architecture: new(Description: "Pools patches, gathers HD tokens and projects to decoder width."));
        return builder.Build();

        string Temporary(params int[] shape)
        {
            var id = "vision.node." + index++ + "_output";
            descriptors.Add(id, F(shape));
            builder.AddResource(id, id, GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, descriptors[id]);
            return id;
        }
        void Emit(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, GraphOperationId operation, NodeResourceBinding[] bindings, IReadOnlyDictionary<string, string>? attributes = null)
        {
            var output = bindings.FirstOrDefault(binding => binding.Access == GraphResourceAccess.Write &&
                binding.Resource.Value.EndsWith("_output", StringComparison.Ordinal));
            var id = output is null ? "vision.node." + index++ : output.Resource.Value[..^"_output".Length];
            scope.AddNode(id, operation, bindings, null, attributes);
        }
        string Weight(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string name, bool fp32 = false)
        {
            if (!weights.TryGetValue(name, out var id))
            {
                var tensor = tensors.GetRequired(name);
                id = "weight." + name;
                var descriptor = new TensorDescriptor(tensor.DataType == TensorDataType.Float16 ? GraphElementType.Float16 : GraphElementType.Float32,
                    tensor.Dimensions);
                builder.AddResource(id, name, GraphResourceKind.Weight, GraphResourceLifetime.Model, descriptor, name);
                descriptors.Add(id, descriptor);
                weights.Add(name, id);
            }
            if (!fp32 || descriptors[id].ElementType == GraphElementType.Float32) return id;
            if (converted.TryGetValue(name, out var existing)) return existing;
            var destination = Temporary(descriptors[id].Dimensions.ToArray());
            Emit(scope, PortableTensorOperationContracts.CastFp16ToFp32,
                [GraphBindings.Read("input", id), GraphBindings.Write("output", destination)]);
            converted.Add(name, destination);
            return destination;
        }
        string Norm(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string source, string prefix)
        {
            var output = Temporary(descriptors[source].Dimensions.ToArray());
            var weight = Weight(scope, prefix + ".weight");
            var bias = Weight(scope, prefix + ".bias");
            Emit(scope, Phi4VisionGraphOperations.LayerNorm,
                [GraphBindings.Read("input", source), GraphBindings.Read("weight", weight), GraphBindings.Read("bias", bias),
                 GraphBindings.Write("output", output)], Attr(("epsilon", 1e-6f)));
            return output;
        }
        string Linear(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string source, string prefix, int outputWidth)
        {
            var shape = descriptors[source].Dimensions.ToArray();
            var inputWidth = shape[^1];
            var rows = shape[..^1].Aggregate(1, (count, size) => checked(count * size));
            shape[^1] = outputWidth;
            var output = Temporary(shape);
            var weight = Weight(scope, prefix + ".weight", fp32: true);
            var bias = Weight(scope, prefix + ".bias", fp32: true);
            Emit(scope, PrimitiveGraphOperations.Affine,
                [View("left", source, GraphResourceAccess.Read, rows, inputWidth), GraphBindings.Read("right", weight),
                 GraphBindings.Read("bias", bias), View("output", output, GraphResourceAccess.Write, rows, outputWidth)],
                Attr(("transpose_left", "false"), ("transpose_right", "true")));
            return output;
        }
        string Add(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string left, string right)
        {
            var output = Temporary(descriptors[left].Dimensions.ToArray());
            scope.Add(output[..^"_output".Length], left, right, output);
            return output;
        }
        string Gelu(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string source)
        {
            var output = Temporary(descriptors[source].Dimensions.ToArray());
            Emit(scope, Phi4VisionGraphOperations.Gelu, [GraphBindings.Read("input", source), GraphBindings.Write("output", output)]);
            return output;
        }
    }

    /// <inheritdoc/>
    private static TensorDescriptor F(params int[] dimensions) => new(GraphElementType.Float32, dimensions);
    private static NodeResourceBinding View(string port, string source, GraphResourceAccess access, params int[] dimensions) =>
        new(port, new(source), access, View: new(0, F(dimensions)));
    private static IReadOnlyDictionary<string, string> Attr(params (string Name, object Value)[] values) =>
        values.ToDictionary(value => value.Name, value => Convert.ToString(value.Value, CultureInfo.InvariantCulture)!, StringComparer.Ordinal);
}
