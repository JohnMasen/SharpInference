using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Architectures.Phi4;

public sealed class Phi4VisionGraphModule : IModelGraphModule, IGraphOperationValidator
{
    private readonly int crops;
    private readonly int imageTokens;
    private readonly int heads;
    private readonly int layers;

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

    public string ArchitectureId => "phi4.vision";
    public IGraphOperationValidator OperationValidator => this;
    public bool CanLoad(IModelTensorCatalog tensors) => tensors.TryGet("v.patch_embd.weight", out _) &&
        tensors.TryGet("v.position_embd.weight", out _) && tensors.TryGet($"v.blk.{layers - 1}.ln2.weight", out _) &&
        tensors.TryGet("mm.2.weight", out _);

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

    public LogicalGraph Build(IModelTensorCatalog tensors)
    {
        var metadata = ReadMetadata(tensors);
        var dimensions = metadata.Dimensions.ToDictionary(value => value.Key, value => checked((int)value.Value), StringComparer.Ordinal);
        var width = dimensions["width"];
        var grid = dimensions["grid"];
        var cropSize = dimensions["crop_size"];
        var builder = new LogicalGraphBuilder(new(ArchitectureId, 2, "forward"),
                new GraphModelSignature(ArchitectureId, "phi4.vision.empty@1", dimensions))
            .AddRegion("root", GraphRegionTypes.Graph, "Vision encoder and HD projection", architecture: new(
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
        builder.AddRegion("blocks", GraphRegionTypes.Architecture, "Vision Blocks", "root",
            architecture: new(Description: "Ordered vision transformer Block instances.", DefaultCollapsed: true, RepeatGroup: true));
        var hidden = builder.WithRegion("embedding", GraphRegionTypes.Stage, "Patch Embedding", _ =>
        {
            var embedded = Temporary(crops, grid, grid, width);
            var patch = Weight("v.patch_embd.weight");
            var bias = Weight("v.patch_embd.bias");
            var position = Weight("v.position_embd.weight");
            Emit(Phi4VisionGraphOperations.PatchEmbedding,
                [GraphBindings.Read("pixels", "pixels"), GraphBindings.Read("mask", "mask"), GraphBindings.Read("weight", patch),
                 GraphBindings.Read("bias", bias), GraphBindings.Read("position", position), GraphBindings.Write("output", embedded)],
                Attr(("crop_size", cropSize), ("patch_size", dimensions["patch_size"]), ("width", width)));
            return embedded;
        }, parentId: "root", architecture: new(Description: "Projects RGB patches, applies patch masks and adds positional embeddings."));
        for (var layer = 0; layer < layers; layer++)
        {
            var layerRegion = $"layer.{layer}";
            builder.AddRegion(layerRegion, GraphRegionTypes.Layer, "Vision Block", "blocks",
                architecture: new(Description: "Pre-normalized masked self-attention and GELU feed forward with residual connections."));
            var attentionRegion = layerRegion + ".attention";
            builder.AddRegion(attentionRegion, GraphRegionTypes.Stage, "Attention", layerRegion, role: "attention",
                architecture: new(Description: "Masked multi-head attention over patch tokens."));
            var prefix = $"v.blk.{layer}.";
            var (normalized, projections) = builder.WithRegion(attentionRegion + ".norm", GraphRegionTypes.Architecture, "LayerNorm", _ =>
            {
                var normalized = Norm(hidden, prefix + "ln1");
                var projections = attentionRegion + ".qkv";
                return (normalized, projections);
            }, parentId: attentionRegion, architecture: new(null, "Normalizes patch features before attention.", null));
            var (query, key, value) = builder.WithRegion(projections, GraphRegionTypes.Architecture, "QKV Projections", _ =>
            {
                var query = builder.WithRegion(projections + ".q", GraphRegionTypes.Architecture, "Query", _ => Linear(normalized, prefix + "attn_q", width), architecture: new(null, "Projects query features.", null));
                var key = builder.WithRegion(projections + ".k", GraphRegionTypes.Architecture, "Key", _ => Linear(normalized, prefix + "attn_k", width), architecture: new(null, "Projects key features.", null));
                var value = builder.WithRegion(projections + ".v", GraphRegionTypes.Architecture, "Value", _ => Linear(normalized, prefix + "attn_v", width), architecture: new(null, "Projects value features.", null));
                return (query, key, value);
            }, parentId: attentionRegion, architecture: new(null, "Projects separate query, key and value branches.", null));
            var attention = builder.WithRegion(attentionRegion + ".attention", GraphRegionTypes.Architecture, "Masked Attention", _ =>
            {
                var attention = Temporary(crops, grid, grid, width);
                var tokens = checked(grid * grid);
                Emit(Phi4VisionGraphOperations.Attention, [View("query", query, GraphResourceAccess.Read, crops, tokens, width), View("key", key, GraphResourceAccess.Read, crops, tokens, width), View("value", value, GraphResourceAccess.Read, crops, tokens, width), View("mask", "mask", GraphResourceAccess.Read, crops, tokens), View("output", attention, GraphResourceAccess.Write, crops, tokens, width)], Attr(("heads", heads)));
                return attention;
            }, parentId: attentionRegion, architecture: new(null, "Computes masked multi-head attention.", "attention = softmax(mask(Q K^T / sqrt(head_size))) V"));
            var attentionOutput = builder.WithRegion(attentionRegion + ".output", GraphRegionTypes.Architecture, "Output Projection", _ => Linear(attention, prefix + "attn_out", width), parentId: attentionRegion, architecture: new(null, "Projects attended features back to vision width.", null));
            builder.WithRegion(attentionRegion + ".residual", GraphRegionTypes.Architecture, "Attention Residual", _ =>
            {
                hidden = Add(hidden, attentionOutput);
            }, parentId: attentionRegion, architecture: new("residual-add", "Adds the attention output.", "x <- x + attention_output"));
            var ffnRegion = layerRegion + ".ffn";
            builder.AddRegion(ffnRegion, GraphRegionTypes.Stage, "Feed forward", layerRegion, role: "ffn",
                architecture: new(Description: "Pre-normalized GELU feed forward."));
            var ffnInput = builder.WithRegion(ffnRegion + ".norm", GraphRegionTypes.Architecture, "LayerNorm", _ => Norm(hidden, prefix + "ln2"), parentId: ffnRegion, architecture: new(null, "Normalizes the feed-forward input.", null));
            var up = builder.WithRegion(ffnRegion + ".up", GraphRegionTypes.Architecture, "Up Projection", _ => Linear(ffnInput, prefix + "ffn_up", dimensions["intermediate"]), parentId: ffnRegion, architecture: new(null, "Expands to the feed-forward width.", null));
            var activated = builder.WithRegion(ffnRegion + ".activation", GraphRegionTypes.Architecture, "GELU", _ => Gelu(up), parentId: ffnRegion, architecture: new(null, "Applies GELU.", null));
            var ffnOutput = builder.WithRegion(ffnRegion + ".down", GraphRegionTypes.Architecture, "Down Projection", _ => Linear(activated, prefix + "ffn_down", width), parentId: ffnRegion, architecture: new(null, "Projects back to vision width.", null));
            builder.WithRegion(ffnRegion + ".residual", GraphRegionTypes.Architecture, "FFN Residual", _ =>
            {
                hidden = Add(hidden, ffnOutput);
            }, parentId: ffnRegion, architecture: new("residual-add", "Adds the feed-forward output.", "x <- x + ffn_output"));
        }
        var outputRegion = "output";
        builder.AddRegion(outputRegion, GraphRegionTypes.Stage, "HD Projection", "root",
            architecture: new(Description: "Pools patches, gathers HD tokens and projects to decoder width."));
        var compressed = builder.WithRegion(outputRegion + ".pool", GraphRegionTypes.Architecture, "2x2 Pooling", _ =>
        {
            var compressed = Temporary(crops, grid / 2, grid / 2, width);
            Emit(Phi4VisionGraphOperations.Pool2x2, [GraphBindings.Read("input", hidden), GraphBindings.Write("output", compressed)]);
            return compressed;
        }, parentId: outputRegion, architecture: new(null, "Compresses spatial patch tokens.", null));
        var gathered = builder.WithRegion(outputRegion + ".gather", GraphRegionTypes.Architecture, "HD Token Gathering", _ =>
        {
            var gathered = Temporary(imageTokens, width);
            var subSeparator = Weight("v.sub_GN");
            var globalSeparator = Weight("v.glb_GN");
            Emit(Phi4VisionGraphOperations.HdGather, [View("input", compressed, GraphResourceAccess.Read, checked(crops * (grid / 2) * (grid / 2)), width), GraphBindings.Read("mapping", "mapping"), GraphBindings.Read("sub_separator", subSeparator), GraphBindings.Read("global_separator", globalSeparator), GraphBindings.Write("output", gathered)]);
            return gathered;
        }, parentId: outputRegion, architecture: new(null, "Assembles local/global image tokens and separator embeddings.", null));
        builder.WithRegion(outputRegion + ".projector", GraphRegionTypes.Architecture, "Multimodal Projector", _ =>
        {
            var projected = Linear(Gelu(Linear(gathered, "mm.0", dimensions["embedding_width"])), "mm.2", dimensions["embedding_width"]);
            Emit(PrimitiveGraphOperations.Copy, [GraphBindings.Read("input", projected), GraphBindings.Write("output", "output")]);
        }, parentId: outputRegion, architecture: new(null, "Two linear projections with GELU map image tokens to decoder embeddings.", null));
        var graph = GraphLayerReuse.Extract(builder.BuildSequential(), region =>
            region.Type == GraphRegionTypes.Layer ? "Phi4.Vision.Layer" : null);
        GraphValidator.Validate(graph, this);
        return graph;

        string Temporary(params int[] shape)
        {
            var id = "vision.node." + index++ + "_output";
            descriptors.Add(id, F(shape));
            builder.AddResource(id, id, GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, descriptors[id]);
            return id;
        }
        void Emit(GraphOperationId operation, NodeResourceBinding[] bindings, IReadOnlyDictionary<string, string>? attributes = null)
        {
            var output = bindings.FirstOrDefault(binding => binding.Access == GraphResourceAccess.Write &&
                binding.Resource.Value.EndsWith("_output", StringComparison.Ordinal));
            var id = output is null ? "vision.node." + index++ : output.Resource.Value[..^"_output".Length];
            builder.AddNode(id, operation, bindings, null, attributes);
        }
        string Weight(string name, bool fp32 = false)
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
            Emit(PortableTensorOperationContracts.CastFp16ToFp32,
                [GraphBindings.Read("input", id), GraphBindings.Write("output", destination)]);
            converted.Add(name, destination);
            return destination;
        }
        string Norm(string source, string prefix)
        {
            var output = Temporary(descriptors[source].Dimensions.ToArray());
            var weight = Weight(prefix + ".weight");
            var bias = Weight(prefix + ".bias");
            Emit(Phi4VisionGraphOperations.LayerNorm,
                [GraphBindings.Read("input", source), GraphBindings.Read("weight", weight), GraphBindings.Read("bias", bias),
                 GraphBindings.Write("output", output)], Attr(("epsilon", 1e-6f)));
            return output;
        }
        string Linear(string source, string prefix, int outputWidth)
        {
            var shape = descriptors[source].Dimensions.ToArray();
            var inputWidth = shape[^1];
            var rows = shape[..^1].Aggregate(1, (count, size) => checked(count * size));
            shape[^1] = outputWidth;
            var output = Temporary(shape);
            var weight = Weight(prefix + ".weight", fp32: true);
            var bias = Weight(prefix + ".bias", fp32: true);
            Emit(PrimitiveGraphOperations.Affine,
                [View("left", source, GraphResourceAccess.Read, rows, inputWidth), GraphBindings.Read("right", weight),
                 GraphBindings.Read("bias", bias), View("output", output, GraphResourceAccess.Write, rows, outputWidth)],
                Attr(("transpose_left", "false"), ("transpose_right", "true")));
            return output;
        }
        string Add(string left, string right)
        {
            var output = Temporary(descriptors[left].Dimensions.ToArray());
            builder.Add(output[..^"_output".Length], left, right, output);
            return output;
        }
        string Gelu(string source)
        {
            var output = Temporary(descriptors[source].Dimensions.ToArray());
            Emit(Phi4VisionGraphOperations.Gelu, [GraphBindings.Read("input", source), GraphBindings.Write("output", output)]);
            return output;
        }
    }

    public void Validate(GraphOperationValidationContext context) => Phi4VisionOperationValidation.Validate(context);
    private static TensorDescriptor F(params int[] dimensions) => new(GraphElementType.Float32, dimensions);
    private static NodeResourceBinding View(string port, string source, GraphResourceAccess access, params int[] dimensions) =>
        new(port, new(source), access, View: new(0, F(dimensions)));
    private static IReadOnlyDictionary<string, string> Attr(params (string Name, object Value)[] values) =>
        values.ToDictionary(value => value.Name, value => Convert.ToString(value.Value, CultureInfo.InvariantCulture)!, StringComparer.Ordinal);
}
