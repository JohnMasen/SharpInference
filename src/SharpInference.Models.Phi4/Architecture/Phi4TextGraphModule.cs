using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Architectures.Phi4;

public sealed class Phi4TextGraphModule : IModelGraphModule, IGraphOperationValidator
{
    private readonly int context;
    private readonly bool embeddingInput;
    private readonly Phi4Adapter adapter;
    private readonly float? adapterScale;

    public Phi4TextGraphModule(int maximumContext = 4096, bool embeddingInput = false,
        Phi4Adapter adapter = Phi4Adapter.None, float? adapterScale = null)
    {
        if (maximumContext <= 0) throw new ArgumentOutOfRangeException(nameof(maximumContext));
        if (!Enum.IsDefined(adapter)) throw new ArgumentOutOfRangeException(nameof(adapter));
        if (adapterScale is { } scale && (!float.IsFinite(scale) || scale <= 0))
            throw new ArgumentException("An explicit adapter scale must be finite and positive.", nameof(adapterScale));
        context = maximumContext;
        this.embeddingInput = embeddingInput;
        this.adapter = adapter;
        this.adapterScale = adapterScale;
    }

    public string ArchitectureId => "phi4.text";
    public IGraphOperationValidator OperationValidator => this;
    public bool CanLoad(IModelTensorCatalog tensors) => tensors is Phi4PackageTensorCatalog &&
        tensors.TryGet("text.token_embd.weight", out _) && tensors.TryGet("text.output_norm.weight", out _);

    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        if (tensors is not Phi4PackageTensorCatalog package)
            throw new InvalidDataException("Phi4 text requires a model-owned package catalog with decoder metadata.");
        var embedding = tensors.GetRequired("text.token_embd.weight");
        if (embedding.Dimensions.Count != 2) throw new InvalidDataException("Phi4 embedding must be vocabulary-by-width.");
        var vocabulary = embedding.Dimensions[0];
        var width = embedding.Dimensions[1];
        if (vocabulary <= 0) throw new InvalidDataException("Phi4 vocabulary must be positive.");
        var p = package.TextParameters;
        p.Validate(width);
        var kv = checked(p.KeyValueHeads * (width / p.QueryHeads));
        var up = tensors.GetRequired("text.blk.0.ffn_up.weight");
        if (up.Dimensions.Count != 2 || up.Dimensions[0] <= 0 || up.Dimensions[0] % 2 != 0)
            throw new InvalidDataException("Phi4 gate/up must contain equal positive gate and value halves.");
        var intermediate = up.Dimensions[0] / 2;
        Require("text.token_embd.weight", [vocabulary, width]);
        Require("text.output_norm.weight", [width]);
        Require("text.rope_frequencies", [p.RotarySize / 2]);
        for (var layer = 0; layer < p.Layers; layer++)
        {
            var prefix = $"text.blk.{layer}.";
            Require(prefix + "attn_norm.weight", [width]);
            Require(prefix + "ffn_norm.weight", [width]);
            Linear(prefix + "attn_qkv.weight", width, checked(width + 2 * kv));
            Linear(prefix + "attn_output.weight", width, width);
            Linear(prefix + "ffn_up.weight", width, checked(2 * intermediate));
            Linear(prefix + "ffn_down.weight", intermediate, width);
        }
        return new(ArchitectureId, new Dictionary<string, long>
        {
            ["embedding_width"] = width,
            ["vocabulary"] = vocabulary,
            ["intermediate"] = intermediate,
            ["context"] = context,
            ["layers"] = p.Layers,
            ["query_heads"] = p.QueryHeads,
            ["kv_heads"] = p.KeyValueHeads,
            ["head_size"] = width / p.QueryHeads,
        });

        void Require(string name, int[] shape)
        {
            var tensor = tensors.GetRequired(name);
            if (!tensor.Dimensions.SequenceEqual(shape) || tensor.DataType is not (TensorDataType.Float16 or TensorDataType.Float32))
                throw new InvalidDataException($"Phi4 text tensor '{name}' has incompatible type or shape.");
        }
        void Linear(string name, int input, int output)
        {
            Require(name, [output, input]);
            if (adapter == Phi4Adapter.None) return;
            var a = tensors.GetRequired(AdapterName(name) + ".lora_a");
            if (a.Dimensions.Count != 2 || a.Dimensions[0] <= 0)
                throw new InvalidDataException("Phi4 adapter rank must be positive.");
            Require(a.Name, [a.Dimensions[0], input]);
            Require(AdapterName(name) + ".lora_b", [output, a.Dimensions[0]]);
            _ = AdapterScale(package, a.Dimensions[0]);
        }
    }

    public LogicalGraph Build(IModelTensorCatalog tensors)
    {
        var dimensions = ReadMetadata(tensors).Dimensions.ToDictionary(value => value.Key, value => checked((int)value.Value));
        var p = ((Phi4PackageTensorCatalog)tensors).TextParameters;
        var width = dimensions["embedding_width"];
        var head = dimensions["head_size"];
        var kv = checked(p.KeyValueHeads * head);
        var intermediate = dimensions["intermediate"];
        var abi = $"phi4.text.kv@1:{p.Layers}:{p.QueryHeads}:{p.KeyValueHeads}:{head}:{context}:{adapter}";
        var builder = new LogicalGraphBuilder(new(ArchitectureId, 2, "forward"),
                new GraphModelSignature(ArchitectureId, abi, dimensions))
            .AddRegion("root", GraphRegionTypes.Graph, "Phi4 decoder", architecture: new(
                Description: "Token embedding, grouped-query decoder Blocks and tied language-model head.",
                DefaultView: GraphArchitectureView.Architecture, Step: GraphArchitectureStep.Token))
            .SetStateSchema(new(abi))
            .AddResource("position", "Position", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true)
            .AddResource("logits", "Logits", GraphResourceKind.Output, GraphResourceLifetime.External,
                F(dimensions["vocabulary"]), graphOutput: true);
        var descriptors = new Dictionary<string, TensorDescriptor>();
        var weights = new Dictionary<string, string>();
        var converted = new Dictionary<string, string>();
        var index = 0;
        builder.AddRegion("blocks", GraphRegionTypes.Architecture, "Decoder Blocks", "root",
            architecture: new(Description: "Ordered transformer decoder Block instances.", DefaultCollapsed: true, RepeatGroup: true));
        var hidden = builder.WithRegion("embedding", GraphRegionTypes.Stage, "Embedding", _ =>
        {
            if (embeddingInput)
            {
                builder.AddResource("embedding", "Embedding", GraphResourceKind.Input, GraphResourceLifetime.External, F(width), graphInput: true);
                return "embedding";
            }
            builder.AddResource("token", "Token", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true);
            var embedded = Temporary(width);
            var table = Weight("text.token_embd.weight");
            Emit(PrimitiveGraphOperations.GatherRow,
                [GraphBindings.Read("table", table), GraphBindings.Read("index", "token"), GraphBindings.Write("output", embedded)]);
            return embedded;
        }, parentId: "root", architecture: new(Description: "Looks up token embeddings, or accepts a model-owned embedding input."));
        for (var layer = 0; layer < p.Layers; layer++)
        {
            var layerRegion = $"layer.{layer}";
            builder.AddRegion(layerRegion, GraphRegionTypes.Layer, "Decoder Block", "blocks",
                architecture: new(Description: "Pre-normalized grouped-query attention and SwiGLU feed forward with residual connections."));
            var attentionRegion = layerRegion + ".attention";
            builder.AddRegion(attentionRegion, GraphRegionTypes.Stage, "Attention", layerRegion, role: "attention",
                architecture: new(Description: "Grouped-query attention with RoPE and persistent KV cache."));
            var prefix = $"text.blk.{layer}.";
            var attentionInput = builder.WithRegion(attentionRegion + ".norm", GraphRegionTypes.Architecture, "RMSNorm", _ => Norm(hidden, prefix + "attn_norm.weight"), parentId: attentionRegion, architecture: new(null, "Normalizes the attention input.", "x_norm = x / sqrt(mean(x^2) + epsilon) .* weight"));
            var qkv = builder.WithRegion(attentionRegion + ".qkv", GraphRegionTypes.Architecture, "QKV Projection", _ => Linear(attentionInput, prefix + "attn_qkv.weight", width + 2 * kv), parentId: attentionRegion, architecture: new(null, "Projects query, key and value; includes the selected LoRA adapter when enabled.", null));
            var (query, keys, values) = builder.WithRegion(attentionRegion + ".rope-cache", GraphRegionTypes.Architecture, "RoPE / KV Cache Write", _ =>
            {
                var query = Temporary(width);
                var keys = Cache($"key_cache.{layer}");
                var values = Cache($"value_cache.{layer}");
                var frequencies = Weight("text.rope_frequencies", fp32: true);
                Emit(Phi4TextGraphOperations.RopeKeyValueWrite, [GraphBindings.Read("qkv", qkv), GraphBindings.Read("position", "position"), GraphBindings.Read("frequencies", frequencies), GraphBindings.Write("query", query), new("key_cache", new(keys), GraphResourceAccess.ReadWrite), new("value_cache", new(values), GraphResourceAccess.ReadWrite)], Attr(("head_size", head), ("rotary_size", p.RotarySize), ("rope_scale", p.RopeScale)));
                return (query, keys, values);
            }, parentId: attentionRegion, architecture: new(null, "Rotates query/key and writes the current key/value cache row.", null));
            var scores = builder.WithRegion(attentionRegion + ".scores", GraphRegionTypes.Architecture, "Grouped-Query Scores", _ =>
            {
                var scores = Temporary(p.QueryHeads, context);
                Emit(Phi4TextGraphOperations.GroupedQueryScores, [GraphBindings.Read("query", query), GraphBindings.Read("key_cache", keys), GraphBindings.Read("position", "position"), GraphBindings.Write("scores", scores)]);
                return scores;
            }, parentId: attentionRegion, architecture: new(null, "Computes scores against the cached keys.", "scores = Q K_cache^T / sqrt(head_size)"));
            var probabilities = builder.WithRegion(attentionRegion + ".softmax", GraphRegionTypes.Architecture, "Causal Softmax", _ =>
            {
                var probabilities = Temporary(p.QueryHeads, context);
                Emit(Phi4TextGraphOperations.CausalSoftmax, [GraphBindings.Read("scores", scores), GraphBindings.Read("position", "position"), GraphBindings.Write("probabilities", probabilities)]);
                return probabilities;
            }, parentId: attentionRegion, architecture: new(null, "Masks future positions and normalizes attention scores.", "P = softmax(causal_mask(scores))"));
            var attention = builder.WithRegion(attentionRegion + ".values", GraphRegionTypes.Architecture, "Grouped-Query Values", _ =>
            {
                var attention = Temporary(width);
                Emit(Phi4TextGraphOperations.GroupedQueryValues, [GraphBindings.Read("probabilities", probabilities), GraphBindings.Read("value_cache", values), GraphBindings.Read("position", "position"), GraphBindings.Write("output", attention)]);
                return attention;
            }, parentId: attentionRegion, architecture: new(null, "Reads cached values with attention probabilities.", "attention = P V_cache"));
            var attentionOutput = builder.WithRegion(attentionRegion + ".output", GraphRegionTypes.Architecture, "Output Projection", _ => Linear(attention, prefix + "attn_output.weight", width), parentId: attentionRegion, architecture: new(null, "Projects attention back to decoder width, including the selected adapter.", null));
            builder.WithRegion(attentionRegion + ".residual", GraphRegionTypes.Architecture, "Attention Residual", _ =>
            {
                hidden = Add(hidden, attentionOutput);
            }, parentId: attentionRegion, architecture: new("residual-add", "Adds the attention output.", "x <- x + attention_output"));
            var ffnRegion = layerRegion + ".ffn";
            builder.AddRegion(ffnRegion, GraphRegionTypes.Stage, "Feed forward", layerRegion, role: "ffn",
                architecture: new(Description: "Pre-normalized SwiGLU feed forward."));
            var ffnInput = builder.WithRegion(ffnRegion + ".norm", GraphRegionTypes.Architecture, "RMSNorm", _ => Norm(hidden, prefix + "ffn_norm.weight"), parentId: ffnRegion, architecture: new(null, "Normalizes the feed-forward input.", null));
            var gate = builder.WithRegion(ffnRegion + ".up", GraphRegionTypes.Architecture, "Gate / Up Projection", _ => Linear(ffnInput, prefix + "ffn_up.weight", 2 * intermediate), parentId: ffnRegion, architecture: new(null, "Projects paired gate and value channels, including the selected adapter.", null));
            var activated = builder.WithRegion(ffnRegion + ".activation", GraphRegionTypes.Architecture, "SwiGLU", _ =>
            {
                var activated = Temporary(intermediate);
                Emit(Phi4TextGraphOperations.SwiGlu, [GraphBindings.Read("gate_up", gate), GraphBindings.Write("output", activated)]);
                return activated;
            }, parentId: ffnRegion, architecture: new(null, "Gates the value half with SiLU of the gate half.", "h = silu(gate) .* up"));
            var ffnOutput = builder.WithRegion(ffnRegion + ".down", GraphRegionTypes.Architecture, "Down Projection", _ => Linear(activated, prefix + "ffn_down.weight", width), parentId: ffnRegion, architecture: new(null, "Projects back to decoder width, including the selected adapter.", null));
            builder.WithRegion(ffnRegion + ".residual", GraphRegionTypes.Architecture, "FFN Residual", _ =>
            {
                hidden = Add(hidden, ffnOutput);
            }, parentId: ffnRegion, architecture: new("residual-add", "Adds the feed-forward output.", "x <- x + ffn_output"));
        }
        builder.WithRegion("output", GraphRegionTypes.Stage, "LM Head", _ =>
        {
            var normalized = Norm(hidden, "text.output_norm.weight");
            builder.MatVec("text.node." + index++, Weight("text.token_embd.weight"), normalized, "logits");
        }, parentId: "root", architecture: new(Description: "Final RMSNorm and vocabulary projection with the tied embedding matrix."));
        var graph = GraphLayerReuse.Extract(builder.BuildSequential(), region =>
            region.Type == GraphRegionTypes.Layer ? "Phi4.Text.Layer" : null);
        GraphValidator.Validate(graph, this);
        return graph;

        string Temporary(params int[] shape)
        {
            var id = "text.node." + index++ + "_output";
            descriptors.Add(id, F(shape));
            builder.AddResource(id, id, GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, descriptors[id]);
            return id;
        }
        string Cache(string id)
        {
            builder.AddResource(id, id, GraphResourceKind.SessionState, GraphResourceLifetime.Session, F(context, p.KeyValueHeads, head));
            builder.AddStateSlot(id, id);
            return id;
        }
        void Emit(GraphOperationId operation, NodeResourceBinding[] bindings, IReadOnlyDictionary<string, string>? attributes = null)
        {
            var output = bindings.FirstOrDefault(binding => binding.Access == GraphResourceAccess.Write &&
                binding.Resource.Value.EndsWith("_output", StringComparison.Ordinal));
            var id = output is null ? "text.node." + index++ : output.Resource.Value[..^"_output".Length];
            builder.AddNode(id, operation, bindings, null, attributes);
        }
        string Weight(string name, bool fp32 = false)
        {
            if (!weights.TryGetValue(name, out var id))
            {
                var tensor = tensors.GetRequired(name);
                id = "weight." + name;
                var descriptor = new TensorDescriptor(tensor.DataType == TensorDataType.Float16 ? GraphElementType.Float16 : GraphElementType.Float32, tensor.Dimensions);
                builder.AddResource(id, name, GraphResourceKind.Weight, GraphResourceLifetime.Model, descriptor, name);
                descriptors.Add(id, descriptor);
                weights.Add(name, id);
            }
            if (!fp32 || descriptors[id].ElementType == GraphElementType.Float32) return id;
            if (converted.TryGetValue(name, out var existing)) return existing;
            var destination = Temporary(descriptors[id].Dimensions.ToArray());
            Emit(PortableTensorOperationContracts.CastFp16ToFp32, [GraphBindings.Read("input", id), GraphBindings.Write("output", destination)]);
            converted.Add(name, destination);
            return destination;
        }
        string Norm(string source, string name)
        {
            var output = Temporary(width);
            var weight = Weight(name, fp32: true);
            Emit(Phi4TextGraphOperations.RmsNorm, [GraphBindings.Read("input", source), GraphBindings.Read("weight", weight), GraphBindings.Write("output", output)],
                Attr(("epsilon", p.RmsEpsilon)));
            return output;
        }
        string Project(string source, string name, int outputWidth)
        {
            var output = Temporary(outputWidth);
            var weight = Weight(name);
            builder.MatVec(output[..^"_output".Length], weight, source, output);
            return output;
        }
        string Linear(string source, string name, int outputWidth)
        {
            var output = Project(source, name, outputWidth);
            if (adapter == Phi4Adapter.None) return output;
            var a = AdapterName(name) + ".lora_a";
            var low = Project(source, a, tensors.GetRequired(a).Dimensions[0]);
            var update = Project(low, AdapterName(name) + ".lora_b", outputWidth);
            var combined = Temporary(outputWidth);
            Emit(Phi4TextGraphOperations.ScaledAdd, [GraphBindings.Read("input", output), GraphBindings.Read("update", update), GraphBindings.Write("output", combined)],
                Attr(("scale", AdapterScale((Phi4PackageTensorCatalog)tensors, tensors.GetRequired(a).Dimensions[0]))));
            return combined;
        }
        string Add(string left, string right)
        {
            var output = Temporary(width);
            builder.Add(output[..^"_output".Length], left, right, output);
            return output;
        }
    }

    private float AdapterScale(Phi4PackageTensorCatalog package, int rank)
    {
        var alpha = adapter == Phi4Adapter.Vision ? package.VisionAdapterAlpha : package.SpeechAdapterAlpha;
        var scale = adapterScale ?? (alpha is { } value ? value / rank : throw new InvalidDataException("Phi4 adapter alpha metadata is missing."));
        return float.IsFinite(scale) && scale > 0 ? scale : throw new InvalidDataException("Phi4 adapter alpha/rank scale is invalid.");
    }
    private string AdapterName(string textName) => (adapter == Phi4Adapter.Vision ? "vision-adapter." : "speech-adapter.") + textName[5..];
    public void Validate(GraphOperationValidationContext context) => Phi4TextOperationValidation.Validate(context);
    private static TensorDescriptor F(params int[] shape) => new(GraphElementType.Float32, shape);
    private static IReadOnlyDictionary<string, string> Attr(params (string Name, object Value)[] values) =>
        values.ToDictionary(value => value.Name, value => Convert.ToString(value.Value, CultureInfo.InvariantCulture)!);
}
