using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Architectures.Phi4;

/// <summary>Builds and validates a stateful Phi-4 decoder graph with token or embedding input and optional LoRA adaptation.</summary>
public sealed class Phi4TextGraphModule : IPhi4GraphModule
{
    private readonly int context;
    private readonly bool embeddingInput;
    private readonly Phi4Adapter adapter;
    private readonly float? adapterScale;

    /// <summary>Initializes a decoder graph with the specified context capacity, input mode, and optional adapter scaling.</summary>
    public Phi4TextGraphModule(int maximumContext = 4096, bool embeddingInput = false,
        Phi4Adapter adapter = Phi4Adapter.None, float? adapterScale = null)
    {
        if (maximumContext <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumContext));
        }

        if (!Enum.IsDefined(adapter))
        {
            throw new ArgumentOutOfRangeException(nameof(adapter));
        }

        if (adapterScale is { } scale && (!float.IsFinite(scale) || scale <= 0))
        {
            throw new ArgumentException("An explicit adapter scale must be finite and positive.", nameof(adapterScale));
        }

        context = maximumContext;
        this.embeddingInput = embeddingInput;
        this.adapter = adapter;
        this.adapterScale = adapterScale;
    }

    /// <inheritdoc/>
    public string ArchitectureId => "phi4.text";
    /// <inheritdoc/>
    /// <inheritdoc/>
    public bool CanLoad(IModelTensorCatalog tensors) => tensors is Phi4PackageTensorCatalog &&
        tensors.TryGet("text.token_embd.weight", out _) && tensors.TryGet("text.output_norm.weight", out _);

    /// <inheritdoc/>
    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        if (tensors is not Phi4PackageTensorCatalog package)
        {
            throw new InvalidDataException("Phi4 text requires a model-owned package catalog with decoder metadata.");
        }

        var embedding = tensors.GetRequired("text.token_embd.weight");
        if (embedding.Dimensions.Count != 2)
        {
            throw new InvalidDataException("Phi4 embedding must be vocabulary-by-width.");
        }

        var vocabulary = embedding.Dimensions[0];
        var width = embedding.Dimensions[1];
        if (vocabulary <= 0)
        {
            throw new InvalidDataException("Phi4 vocabulary must be positive.");
        }

        var p = package.TextParameters;
        p.Validate(width);
        var kv = checked(p.KeyValueHeads * (width / p.QueryHeads));
        var up = tensors.GetRequired("text.blk.0.ffn_up.weight");
        if (up.Dimensions.Count != 2 || up.Dimensions[0] <= 0 || up.Dimensions[0] % 2 != 0)
        {
            throw new InvalidDataException("Phi4 gate/up must contain equal positive gate and value halves.");
        }

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
            {
                throw new InvalidDataException($"Phi4 text tensor '{name}' has incompatible type or shape.");
            }
        }
        void Linear(string name, int input, int output)
        {
            Require(name, [output, input]);
            if (adapter == Phi4Adapter.None)
            {
                return;
            }

            var a = tensors.GetRequired(AdapterName(name) + ".lora_a");
            if (a.Dimensions.Count != 2 || a.Dimensions[0] <= 0)
            {
                throw new InvalidDataException("Phi4 adapter rank must be positive.");
            }

            Require(a.Name, [a.Dimensions[0], input]);
            Require(AdapterName(name) + ".lora_b", [output, a.Dimensions[0]]);
            _ = AdapterScale(package, a.Dimensions[0]);
        }
    }

    /// <inheritdoc/>
    public LogicalGraphDefinition Build(IModelTensorCatalog tensors)
    {
        var dimensions = ReadMetadata(tensors).Dimensions.ToDictionary(value => value.Key, value => checked((int)value.Value));
        var p = ((Phi4PackageTensorCatalog)tensors).TextParameters;
        var width = dimensions["embedding_width"];
        var head = dimensions["head_size"];
        var kv = checked(p.KeyValueHeads * head);
        var intermediate = dimensions["intermediate"];
        var abi = $"phi4.text.kv@1:{p.Layers}:{p.QueryHeads}:{p.KeyValueHeads}:{head}:{context}:{adapter}";
        var builder = new Phi4GraphDefinitionBuilder(new(ArchitectureId, 2, "forward"),
                new GraphModelSignature(ArchitectureId, abi, dimensions))
            .Root("root", GraphSectionTypes.Graph, "Phi4 decoder", architecture: new(
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
        var hidden = builder.SectionValue("embedding", GraphSectionTypes.Stage, "Embedding", scope =>
        {
            if (embeddingInput)
            {
                builder.AddResource("embedding", "Embedding", GraphResourceKind.Input, GraphResourceLifetime.External, F(width), graphInput: true);
                return "embedding";
            }
            builder.AddResource("token", "Token", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true);
            var embedded = Temporary(width);
            var table = Weight(scope, "text.token_embd.weight");
            Emit(scope, PrimitiveGraphOperations.GatherRow,
                [GraphBindings.Read("table", table), GraphBindings.Read("index", "token"), GraphBindings.Write("output", embedded)]);
            return embedded;
        }, architecture: new(Description: "Looks up token embeddings, or accepts a model-owned embedding input."));
        builder.Section("blocks", GraphSectionTypes.Architecture, "Decoder Blocks", blocks =>
        {
            for (var layer = 0; layer < p.Layers; layer++)
            {
                var layerRegion = $"layer.{layer}";
                blocks.Section(layerRegion, GraphSectionTypes.Layer, "Decoder Block", layerScope =>
                {
                    var attentionRegion = layerRegion + ".attention";
                    var prefix = $"text.blk.{layer}.";
                    layerScope.Section(attentionRegion, GraphSectionTypes.Stage, "Attention", attentionScope =>
                    {
            var attentionInput = attentionScope.SectionValue(attentionRegion + ".norm", GraphSectionTypes.Architecture, "RMSNorm", scope => Norm(scope, hidden, prefix + "attn_norm.weight"), architecture: new(null, "Normalizes the attention input.", "x_norm = x / sqrt(mean(x^2) + epsilon) .* weight"));
            var qkv = attentionScope.SectionValue(attentionRegion + ".qkv", GraphSectionTypes.Architecture, "QKV Projection", scope => Linear(scope, attentionInput, prefix + "attn_qkv.weight", width + 2 * kv), architecture: new(null, "Projects query, key and value; includes the selected LoRA adapter when enabled.", null));
            var (query, keys, values) = attentionScope.SectionValue(attentionRegion + ".rope-cache", GraphSectionTypes.Architecture, "RoPE / KV Cache Write", scope =>
            {
                var query = Temporary(width);
                var keys = Cache($"key_cache.{layer}");
                var values = Cache($"value_cache.{layer}");
                var frequencies = Weight(scope, "text.rope_frequencies", fp32: true);
                Emit(scope, Phi4TextGraphOperations.RopeKeyValueWrite, [GraphBindings.Read("qkv", qkv), GraphBindings.Read("position", "position"), GraphBindings.Read("frequencies", frequencies), GraphBindings.Write("query", query), new("key_cache", new(keys), GraphResourceAccess.ReadWrite), new("value_cache", new(values), GraphResourceAccess.ReadWrite)], Attr(("head_size", head), ("rotary_size", p.RotarySize), ("rope_scale", p.RopeScale)));
                return (query, keys, values);
            }, architecture: new(null, "Rotates query/key and writes the current key/value cache row.", null));
            var scores = attentionScope.SectionValue(attentionRegion + ".scores", GraphSectionTypes.Architecture, "Grouped-Query Scores", scope =>
            {
                var scores = Temporary(p.QueryHeads, context);
                Emit(scope, Phi4TextGraphOperations.GroupedQueryScores, [GraphBindings.Read("query", query), GraphBindings.Read("key_cache", keys), GraphBindings.Read("position", "position"), GraphBindings.Write("scores", scores)]);
                return scores;
            }, architecture: new(null, "Computes scores against the cached keys.", "scores = Q K_cache^T / sqrt(head_size)"));
            var probabilities = attentionScope.SectionValue(attentionRegion + ".softmax", GraphSectionTypes.Architecture, "Causal Softmax", scope =>
            {
                var probabilities = Temporary(p.QueryHeads, context);
                Emit(scope, Phi4TextGraphOperations.CausalSoftmax, [GraphBindings.Read("scores", scores), GraphBindings.Read("position", "position"), GraphBindings.Write("probabilities", probabilities)]);
                return probabilities;
            }, architecture: new(null, "Masks future positions and normalizes attention scores.", "P = softmax(causal_mask(scores))"));
            var attention = attentionScope.SectionValue(attentionRegion + ".values", GraphSectionTypes.Architecture, "Grouped-Query Values", scope =>
            {
                var attention = Temporary(width);
                Emit(scope, Phi4TextGraphOperations.GroupedQueryValues, [GraphBindings.Read("probabilities", probabilities), GraphBindings.Read("value_cache", values), GraphBindings.Read("position", "position"), GraphBindings.Write("output", attention)]);
                return attention;
            }, architecture: new(null, "Reads cached values with attention probabilities.", "attention = P V_cache"));
            var attentionOutput = attentionScope.SectionValue(attentionRegion + ".output", GraphSectionTypes.Architecture, "Output Projection", scope => Linear(scope, attention, prefix + "attn_output.weight", width), architecture: new(null, "Projects attention back to decoder width, including the selected adapter.", null));
                        attentionScope.Section(attentionRegion + ".residual", GraphSectionTypes.Architecture, "Attention Residual", scope =>
                        {
                            hidden = Add(scope, hidden, attentionOutput);
                        }, architecture: new("residual-add", "Adds the attention output.", "x <- x + attention_output"));
                    }, role: "attention", architecture: new(Description: "Grouped-query attention with RoPE and persistent KV cache."));
                    var ffnRegion = layerRegion + ".ffn";
                    layerScope.Section(ffnRegion, GraphSectionTypes.Stage, "Feed forward", ffnScope =>
                    {
            var ffnInput = ffnScope.SectionValue(ffnRegion + ".norm", GraphSectionTypes.Architecture, "RMSNorm", scope => Norm(scope, hidden, prefix + "ffn_norm.weight"), architecture: new(null, "Normalizes the feed-forward input.", null));
            var gate = ffnScope.SectionValue(ffnRegion + ".up", GraphSectionTypes.Architecture, "Gate / Up Projection", scope => Linear(scope, ffnInput, prefix + "ffn_up.weight", 2 * intermediate), architecture: new(null, "Projects paired gate and value channels, including the selected adapter.", null));
            var activated = ffnScope.SectionValue(ffnRegion + ".activation", GraphSectionTypes.Architecture, "SwiGLU", scope =>
            {
                var activated = Temporary(intermediate);
                Emit(scope, Phi4TextGraphOperations.SwiGlu, [GraphBindings.Read("gate_up", gate), GraphBindings.Write("output", activated)]);
                return activated;
            }, architecture: new(null, "Gates the value half with SiLU of the gate half.", "h = silu(gate) .* up"));
            var ffnOutput = ffnScope.SectionValue(ffnRegion + ".down", GraphSectionTypes.Architecture, "Down Projection", scope => Linear(scope, activated, prefix + "ffn_down.weight", width), architecture: new(null, "Projects back to decoder width, including the selected adapter.", null));
                        ffnScope.Section(ffnRegion + ".residual", GraphSectionTypes.Architecture, "FFN Residual", scope =>
                        {
                            hidden = Add(scope, hidden, ffnOutput);
                        }, architecture: new("residual-add", "Adds the feed-forward output.", "x <- x + ffn_output"));
                    }, role: "ffn", architecture: new(Description: "Pre-normalized SwiGLU feed forward."));
                }, architecture: new(Description: "Pre-normalized grouped-query attention and SwiGLU feed forward with residual connections."));
            }
        }, architecture: new(Description: "Ordered transformer decoder Block instances.", DefaultCollapsed: true, RepeatGroup: true));
        builder.Section("output", GraphSectionTypes.Stage, "LM Head", scope =>
        {
            var normalized = Norm(scope, hidden, "text.output_norm.weight");
            scope.MatVec("text.node." + index++, Weight(scope, "text.token_embd.weight"), normalized, "logits");
        }, architecture: new(Description: "Final RMSNorm and vocabulary projection with the tied embedding matrix."));
        return builder.Build();

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
        void Emit(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, GraphOperationId operation, NodeResourceBinding[] bindings, IReadOnlyDictionary<string, string>? attributes = null)
        {
            var output = bindings.FirstOrDefault(binding => binding.Access == GraphResourceAccess.Write &&
                binding.Resource.Value.EndsWith("_output", StringComparison.Ordinal));
            var id = output is null ? "text.node." + index++ : output.Resource.Value[..^"_output".Length];
            scope.AddNode(id, operation, bindings, null, attributes);
        }
        string Weight(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string name, bool fp32 = false)
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
            if (!fp32 || descriptors[id].ElementType == GraphElementType.Float32)
            {
                return id;
            }

            if (converted.TryGetValue(name, out var existing))
            {
                return existing;
            }

            var destination = Temporary(descriptors[id].Dimensions.ToArray());
            Emit(scope, PortableTensorOperationContracts.CastFp16ToFp32, [GraphBindings.Read("input", id), GraphBindings.Write("output", destination)]);
            converted.Add(name, destination);
            return destination;
        }
        string Norm(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string source, string name)
        {
            var output = Temporary(width);
            var weight = Weight(scope, name, fp32: true);
            Emit(scope, Phi4TextGraphOperations.RmsNorm, [GraphBindings.Read("input", source), GraphBindings.Read("weight", weight), GraphBindings.Write("output", output)],
                Attr(("epsilon", p.RmsEpsilon)));
            return output;
        }
        string Project(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string source, string name, int outputWidth)
        {
            var output = Temporary(outputWidth);
            var weight = Weight(scope, name);
            scope.MatVec(output[..^"_output".Length], weight, source, output);
            return output;
        }
        string Linear(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string source, string name, int outputWidth)
        {
            var output = Project(scope, source, name, outputWidth);
            if (adapter == Phi4Adapter.None)
            {
                return output;
            }

            var a = AdapterName(name) + ".lora_a";
            var low = Project(scope, source, a, tensors.GetRequired(a).Dimensions[0]);
            var update = Project(scope, low, AdapterName(name) + ".lora_b", outputWidth);
            var combined = Temporary(outputWidth);
            Emit(scope, Phi4TextGraphOperations.ScaledAdd, [GraphBindings.Read("input", output), GraphBindings.Read("update", update), GraphBindings.Write("output", combined)],
                Attr(("scale", AdapterScale((Phi4PackageTensorCatalog)tensors, tensors.GetRequired(a).Dimensions[0]))));
            return combined;
        }
        string Add(Phi4GraphDefinitionBuilder.Phi4DefinitionScope scope, string left, string right)
        {
            var output = Temporary(width);
            scope.Add(output[..^"_output".Length], left, right, output);
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
    /// <inheritdoc/>
    private static TensorDescriptor F(params int[] shape) => new(GraphElementType.Float32, shape);
    private static IReadOnlyDictionary<string, string> Attr(params (string Name, object Value)[] values) =>
        values.ToDictionary(value => value.Name, value => Convert.ToString(value.Value, CultureInfo.InvariantCulture)!);
}
