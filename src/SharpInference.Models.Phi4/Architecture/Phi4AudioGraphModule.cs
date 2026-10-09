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
            .AddRegion("root", GraphRegionTypes.Graph, "Audio encoder and projection");
        var resources = new Dictionary<string, TensorDescriptor>(StringComparer.Ordinal);
        var weights = new Dictionary<string, string>(StringComparer.Ordinal);
        var zeros = new Dictionary<int, string>();
        string? previous = null;
        var index = 0;
        Append(subsampling.Build(tensors), "subsampling", new Dictionary<string, string> { ["output"] = "flattened" });
        var hidden = Linear("flattened", "a.conv1d.out", width);
        var relative = Weight("a.rel_attn_bias");
        for (var layer = 0; layer < layers; layer++)
        {
            var prefix = $"a.blk.{layer}.";
            hidden = Residual(hidden, FeedForward(hidden, prefix + "ffn_in"), 0.5f);
            var normalized = Norm(hidden, prefix + "ln_att");
            var query = Linear(normalized, prefix + "attn_q", width);
            var key = Linear(normalized, prefix + "attn_k", width);
            var value = Linear(normalized, prefix + "attn_v", width);
            var attention = Temporary(tokens, width);
            Emit(Phi4AudioGraphOperations.RelativeAttention,
                [GraphBindings.Read("query", query), GraphBindings.Read("key", key), GraphBindings.Read("value", value),
                 GraphBindings.Read("relative_bias", relative), GraphBindings.Read("frame_count", "frame_count"),
                 GraphBindings.Write("output", attention)],
                Attr(("tokens", tokens), ("width", width), ("heads", dimensions["heads"]), ("subsampling", 8)));
            hidden = Residual(hidden, Linear(attention, prefix + "attn_out", width), 1f);
            normalized = Norm(hidden, prefix + "conv.ln");
            var wide = Linear(normalized, prefix + "conv.glu.pw", checked(width * 2));
            var gated = SwiGlu(wide, width, Weight(prefix + "conv.glu.b1"), Weight(prefix + "conv.glu.b2"));
            var depthwise = Conv1D(gated, prefix + "conv.dw", 3, 2, width);
            var middle = Conv1D(depthwise, prefix + "conv.pw_mid", 1, 0, 1);
            middle = Activate(middle, width, "swish");
            hidden = Residual(hidden, Conv1D(middle, prefix + "conv.pw_ext", 1, 0, 1), 1f);
            hidden = Residual(hidden, FeedForward(hidden, prefix + "ffn_out"), 0.5f);
            hidden = Norm(hidden, prefix + "ln");
        }
        Append(projection.Build(tensors), "projection", new Dictionary<string, string> { ["audio_hidden"] = hidden });
        var graph = builder.Build();
        GraphValidator.Validate(graph, this);
        return graph;

        void Append(LogicalGraph component, string tag, IReadOnlyDictionary<string, string> aliases)
        {
            string Map(ResourceId id) => aliases.GetValueOrDefault(id.Value, id.Value);
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
            var predecessor = previous;
            foreach (var node in component.Nodes)
            {
                var id = tag + "." + node.Id.Value;
                var dependencies = node.Dependencies.Select(dependency => tag + "." + dependency.Value).ToList();
                if (dependencies.Count == 0 && predecessor is not null) dependencies.Add(predecessor);
                builder.AddNode(id, node.Operation, "root", node.Resources.Select(binding => binding with { Resource = new(Map(binding.Resource)) }),
                    dependencies, node.Attributes, node.Requirements);
                previous = id;
            }
        }
        string Temporary(params int[] shape)
        {
            var id = "conformer.buffer." + index++;
            resources.Add(id, F(shape));
            builder.AddResource(id, id, GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, resources[id]);
            return id;
        }
        void Emit(GraphOperationId operation, NodeResourceBinding[] bindings, IReadOnlyDictionary<string, string>? attributes = null)
        {
            var id = "conformer.node." + index++;
            builder.AddNode(id, operation, "root", bindings, previous is null ? [] : [previous], attributes);
            previous = id;
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
            Emit(PrimitiveGraphOperations.Affine,
                [GraphBindings.Read("left", source), GraphBindings.Read("right", weight), GraphBindings.Read("bias", bias),
                 GraphBindings.Write("output", output)], Attr(("transpose_left", "false"), ("transpose_right", "true")));
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
        string FeedForward(string source, string prefix)
        {
            var wide = Linear(Norm(source, prefix + ".ln"), prefix + ".up", checked(intermediate * 2));
            var gated = SwiGlu(wide, intermediate, Zero(intermediate), Zero(intermediate));
            return Linear(gated, prefix + ".down", width);
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
