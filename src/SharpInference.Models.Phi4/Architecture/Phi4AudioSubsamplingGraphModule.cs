using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Architectures.Phi4;

public sealed class Phi4AudioSubsamplingGraphModule : IModelGraphModule, IGraphOperationValidator
{
    private static readonly string[] Prefixes = ["a.conv1d.0", "a.conv1d.2", "a.conv1d.3", "a.conv1d.5", "a.conv1d.6"];
    private readonly int frames;

    public Phi4AudioSubsamplingGraphModule(int frames = 352)
    {
        if (frames <= 0) throw new ArgumentOutOfRangeException(nameof(frames));
        this.frames = frames;
    }

    public string ArchitectureId => "phi4.audio.subsampling";
    public IGraphOperationValidator OperationValidator => this;

    public bool CanLoad(IModelTensorCatalog tensors) =>
        tensors.TryGet("a.global_mean", out _) && tensors.TryGet("a.global_invstd", out _) &&
        Prefixes.All(prefix => tensors.TryGet(prefix + ".weight", out _) && tensors.TryGet(prefix + ".bias", out _));

    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var mean = tensors.GetRequired("a.global_mean");
        if (mean.Dimensions.Count != 1 || mean.Dimensions[0] <= 0)
            throw new InvalidDataException("Phi4 audio feature means must be a positive vector.");
        var features = mean.Dimensions[0];
        Require("a.global_mean", [features]);
        Require("a.global_invstd", [features]);
        var first = tensors.GetRequired(Prefixes[0] + ".weight");
        if (first.Dimensions.Count != 4 || first.Dimensions[0] <= 0)
            throw new InvalidDataException("Phi4 audio subsampling requires a four-dimensional first convolution weight.");
        var channels = first.Dimensions[0];
        for (var index = 0; index < Prefixes.Length; index++)
        {
            var kernel = index is 2 or 4 ? 1 : 3;
            var inputChannels = index == 0 || index is 1 or 3 ? 1 : channels;
            Require(Prefixes[index] + ".weight", [channels, inputChannels, kernel, kernel]);
            Require(Prefixes[index] + ".bias", [channels]);
        }
        return new(ArchitectureId, new Dictionary<string, long>
        {
            ["frames"] = frames,
            ["features"] = features,
            ["channels"] = channels,
        });

        void Require(string name, int[] dimensions)
        {
            var tensor = tensors.GetRequired(name);
            if (tensor.DataType is not (TensorDataType.Float16 or TensorDataType.Float32) ||
                !tensor.Dimensions.SequenceEqual(dimensions))
                throw new InvalidDataException($"Phi4 audio tensor '{name}' has an incompatible type or shape.");
        }
    }

    public LogicalGraph Build(IModelTensorCatalog tensors)
    {
        var metadata = ReadMetadata(tensors);
        var features = checked((int)metadata.Dimensions["features"]);
        var channels = checked((int)metadata.Dimensions["channels"]);
        var builder = new LogicalGraphBuilder(new(ArchitectureId, 2, "phi4.audio.subsampling.forward"),
                new GraphModelSignature("phi4.audio.subsampling", "phi4.audio.subsampling.empty@1",
                    metadata.Dimensions.ToDictionary(value => value.Key, value => checked((int)value.Value), StringComparer.Ordinal)))
            .AddRegion("root", GraphRegionTypes.Graph, "Audio subsampling", architecture: new(
                Description: "Feature normalization, strided/depthwise/pointwise convolutions and token-major flattening.",
                DefaultView: GraphArchitectureView.Architecture))
            .AddResource("audio_features", "Audio features", GraphResourceKind.Input, GraphResourceLifetime.External,
                F(frames, features), graphInput: true)
            .AddResource("frame_count", "Valid frame count", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true);
        builder.WithRegion("normalization", GraphRegionTypes.Architecture, "Feature Normalization", _ =>
        {
            Temporary("normalize_output", frames, features);
            var mean = Weight("a.global_mean");
            var inverse = Weight("a.global_invstd");
            Emit("normalize", Phi4AudioGraphOperations.NormalizeFeatures,
                [GraphBindings.Read("input", "audio_features"), GraphBindings.Read("mean", mean),
                 GraphBindings.Read("inverse_standard_deviation", inverse), GraphBindings.Read("frame_count", "frame_count"),
                 GraphBindings.Write("output", "normalize_output")], Attr(("frames", frames), ("features", features)));
        }, parentId: "root", architecture: new(Description: "Normalizes valid audio frames with global statistics.",
            Formula: "x_norm = (x - mean) .* inverse_std"));
        var source = "normalize_output";
        var height = frames;
        var width = features;
        var inputChannels = 1;
        for (var index = 0; index < Prefixes.Length; index++)
        {
            var kernel = index is 2 or 4 ? 1 : 3;
            var stride = kernel == 3 ? 2 : 1;
            var padding = kernel == 3 ? 1 : 0;
            var groups = index is 1 or 3 ? channels : 1;
            builder.WithRegion($"convolution.{index}", GraphRegionTypes.Architecture, $"Convolution {index}", _ =>
            {
                var outputHeight = stride == 2 ? Half(height) : height;
                var outputWidth = stride == 2 ? Half(width) : width;
                var destination = $"convolution_{index}_output";
                Temporary(destination, outputHeight, outputWidth, channels);
                var weight = Weight(Prefixes[index] + ".weight");
                var bias = Weight(Prefixes[index] + ".bias");
                Emit($"convolution_{index}", Phi4AudioGraphOperations.Conv2D,
                    [new("input", new(source), GraphResourceAccess.Read, View: new(0, F(height, width, inputChannels))),
                 GraphBindings.Read("weight", weight), GraphBindings.Read("bias", bias), GraphBindings.Write("output", destination)],
                    Attr(("input_height", height), ("input_width", width), ("input_channels", inputChannels),
                        ("output_height", outputHeight), ("output_width", outputWidth), ("output_channels", channels),
                        ("kernel_height", kernel), ("kernel_width", kernel), ("padding", padding), ("stride", stride),
                        ("groups", groups), ("activation", index is 1 or 3 ? "none" : "relu")));
                source = destination;
                height = outputHeight;
                width = outputWidth;
                inputChannels = channels;
            }, parentId: "root", architecture: new(Description: $"Kernel {kernel}, stride {stride}, groups {groups}; includes weight conversion and the configured activation."));
        }
        builder.AddResource("output", "Subsampled audio features", GraphResourceKind.Output, GraphResourceLifetime.External,
            F(height, checked(width * channels)), graphOutput: true);
        builder.WithRegion("flattening", GraphRegionTypes.Architecture, "Token-Major Flattening", _ =>
        {
            Emit("flatten", Phi4AudioGraphOperations.FlattenSubsampling,
                [GraphBindings.Read("input", source), GraphBindings.Write("output", "output")],
                Attr(("tokens", height), ("frequency", width), ("channels", channels)));
        }, parentId: "root", architecture: new(Description: "Flattens frequency/channel features into the Conformer token width."));
        var graph = builder.BuildSequential();
        GraphValidator.Validate(graph, this);
        return graph;

        void Temporary(string id, params int[] dimensions) => builder.AddResource(id, id,
            GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, F(dimensions));
        string Weight(string name)
        {
            var tensor = tensors.GetRequired(name);
            var id = "weight." + name;
            builder.AddResource(id, name, GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(tensor.DataType == TensorDataType.Float16 ? GraphElementType.Float16 : GraphElementType.Float32,
                    tensor.Dimensions), name);
            if (tensor.DataType == TensorDataType.Float32) return id;
            var converted = "cast." + name + "_output";
            Temporary(converted, tensor.Dimensions.ToArray());
            Emit("cast." + name, PortableTensorOperationContracts.CastFp16ToFp32,
                [GraphBindings.Read("input", id), GraphBindings.Write("output", converted)], new Dictionary<string, string>());
            return converted;
        }
        void Emit(string id, GraphOperationId operation, NodeResourceBinding[] bindings, IReadOnlyDictionary<string, string> attributes)
        {
            builder.AddNode(id, operation, bindings, null, attributes);
        }
    }

    public void Validate(GraphOperationValidationContext context) => Phi4AudioSubsamplingValidation.Validate(context);

    private static TensorDescriptor F(params int[] dimensions) => new(GraphElementType.Float32, dimensions);
    private static int Half(int value) => value / 2 + value % 2;
    private static IReadOnlyDictionary<string, string> Attr(params (string Name, object Value)[] values) =>
        values.ToDictionary(value => value.Name, value => Convert.ToString(value.Value, CultureInfo.InvariantCulture)!, StringComparer.Ordinal);
}
