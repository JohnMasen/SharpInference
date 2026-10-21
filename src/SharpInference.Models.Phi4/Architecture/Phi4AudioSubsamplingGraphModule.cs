using SharpInference.Instructions;
using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;
using SharpInference.Graphs.Logical;

namespace SharpInference.Architectures.Phi4;

/// <summary>Builds the Phi-4 audio subsampling graph package for explicit caller validation.</summary>
public sealed class Phi4AudioSubsamplingGraphModule : IPhi4GraphModule
{
    private static readonly string[] Prefixes = ["a.conv1d.0", "a.conv1d.2", "a.conv1d.3", "a.conv1d.5", "a.conv1d.6"];
    private readonly int frames;

    /// <summary>Initializes the audio subsampling graph with the specified input frame capacity.</summary>
    public Phi4AudioSubsamplingGraphModule(int frames = 352)
    {
        if (frames <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(frames));
        }

        this.frames = frames;
    }

    /// <inheritdoc/>
    public string ArchitectureId => "phi4.audio.subsampling";
    /// <inheritdoc/>

    /// <inheritdoc/>
    public bool CanLoad(IModelTensorCatalog tensors) =>
        tensors.TryGet("a.global_mean", out _) && tensors.TryGet("a.global_invstd", out _) &&
        Prefixes.All(prefix => tensors.TryGet(prefix + ".weight", out _) && tensors.TryGet(prefix + ".bias", out _));

    /// <inheritdoc/>
    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var mean = tensors.GetRequired("a.global_mean");
        if (mean.Dimensions.Count != 1 || mean.Dimensions[0] <= 0)
        {
            throw new InvalidDataException("Phi4 audio feature means must be a positive vector.");
        }

        var features = mean.Dimensions[0];
        Require("a.global_mean", [features]);
        Require("a.global_invstd", [features]);
        var first = tensors.GetRequired(Prefixes[0] + ".weight");
        if (first.Dimensions.Count != 4 || first.Dimensions[0] <= 0)
        {
            throw new InvalidDataException("Phi4 audio subsampling requires a four-dimensional first convolution weight.");
        }

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
            {
                throw new InvalidDataException($"Phi4 audio tensor '{name}' has an incompatible type or shape.");
            }
        }
    }

    /// <inheritdoc/>
    public GraphPackage<LogicalGraph> Build(IModelTensorCatalog tensors)
    {
        var metadata = ReadMetadata(tensors);
        var features = checked((int)metadata.Dimensions["features"]);
        var channels = checked((int)metadata.Dimensions["channels"]);
        var builder = new Phi4GraphComposer(new(ArchitectureId, 2, "phi4.audio.subsampling.forward"),
                new GraphModelSignature("phi4.audio.subsampling", "phi4.audio.subsampling.empty@1",
                    metadata.Dimensions.ToDictionary(value => value.Key, value => checked((int)value.Value), StringComparer.Ordinal)))
            .Root("root", GraphSectionTypes.Graph, "Audio subsampling", architecture: new(
                Description: "Feature normalization, strided/depthwise/pointwise convolutions and token-major flattening.",
                DefaultView: GraphArchitectureView.Architecture))
            .Resource("audio_features", F(frames, features), GraphIO.In, name: "Audio features")
            .Resource("frame_count", new TensorDescriptor(ElementType.Int32, [1]), GraphIO.In, name: "Valid frame count");
        builder.Section("normalization", GraphSectionTypes.Architecture, "Feature Normalization", scope =>
        {
            Temporary("normalize_output", frames, features);
            var mean = Weight(scope, "a.global_mean");
            var inverse = Weight(scope, "a.global_invstd");
            Emit(scope, "normalize", Phi4AudioGraphOperations.NormalizeFeatures,
                [GraphBindings.Read("input", "audio_features"), GraphBindings.Read("mean", mean),
                 GraphBindings.Read("inverse_standard_deviation", inverse), GraphBindings.Read("frame_count", "frame_count"),
                 GraphBindings.Write("output", "normalize_output")], Attr(("frames", frames), ("features", features)));
        }, architecture: new(Description: "Normalizes valid audio frames with global statistics.",
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
            builder.Section($"convolution.{index}", GraphSectionTypes.Architecture, $"Convolution {index}", scope =>
            {
                var outputHeight = stride == 2 ? Half(height) : height;
                var outputWidth = stride == 2 ? Half(width) : width;
                var destination = $"convolution_{index}_output";
                Temporary(destination, outputHeight, outputWidth, channels);
                var weight = Weight(scope, Prefixes[index] + ".weight");
                var bias = Weight(scope, Prefixes[index] + ".bias");
                Emit(scope, $"convolution_{index}", Phi4AudioGraphOperations.Conv2D,
                    [new("input", source, ResourceAccess.Read, View: new(0, F(height, width, inputChannels))),
                 GraphBindings.Read("weight", weight), GraphBindings.Read("bias", bias), GraphBindings.Write("output", destination)],
                    Attr(("input_height", height), ("input_width", width), ("input_channels", inputChannels),
                        ("output_height", outputHeight), ("output_width", outputWidth), ("output_channels", channels),
                        ("kernel_height", kernel), ("kernel_width", kernel), ("padding", padding), ("stride", stride),
                        ("groups", groups), ("activation", index is 1 or 3 ? "none" : "relu")));
                source = destination;
                height = outputHeight;
                width = outputWidth;
                inputChannels = channels;
            }, architecture: new(Description: $"Kernel {kernel}, stride {stride}, groups {groups}; includes weight conversion and the configured activation."));
        }
        builder.Resource("output", F(height, checked(width * channels)), GraphIO.Out, name: "Subsampled audio features");
        builder.Section("flattening", GraphSectionTypes.Architecture, "Token-Major Flattening", scope =>
        {
            Emit(scope, "flatten", Phi4AudioGraphOperations.FlattenSubsampling,
                [GraphBindings.Read("input", source), GraphBindings.Write("output", "output")],
                Attr(("tokens", height), ("frequency", width), ("channels", channels)));
        }, architecture: new(Description: "Flattens frequency/channel features into the Conformer token width."));
        return builder.Build();

        void Temporary(string id, params int[] dimensions) => builder.Resource(id, F(dimensions));
        string Weight(Phi4GraphComposer.Scope scope, string name)
        {
            var tensor = tensors.GetRequired(name);
            var id = "weight." + name;
            builder.Resource(id, new TensorDescriptor(tensor.DataType == TensorDataType.Float16 ? ElementType.Float16 : ElementType.Float32,
                    tensor.Dimensions), scope: ResourceScope.Global, usage: ResourceUsage.Weight, permission: ResourcePermission.ReadOnly, bindingKey: name, name: name);
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
        static void Emit(Phi4GraphComposer.Scope scope, string id, OperationId operation, NodeResourceBinding[] bindings, IReadOnlyDictionary<string, string> attributes)
        {
            scope.AddNode(id, operation, bindings, null, attributes);
        }
    }

    /// <inheritdoc/>

    private static TensorDescriptor F(params int[] dimensions) => new(ElementType.Float32, dimensions);
    private static int Half(int value) => value / 2 + value % 2;
    private static IReadOnlyDictionary<string, string> Attr(params (string Name, object Value)[] values) =>
        values.ToDictionary(value => value.Name, value => Convert.ToString(value.Value, CultureInfo.InvariantCulture)!, StringComparer.Ordinal);
}
