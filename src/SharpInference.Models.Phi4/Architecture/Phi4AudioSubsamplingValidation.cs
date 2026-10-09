using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Architectures.Phi4;

internal static class Phi4AudioSubsamplingValidation
{
    internal static void Validate(GraphOperationValidationContext context)
    {
        if (!context.Operation.Name.StartsWith("phi4.audio.", StringComparison.Ordinal)) return;
        if (!NumericTypeCompatibility.Satisfies(new(GraphElementType.Float32, GraphElementType.Float32), context.Requirements))
            throw Error("requires FP32 arithmetic and accumulation");
        if (context.Operation == Phi4AudioGraphOperations.NormalizeFeatures)
        {
            Ports(["input", "mean", "inverse_standard_deviation", "frame_count"]);
            Attributes(["frames", "features"]);
            var frames = Positive("frames");
            var features = Positive("features");
            Shape("input", [frames, features]);
            Shape("mean", [features]);
            Shape("inverse_standard_deviation", [features]);
            Shape("frame_count", [1], GraphElementType.Int32);
            Shape("output", [frames, features]);
        }
        else if (context.Operation == Phi4AudioGraphOperations.Conv2D)
        {
            Ports(["input", "weight", "bias"]);
            Attributes(["input_height", "input_width", "input_channels", "output_height", "output_width", "output_channels",
                "kernel_height", "kernel_width", "padding", "stride", "groups", "activation"]);
            var height = Positive("input_height");
            var width = Positive("input_width");
            var inputChannels = Positive("input_channels");
            var outputChannels = Positive("output_channels");
            var kernelHeight = Positive("kernel_height");
            var kernelWidth = Positive("kernel_width");
            var stride = Positive("stride");
            var groups = Positive("groups");
            if (!int.TryParse(context.Attributes["padding"], NumberStyles.None, CultureInfo.InvariantCulture, out var padding) ||
                inputChannels % groups != 0 || outputChannels % groups != 0 ||
                context.Attributes["activation"] is not ("none" or "relu"))
                throw Error("has invalid padding, groups or activation");
            var outputHeight = Positive("output_height");
            var outputWidth = Positive("output_width");
            var heightNumerator = (long)height + 2L * padding - kernelHeight;
            var widthNumerator = (long)width + 2L * padding - kernelWidth;
            if (heightNumerator < 0 || widthNumerator < 0 ||
                heightNumerator / stride + 1 != outputHeight || widthNumerator / stride + 1 != outputWidth)
                throw Error("has incompatible convolution dimensions");
            Shape("input", [height, width, inputChannels]);
            Shape("weight", [outputChannels, inputChannels / groups, kernelHeight, kernelWidth]);
            Shape("bias", [outputChannels]);
            Shape("output", [outputHeight, outputWidth, outputChannels]);
        }
        else if (context.Operation == Phi4AudioGraphOperations.FlattenSubsampling)
        {
            Ports(["input"]);
            Attributes(["tokens", "frequency", "channels"]);
            var tokens = Positive("tokens");
            var frequency = Positive("frequency");
            var channels = Positive("channels");
            Shape("input", [tokens, frequency, channels]);
            Shape("output", [tokens, checked(frequency * channels)]);
        }
        else throw new NotSupportedException($"No Phi4 audio subsampling contract is registered for '{context.Operation}'.");

        InvalidDataException Error(string reason) => new($"Phi4 audio node '{context.NodeId}' {reason}.");
        void Ports(string[] inputs)
        {
            var expected = inputs.Append("output").ToHashSet(StringComparer.Ordinal);
            if (context.Bindings.Count != expected.Count || !expected.SetEquals(context.Bindings.Select(binding => binding.Port)) ||
                context.Bindings.Any(binding => binding.Access != (binding.Port == "output" ? GraphResourceAccess.Write : GraphResourceAccess.Read)))
                throw Error("has incompatible ports or access");
        }
        void Attributes(string[] names)
        {
            if (!names.ToHashSet(StringComparer.Ordinal).SetEquals(context.Attributes.Keys))
                throw Error("has incompatible attributes");
        }
        int Positive(string name) => int.TryParse(context.Attributes[name], NumberStyles.None, CultureInfo.InvariantCulture, out var value) && value > 0
            ? value : throw Error($"requires positive '{name}'");
        void Shape(string port, int[] dimensions, GraphElementType type = GraphElementType.Float32)
        {
            var tensor = context.Tensor(port);
            if (tensor.ElementType != type || tensor.Layout != "dense" || !tensor.Dimensions.SequenceEqual(dimensions))
                throw Error($"has an incompatible '{port}' tensor");
        }
    }
}
