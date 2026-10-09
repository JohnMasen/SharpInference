using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Architectures.Phi4;

internal static class Phi4AudioOperationValidation
{
    internal static void Validate(GraphOperationValidationContext context)
    {
        if (!context.Operation.Name.StartsWith("phi4.audio.", StringComparison.Ordinal)) return;
        if (context.Operation == Phi4AudioGraphOperations.NormalizeFeatures ||
            context.Operation == Phi4AudioGraphOperations.Conv2D ||
            context.Operation == Phi4AudioGraphOperations.FlattenSubsampling)
        {
            Phi4AudioSubsamplingValidation.Validate(context);
            return;
        }
        if (!NumericTypeCompatibility.Satisfies(new(GraphElementType.Float32, GraphElementType.Float32), context.Requirements))
            throw Error("requires FP32 arithmetic and accumulation");
        if (context.Operation == Phi4AudioGraphOperations.LayerNorm)
        {
            Ports(["input", "weight", "bias"]);
            Attributes(["rows", "width", "epsilon"]);
            var rows = Positive("rows");
            var width = Positive("width");
            if (Finite("epsilon") <= 0) throw Error("requires positive epsilon");
            Shape("input", [rows, width]);
            Shape("weight", [width]);
            Shape("bias", [width]);
            Shape("output", [rows, width]);
        }
        else if (context.Operation == Phi4AudioGraphOperations.SwiGlu)
        {
            Ports(["input", "bias_first", "bias_second"]);
            Attributes(["rows", "width"]);
            var rows = Positive("rows");
            var width = Positive("width");
            Shape("input", [rows, checked(2 * width)]);
            Shape("bias_first", [width]);
            Shape("bias_second", [width]);
            Shape("output", [rows, width]);
        }
        else if (context.Operation == Phi4AudioGraphOperations.Residual)
        {
            Ports(["hidden", "update"]);
            Attributes(["count", "scale"]);
            var count = Positive("count");
            _ = Finite("scale");
            Shape("hidden", [count]);
            Shape("update", [count]);
            Shape("output", [count]);
        }
        else if (context.Operation == Phi4AudioGraphOperations.BiasActivation)
        {
            Ports(["input", "bias"]);
            Attributes(["count", "width", "activation"]);
            var count = Positive("count");
            var width = Positive("width");
            if (count % width != 0 || context.Attributes["activation"] is not ("none" or "relu" or "swish" or "gelu"))
                throw Error("has incompatible width or activation");
            Shape("input", [count]);
            Shape("bias", [width]);
            Shape("output", [count]);
        }
        else if (context.Operation == Phi4AudioGraphOperations.RelativeAttention)
        {
            Ports(["query", "key", "value", "relative_bias", "frame_count"]);
            Attributes(["tokens", "width", "heads", "subsampling"]);
            var tokens = Positive("tokens");
            var width = Positive("width");
            var heads = Positive("heads");
            _ = Positive("subsampling");
            if (width % heads != 0) throw Error("requires width divisible by heads");
            foreach (var port in new[] { "query", "key", "value", "output" }) Shape(port, [tokens, width]);
            Shape("relative_bias", [1000, heads]);
            Shape("frame_count", [1], GraphElementType.Int32);
        }
        else if (context.Operation == Phi4AudioGraphOperations.Conv1D)
        {
            Ports(["input", "weight", "bias"]);
            Attributes(["input_length", "input_channels", "output_length", "output_channels", "kernel",
                "padding", "stride", "groups", "activation"]);
            var length = Positive("input_length");
            var inputChannels = Positive("input_channels");
            var outputChannels = Positive("output_channels");
            var kernel = Positive("kernel");
            var stride = Positive("stride");
            var groups = Positive("groups");
            var outputLength = Positive("output_length");
            if (!int.TryParse(context.Attributes["padding"], NumberStyles.None, CultureInfo.InvariantCulture, out var padding) ||
                inputChannels % groups != 0 || outputChannels % groups != 0 ||
                context.Attributes["activation"] is not ("none" or "relu"))
                throw Error("has incompatible padding, groups or activation");
            var numerator = (long)length + 2L * padding - kernel;
            if (numerator < 0 || numerator / stride + 1 != outputLength) throw Error("has incompatible convolution dimensions");
            Shape("input", [length, inputChannels]);
            Shape("weight", [outputChannels, inputChannels / groups, kernel]);
            Shape("bias", [outputChannels]);
            Shape("output", [outputLength, outputChannels]);
        }
        else throw new NotSupportedException($"No Phi4 audio contract is registered for '{context.Operation}'.");

        InvalidDataException Error(string reason) => new($"Phi4 audio node '{context.NodeId}' {reason}.");
        void Ports(string[] inputs)
        {
            var names = inputs.Append("output").ToHashSet(StringComparer.Ordinal);
            if (context.Bindings.Count != names.Count || !names.SetEquals(context.Bindings.Select(binding => binding.Port)) ||
                context.Bindings.Any(binding => binding.InitializedBeforeRead ||
                    binding.Access != (binding.Port == "output" ? GraphResourceAccess.Write : GraphResourceAccess.Read)))
                throw Error("has incompatible ports or access");
        }
        void Attributes(string[] names)
        {
            if (!names.ToHashSet(StringComparer.Ordinal).SetEquals(context.Attributes.Keys))
                throw Error("has incompatible attributes");
        }
        int Positive(string name) => int.TryParse(context.Attributes[name], NumberStyles.None, CultureInfo.InvariantCulture, out var value) && value > 0
            ? value : throw Error($"requires positive '{name}'");
        float Finite(string name) => float.TryParse(context.Attributes[name], NumberStyles.Float, CultureInfo.InvariantCulture, out var value) && float.IsFinite(value)
            ? value : throw Error($"requires finite '{name}'");
        void Shape(string port, int[] dimensions, GraphElementType type = GraphElementType.Float32)
        {
            var tensor = context.Tensor(port);
            if (tensor.ElementType != type || tensor.Layout != "dense" || !tensor.Dimensions.SequenceEqual(dimensions))
                throw Error($"has an incompatible '{port}' tensor");
        }
    }
}
