using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Architectures.Phi4;

internal static class Phi4VisionOperationValidation
{
    internal static void Validate(GraphOperationValidationContext context)
    {
        if (!context.Operation.Name.StartsWith("phi4.vision.", StringComparison.Ordinal)) return;
        if (!NumericTypeCompatibility.Satisfies(new(GraphElementType.Float32, GraphElementType.Float32), context.Requirements))
            throw Error("requires FP32 arithmetic and accumulation");
        if (context.Operation == Phi4VisionGraphOperations.PatchEmbedding)
        {
            Ports(["pixels", "mask", "weight", "bias", "position"]);
            Attributes(["crop_size", "patch_size", "width"]);
            var cropSize = Positive("crop_size");
            var patchSize = Positive("patch_size");
            var width = Positive("width");
            if (cropSize % patchSize != 0) throw Error("requires crop size divisible by patch size");
            var grid = cropSize / patchSize;
            var output = context.Tensor("output");
            if (output.Dimensions.Count != 4) throw Error("requires a four-dimensional patch output");
            var crops = output.Dimensions[0];
            Shape("pixels", [crops, 3, cropSize, cropSize]);
            Shape("mask", [crops, grid, grid]);
            Shape("weight", [width, 3, patchSize, patchSize], GraphElementType.Float16);
            Shape("bias", [width], GraphElementType.Float16);
            Shape("position", [checked(grid * grid), width], GraphElementType.Float16);
            Shape("output", [crops, grid, grid, width]);
        }
        else if (context.Operation == Phi4VisionGraphOperations.LayerNorm)
        {
            Ports(["input", "weight", "bias"]);
            Attributes(["epsilon"]);
            if (!float.TryParse(context.Attributes["epsilon"], NumberStyles.Float, CultureInfo.InvariantCulture, out var epsilon) ||
                !float.IsFinite(epsilon) || epsilon <= 0) throw Error("requires finite positive epsilon");
            var input = context.Tensor("input");
            if (input.Dimensions.Count < 2) throw Error("requires rows and channels");
            Shape("input", input.Dimensions.ToArray());
            Shape("output", input.Dimensions.ToArray());
            Shape("weight", [input.Dimensions[^1]], GraphElementType.Float16);
            Shape("bias", [input.Dimensions[^1]], GraphElementType.Float16);
        }
        else if (context.Operation == Phi4VisionGraphOperations.Attention)
        {
            Ports(["query", "key", "value", "mask"]);
            Attributes(["heads"]);
            var heads = Positive("heads");
            var query = context.Tensor("query");
            if (query.Dimensions.Count != 3 || query.Dimensions[2] % heads != 0)
                throw Error("requires [crops, tokens, width] and width divisible by heads");
            foreach (var port in new[] { "query", "key", "value", "output" }) Shape(port, query.Dimensions.ToArray());
            Shape("mask", [query.Dimensions[0], query.Dimensions[1]]);
        }
        else if (context.Operation == Phi4VisionGraphOperations.Gelu)
        {
            Ports(["input"]);
            Attributes([]);
            var input = context.Tensor("input");
            Shape("input", input.Dimensions.ToArray());
            Shape("output", input.Dimensions.ToArray());
        }
        else if (context.Operation == Phi4VisionGraphOperations.Pool2x2)
        {
            Ports(["input"]);
            Attributes([]);
            var input = context.Tensor("input");
            if (input.Dimensions.Count != 4 || input.Dimensions[1] != input.Dimensions[2] || input.Dimensions[1] % 2 != 0)
                throw Error("requires a four-dimensional even square patch grid");
            Shape("input", input.Dimensions.ToArray());
            Shape("output", [input.Dimensions[0], input.Dimensions[1] / 2, input.Dimensions[2] / 2, input.Dimensions[3]]);
        }
        else if (context.Operation == Phi4VisionGraphOperations.HdGather)
        {
            Ports(["input", "mapping", "sub_separator", "global_separator"]);
            Attributes([]);
            var input = context.Tensor("input");
            var output = context.Tensor("output");
            if (input.Dimensions.Count != 2 || output.Dimensions.Count != 2 || input.Dimensions[1] != output.Dimensions[1])
                throw Error("requires compatible flattened patch and HD embedding matrices");
            Shape("input", input.Dimensions.ToArray());
            Shape("output", output.Dimensions.ToArray());
            Shape("mapping", [output.Dimensions[0]], GraphElementType.Int32);
            Shape("sub_separator", [output.Dimensions[1]], GraphElementType.Float16);
            Shape("global_separator", [output.Dimensions[1]], GraphElementType.Float16);
        }
        else throw new NotSupportedException($"No Phi4 vision contract is registered for '{context.Operation}'.");

        InvalidDataException Error(string reason) => new($"Phi4 vision node '{context.NodeId}' {reason}.");
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
            if (!names.ToHashSet(StringComparer.Ordinal).SetEquals(context.Attributes.Keys)) throw Error("has incompatible attributes");
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
