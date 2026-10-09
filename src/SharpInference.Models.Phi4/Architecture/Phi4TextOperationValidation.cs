using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Architectures.Phi4;

internal static class Phi4TextOperationValidation
{
    internal static void Validate(GraphOperationValidationContext c)
    {
        if (!c.Operation.Name.StartsWith("phi4.text.", StringComparison.Ordinal)) return;
        if (!NumericTypeCompatibility.Satisfies(new(GraphElementType.Float32, GraphElementType.Float32), c.Requirements))
            throw Error("requires FP32 arithmetic and accumulation");
        if (c.Operation == Phi4TextGraphOperations.RmsNorm)
        {
            Ports(["input", "weight"], "output");
            Attributes("epsilon");
            PositiveFloat("epsilon");
            Vector("input");
            Shape("weight", c.Tensor("input").Dimensions);
            Shape("output", c.Tensor("input").Dimensions);
        }
        else if (c.Operation == Phi4TextGraphOperations.SwiGlu)
        {
            Ports(["gate_up"], "output");
            Attributes();
            Vector("output");
            Shape("gate_up", [checked(2 * c.Tensor("output").Dimensions[0])]);
        }
        else if (c.Operation == Phi4TextGraphOperations.ScaledAdd)
        {
            Ports(["input", "update"], "output");
            Attributes("scale");
            PositiveFloat("scale");
            Vector("input");
            Shape("update", c.Tensor("input").Dimensions);
            Shape("output", c.Tensor("input").Dimensions);
        }
        else if (c.Operation == Phi4TextGraphOperations.RopeKeyValueWrite)
        {
            Ports(["qkv", "position", "frequencies"], "query", ["key_cache", "value_cache"]);
            Attributes("head_size", "rotary_size", "rope_scale");
            var head = PositiveInt("head_size");
            var rotary = PositiveInt("rotary_size");
            PositiveFloat("rope_scale");
            Cache("key_cache");
            Shape("value_cache", c.Tensor("key_cache").Dimensions);
            Vector("query");
            var query = c.Tensor("query").Dimensions[0];
            var cache = c.Tensor("key_cache").Dimensions;
            if (cache[2] != head || rotary > head || rotary % 2 != 0 || query % head != 0 || query / head % cache[1] != 0)
                throw Error("has incompatible rotary or head dimensions");
            Shape("qkv", [checked(query + 2 * cache[1] * head)]);
            Shape("frequencies", [rotary / 2]);
            Shape("position", [1], GraphElementType.Int32);
        }
        else if (c.Operation == Phi4TextGraphOperations.GroupedQueryScores)
        {
            Ports(["query", "key_cache", "position"], "scores");
            Attributes();
            Cache("key_cache");
            Matrix("scores");
            var cache = c.Tensor("key_cache").Dimensions;
            var scores = c.Tensor("scores").Dimensions;
            if (scores[1] != cache[0] || scores[0] % cache[1] != 0) throw Error("has incompatible GQA scores");
            Shape("query", [checked(scores[0] * cache[2])]);
            Shape("position", [1], GraphElementType.Int32);
        }
        else if (c.Operation == Phi4TextGraphOperations.CausalSoftmax)
        {
            Ports(["scores", "position"], "probabilities");
            Attributes();
            Matrix("scores");
            Shape("probabilities", c.Tensor("scores").Dimensions);
            Shape("position", [1], GraphElementType.Int32);
        }
        else if (c.Operation == Phi4TextGraphOperations.GroupedQueryValues)
        {
            Ports(["probabilities", "value_cache", "position"], "output");
            Attributes();
            Cache("value_cache");
            Matrix("probabilities");
            var cache = c.Tensor("value_cache").Dimensions;
            var probabilities = c.Tensor("probabilities").Dimensions;
            if (probabilities[1] != cache[0] || probabilities[0] % cache[1] != 0) throw Error("has incompatible GQA values");
            Shape("output", [checked(probabilities[0] * cache[2])]);
            Shape("position", [1], GraphElementType.Int32);
        }
        else throw new NotSupportedException($"No Phi4 text contract is registered for '{c.Operation}'.");

        InvalidDataException Error(string reason) => new($"Phi4 text node '{c.NodeId}' {reason}.");
        void Ports(string[] reads, string output, string[]? readWrites = null)
        {
            readWrites ??= [];
            var names = reads.Concat(readWrites).Append(output).ToHashSet(StringComparer.Ordinal);
            if (c.Bindings.Count != names.Count || !names.SetEquals(c.Bindings.Select(binding => binding.Port)) ||
                c.Bindings.Any(binding => binding.InitializedBeforeRead || binding.Access !=
                    (binding.Port == output ? GraphResourceAccess.Write : readWrites.Contains(binding.Port) ? GraphResourceAccess.ReadWrite : GraphResourceAccess.Read)))
                throw Error("has incompatible ports or access");
        }
        void Attributes(params string[] names)
        {
            if (!names.ToHashSet(StringComparer.Ordinal).SetEquals(c.Attributes.Keys)) throw Error("has incompatible attributes");
        }
        void Shape(string port, IReadOnlyList<int> shape, GraphElementType type = GraphElementType.Float32)
        {
            var tensor = c.Tensor(port);
            if (tensor.ElementType != type || tensor.Layout != "dense" || !tensor.Dimensions.SequenceEqual(shape))
                throw Error($"has an incompatible '{port}' tensor");
        }
        void Rank(string port, int rank)
        {
            var tensor = c.Tensor(port);
            if (tensor.Dimensions.Count != rank) throw Error($"requires rank {rank} for '{port}'");
            Shape(port, tensor.Dimensions);
        }
        void Vector(string port) => Rank(port, 1);
        void Matrix(string port) => Rank(port, 2);
        void Cache(string port) => Rank(port, 3);
        int PositiveInt(string name) => int.TryParse(c.Attributes[name], NumberStyles.None, CultureInfo.InvariantCulture, out var value) && value > 0
            ? value : throw Error($"requires positive '{name}'");
        float PositiveFloat(string name) => float.TryParse(c.Attributes[name], NumberStyles.Float, CultureInfo.InvariantCulture, out var value) && float.IsFinite(value) && value > 0
            ? value : throw Error($"requires positive finite '{name}'");
    }
}
