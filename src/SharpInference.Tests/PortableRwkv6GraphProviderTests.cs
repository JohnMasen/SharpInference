using System.Diagnostics;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Backends.Cpu;
using SharpInference.Graphs;
using Xunit.Abstractions;

namespace SharpInference.Tests;

public sealed class PortableRwkv6GraphProviderTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(RwkvTensorDataType.Float32, 1)]
    [InlineData(RwkvTensorDataType.Float16, 2)]
    public void BuildsCompletePrimitiveGraph(RwkvTensorDataType type, int layers)
    {
        var catalog = new Catalog(type, layers);
        var graph = new PortableRwkv6GraphProvider().Build(catalog);
        GraphValidator.Validate(graph);
        PortableTensorOperationContracts.ValidateGraph(graph);
        Assert.Equal("rwkv-6", graph.Identity.ArchitectureId);
        Assert.Equal("rwkv6.forward-token.portable", graph.Identity.Name);
        Assert.Equal("RWKV6_State", graph.GraphState.Schema.Name);
        Assert.Equal("rwkv-6.state.fp32@1", graph.Model.StateAbiId);
        Assert.Equal(["token"], graph.Inputs.Select(id => id.Value));
        Assert.Equal(["logits"], graph.Outputs.Select(id => id.Value));
        Assert.Equal(3 * layers, graph.GraphState.Slots.Count);
        var operations = PrimitiveGraphOperations.CreateStandardDescriptions(false)
            .Select(item => item.Operation)
            .Concat(PortableTensorOperationContracts.Contracts.Select(item => item.Operation))
            .ToHashSet();
        Assert.All(graph.Nodes, node => Assert.Contains(node.Operation, operations));
        Assert.Equal(type == RwkvTensorDataType.Float16 ? catalog.Names.Count : 0,
            graph.Nodes.Count(node => node.Operation == PortableTensorOperationContracts.CastFp16ToFp32));
        Assert.Equal(layers, graph.Nodes.Count(node => node.Operation == PrimitiveGraphOperations.Copy &&
            node.Resources.Any(binding => binding.Port == "output" &&
                binding.Resource.Value.EndsWith(".wkv", StringComparison.Ordinal))));
        Assert.Equal(layers, graph.Nodes.Count(node =>
            node.Operation == PortableTensorOperationContracts.BatchedMatVec));
        Assert.Equal(layers, graph.Nodes.Count(node =>
            node.Operation == PortableTensorOperationContracts.HeadOuter));
        for (var i = 1; i < graph.Nodes.Count; i++)
            Assert.Equal([graph.Nodes[i - 1].Id], graph.Nodes[i].Dependencies);
        Assert.Equal(Enumerable.Range(0, layers).SelectMany(layer => new[]
        {
            $"state.{layer}.ffn-previous", $"state.{layer}.att-previous", $"state.{layer}.wkv",
        }), graph.GraphState.Slots.Select(slot => slot.Name));
    }

    [Theory]
    [InlineData(RwkvTensorDataType.Float32, 1)]
    [InlineData(RwkvTensorDataType.Float32, 2)]
    [InlineData(RwkvTensorDataType.Float16, 1)]
    [InlineData(RwkvTensorDataType.Float16, 2)]
    public void MatchesScalarReferenceForTwoTokens(RwkvTensorDataType type, int layers)
    {
        var catalog = new Catalog(type, layers);
        var graph = new PortableRwkv6GraphProvider().Build(catalog);
        var executor = new CpuPrimitiveGraphExecutor(graph, catalog);
        var initial = graph.GraphState.Slots.Select((slot, index) =>
        {
            var dimensions = graph.Resources.Single(resource => resource.Id == slot.Resource).Tensor.Dimensions;
            var count = dimensions.Aggregate(1, (product, dimension) => product * dimension);
            return new GraphStateValue(slot.Name, dimensions,
                Enumerable.Range(0, count).Select(i => 0.015f * (index + 1) * (i - 2)).ToArray());
        }).ToArray();
        IReadOnlyList<GraphStateValue> portableState = initial;
        IReadOnlyList<GraphStateValue> referenceState = initial;
        foreach (var token in new[] { 2, 4 })
        {
            var result = executor.Execute(
                new Dictionary<ResourceId, Array> { [new("token")] = new[] { token } }, portableState);
            var (expected, nextState) = ReferenceForward(catalog, token, referenceState);
            AssertClose(expected, Assert.IsType<float[]>(result.Outputs[new("logits")]), 3e-4f);
            AssertClose(nextState.SelectMany(entry => entry.Values).ToArray(),
                result.State.SelectMany(entry => entry.Values).ToArray(), 3e-4f);
            portableState = result.State;
            referenceState = nextState;
        }
    }

    [Theory]
    [InlineData(RwkvTensorDataType.Float32, OptimizationBoundary.Off)]
    [InlineData(RwkvTensorDataType.Float32, OptimizationBoundary.Unrestricted)]
    [InlineData(RwkvTensorDataType.Float16, OptimizationBoundary.Off)]
    [InlineData(RwkvTensorDataType.Float16, OptimizationBoundary.Unrestricted)]
    public void EightTokensMatchScalarReferenceAndReportWarmThroughput(
        RwkvTensorDataType type, OptimizationBoundary optimization)
    {
        const int layers = 2;
        const float tolerance = 3e-4f;
        int[] tokens = [2, 4, 1, 0, 3, 2, 1, 4];
        var catalog = new Catalog(type, layers);
        var logical = new PortableRwkv6GraphProvider().Build(catalog);
        var execution = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(optimization,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded));
        var executor = new CpuPrimitiveGraphExecutor(execution, catalog);
        var initial = logical.GraphState.Slots.Select((slot, index) =>
        {
            var dimensions = logical.Resources.Single(resource => resource.Id == slot.Resource).Tensor.Dimensions;
            var count = dimensions.Aggregate(1, (product, dimension) => product * dimension);
            return new GraphStateValue(slot.Name, dimensions,
                Enumerable.Range(0, count).Select(i => 0.015f * (index + 1) * (i - 2)).ToArray());
        }).ToArray();
        var inputs = tokens.Select(token => (IReadOnlyDictionary<ResourceId, Array>)
            new Dictionary<ResourceId, Array> { [new("token")] = new[] { token } }).ToArray();

        IReadOnlyList<GraphStateValue> portableState = initial;
        IReadOnlyList<GraphStateValue> referenceState = initial;
        for (var step = 0; step < tokens.Length; step++)
        {
            var result = executor.Execute(inputs[step], portableState);
            var (expected, nextState) = ReferenceForward(catalog, tokens[step], referenceState);
            AssertClose(expected, Assert.IsType<float[]>(result.Outputs[new("logits")]), tolerance);
            AssertClose(nextState.SelectMany(entry => entry.Values).ToArray(),
                result.State.SelectMany(entry => entry.Values).ToArray(), tolerance);
            portableState = result.State;
            referenceState = nextState;
        }

        // Warm both paths with a complete eight-token sequence before timing.
        referenceState = initial;
        portableState = initial;
        for (var step = 0; step < tokens.Length; step++)
        {
            portableState = executor.Execute(inputs[step], portableState).State;
            (_, referenceState) = ReferenceForward(catalog, tokens[step], referenceState);
        }

        referenceState = initial;
        var stopwatch = Stopwatch.StartNew();
        foreach (var token in tokens)
            (_, referenceState) = ReferenceForward(catalog, token, referenceState);
        stopwatch.Stop();
        var referenceMilliseconds = stopwatch.Elapsed.TotalMilliseconds;

        portableState = initial;
        stopwatch.Restart();
        foreach (var input in inputs)
            portableState = executor.Execute(input, portableState).State;
        stopwatch.Stop();
        var portableMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
        output.WriteLine(
            $"RWKV6 {type}, optimization={optimization}, nodes={execution.Nodes.Count}/{logical.Nodes.Count}, " +
            $"layers={layers}, width=4, " +
            $"8 tokens: scalar-reference={referenceMilliseconds:F3} ms ({tokens.Length / (referenceMilliseconds / 1000):F1} tok/s), " +
            $"portable={portableMilliseconds:F3} ms ({tokens.Length / (portableMilliseconds / 1000):F1} tok/s); " +
            $"logits/state tolerance={tolerance:G}");
    }

    [Fact]
    public void RejectsInvalidGroupedProjectionInsteadOfEmittingAMisleadingGraph()
    {
        var catalog = new Catalog(RwkvTensorDataType.Float32, 1, invalidMaa: true);
        Assert.Throws<InvalidDataException>(() => new PortableRwkv6GraphProvider().Build(catalog));
    }

    private static (float[] Logits, GraphStateValue[] State) ReferenceForward(
        Catalog catalog, int token, IReadOnlyList<GraphStateValue> previous)
    {
        const int width = 4;
        const int headSize = 2;
        var state = previous.Select(entry =>
            new GraphStateValue(entry.Name, entry.Dimensions, (float[])entry.Values.Clone())).ToArray();
        var slots = state.ToDictionary(entry => entry.Name, entry => entry.Values);
        var cache = new Dictionary<string, float[]>(StringComparer.Ordinal);
        float[] Weight(string name)
        {
            if (cache.TryGetValue(name, out var values)) return values;
            var tensor = catalog.GetRequired(name);
            values = tensor.DataType == RwkvTensorDataType.Float16
                ? tensor.HalfValues.ToArray().Select(value => (float)value).ToArray()
                : tensor.FloatValues.ToArray();
            cache.Add(name, values);
            return values;
        }
        float[] Project(string name, float[] input)
        {
            var tensor = catalog.GetRequired(name);
            Assert.Equal(input.Length, tensor.Dimensions[0]);
            var matrix = Weight(name);
            var result = new float[tensor.Dimensions[1]];
            for (var row = 0; row < result.Length; row++)
                for (var column = 0; column < input.Length; column++)
                    result[row] += matrix[row * input.Length + column] * input[column];
            return result;
        }
        static float[] Map(float[] input, Func<float, float> function)
        {
            var result = new float[input.Length];
            for (var i = 0; i < input.Length; i++) result[i] = function(input[i]);
            return result;
        }
        static float[] Zip(float[] left, float[] right, Func<float, float, float> function)
        {
            var result = new float[left.Length];
            for (var i = 0; i < result.Length; i++) result[i] = function(left[i], right[i]);
            return result;
        }
        static float[] Add(float[] left, float[] right) => Zip(left, right, (x, y) => x + y);
        static float[] Multiply(float[] left, float[] right) => Zip(left, right, (x, y) => x * y);
        static float[] Mix(float[] x, float[] previousValue, float[] coefficient)
        {
            var result = new float[x.Length];
            for (var i = 0; i < result.Length; i++)
                result[i] = x[i] + (previousValue[i] - x[i]) * coefficient[i];
            return result;
        }
        float[] Normalize(float[] input, string weight, string bias, float epsilon)
        {
            float mean = 0;
            foreach (var value in input) mean += value;
            mean /= input.Length;
            float variance = 0;
            foreach (var value in input)
            {
                var delta = value - mean;
                variance += delta * delta;
            }
            var scale = 1f / MathF.Sqrt(variance / input.Length + epsilon);
            var weights = Weight(weight);
            var biases = Weight(bias);
            var result = new float[input.Length];
            for (var i = 0; i < input.Length; i++)
                result[i] = (input[i] - mean) * scale * weights[i] + biases[i];
            return result;
        }

        var embedding = Weight("emb.weight");
        var x = Normalize(embedding.AsSpan(token * width, width).ToArray(),
            "blocks.0.ln0.weight", "blocks.0.ln0.bias", 1e-5f);
        for (var layer = 0; layer < catalog.LayerCount; layer++)
        {
            var prefix = $"blocks.{layer}.";
            string Att(string suffix) => prefix + "att." + suffix;
            var attPrevious = slots[$"state.{layer}.att-previous"];
            var ffnPrevious = slots[$"state.{layer}.ffn-previous"];
            var matrixState = slots[$"state.{layer}.wkv"];
            var normalized = Normalize(x, prefix + "ln1.weight", prefix + "ln1.bias", 1e-5f);
            var maaInput = Mix(normalized, attPrevious, Weight(Att("time_maa_x")));
            var maaHidden = Map(Project(Att("time_maa_w1"), maaInput), MathF.Tanh);
            var hiddenSize = maaHidden.Length / 5;
            var w2 = Weight(Att("time_maa_w2"));
            var dynamicMix = new float[5][];
            for (var group = 0; group < 5; group++)
            {
                dynamicMix[group] = new float[width];
                for (var row = 0; row < width; row++)
                    for (var column = 0; column < hiddenSize; column++)
                        dynamicMix[group][row] += w2[(group * width + row) * hiddenSize + column] *
                            maaHidden[group * hiddenSize + column];
            }
            float[] Mixed(int group, string suffix) =>
                Mix(normalized, attPrevious, Add(dynamicMix[group], Weight(Att(suffix))));
            var xw = Mixed(0, "time_maa_w");
            var k = Project(Att("key.weight"), Mixed(1, "time_maa_k"));
            var v = Project(Att("value.weight"), Mixed(2, "time_maa_v"));
            var r = Project(Att("receptance.weight"), Mixed(3, "time_maa_r"));
            var gateLinear = Project(Att("gate.weight"), Mixed(4, "time_maa_g"));
            var gate = Map(gateLinear, value => value / (1f + MathF.Exp(-value)));
            var decayHidden = Map(Project(Att("time_decay_w1"), xw), MathF.Tanh);
            var decayOffset = Add(Project(Att("time_decay_w2"), decayHidden), Weight(Att("time_decay")));
            var decay = Map(decayOffset, value => MathF.Exp(-MathF.Exp(value)));
            var first = Weight(Att("time_faaaa"));
            var attention = new float[width];
            for (var head = 0; head < width / headSize; head++)
                for (var i = 0; i < headSize; i++)
                {
                    var vectorIndex = head * headSize + i;
                    for (var j = 0; j < headSize; j++)
                    {
                        var stateIndex = head * headSize * headSize + i * headSize + j;
                        var current = k[vectorIndex] * v[head * headSize + j];
                        var prior = matrixState[stateIndex];
                        attention[head * headSize + j] +=
                            (prior + current * first[vectorIndex]) * r[vectorIndex];
                        matrixState[stateIndex] = prior * decay[vectorIndex] + current;
                    }
                }
            for (var head = 0; head < width / headSize; head++)
            {
                var offset = head * headSize;
                float mean = 0;
                for (var i = 0; i < headSize; i++) mean += attention[offset + i];
                mean /= headSize;
                float variance = 0;
                for (var i = 0; i < headSize; i++)
                {
                    var difference = attention[offset + i] - mean;
                    variance += difference * difference;
                }
                var scale = 1f / MathF.Sqrt(variance / headSize + 64e-5f);
                for (var i = 0; i < headSize; i++)
                    attention[offset + i] = (attention[offset + i] - mean) * scale;
            }
            var groupWeight = Weight(Att("ln_x.weight"));
            var groupBias = Weight(Att("ln_x.bias"));
            for (var i = 0; i < width; i++)
                attention[i] = (attention[i] * groupWeight[i] + groupBias[i]) * gate[i];
            x = Add(x, Project(Att("output.weight"), attention));
            normalized.CopyTo(attPrevious, 0);

            var ffnNormalized = Normalize(x, prefix + "ln2.weight", prefix + "ln2.bias", 1e-5f);
            var keyInput = Mix(ffnNormalized, ffnPrevious, Weight(prefix + "ffn.time_maa_k"));
            var receptanceInput = Mix(ffnNormalized, ffnPrevious, Weight(prefix + "ffn.time_maa_r"));
            var receptance = Map(Project(prefix + "ffn.receptance.weight", receptanceInput),
                value => 1f / (1f + MathF.Exp(-value)));
            var hidden = Map(Project(prefix + "ffn.key.weight", keyInput),
                value => MathF.Max(0f, value) * MathF.Max(0f, value));
            x = Add(x, Multiply(Project(prefix + "ffn.value.weight", hidden), receptance));
            ffnNormalized.CopyTo(ffnPrevious, 0);
        }
        var output = Normalize(x, "ln_out.weight", "ln_out.bias", 1e-5f);
        return (Project("head.weight", output), state);
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && MathF.Abs(expected[i] - actual[i]) <= tolerance,
                $"Element {i}: expected {expected[i]}, actual {actual[i]}");
    }

    private sealed class Catalog : IModelTensorCatalog
    {
        private readonly Dictionary<string, IModelTensor> tensors = new(StringComparer.Ordinal);

        public Catalog(RwkvTensorDataType type, int layers, bool invalidMaa = false)
        {
            LayerCount = layers;
            void Add(string name, params int[] dimensions)
            {
                var count = dimensions.Aggregate(1, (product, dimension) => product * dimension);
                var values = Enumerable.Range(0, count).Select(index =>
                    name.EndsWith(".weight", StringComparison.Ordinal) &&
                    (name.Contains(".ln", StringComparison.Ordinal) ||
                     name == "ln_out.weight")
                        ? 0.9f + (index % 4) * 0.06f
                        : ((index * 7 + name.Length * 3) % 23 - 11) * 0.025f).ToArray();
                tensors.Add(name, new Tensor(name, dimensions, type, values));
            }
            Add("emb.weight", 4, 5);
            Add("head.weight", 4, 5);
            Add("ln_out.weight", 4);
            Add("ln_out.bias", 4);
            Add("blocks.0.ln0.weight", 4);
            Add("blocks.0.ln0.bias", 4);
            for (var layer = 0; layer < layers; layer++)
            {
                var p = $"blocks.{layer}.";
                foreach (var suffix in new[]
                {
                    "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
                    "att.time_maa_x", "att.time_maa_w", "att.time_maa_k",
                    "att.time_maa_v", "att.time_maa_r", "att.time_maa_g",
                    "att.time_decay", "att.ln_x.weight", "att.ln_x.bias",
                    "ffn.time_maa_k", "ffn.time_maa_r",
                }) Add(p + suffix, 4);
                Add(p + "att.time_faaaa", 1, 2, 2);
                Add(p + "att.time_maa_w1", 4, 10);
                Add(p + "att.time_maa_w2", invalidMaa ? 3 : 2, 4, 5);
                Add(p + "att.time_decay_w1", 4, 3);
                Add(p + "att.time_decay_w2", 3, 4);
                foreach (var suffix in new[]
                {
                    "att.receptance.weight", "att.key.weight", "att.value.weight",
                    "att.gate.weight", "att.output.weight", "ffn.receptance.weight",
                }) Add(p + suffix, 4, 4);
                Add(p + "ffn.key.weight", 4, 6);
                Add(p + "ffn.value.weight", 6, 4);
            }
        }

        public int VocabularySize => 5;
        public int EmbeddingSize => 4;
        public int LayerCount { get; }
        public IReadOnlyCollection<string> Names => tensors.Keys;
        public bool TryGet(string name, out IModelTensor tensor) => tensors.TryGetValue(name, out tensor!);
        public IModelTensor GetRequired(string name) => tensors[name];
    }

    private sealed class Tensor(
        string name, int[] dimensions, RwkvTensorDataType type, float[] values) : IModelTensor
    {
        private readonly float[] floats = type == RwkvTensorDataType.Float32 ? values : [];
        private readonly Half[] halves = type == RwkvTensorDataType.Float16
            ? values.Select(value => (Half)value).ToArray() : [];
        public string Name => name;
        public IReadOnlyList<int> Dimensions => dimensions;
        public RwkvTensorDataType DataType => type;
        public ReadOnlySpan<float> FloatValues => floats;
        public ReadOnlySpan<Half> HalfValues => halves;
    }
}
