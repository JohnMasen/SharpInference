using System.Runtime.InteropServices;
using SharpInference.Architectures.Phi4;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Phi4;
using SharpInference.Instructions.Phi4.D3D12;
using SharpInference.Runtime;
using SharpInference.Runtime.Cpu;
using SharpInference.Runtime.D3D12;
using SharpInference.Vm;

namespace SharpInference.Tests;

[Collection(VmGraphGpuCollection.Name)]
public sealed class Phi4VisionGraphTests
{
    [Theory]
    [InlineData(false, 2)]
    [InlineData(true, 2)]
    [InlineData(false, 26)]
    public void CompleteVisionGraphMatchesSyntheticReferenceOnGpu(bool fp32Linear, int layers)
    {
        var file = new VisionFile(fp32Linear, layers);
        var pixels = Enumerable.Range(0, 2 * 3 * 4 * 4).Select(index => (index % 13 - 6) / 16f).ToArray();
        var mask = Enumerable.Range(0, 2 * 4 * 4).Select(index => index < 16 || (index - 16) / 4 < 2 ? 1f : 0f).ToArray();
        int[] mapping = [4, 5, -1, -2, 0, 1, -1, 2, 3, -1];
        var expected = Reference(file, pixels, mask, mapping, layers);
        var module = new Phi4VisionGraphModule(2, mapping.Length, heads: 2, layers: layers);
        var registry = new ModelGraphModuleRegistry();
        registry.Register(module);
        IInstructionCollectionProvider[] collections = [.. D3D12InstructionCollections.Create(), new Phi4D3D12VisionInstructionCollection()];
        using var backend = D3D12VmBackendFactory.Create(instructionCollections: collections);
        using var processor = Processor.Load("vision", new Reader(file), registry, backend);
        Assert.True(file.Disposed);
        var graph = processor.LogicalGraph!;
        Assert.Equal(layers, graph.Nodes.Count(node => node.Operation == Phi4VisionGraphOperations.Attention));
        Assert.Single(graph.Nodes, node => node.Operation == Phi4VisionGraphOperations.PatchEmbedding);
        Assert.Single(graph.Nodes, node => node.Operation == Phi4VisionGraphOperations.HdGather);
        Assert.Equal(new[] { "pixels", "mask", "mapping" }, graph.Inputs.Select(id => id.Value));
        GraphValidator.Validate(GraphXml.DeserializeLogical(GraphXml.Serialize(graph)), module);
        var restoredProgram = VmProgramXml.Deserialize(VmProgramXml.Serialize(backend.Program!));
        Assert.Contains(restoredProgram.Definitions.SelectMany(definition => definition.Nodes).Select(node => node.Instruction).OfType<VmOperator>(),
            operation => operation.IndexBounds.Any(bound => bound.MinimumIndex == -2));
        using var session = Assert.IsAssignableFrom<ITensorProcessorSession>(((IProcessor)processor).CreateSession());
        Check(mapping);
        foreach (var invalid in new[] { -3, 8 })
        {
            var broken = mapping.ToArray();
            broken[^1] = invalid;
            Assert.Throws<ArgumentOutOfRangeException>(() => session.Execute(Inputs(broken)));
            Check(mapping);
        }
        session.Reset();
        Check(mapping);

        Dictionary<string, ReadOnlyMemory<byte>> Inputs(int[] indices) => new()
        {
            ["pixels"] = MemoryMarshal.AsBytes(pixels.AsSpan()).ToArray(),
            ["mask"] = MemoryMarshal.AsBytes(mask.AsSpan()).ToArray(),
            ["mapping"] = MemoryMarshal.AsBytes(indices.AsSpan()).ToArray(),
        };
        void Check(int[] indices)
        {
            var output = session.Execute(Inputs(indices))["output"];
            var actual = MemoryMarshal.Cast<byte, float>(output).ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (var index = 0; index < actual.Length; index++)
            {
                Assert.True(float.IsFinite(actual[index]));
                Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0, 1e-4f + 1e-4f * MathF.Abs(expected[index]));
            }
            Assert.Contains(actual, value => value != 0);
        }
    }

    [Fact]
    public void MissingVisionRegistrationAndUnsupportedCpuTargetAreExplicitlyRejected()
    {
        var graph = new Phi4VisionGraphModule(2, 10, heads: 2, layers: 2).Build(new VisionFile(false, 2));
        using var cpu = CpuVmBackendFactory.Create(instructionCollections:
            [.. CpuInstructionCollections.Create(), new Phi4D3D12VisionInstructionCollection()]);
        Assert.Throws<BackendPreparationException>(() => cpu.Prepare(graph));
        using var gpu = D3D12VmBackendFactory.Create();
        Assert.Throws<BackendPreparationException>(() => gpu.Prepare(graph));
    }

    [Theory]
    [InlineData("phi4.vision.patch-embedding", "patch_size", "3")]
    [InlineData("phi4.vision.layer-norm", "epsilon", "0")]
    [InlineData("phi4.vision.layer-norm", "epsilon", "NaN")]
    [InlineData("phi4.vision.self-attention", "heads", "3")]
    public void VisionContractsRejectMalformedAttributesAndVersions(string operation, string attribute, string value)
    {
        var module = new Phi4VisionGraphModule(2, 10, heads: 2, layers: 2);
        var graph = module.Build(new VisionFile(false, 2));
        var node = graph.Nodes.First(node => node.Operation.Name == operation);
        var attributes = node.Attributes.ToDictionary(pair => pair.Key, pair => pair.Value);
        attributes[attribute] = value;
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        Assert.Throws<InvalidDataException>(() => module.Validate(new(node.Id.Value, node.Operation,
            node.Resources, resources, attributes, node.Requirements)));
        Assert.Throws<NotSupportedException>(() => module.Validate(new(node.Id.Value, node.Operation with { Version = 99 },
            node.Resources, resources, node.Attributes, node.Requirements)));
    }

    [Fact]
    public void NativePrecisionAndPositionGridAreValidatedByTheModel()
    {
        var module = new Phi4VisionGraphModule(2, 10, heads: 2, layers: 2);
        Assert.Throws<InvalidDataException>(() => module.ReadMetadata(new VisionFile(false, 2, invalidPosition: true)));
        Assert.Throws<InvalidDataException>(() => module.ReadMetadata(new VisionFile(false, 2, invalidNativeType: true)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Phi4VisionGraphModule(0, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Phi4VisionGraphModule(2, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Phi4VisionGraphModule(2, 10, layers: 27));
    }

    private static float[] Reference(VisionFile file, float[] pixels, float[] mask, int[] mapping, int layers)
    {
        const int crops = 2;
        const int grid = 4;
        const int width = 4;
        const int tokens = 16;
        var hidden = new float[crops * tokens * width];
        var patch = file.Values("v.patch_embd.weight");
        var patchBias = file.Values("v.patch_embd.bias");
        var position = file.Values("v.position_embd.weight");
        for (var crop = 0; crop < crops; crop++)
        {
            var validRows = 0;
            while (validRows < grid && mask[crop * tokens + validRows * grid] != 0) validRows++;
            var validColumns = 0;
            while (validColumns < grid && mask[crop * tokens + validColumns] != 0) validColumns++;
            for (var y = 0; y < grid; y++)
            for (var x = 0; x < grid; x++)
            for (var channel = 0; channel < width; channel++)
            {
                var sum = patchBias[channel];
                for (var input = 0; input < 3; input++)
                    sum += pixels[(crop * 3 + input) * tokens + y * grid + x] * patch[channel * 3 + input];
                if (y < validRows && x < validColumns)
                    sum += position[((y * grid / validRows) * grid + x * grid / validColumns) * width + channel];
                hidden[(crop * tokens + y * grid + x) * width + channel] = sum;
            }
        }
        for (var layer = 0; layer < layers; layer++)
        {
            var prefix = $"v.blk.{layer}.";
            var normalized = Norm(hidden, prefix + "ln1");
            var query = Linear(normalized, prefix + "attn_q", width, width);
            var key = Linear(normalized, prefix + "attn_k", width, width);
            var value = Linear(normalized, prefix + "attn_v", width, width);
            var attention = new float[hidden.Length];
            for (var crop = 0; crop < crops; crop++)
            for (var token = 0; token < tokens; token++)
            for (var head = 0; head < 2; head++)
            {
                var scores = new float[tokens];
                var maximum = float.NegativeInfinity;
                for (var source = 0; source < tokens; source++)
                {
                    if (mask[crop * tokens + source] == 0) { scores[source] = float.NegativeInfinity; continue; }
                    var score = 0f;
                    for (var channel = 0; channel < 2; channel++)
                        score += query[(crop * tokens + token) * width + head * 2 + channel] *
                            key[(crop * tokens + source) * width + head * 2 + channel];
                    scores[source] = score / MathF.Sqrt(2);
                    maximum = MathF.Max(maximum, scores[source]);
                }
                var denominator = 0f;
                for (var source = 0; source < tokens; source++)
                {
                    scores[source] = mask[crop * tokens + source] == 0 ? 0 : MathF.Exp(scores[source] - maximum);
                    denominator += scores[source];
                }
                for (var channel = 0; channel < 2; channel++)
                {
                    var sum = 0f;
                    for (var source = 0; source < tokens; source++) sum += scores[source] * value[(crop * tokens + source) * width + head * 2 + channel];
                    attention[(crop * tokens + token) * width + head * 2 + channel] = sum / denominator;
                }
            }
            hidden = Add(hidden, Linear(attention, prefix + "attn_out", width, width));
            hidden = Add(hidden, Linear(Gelu(Linear(Norm(hidden, prefix + "ln2"), prefix + "ffn_up", width, 6)), prefix + "ffn_down", 6, width));
        }
        var pooled = new float[crops * 2 * 2 * width];
        for (var crop = 0; crop < crops; crop++)
        for (var y = 0; y < 2; y++)
        for (var x = 0; x < 2; x++)
        for (var channel = 0; channel < width; channel++)
        {
            var top = ((crop * grid + y * 2) * grid + x * 2) * width + channel;
            pooled[((crop * 2 + y) * 2 + x) * width + channel] =
                (hidden[top] + hidden[top + width] + hidden[top + grid * width] + hidden[top + (grid + 1) * width]) * 0.25f;
        }
        var gathered = new float[mapping.Length * width];
        for (var token = 0; token < mapping.Length; token++)
        for (var channel = 0; channel < width; channel++)
            gathered[token * width + channel] = mapping[token] >= 0 ? pooled[mapping[token] * width + channel] :
                file.Values(mapping[token] == -1 ? "v.sub_GN" : "v.glb_GN")[channel];
        return Linear(Gelu(Linear(gathered, "mm.0", width, 5)), "mm.2", 5, 5);

        float[] Linear(float[] input, string prefix, int inputWidth, int outputWidth)
        {
            var weight = file.Values(prefix + ".weight");
            var bias = file.Values(prefix + ".bias");
            var output = new float[input.Length / inputWidth * outputWidth];
            for (var row = 0; row < input.Length / inputWidth; row++)
            for (var column = 0; column < outputWidth; column++)
            {
                var sum = 0f;
                for (var inner = 0; inner < inputWidth; inner++) sum += input[row * inputWidth + inner] * weight[column * inputWidth + inner];
                output[row * outputWidth + column] = sum + bias[column];
            }
            return output;
        }
        float[] Norm(float[] input, string prefix)
        {
            var output = new float[input.Length];
            SharpInference.Instructions.Phi4.Cpu.Phi4CpuAudioNumerics.LayerNorm(input, file.Values(prefix + ".weight"),
                file.Values(prefix + ".bias"), output, input.Length / width, width, 1e-6f);
            return output;
        }
        static float[] Add(float[] left, float[] right) => left.Zip(right, (a, b) => a + b).ToArray();
        static float[] Gelu(float[] input) => input.Select(value =>
            0.5f * value * (1f + MathF.Tanh(0.7978845608028654f * (value + 0.044715f * value * value * value)))).ToArray();
    }

    private sealed class Reader(VisionFile file) : IModelReader
    {
        public IModelFile Open(string path) => file;
    }

    internal static IModelFile CreateSyntheticFile(bool half, int layers, int embeddingWidth) => new VisionFile(!half, layers, embeddingWidth: embeddingWidth);

    private sealed class VisionFile : IModelFile
    {
        private readonly Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        public VisionFile(bool fp32Linear, int layers, bool invalidPosition = false, bool invalidNativeType = false, int embeddingWidth = 5)
        {
            Add("v.patch_embd.weight", [4, 3, 1, 1], half: !invalidNativeType);
            Add("v.patch_embd.bias", [4]);
            Add("v.position_embd.weight", [invalidPosition ? 15 : 16, 4]);
            Add("v.sub_GN", [4], index => (index + 1) / 8f);
            Add("v.glb_GN", [4], index => -(index + 1) / 8f);
            for (var layer = 0; layer < layers; layer++)
            {
                var prefix = $"v.blk.{layer}.";
                foreach (var part in new[] { "ln1", "ln2" })
                {
                    Add(prefix + part + ".weight", [4], index => 1f + index / 16f);
                    Add(prefix + part + ".bias", [4]);
                }
                foreach (var part in new[] { "attn_q", "attn_k", "attn_v", "attn_out" }) Linear(prefix + part, 4, 4);
                Linear(prefix + "ffn_up", 4, 6);
                Linear(prefix + "ffn_down", 6, 4);
            }
            Linear("mm.0", 4, embeddingWidth);
            Linear("mm.2", embeddingWidth, embeddingWidth);
            void Linear(string prefix, int input, int output)
            {
                Add(prefix + ".weight", [output, input], half: !fp32Linear);
                Add(prefix + ".bias", [output], index => (index + 1) / 64f, half: !fp32Linear);
            }
            void Add(string name, int[] dimensions, Func<int, float>? value = null, bool half = true)
            {
                value ??= index => (index % 7 - 3) / 32f;
                tensors.Add(name, new(this, name, dimensions, half,
                    Enumerable.Range(0, dimensions.Aggregate(1, (total, size) => total * size)).Select(value).ToArray()));
            }
        }
        public bool Disposed { get; private set; }
        public string Path => "vision";
        public IReadOnlyCollection<string> Names => tensors.Keys;
        public float[] Values(string name) => tensors[name].Values;
        public bool TryGet(string name, out IModelTensor tensor)
        {
            var found = tensors.TryGetValue(name, out var value);
            tensor = value!;
            return found;
        }
        public IModelTensor GetRequired(string name) => tensors[name];
        public void Dispose() => Disposed = true;
    }

    private sealed class Tensor(VisionFile file, string name, int[] dimensions, bool half, float[] values) : IModelTensor
    {
        private readonly Half[] halves = values.Select(value => (Half)value).ToArray();
        public float[] Values => half ? halves.Select(value => (float)value).ToArray() : values;
        public string Name => name;
        public TensorDataType DataType => half ? TensorDataType.Float16 : TensorDataType.Float32;
        public IReadOnlyList<int> Dimensions => dimensions;
        public ReadOnlySpan<float> FloatValues => file.Disposed ? throw new ObjectDisposedException(nameof(VisionFile)) : values;
        public ReadOnlySpan<Half> HalfValues => file.Disposed ? throw new ObjectDisposedException(nameof(VisionFile)) : halves;
    }
}
