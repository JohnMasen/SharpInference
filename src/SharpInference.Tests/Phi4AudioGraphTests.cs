using System.Runtime.InteropServices;
using SharpInference.Architectures.Phi4;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Phi4;
using SharpInference.Instructions.Phi4.Cpu;
using SharpInference.Instructions.Phi4.D3D12;
using SharpInference.Runtime;
using SharpInference.Runtime.Cpu;
using SharpInference.Runtime.D3D12;

namespace SharpInference.Tests;

[Collection(VmGraphGpuCollection.Name)]
public sealed class Phi4AudioGraphTests
{
    [Theory]
    [InlineData(Phi4AudioProjector.Speech, false, false, 2)]
    [InlineData(Phi4AudioProjector.Speech, true, false, 2)]
    [InlineData(Phi4AudioProjector.Vision, false, false, 2)]
    [InlineData(Phi4AudioProjector.Vision, true, false, 2)]
    [InlineData(Phi4AudioProjector.Speech, false, true, 2)]
    [InlineData(Phi4AudioProjector.Speech, true, true, 2)]
    [InlineData(Phi4AudioProjector.Vision, false, true, 2)]
    [InlineData(Phi4AudioProjector.Vision, true, true, 2)]
    [InlineData(Phi4AudioProjector.Speech, false, false, 24)]
    public void CompleteAudioGraphMatchesDirectExecution(Phi4AudioProjector projector, bool half, bool gpu, int layers)
    {
        var file = new AudioFile(half, layers);
        var input = Enumerable.Range(0, 24 * 8).Select(index => (index % 17 - 7) / 16f).ToArray();
        var expected = ReferenceValues(file, input, 13, layers, projector);
        var module = new Phi4AudioGraphModule(projector, 24, layers);
        var registry = new ModelGraphModuleRegistry();
        registry.Register(module);
        IInstructionCollectionProvider[] collections = gpu
            ? [.. D3D12InstructionCollections.Create(), new Phi4D3D12AudioInstructionCollection()]
            : [.. CpuInstructionCollections.Create(), new Phi4CpuAudioInstructionCollection()];
        using var backend = gpu
            ? D3D12VmBackendFactory.Create(instructionCollections: collections)
            : CpuVmBackendFactory.Create(instructionCollections: collections);
        using var processor = Processor.Load("audio", new Reader(file), registry, backend);
        Assert.True(file.Disposed);
        var graph = processor.LogicalGraph!;
        Assert.Equal(module.ArchitectureId, graph.Model.ModelType);
        Assert.Equal(layers, graph.Nodes.Count(node => node.Operation == Phi4AudioGraphOperations.RelativeAttention));
        Assert.Equal(layers * 3, graph.Nodes.Count(node => node.Operation == Phi4AudioGraphOperations.Conv1D));
        Assert.Equal(5, graph.Nodes.Count(node => node.Operation == Phi4AudioGraphOperations.Conv2D));
        Assert.Equal(new[] { "audio_features", "frame_count" }, graph.Inputs.Select(id => id.Value));
        Assert.Equal("output", Assert.Single(graph.Outputs).Value);
        GraphValidator.Validate(GraphXml.DeserializeLogical(GraphXml.Serialize(graph)), module);
        using var session = Assert.IsAssignableFrom<ITensorProcessorSession>(((IProcessor)processor).CreateSession());
        Check(13, expected);
        Check(24, ReferenceValues(file, input, 24, layers, projector));
        session.Reset();
        Check(13, expected);

        void Check(int frameCount, float[] reference)
        {
            var output = session.Execute(new Dictionary<string, ReadOnlyMemory<byte>>
            {
                ["audio_features"] = MemoryMarshal.AsBytes(input.AsSpan()).ToArray(),
                ["frame_count"] = BitConverter.GetBytes(frameCount),
            })["output"];
            var actual = MemoryMarshal.Cast<byte, float>(output).ToArray();
            Assert.Equal(reference.Length, actual.Length);
            for (var index = 0; index < actual.Length; index++)
            {
                Assert.True(float.IsFinite(actual[index]));
                Assert.InRange(MathF.Abs(reference[index] - actual[index]), 0, 1e-4f + 1e-4f * MathF.Abs(reference[index]));
            }
            Assert.Contains(actual, value => value != 0);
        }
    }

    [Theory]
    [InlineData("phi4.audio.layer-norm", "epsilon", "NaN")]
    [InlineData("phi4.audio.layer-norm", "epsilon", "0")]
    [InlineData("phi4.audio.swi-glu", "width", "0")]
    [InlineData("phi4.audio.residual", "scale", "Infinity")]
    [InlineData("phi4.audio.relative-attention", "heads", "3")]
    [InlineData("phi4.audio.relative-attention", "subsampling", "0")]
    [InlineData("phi4.audio.conv1d", "groups", "0")]
    [InlineData("phi4.audio.conv1d", "padding", "-1")]
    public void ConformerContractsRejectMalformedAttributesAndVersions(string operation, string attribute, string value)
    {
        var module = new Phi4AudioGraphModule(frames: 24, layers: 2);
        var graph = module.Build(new AudioFile(false, 2));
        var node = graph.Nodes.First(node => node.Operation.Name == operation);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var attributes = node.Attributes.ToDictionary(pair => pair.Key, pair => pair.Value);
        attributes[attribute] = value;
        Assert.Throws<InvalidDataException>(() => module.Validate(new(node.Id.Value, node.Operation,
            node.Resources, resources, attributes, node.Requirements)));
        Assert.Throws<NotSupportedException>(() => module.Validate(new(node.Id.Value, node.Operation with { Version = 99 },
            node.Resources, resources, node.Attributes, node.Requirements)));
    }

    [Fact]
    public void CompleteAudioGraphRequiresExplicitInstructionRegistration()
    {
        var module = new Phi4AudioGraphModule(frames: 24, layers: 2);
        var graph = module.Build(new AudioFile(false, 2));
        using var backend = CpuVmBackendFactory.Create();
        Assert.Throws<BackendPreparationException>(() => backend.Prepare(graph));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Phi4AudioGraphModule(layers: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Phi4AudioGraphModule(layers: 25));
    }

    private static float[] ReferenceValues(AudioFile file, float[] input, int frames, int layers, Phi4AudioProjector projector)
    {
        const int tokens = 3;
        const int width = 4;
        const int intermediate = 3;
        var values = new float[input.Length];
        Phi4CpuAudioNumerics.NormalizeFeatures(input, file.Values("a.global_mean"), file.Values("a.global_invstd"), frames, values, 24, 8);
        var height = 24;
        var frequency = 8;
        var channels = 1;
        var subsampleNames = new[] { "a.conv1d.0", "a.conv1d.2", "a.conv1d.3", "a.conv1d.5", "a.conv1d.6" };
        for (var index = 0; index < subsampleNames.Length; index++)
        {
            var kernel = index is 2 or 4 ? 1 : 3;
            var stride = kernel == 3 ? 2 : 1;
            var nextHeight = stride == 2 ? (height + 1) / 2 : height;
            var nextFrequency = stride == 2 ? (frequency + 1) / 2 : frequency;
            var output = new float[nextHeight * nextFrequency * width];
            Phi4CpuAudioNumerics.Conv2D(values, file.Values(subsampleNames[index] + ".weight"), file.Values(subsampleNames[index] + ".bias"),
                output, height, frequency, channels, nextHeight, nextFrequency, width, kernel, kernel,
                kernel == 3 ? 1 : 0, stride, index is 1 or 3 ? width : 1, index is 1 or 3 ? "none" : "relu");
            values = output;
            height = nextHeight;
            frequency = nextFrequency;
            channels = width;
        }
        var flattened = new float[values.Length];
        Phi4CpuAudioNumerics.FlattenSubsampling(values, flattened, tokens, frequency, width);
        var hidden = Linear(flattened, "a.conv1d.out", width * frequency, width);
        for (var layer = 0; layer < layers; layer++)
        {
            var prefix = $"a.blk.{layer}.";
            hidden = Residual(hidden, FeedForward(hidden, prefix + "ffn_in"), 0.5f);
            var normalized = Norm(hidden, prefix + "ln_att");
            var attention = new float[tokens * width];
            Phi4CpuAudioNumerics.RelativeAttention(Linear(normalized, prefix + "attn_q", width, width),
                Linear(normalized, prefix + "attn_k", width, width), Linear(normalized, prefix + "attn_v", width, width),
                file.Values("a.rel_attn_bias"), frames, attention, tokens, width, 2, 8);
            hidden = Residual(hidden, Linear(attention, prefix + "attn_out", width, width), 1f);
            normalized = Norm(hidden, prefix + "conv.ln");
            var wide = Linear(normalized, prefix + "conv.glu.pw", width, width * 2);
            var gated = new float[tokens * width];
            Phi4CpuAudioNumerics.SwiGlu(wide, file.Values(prefix + "conv.glu.b1"), file.Values(prefix + "conv.glu.b2"), gated, tokens, width);
            var depthwise = Conv(gated, prefix + "conv.dw", 3, 2, width);
            var middle = Conv(depthwise, prefix + "conv.pw_mid", 1, 0, 1);
            var activation = new float[middle.Length];
            Phi4CpuAudioNumerics.BiasActivation(middle, new float[width], activation, middle.Length, width, "swish");
            hidden = Residual(hidden, Conv(activation, prefix + "conv.pw_ext", 1, 0, 1), 1f);
            hidden = Residual(hidden, FeedForward(hidden, prefix + "ffn_out"), 0.5f);
            hidden = Norm(hidden, prefix + "ln");
        }
        var projectorPrefix = projector == Phi4AudioProjector.Speech ? "mm.a.mlp" : "mm.a.vis";
        var first = Linear(hidden, projectorPrefix + ".0", width, 6);
        var activated = new float[first.Length];
        Phi4CpuAudioNumerics.BiasActivation(first, new float[6], activated, first.Length, 6, "gelu");
        return Linear(activated, projectorPrefix + ".2", 6, 5);

        float[] Linear(float[] source, string prefix, int inputWidth, int outputWidth)
        {
            var weight = file.Values(prefix + ".weight");
            var bias = file.Values(prefix + ".bias");
            var output = new float[tokens * outputWidth];
            for (var row = 0; row < tokens; row++)
            for (var column = 0; column < outputWidth; column++)
            {
                var sum = 0f;
                for (var inner = 0; inner < inputWidth; inner++) sum += source[row * inputWidth + inner] * weight[column * inputWidth + inner];
                output[row * outputWidth + column] = sum + bias[column];
            }
            return output;
        }
        float[] Norm(float[] source, string prefix)
        {
            var output = new float[tokens * width];
            Phi4CpuAudioNumerics.LayerNorm(source, file.Values(prefix + ".weight"), file.Values(prefix + ".bias"), output, tokens, width, 1e-5f);
            return output;
        }
        float[] Residual(float[] source, float[] update, float scale)
        {
            var output = new float[tokens * width];
            Phi4CpuAudioNumerics.Residual(source, update, output, output.Length, scale);
            return output;
        }
        float[] FeedForward(float[] source, string prefix)
        {
            var wide = Linear(Norm(source, prefix + ".ln"), prefix + ".up", width, intermediate * 2);
            var gated = new float[tokens * intermediate];
            Phi4CpuAudioNumerics.SwiGlu(wide, new float[intermediate], new float[intermediate], gated, tokens, intermediate);
            return Linear(gated, prefix + ".down", intermediate, width);
        }
        float[] Conv(float[] source, string prefix, int kernel, int padding, int groups)
        {
            var length = tokens + 2 * padding - kernel + 1;
            var output = new float[length * width];
            Phi4CpuAudioNumerics.Conv1D(source, file.Values(prefix + ".weight"), file.Values(prefix + ".bias"), output,
                tokens, width, length, width, kernel, padding, 1, groups, "none");
            return output;
        }
    }

    private sealed class Reader(AudioFile file) : IModelReader
    {
        public IModelFile Open(string path) => file;
    }

    internal static IModelFile CreateSyntheticFile(bool half, int layers, int embeddingWidth) => new AudioFile(half, layers, embeddingWidth);

    private sealed class AudioFile : IModelFile
    {
        private readonly Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        public AudioFile(bool half, int layers, int embeddingWidth = 5)
        {
            Add("a.global_mean", [8], index => index / 8f);
            Add("a.global_invstd", [8], index => 0.5f + index / 16f);
            var names = new[] { "a.conv1d.0", "a.conv1d.2", "a.conv1d.3", "a.conv1d.5", "a.conv1d.6" };
            for (var index = 0; index < names.Length; index++)
            {
                var kernel = index is 2 or 4 ? 1 : 3;
                Add(names[index] + ".weight", [4, index == 0 || index is 1 or 3 ? 1 : 4, kernel, kernel]);
                Add(names[index] + ".bias", [4]);
            }
            Linear("a.conv1d.out", 4, 4);
            Add("a.rel_attn_bias", [1000, 2]);
            for (var layer = 0; layer < layers; layer++)
            {
                var prefix = $"a.blk.{layer}.";
                foreach (var part in new[] { "ffn_in", "ffn_out" })
                {
                    Norm(prefix + part + ".ln");
                    Linear(prefix + part + ".up", 4, 6);
                    Linear(prefix + part + ".down", 3, 4);
                }
                foreach (var part in new[] { "ln_att", "conv.ln", "ln" }) Norm(prefix + part);
                foreach (var part in new[] { "attn_q", "attn_k", "attn_v", "attn_out" }) Linear(prefix + part, 4, 4);
                Linear(prefix + "conv.glu.pw", 4, 8);
                Add(prefix + "conv.glu.b1", [4]);
                Add(prefix + "conv.glu.b2", [4]);
                Add(prefix + "conv.dw.weight", [4, 1, 3]);
                Add(prefix + "conv.dw.bias", [4]);
                foreach (var part in new[] { "conv.pw_mid", "conv.pw_ext" })
                {
                    Add(prefix + part + ".weight", [4, 4, 1]);
                    Add(prefix + part + ".bias", [4]);
                }
            }
            foreach (var prefix in new[] { "mm.a.mlp", "mm.a.vis" })
            {
                var scale = prefix == "mm.a.mlp" ? 1f : -2f;
                Linear(prefix + ".0", 4, 6, scale);
                Linear(prefix + ".2", 6, embeddingWidth, scale);
            }
            void Norm(string prefix)
            {
                Add(prefix + ".weight", [4], index => 1f + index / 16f);
                Add(prefix + ".bias", [4]);
            }
            void Linear(string prefix, int input, int output, float scale = 1f)
            {
                Add(prefix + ".weight", [output, input], index => scale * (index % 7 - 3) / 32f);
                Add(prefix + ".bias", [output], index => scale * (index + 1) / 64f);
            }
            void Add(string name, int[] dimensions, Func<int, float>? value = null)
            {
                value ??= index => (index % 7 - 3) / 32f;
                tensors.Add(name, new(this, name, dimensions, half,
                    Enumerable.Range(0, dimensions.Aggregate(1, (total, size) => total * size)).Select(value).ToArray()));
            }
        }
        public bool Disposed { get; private set; }
        public string Path => "audio";
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

    private sealed class Tensor(AudioFile file, string name, int[] dimensions, bool half, float[] values) : IModelTensor
    {
        private readonly Half[] halves = values.Select(value => (Half)value).ToArray();
        public float[] Values => half ? halves.Select(value => (float)value).ToArray() : values;
        public string Name => name;
        public TensorDataType DataType => half ? TensorDataType.Float16 : TensorDataType.Float32;
        public IReadOnlyList<int> Dimensions => dimensions;
        public ReadOnlySpan<float> FloatValues => file.Disposed ? throw new ObjectDisposedException(nameof(AudioFile)) : values;
        public ReadOnlySpan<Half> HalfValues => file.Disposed ? throw new ObjectDisposedException(nameof(AudioFile)) : halves;
    }
}
