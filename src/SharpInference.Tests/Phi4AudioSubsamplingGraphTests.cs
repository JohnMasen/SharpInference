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
public sealed class Phi4AudioSubsamplingGraphTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void RegisteredAudioGraphMatchesDirectNumericsAfterSourceCloses(bool half, bool gpu)
    {
        var file = new AudioFile(half);
        var input = Enumerable.Range(0, 16 * 16).Select(index => (index % 23 - 11) / 8f).ToArray();
        var expected = Reference(file, input, 13);
        var module = new Phi4AudioSubsamplingGraphModule(16);
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
        Assert.Equal(5, processor.LogicalGraph!.Nodes.Count(node => node.Operation == Phi4AudioGraphOperations.Conv2D));
        Assert.Contains(processor.LogicalGraph.Nodes, node => node.Resources.Any(binding => binding.View is not null));
        using var session = Assert.IsAssignableFrom<ITensorProcessorSession>(((IProcessor)processor).CreateSession());
        var output = session.Execute(new Dictionary<string, ReadOnlyMemory<byte>>
        {
            ["audio_features"] = MemoryMarshal.AsBytes(input.AsSpan()).ToArray(),
            ["frame_count"] = BitConverter.GetBytes(13),
        })["output"];
        var actual = MemoryMarshal.Cast<byte, float>(output).ToArray();
        if (!gpu) Assert.Equal(expected, actual);
        else
        {
            Assert.Equal(expected.Length, actual.Length);
            for (var index = 0; index < expected.Length; index++)
            {
                Assert.True(float.IsFinite(actual[index]));
                Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0,
                    1e-5f + 1e-5f * MathF.Abs(expected[index]));
            }
        }
        Assert.Contains(expected, value => value != 0);
    }

    [Fact]
    public void ModelOwnedValidatorRejectsInvalidConvolutionAttributesAndVersion()
    {
        var module = new Phi4AudioSubsamplingGraphModule(16);
        var graph = module.Build(new AudioFile(false));
        var node = graph.Nodes.First(node => node.Operation == Phi4AudioGraphOperations.Conv2D);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var invalid = node.Attributes.ToDictionary(value => value.Key, value => value.Value);
        invalid["groups"] = "0";
        Assert.Throws<InvalidDataException>(() => module.Validate(new(node.Id.Value, node.Operation,
            node.Resources, resources, invalid, node.Requirements)));
        Assert.Throws<NotSupportedException>(() => module.Validate(new(node.Id.Value,
            node.Operation with { Version = 99 }, node.Resources, resources, node.Attributes, node.Requirements)));
    }

    [Fact]
    public void MissingExplicitAudioInstructionRegistrationFailsBeforeExecution()
    {
        var graph = new Phi4AudioSubsamplingGraphModule(16).Build(new AudioFile(false));
        using var backend = CpuVmBackendFactory.Create();
        Assert.Throws<BackendPreparationException>(() => backend.Prepare(graph));
    }

    private static float[] Reference(AudioFile file, float[] input, int validFrames)
    {
        var values = new float[input.Length];
        Phi4CpuAudioNumerics.NormalizeFeatures(input, file.Values("a.global_mean"), file.Values("a.global_invstd"),
            validFrames, values, 16, 16);
        var height = 16;
        var width = 16;
        var channels = 1;
        var names = new[] { "a.conv1d.0", "a.conv1d.2", "a.conv1d.3", "a.conv1d.5", "a.conv1d.6" };
        for (var index = 0; index < names.Length; index++)
        {
            var kernel = index is 2 or 4 ? 1 : 3;
            var stride = kernel == 3 ? 2 : 1;
            var outputHeight = stride == 2 ? (height + 1) / 2 : height;
            var outputWidth = stride == 2 ? (width + 1) / 2 : width;
            var output = new float[outputHeight * outputWidth * 2];
            Phi4CpuAudioNumerics.Conv2D(values, file.Values(names[index] + ".weight"), file.Values(names[index] + ".bias"),
                output, height, width, channels, outputHeight, outputWidth, 2, kernel, kernel,
                kernel == 3 ? 1 : 0, stride, index is 1 or 3 ? 2 : 1, index is 1 or 3 ? "none" : "relu");
            values = output;
            height = outputHeight;
            width = outputWidth;
            channels = 2;
        }
        var flattened = new float[values.Length];
        Phi4CpuAudioNumerics.FlattenSubsampling(values, flattened, height, width, channels);
        return flattened;
    }

    private sealed class Reader(AudioFile file) : IModelReader
    {
        public IModelFile Open(string path) => file;
    }

    private sealed class AudioFile : IModelFile
    {
        private readonly Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        public AudioFile(bool half)
        {
            Add("a.global_mean", [16], index => index / 8f);
            Add("a.global_invstd", [16], index => (index % 4 + 1) / 4f);
            var names = new[] { "a.conv1d.0", "a.conv1d.2", "a.conv1d.3", "a.conv1d.5", "a.conv1d.6" };
            for (var index = 0; index < names.Length; index++)
            {
                var kernel = index is 2 or 4 ? 1 : 3;
                var inputChannels = index == 0 || index is 1 or 3 ? 1 : 2;
                Add(names[index] + ".weight", [2, inputChannels, kernel, kernel], value => (value % 7 - 3) / 16f);
                Add(names[index] + ".bias", [2], value => (value + 1) / 32f);
            }
            void Add(string name, int[] dimensions, Func<int, float> value) =>
                tensors.Add(name, new(this, name, dimensions, half,
                    Enumerable.Range(0, dimensions.Aggregate(1, (total, size) => total * size)).Select(value).ToArray()));
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
        public float[] Values => values;
        public string Name => name;
        public TensorDataType DataType => half ? TensorDataType.Float16 : TensorDataType.Float32;
        public IReadOnlyList<int> Dimensions => dimensions;
        public ReadOnlySpan<float> FloatValues => file.Disposed ? throw new ObjectDisposedException(nameof(AudioFile)) : values;
        public ReadOnlySpan<Half> HalfValues => file.Disposed ? throw new ObjectDisposedException(nameof(AudioFile)) : halves;
    }
}
