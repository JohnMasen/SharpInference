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
public sealed class Phi4AudioProjectionGraphTests
{
    [Theory]
    [InlineData(false, Phi4AudioProjector.Speech)]
    [InlineData(true, Phi4AudioProjector.Speech)]
    [InlineData(false, Phi4AudioProjector.Vision)]
    [InlineData(true, Phi4AudioProjector.Vision)]
    public void GeneratedGraphUsesExplicitOutputResources(bool half, Phi4AudioProjector projector)
    {
        var module = new Phi4AudioProjectionGraphModule(projector, 2);
        using var file = new ProjectionFile(half);
        ModelGraphNamingAssertions.Validate(module.Build(file), module);
    }

    [Theory]
    [InlineData(Phi4AudioProjector.Speech, false, false)]
    [InlineData(Phi4AudioProjector.Speech, true, false)]
    [InlineData(Phi4AudioProjector.Vision, false, false)]
    [InlineData(Phi4AudioProjector.Vision, true, false)]
    [InlineData(Phi4AudioProjector.Speech, false, true)]
    [InlineData(Phi4AudioProjector.Speech, true, true)]
    [InlineData(Phi4AudioProjector.Vision, false, true)]
    [InlineData(Phi4AudioProjector.Vision, true, true)]
    public void RegisteredProjectionMatchesDirectNumericsAfterSourceCloses(Phi4AudioProjector projector, bool half, bool gpu)
    {
        var file = new ProjectionFile(half);
        var input = new[] { -0.5f, 0.25f, 1f, 0.75f, -0.25f, 0.5f };
        var expected = Reference(file, input, projector);
        var module = new Phi4AudioProjectionGraphModule(projector, 2);
        var registry = new ModelGraphModuleRegistry();
        registry.Register(module);
        IInstructionCollectionProvider[] collections = gpu
            ? [.. D3D12InstructionCollections.Create(), new Phi4D3D12AudioInstructionCollection()]
            : [.. CpuInstructionCollections.Create(), new Phi4CpuAudioInstructionCollection()];
        using var backend = gpu
            ? D3D12VmBackendFactory.Create(instructionCollections: collections)
            : CpuVmBackendFactory.Create(instructionCollections: collections);
        using var processor = Processor.Load("projection", new Reader(file), registry, backend);
        Assert.True(file.Disposed);
        var graph = processor.LogicalGraph!;
        ModelGraphNamingAssertions.Validate(graph, module);
        Assert.Equal(module.ArchitectureId, graph.Model.ModelType);
        Assert.Single(graph.Nodes, node => node.Operation == PrimitiveGraphOperations.MatrixMultiply);
        Assert.Single(graph.Nodes, node => node.Operation == PrimitiveGraphOperations.Affine);
        Assert.Single(graph.Nodes, node => node.Operation == Phi4AudioGraphOperations.BiasActivation);
        var roundTrip = GraphXml.DeserializeLogical(GraphXml.Serialize(graph));
        GraphValidator.Validate(roundTrip, module);
        Assert.Equal(graph.Nodes.Count, roundTrip.Nodes.Count);
        using var first = Assert.IsAssignableFrom<ITensorProcessorSession>(((IProcessor)processor).CreateSession());
        using var second = Assert.IsAssignableFrom<ITensorProcessorSession>(((IProcessor)processor).CreateSession());
        Check(first);
        first.Dispose();
        Check(second);

        void Check(ITensorProcessorSession session)
        {
            var output = session.Execute(new Dictionary<string, ReadOnlyMemory<byte>>
            {
                ["audio_hidden"] = MemoryMarshal.AsBytes(input.AsSpan()).ToArray(),
            })["output"];
            var actual = MemoryMarshal.Cast<byte, float>(output).ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (var index = 0; index < expected.Length; index++)
            {
                Assert.True(float.IsFinite(actual[index]));
                Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0,
                    1e-5f + 1e-5f * MathF.Abs(expected[index]));
            }
            Assert.Contains(actual, value => value != 0);
        }
    }

    [Fact]
    public void MissingExplicitProjectionInstructionsFailsBeforeExecution()
    {
        var graph = new Phi4AudioProjectionGraphModule(Phi4AudioProjector.Speech, 2).Build(new ProjectionFile(false));
        using var backend = CpuVmBackendFactory.Create();
        Assert.Throws<BackendPreparationException>(() => backend.Prepare(graph));
    }

    [Theory]
    [InlineData("count", "0")]
    [InlineData("count", "7")]
    [InlineData("width", "0")]
    [InlineData("activation", "relu")]
    public void ModelOwnedValidatorRejectsInvalidActivationAttributes(string attribute, string value)
    {
        var module = new Phi4AudioProjectionGraphModule(Phi4AudioProjector.Speech, 2);
        var graph = module.Build(new ProjectionFile(false));
        var node = graph.Nodes.Single(node => node.Operation == Phi4AudioGraphOperations.BiasActivation);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var attributes = node.Attributes.ToDictionary(pair => pair.Key, pair => pair.Value);
        attributes[attribute] = value;
        Assert.Throws<InvalidDataException>(() => module.Validate(new(node.Id.Value, node.Operation,
            node.Resources, resources, attributes, node.Requirements)));
        Assert.Throws<NotSupportedException>(() => module.Validate(new(node.Id.Value,
            node.Operation with { Version = 99 }, node.Resources, resources, node.Attributes, node.Requirements)));
    }

    [Fact]
    public void ProjectionSelectionAndMatrixDimensionsAreValidated()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Phi4AudioProjectionGraphModule((Phi4AudioProjector)99, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Phi4AudioProjectionGraphModule(Phi4AudioProjector.Speech, 0));
        var module = new Phi4AudioProjectionGraphModule(Phi4AudioProjector.Speech, 2);
        Assert.Throws<InvalidDataException>(() => module.ReadMetadata(new ProjectionFile(false, invalidShape: true)));
    }

    private static float[] Reference(ProjectionFile file, float[] input, Phi4AudioProjector projector)
    {
        var prefix = projector == Phi4AudioProjector.Speech ? "mm.a.mlp" : "mm.a.vis";
        var first = Linear(input, file.Values(prefix + ".0.weight"), 3, 4);
        var hidden = new float[first.Length];
        Phi4CpuAudioNumerics.BiasActivation(first, file.Values(prefix + ".0.bias"), hidden, first.Length, 4, "gelu");
        var output = Linear(hidden, file.Values(prefix + ".2.weight"), 4, 5);
        var bias = file.Values(prefix + ".2.bias");
        for (var index = 0; index < output.Length; index++) output[index] += bias[index % 5];
        return output;

        static float[] Linear(float[] input, float[] weight, int inputWidth, int outputWidth)
        {
            var output = new float[input.Length / inputWidth * outputWidth];
            for (var row = 0; row < input.Length / inputWidth; row++)
            for (var column = 0; column < outputWidth; column++)
            for (var inner = 0; inner < inputWidth; inner++)
                output[row * outputWidth + column] += input[row * inputWidth + inner] * weight[column * inputWidth + inner];
            return output;
        }
    }

    private sealed class Reader(ProjectionFile file) : IModelReader
    {
        public IModelFile Open(string path) => file;
    }

    private sealed class ProjectionFile : IModelFile
    {
        private readonly Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        public ProjectionFile(bool half, bool invalidShape = false)
        {
            foreach (var prefix in new[] { "mm.a.mlp", "mm.a.vis" })
            {
                var scale = prefix == "mm.a.mlp" ? 1f : -2f;
                Add(prefix + ".0.weight", [4, 3], index => scale * (index % 5 - 2) / 8f);
                Add(prefix + ".0.bias", [4], index => scale * (index + 1) / 16f);
                Add(prefix + ".2.weight", [5, invalidShape ? 3 : 4], index => scale * (index % 7 - 3) / 16f);
                Add(prefix + ".2.bias", [5], index => scale * index / 32f);
            }
            void Add(string name, int[] dimensions, Func<int, float> value) =>
                tensors.Add(name, new(this, name, dimensions, half,
                    Enumerable.Range(0, dimensions.Aggregate(1, (total, size) => total * size)).Select(value).ToArray()));
        }
        public bool Disposed { get; private set; }
        public string Path => "projection";
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

    private sealed class Tensor(ProjectionFile file, string name, int[] dimensions, bool half, float[] values) : IModelTensor
    {
        private readonly Half[] halves = values.Select(value => (Half)value).ToArray();
        public float[] Values => half ? halves.Select(value => (float)value).ToArray() : values;
        public string Name => name;
        public TensorDataType DataType => half ? TensorDataType.Float16 : TensorDataType.Float32;
        public IReadOnlyList<int> Dimensions => dimensions;
        public ReadOnlySpan<float> FloatValues => file.Disposed ? throw new ObjectDisposedException(nameof(ProjectionFile)) : values;
        public ReadOnlySpan<Half> HalfValues => file.Disposed ? throw new ObjectDisposedException(nameof(ProjectionFile)) : halves;
    }
}
