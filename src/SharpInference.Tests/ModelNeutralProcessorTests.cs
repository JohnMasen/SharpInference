using System.Runtime.InteropServices;
using SharpInference.Backends.CpuVm;
using SharpInference.Graphs;
using SharpInference.Runtime;
using SharpInference.Runtime.Cpu;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class ModelNeutralProcessorTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RegisteredArrayModelExecutesAfterReaderIsReleased(bool modelOperation)
    {
        var file = new ArrayFile();
        var modules = new ModelGraphModuleRegistry();
        modules.Register(new ArrayModule(modelOperation: modelOperation));
        using var backend = CpuVmBackendFactory.Create(new() { ReuseLocalStorage = false });
        using var processor = Processor.Load("array", new ArrayReader(file), modules, backend);
        Assert.True(file.Disposed);
        Assert.Equal(1, file.WeightReads);
        Assert.Equal(typeof(ModelMetadata), processor.Metadata.GetType());
        Assert.Equal(["samples"], processor.Metadata.Dimensions.Keys);
        Assert.Equal("opaque-cache-v3", processor.LogicalGraph!.Model.StateAbiId);
        Assert.True(processor.Capabilities.Supports(ProcessorInputModality.Tensor));
        Assert.False(processor.Capabilities.Supports(ProcessorInputModality.Text));
        Assert.Null(processor.Capabilities.Text);
        Assert.Throws<InvalidOperationException>(() => backend.Prepare(processor.LogicalGraph));
        Assert.Throws<NotSupportedException>(() => processor.CreateSession());
        using var session = processor.CreateGraphSession();
        using var independent = processor.CreateGraphSession();
        using var generic = Assert.IsAssignableFrom<ITensorProcessorSession>(((IProcessor)processor).CreateSession());
        Assert.Same(processor, generic.Processor);
        var inputs = new Dictionary<ResourceId, ReadOnlyMemory<byte>>
        {
            [new("input")] = MemoryMarshal.AsBytes(new float[] { 1, -2, 3 }.AsSpan()).ToArray(),
        };
        var result = session.Execute(inputs);
        Assert.Equal([11f, 18f, 33f], MemoryMarshal.Cast<byte, float>(result[new("output")]).ToArray());
        session.Dispose();
        Assert.Equal(result[new("output")], independent.Execute(inputs)[new("output")]);
        generic.Reset();
        Assert.Equal(result[new("output")], generic.Execute(new Dictionary<string, ReadOnlyMemory<byte>>
        {
            ["input"] = inputs[new("input")],
        })["output"]);
        Assert.Equal(1, file.WeightReads);
        processor.Dispose();
        Assert.Throws<ObjectDisposedException>(() => session.Execute(new Dictionary<ResourceId, ReadOnlyMemory<byte>>()));
        Assert.Throws<ObjectDisposedException>(() => independent.Reset());
        Assert.Throws<ObjectDisposedException>(() => generic.Execute(new Dictionary<string, ReadOnlyMemory<byte>>()));
    }

    [Fact]
    public void ModuleMetadataMismatchReleasesReaderAndBackend()
    {
        var file = new ArrayFile();
        var modules = new ModelGraphModuleRegistry();
        modules.Register(new ArrayModule(metadataSamples: 4));
        var released = false;
        using var backend = CreateBackend(() => released = true);
        Assert.Throws<InvalidDataException>(() => Processor.Load("array", new ArrayReader(file), modules, backend));
        Assert.True(file.Disposed);
        Assert.True(released);
    }

    private static VmGraphBackend CreateBackend(Action? dispose = null)
    {
        var collections = CpuInstructionCollections.Create();
        return new(VmTarget.Cpu, program =>
        {
            var artifact = new CpuVmCompiler(collections).Compile(program);
            return () => artifact.LoadExecutable();
        }, optimization: new(ReuseLocalStorage: false), disposeCompiler: dispose, generatorCollections: collections);
    }

    private sealed class ArrayModule(int metadataSamples = 3, bool modelOperation = false) : IModelGraphModule
    {
        public string ArchitectureId => "external.array";
        public IGraphOperationLowerer? OperationLowerer => modelOperation ? new ArrayLowerer() : null;
        public bool CanLoad(IModelTensorCatalog tensors) => tensors.Names.SequenceEqual(["offset"]);
        public ModelMetadata ReadMetadata(IModelTensorCatalog tensors) => new(ArchitectureId,
            new Dictionary<string, long> { ["samples"] = metadataSamples });
        public LogicalGraph Build(IModelTensorCatalog tensors) =>
            new LogicalGraphBuilder(new(ArchitectureId, 1, "array.add"),
                    new GraphModelSignature("array", "opaque-cache-v3", new Dictionary<string, int> { ["samples"] = 3 }))
                .AddRegion("root", GraphRegionTypes.Graph, "Root")
                .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [3]), graphInput: true)
                .AddResource("offset", "Offset", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                    new TensorDescriptor(GraphElementType.Float32, [3]), "offset")
                .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [3]), graphOutput: true)
                .AddNode("add", modelOperation ? new GraphOperationId("external.offset", 1) : PrimitiveGraphOperations.Add, "root",
                    [GraphBindings.Read("left", "input"), GraphBindings.Read("right", "offset"),
                     GraphBindings.Write("output", "output")])
                .Build();
    }

    private sealed class ArrayLowerer : IGraphOperationLowerer
    {
        public LogicalGraph Lower(LogicalGraph graph) => new(graph.Identity, graph.Model, graph.Resources, graph.Regions,
            graph.Nodes.Select(node => node.Operation == new GraphOperationId("external.offset", 1)
                ? node with { Operation = PrimitiveGraphOperations.Add }
                : throw new NotSupportedException($"Unsupported array operation '{node.Operation}'.")),
            graph.Inputs, graph.Outputs, graph.GraphState);
    }

    private sealed class ArrayReader(ArrayFile file) : IModelReader
    {
        public IModelFile Open(string path) => file;
    }

    private sealed class ArrayFile : IModelFile
    {
        public bool Disposed { get; private set; }
        public int WeightReads { get; set; }
        public string Path => "array";
        public IReadOnlyCollection<string> Names => ["offset"];
        public bool TryGet(string name, out IModelTensor tensor)
        {
            tensor = new ArrayTensor(this);
            return name == "offset";
        }
        public IModelTensor GetRequired(string name) => TryGet(name, out var tensor) ? tensor : throw new InvalidDataException(name);
        public void Dispose() => Disposed = true;
    }

    private sealed class ArrayTensor(ArrayFile file) : IModelTensor
    {
        public string Name => "offset";
        public TensorDataType DataType => TensorDataType.Float32;
        public IReadOnlyList<int> Dimensions => [3];
        public ReadOnlySpan<float> FloatValues
        {
            get
            {
                ObjectDisposedException.ThrowIf(file.Disposed, file);
                file.WeightReads++;
                return [10f, 20f, 30f];
            }
        }
        public ReadOnlySpan<Half> HalfValues => throw new NotSupportedException();
    }
}
