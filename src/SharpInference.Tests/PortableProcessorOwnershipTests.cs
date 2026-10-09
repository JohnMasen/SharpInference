using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;
using SharpInference.Runtime;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.Tests;

public sealed class PortableProcessorOwnershipTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MetadataFailure_ReleasesReaderAndPreservesCleanupFailure(bool cleanupFailure)
    {
        var metadataError = new InvalidDataException("Invalid model metadata.");
        var cleanupError = cleanupFailure ? new IOException("Reader cleanup failed.") : null;
        var reader = new TrackedModelFile(cleanupError);
        var builder = new ProcessorPipelineBuilder("virtual-model")
            .UseReader(new TrackedReader(reader), new FailingMetadataReader(metadataError));
        if (cleanupFailure)
        {
            var error = Assert.Throws<AggregateException>(() => builder.Build());
            Assert.Equal(2, error.InnerExceptions.Count);
            Assert.Same(metadataError, error.InnerExceptions[0]);
            Assert.Same(cleanupError, error.InnerExceptions[1]);
        }
        else
        {
            Assert.Same(metadataError, Assert.Throws<InvalidDataException>(() => builder.Build()));
        }
        Assert.True(reader.Disposed);
        Assert.Equal(1, reader.DisposeCalls);
    }

    [Fact]
    public void ProviderOverload_DoesNotRequireAnArchitectureDetector()
    {
        var path = TestModelLoader.GetPath(TestModel.Rwkv7Fp32);
        LogicalGraph graph;
        float[] firstRow;
        using (var catalog = GgmlModelFile.Open(path))
        {
            var metadata = RwkvModelMetadata.FromModelMetadata(new Rwkv7ModelModule().ReadMetadata(catalog));
            var weight = catalog.GetRequired("head.weight");
            firstRow = weight.FloatValues[..metadata.VocabularySize].ToArray();
            graph = new LogicalGraphBuilder(
                    new GraphIdentity("experimental-graph", 1, "test.forward"),
                    TestGraphSignatures.Create(metadata.VocabularySize, metadata.EmbeddingSize,
                        metadata.LayerCount, metadata.HeadCount, metadata.HeadSize,
                        "experimental-graph.state.fp32@1"))
                .AddRegion("graph", GraphRegionTypes.Graph, "Graph")
                .AddResource("token", "Token", GraphResourceKind.Input,
                    GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Int32, [1]),
                    graphInput: true)
                .AddResource("logits", "Logits", GraphResourceKind.Output,
                    GraphResourceLifetime.External,
                    new TensorDescriptor(GraphElementType.Float32, [metadata.VocabularySize]),
                    graphOutput: true)
                .AddResource("weight", "Weight", GraphResourceKind.Weight,
                    GraphResourceLifetime.Model,
                    new TensorDescriptor(GraphElementType.Float32, weight.Dimensions), "head.weight")
                .AddNode("gather", PrimitiveGraphOperations.GatherRow, "graph",
                    [GraphBindings.Read("table", "weight"), GraphBindings.Read("index", "token"),
                     GraphBindings.Write("output", "logits")])
                .Build();
        }

        using var processor = Processor.LoadGraph(path, new SuppliedLogicalGraphProvider(graph),
            SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create());
        Assert.Equal("experimental-graph", processor.Metadata.ArchitectureId);
        using var session = processor.CreateSession();
        Assert.Equal(firstRow, session.ForwardToken(0).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ProviderOverload_LoadsRealModelWithoutKeepingReader(bool rwkv7)
    {
        var path = TestModelLoader.GetPath(rwkv7 ? TestModel.Rwkv7Fp16 : TestModel.Rwkv6);
        ILogicalGraphProvider provider = rwkv7
            ? new PortableRwkv7GraphProvider()
            : new PortableRwkv6GraphProvider();
        using var processor = Processor.LoadGraph(path, provider, SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create());
        using var session = processor.CreateSession();
        var logits = session.ForwardToken(0).ToArray();
        Assert.Equal(processor.Metadata.Dimensions["vocabulary"], logits.Length);
        Assert.All(logits, value => Assert.True(float.IsFinite(value)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Processor_ReleasesReaderBeforeFirstInference(bool gpu)
    {
        using var device = gpu ? CreateDevice() : null;
        if (gpu && device is null) return;
        var graph = new LogicalGraphBuilder(
                new GraphIdentity("test-graph", 1, "test.forward"),
                TestGraphSignatures.Create(2, 2, 1, 1, 2, "test-graph.state.fp32@1"))
            .AddRegion("graph", GraphRegionTypes.Graph, "Graph")
            .AddResource("token", "Token", GraphResourceKind.Input,
                GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Int32, [1]),
                graphInput: true)
            .AddResource("logits", "Logits", GraphResourceKind.Output,
                GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Float32, [2]),
                graphOutput: true)
            .AddResource("weight", "Weight", GraphResourceKind.Weight,
                GraphResourceLifetime.Model, new TensorDescriptor(GraphElementType.Float32, [2, 2]),
                "table")
            .AddNode("gather", PrimitiveGraphOperations.GatherRow, "graph",
                [GraphBindings.Read("table", "weight"), GraphBindings.Read("index", "token"),
                 GraphBindings.Write("output", "logits")])
            .Build();
        var reader = new TrackedModelFile();
        var backend = gpu ? SharpInference.Runtime.D3D12.D3D12VmBackendFactory.Create() : SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create();
        using var processor = new ProcessorPipelineBuilder("virtual-model")
            .UseReader(new TrackedReader(reader), new GraphArchitectureMetadataReader(graph))
            .UseProvider(new SuppliedLogicalGraphProvider(graph))
            .UseBackend(backend)
            .UsePortableGraphArchitecture()
            .Build();
        Assert.True(reader.Disposed);
        Assert.Equal(1, reader.DisposeCalls);
        using var session = processor.CreateSession();
        Assert.Equal([1f, 2f], session.ForwardToken(0).ToArray());
        Assert.Equal([3f, 4f], session.ForwardToken(1).ToArray());
    }

    private static ID3D12Device? CreateDevice()
    {
        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory6>();
        for (uint index = 0; factory.EnumAdapterByGpuPreference(
                 index, GpuPreference.HighPerformance, out IDXGIAdapter1? adapter).Success; index++)
        {
            using (adapter)
            {
                if ((adapter!.Description1.Flags & AdapterFlags.Software) == 0)
                    return D3D12CreateDevice<ID3D12Device>(adapter, FeatureLevel.Level_11_0);
            }
        }
        return null;
    }

    private sealed class TrackedReader(TrackedModelFile file) : IModelReader
    {
        public IModelFile Open(string path) => file;
    }

    private sealed class FailingMetadataReader(Exception error) : IArchitectureMetadataReader
    {
        public ModelMetadata Read(IModelTensorCatalog catalog) => throw error;
    }

    private sealed class TrackedModelFile(Exception? cleanupError = null) : IModelFile
    {
        public bool Disposed { get; private set; }
        public int DisposeCalls { get; private set; }
        public string Path => "virtual-model";
        public int VocabularySize => 2;
        public int EmbeddingSize => 2;
        public int LayerCount => 1;
        public IReadOnlyCollection<string> Names => ["table"];
        public bool TryGet(string name, out IModelTensor tensor)
        {
            if (name == "table")
            {
                tensor = new TrackedTensor(this);
                return true;
            }
            tensor = null!;
            return false;
        }
        public IModelTensor GetRequired(string name) =>
            TryGet(name, out var tensor) ? tensor : throw new InvalidDataException(name);
        public void Dispose()
        {
            Disposed = true;
            DisposeCalls++;
            if (cleanupError is not null) throw cleanupError;
        }
    }

    private sealed class TrackedTensor(TrackedModelFile owner) : IModelTensor
    {
        public string Name => "table";
        public TensorDataType DataType => TensorDataType.Float32;
        public IReadOnlyList<int> Dimensions => [2, 2];
        public ReadOnlySpan<float> FloatValues =>
            owner.Disposed ? throw new ObjectDisposedException(nameof(TrackedModelFile)) : [1f, 2f, 3f, 4f];
        public ReadOnlySpan<Half> HalfValues => [];
    }
}
