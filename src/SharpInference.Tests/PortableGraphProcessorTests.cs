using SharpInference.Graphs;
using SharpInference.Runtime;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.Tests;

public sealed class PortableGraphProcessorTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SuppliedPrimitiveGraph_ExecutesWithStateOnCpuAndGpu(bool gpu)
    {
        using var device = gpu ? CreateDevice() : null;
        if (gpu && device is null) return;

        var path = TestModelLoader.GetPath(TestModel.Rwkv7Fp32);
        LogicalGraph graph;
        float[] firstRow;
        using (var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv7Fp32))
        {
            var metadata = new CatalogArchitectureMetadataReader().Read(catalog);
            var weight = catalog.GetRequired("head.weight");
            Assert.Equal(metadata.VocabularySize, weight.Dimensions[1]);
            firstRow = weight.FloatValues[..metadata.VocabularySize].ToArray();
            var vector = new TensorDescriptor(GraphElementType.Float32, [metadata.VocabularySize]);
            graph = new LogicalGraphBuilder(
                    new GraphIdentity(metadata.ArchitectureId, 1, "test.primitive"),
                    new GraphModelSignature(metadata.VocabularySize, metadata.EmbeddingSize,
                        metadata.LayerCount, metadata.HeadCount, metadata.HeadSize,
                        $"{metadata.ArchitectureId}.state.fp32@1"))
                .SetStateSchema(new StateSchema("Primitive_State"))
                .AddRegion("graph", GraphRegionTypes.Graph, "Primitive graph")
                .AddResource("token", "Token", GraphResourceKind.Input,
                    GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Int32, [1]),
                    graphInput: true)
                .AddResource("logits", "Logits", GraphResourceKind.Output,
                    GraphResourceLifetime.External, vector, graphOutput: true)
                .AddResource("weight", "Head weight", GraphResourceKind.Weight,
                    GraphResourceLifetime.Model, new TensorDescriptor(GraphElementType.Float32, weight.Dimensions),
                    "head.weight")
                .AddResource("row", "Selected row", GraphResourceKind.Temporary,
                    GraphResourceLifetime.Invocation, vector)
                .AddResource("state", "Previous row", GraphResourceKind.SessionState,
                    GraphResourceLifetime.Session, vector)
                .AddStateSlot("state", "state")
                .AddNode("gather", PrimitiveGraphOperations.GatherRow, "graph",
                    [GraphBindings.Read("table", "weight"), GraphBindings.Read("index", "token"),
                     GraphBindings.Write("output", "row")])
                .AddNode("sum", PrimitiveGraphOperations.Add, "graph",
                    [GraphBindings.Read("left", "row"), GraphBindings.Read("right", "state"),
                     GraphBindings.Write("output", "logits")], ["gather"])
                .AddNode("store", PrimitiveGraphOperations.Copy, "graph",
                    [GraphBindings.Read("input", "row"), GraphBindings.Write("output", "state")], ["sum"])
                .Build();
        }

        var backend = gpu ? VmBackendFactory.CreateD3D12() : VmBackendFactory.CreateCpu();
        using var model = Processor.LoadGraph(path, graph, backend);
        using var original = model.CreateSession();
        var first = original.ForwardToken(0).ToArray();
        Assert.Equal(firstRow.Length, first.Length);
        Assert.All(first.Zip(firstRow), pair => Assert.InRange(MathF.Abs(pair.First - pair.Second), 0, 1e-4f));

        using var fork = original.Fork();
        var continuation = original.ForwardToken(0).ToArray();
        Assert.All(continuation.Zip(firstRow), pair =>
            Assert.InRange(MathF.Abs(pair.First - 2 * pair.Second), 0, 1e-4f));
        var forkContinuation = fork.ForwardToken(0).ToArray();
        Assert.Equal(continuation, forkContinuation);

        using var snapshot = new MemoryStream();
        original.SaveState(snapshot);
        snapshot.Position = 0;
        using var restored = model.CreateSession();
        restored.LoadState(snapshot);
        Assert.Equal(original.ForwardToken(0).ToArray(), restored.ForwardToken(0).ToArray());
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
}
