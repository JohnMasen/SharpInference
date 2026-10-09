using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class ModelGraphModuleTests
{
    [Theory]
    [Trait("Category", "ExternalModel")]
    [InlineData(false)]
    [InlineData(true)]
    public void ExplicitRegistryRecognizesRwkv7FormatsWithoutArchitectureSwitches(bool half)
    {
        using var catalog = TestModelLoader.OpenCatalog(half ? TestModel.Rwkv7Fp16 : TestModel.Rwkv7Fp32);
        var registry = new ModelGraphModuleRegistry();
        registry.Register(new Rwkv6ModelModule());
        registry.Register(new Rwkv7ModelModule());
        var selection = registry.Build(catalog);
        Assert.IsType<Rwkv7ModelModule>(selection.Module);
        Assert.Equal("rwkv-7", selection.Metadata.ArchitectureId);
        Assert.Equal(catalog.VocabularySize, selection.Metadata.Dimensions["vocabulary"]);
        Assert.Equal("RWKV7_State", selection.Graph.GraphState.Schema.Name);
        Assert.Equal("rwkv-7.state.fp32@1", selection.Graph.Model.StateAbiId);
        Assert.Equal(selection.Metadata.Dimensions["embedding"],
            selection.Metadata.Dimensions["attentionHeads"] * selection.Metadata.Dimensions["attentionHeadSize"]);
    }

    [Fact]
    [Trait("Category", "ExternalModel")]
    public void Rwkv6ModuleResolvesAndBuildsItsOwnGraph()
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv6);
        var module = new Rwkv6ModelModule();
        var registry = new ModelGraphModuleRegistry();
        registry.Register(module);
        var selection = registry.Build(catalog);
        Assert.Same(module, selection.Module);
        Assert.Equal("rwkv-6", selection.Metadata.ArchitectureId);
        Assert.Equal(catalog.VocabularySize, selection.Metadata.Dimensions["vocabulary"]);
        Assert.Equal("RWKV6_State", selection.Graph.GraphState.Schema.Name);
        Assert.Equal(selection.Graph.Model, RwkvGraphSignatures.Create(catalog.VocabularySize,
            catalog.EmbeddingSize, catalog.LayerCount,
            checked((int)selection.Metadata.Dimensions["attentionHeads"]),
            checked((int)selection.Metadata.Dimensions["attentionHeadSize"]), "rwkv-6.state.fp32@1"));
    }

    [Fact]
    public void RegistrationReportsUnknownDuplicateAndAmbiguousModules()
    {
        var registry = new ModelGraphModuleRegistry();
        Assert.Throws<NotSupportedException>(() => registry.Resolve(new EmptyCatalog()));
        registry.Register(new Module("external-a"));
        Assert.Throws<InvalidOperationException>(() => registry.Register(new Module("external-a")));
        Assert.Throws<NotSupportedException>(() => registry.GetRequired("external-b"));
        registry.Register(new Module("external-b"));
        Assert.Throws<InvalidDataException>(() => registry.Resolve(new EmptyCatalog()));
    }

    [Fact]
    public void NonLanguageModuleBuildsWithoutChangingSharedCode()
    {
        var module = new Module("external-counter");
        var registry = new ModelGraphModuleRegistry();
        registry.Register(module);
        var selection = registry.Build(new EmptyCatalog());
        Assert.Equal(["counterWidth"], selection.Metadata.Dimensions.Keys);
        Assert.Equal("external-counter", selection.Graph.Identity.ArchitectureId);
        Assert.Empty(selection.Graph.GraphState);
    }

    private sealed class EmptyCatalog : IModelTensorCatalog
    {
        public IReadOnlyCollection<string> Names => Array.Empty<string>();
        public bool TryGet(string name, out IModelTensor tensor) { tensor = null!; return false; }
        public IModelTensor GetRequired(string name) => throw new InvalidDataException(name);
    }

    private sealed class Module(string id) : IModelGraphModule
    {
        public string ArchitectureId => id;
        public bool CanLoad(IModelTensorCatalog tensors) => true;
        public ModelMetadata ReadMetadata(IModelTensorCatalog tensors) =>
            new(id, new Dictionary<string, long> { ["counterWidth"] = 3 });
        public LogicalGraph Build(IModelTensorCatalog tensors) =>
            new LogicalGraphBuilder(new(id, 1, "counter"),
                    new GraphModelSignature("counter", "counter.empty@1", new Dictionary<string, int> { ["counterWidth"] = 3 }))
                .AddRegion("root", GraphRegionTypes.Graph, "Root").Build();
    }
}
