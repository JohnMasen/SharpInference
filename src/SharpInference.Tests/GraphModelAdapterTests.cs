using SharpInference.Instructions;
using SharpInference;
using SharpInference.Architectures.Phi4;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

namespace SharpInference.Tests;

public sealed class GraphModelAdapterTests
{
    [Theory]
    [InlineData(TensorDataType.Float32)]
    [InlineData(TensorDataType.Float16)]
    public void Rwkv7_PublicModuleBuildsWithoutTensorValues(TensorDataType dataType)
    {
        var catalog = Rwkv7Catalog(dataType);
        var module = new Rwkv7ModelModule();

        Assert.True(module.CanLoad(catalog));
        var metadata = module.ReadMetadata(catalog);
        Assert.Equal("rwkv-7", metadata.ArchitectureId);
        Assert.Equal(3L, metadata.Dimensions["vocabulary"]);
        Assert.Equal(2L, metadata.Dimensions["embedding"]);
        Assert.Equal(1L, metadata.Dimensions["layers"]);
        Assert.Equal(2L, metadata.Dimensions["attentionHeads"]);
        Assert.Equal(1L, metadata.Dimensions["attentionHeadSize"]);

        var package = module.Build(catalog);
        var graph = Assert.IsType<LogicalGraph>(package.MainGraph);

        Assert.Equal("rwkv7.forward-token.portable", package.MainGraphId);
        Assert.Same(graph, Assert.Single(package));
        Assert.Equal(GraphCompileBoundary.Atomic, graph.CompileBoundary);
        Assert.DoesNotContain(package.Validate(), item => item.Severity == GraphDiagnosticSeverity.Error);
        AssertResource(graph, "token", GraphIO.In, ElementType.Int32, [1]);
        AssertResource(graph, "logits", GraphIO.Out, ElementType.Float32, [3]);
        foreach (var name in catalog.Names)
        {
            var resource = Assert.Single(graph.Resources, resource => resource.Id == $"weight.{name}");
            Assert.Equal(ResourceScope.Global, resource.Scope);
            Assert.Equal(ResourceUsage.Weight, resource.Usage);
            Assert.Equal(ResourcePermission.ReadOnly, resource.Permission);
            Assert.Equal(name, resource.BindingKey);
            Assert.Equal(dataType == TensorDataType.Float16 ? ElementType.Float16 : ElementType.Float32,
                resource.Tensor.ElementType);
            Assert.Equal(catalog.GetRequired(name).Dimensions, resource.Tensor.Dimensions);
        }

        var states = graph.Resources.Where(resource => resource.IsState).ToArray();
        Assert.Equal(3, states.Length);
        AssertResource(graph, "state.0.att-previous", GraphIO.InOut, ElementType.Float32, [2]);
        AssertResource(graph, "state.0.ffn-previous", GraphIO.InOut, ElementType.Float32, [2]);
        AssertResource(graph, "state.0.wkv", GraphIO.InOut, ElementType.Float32, [2, 1, 1]);
        foreach (var state in states)
        {
            Assert.Equal(ResourceScope.Session, state.Scope);
            Assert.Equal(state.Id, state.BindingKey);
            var accesses = graph.Actions.SelectMany(action => action.Bindings)
                .Where(binding => binding.Target.TargetProperty == state.Id).ToArray();
            Assert.Contains(accesses, binding => binding.Access == ResourceAccess.ReadWrite);
            Assert.Contains(accesses, binding => binding.Access == ResourceAccess.Write);
            Assert.DoesNotContain(accesses, binding => binding.Access == ResourceAccess.Read);
        }

        AssertSerialDependencies(graph);
        Assert.Contains(graph.Actions, node => node.Metadata.ContainsKey("architecture.formula"));
        AssertRoundTrip(package);
    }

    [Theory]
    [InlineData(Phi4AudioProjector.Speech, TensorDataType.Float32)]
    [InlineData(Phi4AudioProjector.Speech, TensorDataType.Float16)]
    [InlineData(Phi4AudioProjector.Vision, TensorDataType.Float32)]
    [InlineData(Phi4AudioProjector.Vision, TensorDataType.Float16)]
    public void Phi4_PublicAudioProjectorPreservesBindingsAndViews(
        Phi4AudioProjector projector, TensorDataType dataType)
    {
        var prefix = projector == Phi4AudioProjector.Speech ? "mm.a.mlp" : "mm.a.vis";
        var catalog = new SyntheticCatalog(
            new SyntheticTensor(prefix + ".0.weight", dataType, [3, 2]),
            new SyntheticTensor(prefix + ".0.bias", dataType, [3]),
            new SyntheticTensor(prefix + ".2.weight", dataType, [4, 3]),
            new SyntheticTensor(prefix + ".2.bias", dataType, [4]));
        var module = new Phi4AudioProjectionGraphModule(projector, tokens: 2);

        Assert.True(module.CanLoad(catalog));
        var metadata = module.ReadMetadata(catalog);
        Assert.Equal(2L, metadata.Dimensions["tokens"]);
        Assert.Equal(2L, metadata.Dimensions["audio_width"]);
        Assert.Equal(3L, metadata.Dimensions["hidden_width"]);
        Assert.Equal(4L, metadata.Dimensions["embedding_width"]);

        var package = module.Build(catalog);
        var graph = Assert.IsType<LogicalGraph>(package.MainGraph);

        Assert.Equal(module.ArchitectureId, package.MainGraphId);
        Assert.Same(graph, Assert.Single(package));
        Assert.Equal(GraphCompileBoundary.Atomic, graph.CompileBoundary);
        Assert.DoesNotContain(package.Validate(), item => item.Severity == GraphDiagnosticSeverity.Error);
        AssertResource(graph, "audio_hidden", GraphIO.In, ElementType.Float32, [2, 2]);
        AssertResource(graph, "output", GraphIO.Out, ElementType.Float32, [2, 4]);
        Assert.Equal(dataType == TensorDataType.Float16 ? 7 : 3, graph.Actions.Count);
        foreach (var name in catalog.Names)
        {
            var resource = Assert.Single(graph.Resources, resource => resource.Id == $"weight.{name}");
            Assert.Equal(ResourceScope.Global, resource.Scope);
            Assert.Equal(ResourceUsage.Weight, resource.Usage);
            Assert.Equal(ResourcePermission.ReadOnly, resource.Permission);
            Assert.Equal(name, resource.BindingKey);
            Assert.Equal(dataType == TensorDataType.Float16 ? ElementType.Float16 : ElementType.Float32,
                resource.Tensor.ElementType);
        }

        var activation = Assert.IsType<OperationAction>(
            Assert.Single(graph.Actions, node => node.Id == "projector_gelu"));
        Assert.Equal("h = gelu(x W_up^T + b_up)", activation.Metadata["architecture.formula"]);
        Assert.Equal("root/activation", activation.Metadata["architecture.path"]);
        Assert.Equal("gelu", activation.Attributes["activation"]);
        Assert.Equal("6", activation.Attributes["count"]);
        Assert.Equal("3", activation.Attributes["width"]);
        foreach (var port in new[] { "input", "output" })
        {
            Assert.Equal("0", activation.Attributes[$"view.{port}.offset"]);
            Assert.Equal("Float32", activation.Attributes[$"view.{port}.element_type"]);
            Assert.Equal("6", activation.Attributes[$"view.{port}.dimensions"]);
            Assert.Equal("dense", activation.Attributes[$"view.{port}.layout"]);
        }

        var up = Assert.IsType<OperationAction>(
            Assert.Single(graph.Actions, node => node.Id == "projector_up"));
        Assert.Equal("True", up.Attributes["transpose_right"]);
        Assert.Contains(up.Bindings, binding =>
            binding.SourceProperty == "left" && binding.Target.TargetProperty == "audio_hidden" &&
            binding.Access == ResourceAccess.Read);
        var down = Assert.Single(graph.Actions, node => node.Id == "projector_down");
        Assert.Contains(down.Bindings, binding =>
            binding.SourceProperty == "output" && binding.Target.TargetProperty == "output" &&
            binding.Access == ResourceAccess.Write);
        AssertSerialDependencies(graph);
        AssertRoundTrip(package);
    }

    private static void AssertResource(LogicalGraph graph, string id,
        GraphIO graphIO, ElementType type, int[] dimensions)
    {
        var resource = Assert.Single(graph.Resources, resource => resource.Id == id);
        Assert.Equal(graphIO, resource.GraphIO);
        Assert.Equal(type, resource.Tensor.ElementType);
        Assert.Equal(dimensions, resource.Tensor.Dimensions);
    }

    private static void AssertSerialDependencies(LogicalGraph graph)
    {
        Assert.NotEmpty(graph.Actions);
        Assert.Equal(graph.Actions.Count - 1, graph.GraphFlow.Count);
        for (var index = 1; index < graph.Actions.Count; index++)
        {
            Assert.Contains(graph.GraphFlow, flow =>
                flow.Source == new ActionRef(graph.Actions[index - 1]) &&
                flow.Target == new ActionRef(graph.Actions[index]));
        }
    }

    private static void AssertRoundTrip(GraphPackage<LogicalGraph> package)
    {
        var xml = GraphPackageXml.Serialize(package);
        var restored = GraphPackageXml.Deserialize<LogicalGraph>(xml);
        Assert.Equal(package.MainGraphId, restored.MainGraphId);
        Assert.Equal(xml, GraphPackageXml.Serialize(restored));
        Assert.DoesNotContain(restored.Validate(), item => item.Severity == GraphDiagnosticSeverity.Error);
        Assert.Equal(xml, GraphPackageXml.Serialize(package.Snapshot()));
    }

    private static SyntheticCatalog Rwkv7Catalog(TensorDataType dataType)
    {
        var tensors = new List<SyntheticTensor>();
        void Add(string name, params int[] dimensions) => tensors.Add(new(name, dataType, dimensions));
        Add("emb.weight", 2, 3);
        Add("head.weight", 2, 3);
        foreach (var name in new[]
        {
            "blocks.0.ln0", "blocks.0.ln1", "blocks.0.ln2", "blocks.0.att.ln_x", "ln_out",
        })
        {
            Add(name + ".weight", 2);
            Add(name + ".bias", 2);
        }
        Add("blocks.0.att.r_k", 1, 2);
        Add("blocks.0.att.x_rwkvag", 6, 2);
        foreach (var name in new[] { "receptance.weight", "key.weight", "value.weight", "output.weight" })
        {
            Add("blocks.0.att." + name, 2, 2);
        }
        foreach (var name in new[] { "w", "a", "g" })
        {
            Add("blocks.0.att." + name + "1", 2, 1);
            Add("blocks.0.att." + name + "2", 1, 2);
        }
        foreach (var name in new[] { "w0", "a0", "k_k", "k_a" })
        {
            Add("blocks.0.att." + name, 2);
        }
        Add("blocks.0.ffn.x_k", 2);
        Add("blocks.0.ffn.key.weight", 2, 3);
        Add("blocks.0.ffn.value.weight", 3, 2);
        return new(tensors.ToArray());
    }

    private sealed class SyntheticCatalog(params SyntheticTensor[] tensors) : IModelTensorCatalog
    {
        private readonly Dictionary<string, IModelTensor> values =
            tensors.ToDictionary(tensor => tensor.Name, tensor => (IModelTensor)tensor, StringComparer.Ordinal);

        public IReadOnlyCollection<string> Names => values.Keys;
        public bool TryGet(string name, out IModelTensor tensor) => values.TryGetValue(name, out tensor!);
        public IModelTensor GetRequired(string name) => values[name];
    }

    private sealed record SyntheticTensor(
        string Name, TensorDataType DataType, IReadOnlyList<int> Dimensions) : IModelTensor
    {
        public ReadOnlySpan<float> FloatValues =>
            throw new InvalidOperationException("Graph construction must not read tensor values.");
        public ReadOnlySpan<Half> HalfValues =>
            throw new InvalidOperationException("Graph construction must not read tensor values.");
    }
}
