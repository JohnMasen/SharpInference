using SharpInference.Architectures.Rwkv7;
using SharpInference.Backends.Cpu;
using SharpInference.Gguf;
using SharpInference.Graphs;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class PortableRwkv7GraphProviderTests
{
    [Theory]
    [InlineData(RwkvTensorDataType.Float32, 1)]
    [InlineData(RwkvTensorDataType.Float16, 2)]
    public void BuildsCompletePrimitiveGraphWithVersionedTensorContracts(
        RwkvTensorDataType weightType, int layerCount)
    {
        var catalog = new Catalog(layerCount, weightType);
        var graph = new PortableRwkv7GraphProvider().Build(catalog);
        GraphValidator.Validate(graph);
        PortableTensorOperationContracts.ValidateGraph(graph);
        Assert.Equal("rwkv7.forward-token.portable", graph.Identity.Name);
        Assert.Equal("rwkv-7", graph.Identity.ArchitectureId);
        Assert.Equal("RWKV7_State", graph.GraphState.Schema.Name);
        Assert.Equal("rwkv-7.state.fp32@1", graph.Model.StateAbiId);
        Assert.Equal(["token"], graph.Inputs.Select(input => input.Value));
        Assert.Equal(["logits"], graph.Outputs.Select(output => output.Value));
        Assert.Equal(3 * layerCount, graph.GraphState.Slots.Count);
        Assert.Equal(Enumerable.Range(0, layerCount).SelectMany(layer =>
            new[] { $"state.{layer}.ffn-previous", $"state.{layer}.att-previous",
                $"state.{layer}.wkv" }), graph.GraphState.Slots.Select(slot => slot.Name));
        Assert.Equal([2, 4, 4], graph.Resources.Single(resource =>
            resource.Id.Value == "state.0.wkv").Tensor.Dimensions);
        Assert.All(graph.Resources.Where(resource => resource.Kind == GraphResourceKind.Weight),
            resource => Assert.Equal(resource.BindingKey,
                resource.Id.Value["weight.".Length..]));

        var primitives = PrimitiveGraphOperations.CreateStandardDescriptions(false)
            .Select(description => description.Operation)
            .Concat(PortableTensorOperationContracts.Contracts.Select(contract => contract.Operation))
            .ToHashSet();
        Assert.All(graph.Nodes, node => Assert.Contains(node.Operation, primitives));
        Assert.DoesNotContain(graph.Nodes, node =>
            node.Operation.Name.StartsWith("rwkv7.", StringComparison.Ordinal) ||
            node.Operation.Name.StartsWith("rwkv.", StringComparison.Ordinal));
        for (var index = 1; index < graph.Nodes.Count; index++)
            Assert.Equal([graph.Nodes[index - 1].Id], graph.Nodes[index].Dependencies);

        var casts = graph.Nodes.Count(node =>
            node.Operation == PortableTensorOperationContracts.CastFp16ToFp32);
        Assert.Equal(weightType == RwkvTensorDataType.Float16 ? catalog.Names.Count : 0, casts);
        Assert.Equal(layerCount, graph.Nodes.Count(node =>
            node.Operation == PrimitiveGraphOperations.Copy &&
            node.Resources.Any(binding => binding.Access == GraphResourceAccess.Write &&
                binding.Resource.Value.EndsWith(".wkv", StringComparison.Ordinal))));
        Assert.Equal(6 * layerCount, graph.Nodes.Count(node =>
            node.Operation == PortableTensorOperationContracts.Slice));
        Assert.Contains(graph.Nodes, node => node.Operation == PortableTensorOperationContracts.Fill &&
            node.Attributes["value"] == (-0.606531f).ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        Assert.Contains(graph.Nodes, node => node.Operation == PortableTensorOperationContracts.Fill &&
            node.Attributes["value"] == (64e-5f).ToString("R", System.Globalization.CultureInfo.InvariantCulture));

    }

    [Fact]
    public void WkvUpdateIsExplicitMatrixAlgebraWithOldStateProjectionAndNewStateOutput()
    {
        var graph = new PortableRwkv7GraphProvider().Build(new Catalog(1, RwkvTensorDataType.Float32));
        var producers = graph.Nodes.SelectMany(node => node.Resources
            .Where(binding => binding.Access == GraphResourceAccess.Write &&
                !binding.Resource.Value.StartsWith("state.", StringComparison.Ordinal))
            .Select(binding => (binding.Resource, node)))
            .ToDictionary(item => item.Resource, item => item.node);
        LogicalNode Produces(ResourceId resource) => producers[resource];
        ResourceId Read(LogicalNode node, string port) => node.Resources.Single(binding => binding.Port == port).Resource;

        var store = graph.Nodes.Single(node => node.Operation == PrimitiveGraphOperations.Copy &&
            node.Resources.Any(binding => binding.Port == "output" &&
                binding.Resource.Value == "state.0.wkv"));
        var updated = Read(store, "input");
        var sum = Produces(updated);
        Assert.Equal(PrimitiveGraphOperations.Add, sum.Operation);
        var oldTerms = Produces(Read(sum, "left"));
        Assert.Equal(PrimitiveGraphOperations.Add, oldTerms.Operation);
        var decayTerm = Produces(Read(oldTerms, "left"));
        Assert.Equal(PrimitiveGraphOperations.Multiply, decayTerm.Operation);
        Assert.Equal("state.0.wkv", Read(decayTerm, "left").Value);
        Assert.Equal(PortableTensorOperationContracts.Broadcast,
            Produces(Read(decayTerm, "right")).Operation);
        Assert.Equal(PortableTensorOperationContracts.HeadOuter,
            Produces(Read(oldTerms, "right")).Operation);
        var adaptationTerm = Produces(Read(sum, "right"));
        Assert.Equal(PortableTensorOperationContracts.HeadOuter, adaptationTerm.Operation);
        var negativeProjection = Produces(Read(adaptationTerm, "left"));
        Assert.Equal(PrimitiveGraphOperations.Subtract, negativeProjection.Operation);
        var projection = Produces(Read(negativeProjection, "right"));
        Assert.Equal(PortableTensorOperationContracts.BatchedMatVec, projection.Operation);
        Assert.Equal("state.0.wkv", Read(projection, "matrix").Value);
        Assert.Equal(Read(projection, "vector"),
            Read(Produces(Read(adaptationTerm, "right")), "left"));
        Assert.Contains(graph.Nodes, node =>
            node.Operation == PortableTensorOperationContracts.BatchedMatVec &&
            node.Resources.Any(binding => binding.Port == "matrix" &&
                binding.Resource == updated));
        Assert.True(graph.Nodes.ToList().IndexOf(projection) < graph.Nodes.ToList().IndexOf(store));
    }

    [Fact]
    public void RejectsInvalidLowRankProjectionShapeInsteadOfProducingMisleadingGraph()
    {
        var catalog = new Catalog(1, RwkvTensorDataType.Float32, invalidW2: true);
        Assert.Throws<InvalidDataException>(() => new PortableRwkv7GraphProvider().Build(catalog));
    }

    [Theory]
    [Trait("Category", "ExternalModel")]
    [InlineData("FP32")]
    [InlineData("FP16")]
    public void TinyModel_PortablePipelineMatchesDirectExecutorLogitsAndState(string format)
    {
        var path = TestModelLoader.GetPath(format == "FP32"
            ? TestModel.Rwkv7Fp32 : TestModel.Rwkv7Fp16);
        using var catalog = GgmlModelFile.Open(path);
        var graph = new PortableRwkv7GraphProvider().Build(catalog);
        var executor = new CpuPrimitiveGraphExecutor(graph, catalog);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var state = graph.GraphState.Slots.Select(slot =>
        {
            var dims = resources[slot.Resource].Tensor.Dimensions.ToArray();
            return new GraphStateValue(slot.Name, dims,
                new float[checked(dims.Aggregate(1, (product, dimension) =>
                    checked(product * dimension)))]);
        }).ToArray();
        using var processor = Processor.LoadGraph(path, graph, CpuPrimitiveGraphBackend.Instance);
        using var session = processor.CreateSession();
        foreach (var token in new[] { (int)'"', (int)'i', (int)'n' })
        {
            var expectedLogits = session.ForwardToken(token).ToArray();
            var actual = executor.Execute(
                new Dictionary<ResourceId, Array> { [new("token")] = new[] { token } }, state);
            var actualLogits = Assert.IsType<float[]>(actual.Outputs[new ResourceId("logits")]);
            var expectedState = StateSnapshotAssertions.Capture(session);
            Assert.Equal(expectedState.Tensors.Select(tensor => tensor.Name),
                actual.State.Select(entry => entry.Name));
            var expectedValues = StateSnapshotAssertions.Values(expectedState);
            var actualValues = actual.State.SelectMany(entry => entry.Values).ToArray();
            Near(expectedValues, actualValues, 0.001f, $"token {token} state");
            Near(expectedLogits, actualLogits, 0.001f, $"token {token} logits");
            state = actual.State.ToArray();
        }

        static void Near(float[] expected, float[] actual, float tolerance, string label)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (var i = 0; i < actual.Length; i++)
                Assert.True(float.IsFinite(actual[i]) &&
                    MathF.Abs(expected[i] - actual[i]) <= tolerance * MathF.Max(1, MathF.Abs(expected[i])),
                    $"{label}[{i}]: expected {expected[i]}, actual {actual[i]}");
        }
    }

    private sealed class Catalog : IModelTensorCatalog
    {
        private readonly Dictionary<string, IModelTensor> tensors = new(StringComparer.Ordinal);

        public Catalog(int layerCount, RwkvTensorDataType type, bool invalidW2 = false)
        {
            LayerCount = layerCount;
            void Add(string name, params int[] dimensions) =>
                tensors.Add(name, new Tensor(name, dimensions, type));
            Add("emb.weight", 8, 32);
            Add("head.weight", 8, 32);
            Add("ln_out.weight", 8);
            Add("ln_out.bias", 8);
            Add("blocks.0.ln0.weight", 8);
            Add("blocks.0.ln0.bias", 8);
            for (var index = 0; index < layerCount; index++)
            {
                var prefix = $"blocks.{index}.";
                foreach (var name in new[]
                {
                    "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias", "att.w0",
                    "att.a0", "att.k_k", "att.k_a", "att.ln_x.weight", "att.ln_x.bias",
                    "ffn.x_k",
                }) Add(prefix + name, 8);
                Add(prefix + "att.x_rwkvag", 8, 6);
                Add(prefix + "att.r_k", 4, 2);
                foreach (var name in new[]
                {
                    "att.receptance.weight", "att.key.weight", "att.value.weight",
                    "att.output.weight",
                }) Add(prefix + name, 8, 8);
                foreach (var (family, rank) in new[]
                {
                    ("w", index == 0 ? 2 : 3), ("a", 3), ("g", 4),
                })
                {
                    Add(prefix + $"att.{family}1", 8, rank);
                    Add(prefix + $"att.{family}2", family == "w" && invalidW2 ? rank + 1 : rank, 8);
                }
                if (index > 0)
                {
                    Add(prefix + "att.v0", 8);
                    Add(prefix + "att.v1", 8, 2);
                    Add(prefix + "att.v2", 2, 8);
                }
                Add(prefix + "ffn.key.weight", 8, 16);
                Add(prefix + "ffn.value.weight", 16, 8);
            }
        }

        public int VocabularySize => 32;
        public int EmbeddingSize => 8;
        public int LayerCount { get; }
        public IReadOnlyCollection<string> Names => tensors.Keys;
        public bool TryGet(string name, out IModelTensor tensor) => tensors.TryGetValue(name, out tensor!);
        public IModelTensor GetRequired(string name) => tensors[name];
    }

    private sealed record Tensor(string Name, IReadOnlyList<int> Dimensions, RwkvTensorDataType DataType)
        : IModelTensor
    {
        public ReadOnlySpan<float> FloatValues => [];
        public ReadOnlySpan<Half> HalfValues => [];
    }
}
