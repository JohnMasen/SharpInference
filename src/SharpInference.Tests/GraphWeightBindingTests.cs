using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Backends.Cpu;
using SharpInference.Graphs;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class GraphWeightBindingTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PortableGraph_BindsWeightsByCatalogKeyAfterResourceIdsAreRenamed(bool rwkv7)
    {
        var model = rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6;
        var path = TestModelLoader.GetPath(model);
        var backend = CpuPrimitiveGraphBackend.Instance;
        using var catalog = TestModelLoader.OpenCatalog(model);
        var logical = Provider(rwkv7).Build(catalog);
        var execution = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Unrestricted), backend.KernelCatalog);
        var renamed = XmlOnly(AliasWeights(execution));
        var weights = renamed.Resources.Where(resource => resource.Kind == GraphResourceKind.Weight).ToArray();
        Assert.NotEmpty(weights);
        Assert.All(weights, weight =>
        {
            Assert.StartsWith("parameter.", weight.Id.Value, StringComparison.Ordinal);
            Assert.True(catalog.TryGet(weight.BindingKey!, out _));
        });
        GraphValidator.Validate(renamed);

        using var baseline = Processor.LoadGraph(path, logical, backend);
        using var imported = new ProcessorPipelineBuilder(path)
            .UseReader(new GgmlModelReader(), new GraphArchitectureMetadataReader(logical))
            .UseExecutionGraph(_ => renamed)
            .UseBackend(backend)
            .UsePortableGraphArchitecture()
            .Build();
        using var expected = baseline.CreateSession();
        using var actual = imported.CreateSession();
        foreach (var token in new[] { 0, 1, 0 })
        {
            Assert.Equal(expected.ForwardToken(token).ToArray(), actual.ForwardToken(token).ToArray());
            StateSnapshotAssertions.Equal(StateSnapshotAssertions.Capture(expected),
                StateSnapshotAssertions.Capture(actual));
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PortableGraph_RejectsWeightShapeMismatchBeforePreparation(bool rwkv7)
    {
        var model = rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6;
        using var catalog = TestModelLoader.OpenCatalog(model);
        var logical = Provider(rwkv7).Build(catalog);
        var head = logical.Resources.Single(resource =>
            resource.Kind == GraphResourceKind.Weight && resource.BindingKey == "head.weight");
        var resources = logical.Resources.Select(resource => resource.Id == head.Id
            ? resource with { Tensor = new TensorDescriptor(resource.Tensor.ElementType, [1]) }
            : resource);
        var invalid = new LogicalGraph(logical.Identity, logical.Model, resources,
            logical.Regions, logical.Nodes, logical.Inputs, logical.Outputs, logical.GraphState);

        Assert.Throws<InvalidDataException>(() =>
            Processor.LoadGraph(TestModelLoader.GetPath(model), invalid, CpuPrimitiveGraphBackend.Instance));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PortableGraph_UsesSemanticWeightPorts(bool rwkv7)
    {
        using var catalog = TestModelLoader.OpenCatalog(rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6);
        var graph = Provider(rwkv7).Build(catalog);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var weightArguments = graph.Nodes.SelectMany(node => node.Resources)
            .Where(binding => resources[binding.Resource].Kind == GraphResourceKind.Weight)
            .ToArray();
        Assert.NotEmpty(weightArguments);
        Assert.All(weightArguments, binding => Assert.DoesNotMatch(@"^node\d+\.", binding.Port));
    }

    private static ILogicalGraphProvider Provider(bool rwkv7) =>
        rwkv7 ? new PortableRwkv7GraphProvider() : new PortableRwkv6GraphProvider();

    private static ExecutionGraph AliasWeights(ExecutionGraph graph)
    {
        var renamed = graph.Resources.Where(resource => resource.Kind == GraphResourceKind.Weight)
            .Select((resource, index) => (resource.Id, NewId: new ResourceId($"parameter.{index:D4}")))
            .ToDictionary(pair => pair.Id, pair => pair.NewId);
        return new ExecutionGraph(graph.Identity, graph.Model,
            graph.Resources.Select(resource => renamed.TryGetValue(resource.Id, out var id)
                ? resource with { Id = id }
                : resource),
            graph.Regions,
            graph.Nodes.Select(node => node with
            {
                Resources = node.Resources.Select(binding => renamed.TryGetValue(binding.Resource, out var id)
                    ? binding with { Resource = id }
                    : binding).ToArray(),
            }),
            graph.Inputs, graph.Outputs, graph.GraphState);
    }

    private static ExecutionGraph XmlOnly(ExecutionGraph graph) =>
        GraphXml.DeserializeExecution(GraphXml.Serialize(new ExecutionGraph(graph.Identity, graph.Model,
            graph.Resources, graph.Regions,
            graph.Nodes.Select(node => node with { Source = ExecutionSourceMap.XmlOnly }),
            graph.Inputs, graph.Outputs, graph.GraphState)));
}
