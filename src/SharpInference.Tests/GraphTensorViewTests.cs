using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using System.Xml.Linq;
using SharpInference.Backends.Cpu;
using SharpInference.Graphs;
using SharpInference.Runtime.Cpu;
using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class GraphTensorViewTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SerializedViewsExecuteWithExactOffsetsOnCpu(bool xml)
    {
        var graph = CreateGraph();
        var serialized = xml ? GraphXml.Serialize(graph) : GraphJson.Serialize(graph);
        graph = xml ? GraphXml.DeserializeLogical(serialized) : GraphJson.DeserializeLogical(serialized);
        using var backend = CpuVmBackendFactory.Create();
        backend.Prepare(graph);
        using var session = backend.CreateGraphSession(new EmptyCatalog());
        var values = Enumerable.Range(1, 10).Select(value => (float)value).ToArray();
        var result = session.Execute(new Dictionary<ResourceId, ReadOnlyMemory<byte>>
        {
            [new("input")] = MemoryMarshal.AsBytes(values.AsSpan()).ToArray(),
        });
        Assert.Equal([7f, 9f, 11f, 13f, 15f, 17f], MemoryMarshal.Cast<byte, float>(result[new("output")]).ToArray());
        var calls = backend.Program!.Definitions.Single(definition => definition.Id == "forward").Nodes
            .Select(node => node.Instruction).OfType<VmCall>().ToArray();
        Assert.Equal([4UL, 16UL, 0UL], calls[0].Arguments.Select(argument => argument.ByteOffset));
        Assert.Equal([16UL, 28UL, 12UL], calls[1].Arguments.Select(argument => argument.ByteOffset));
    }

    [Fact]
    public void ViewFormatsCannotBeDowngradedToLegacyWholeTensorFormats()
    {
        var graph = CreateGraph();
        var json = JsonNode.Parse(GraphJson.Serialize(graph))!.AsObject();
        Assert.Equal("LogicalGraphV2", json["Kind"]!.GetValue<string>());
        json["Kind"] = "LogicalGraph";
        json.Remove("FormatVersion");
        Assert.Throws<InvalidDataException>(() => GraphJson.DeserializeLogical(json.ToJsonString()));
        var xml = XElement.Parse(GraphXml.Serialize(graph));
        Assert.Equal("2", (string?)xml.Attribute("version"));
        xml.SetAttributeValue("version", 1);
        Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeLogical(xml.ToString()));
    }

    [Fact]
    public void ExecutionSerializationPreservesViewShapeAndOffset()
    {
        var execution = new GraphOptimizer().Optimize(CreateGraph(),
            new(OptimizationBoundary.Off, DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded));
        var copies = new[] { GraphJson.DeserializeExecution(GraphJson.Serialize(execution)),
            GraphXml.DeserializeExecution(GraphXml.Serialize(execution)) };
        foreach (var copy in copies)
        {
            Assert.Equal(2, copy.Identity.IrVersion);
            var view = Assert.IsType<GraphTensorView>(copy.Nodes[0].Resources[0].View);
            Assert.Equal(4UL, view.ByteOffset);
            Assert.Equal([3], view.Tensor.Dimensions);
            Assert.Equal(GraphElementType.Float32, view.Tensor.ElementType);
        }
        Assert.Throws<NotSupportedException>(() => new GraphOptimizer().Optimize(CreateGraph()));
        Assert.Throws<NotSupportedException>(() => new CpuPrimitiveGraphExecutor(execution, new EmptyCatalog()));
    }

    [Theory]
    [InlineData("bounds")]
    [InlineData("alignment")]
    [InlineData("type")]
    [InlineData("layout")]
    [InlineData("overflow")]
    [InlineData("ir")]
    [InlineData("overlap")]
    public void InvalidViewsFailBeforeCompilation(string invalid)
    {
        var graph = CreateGraph();
        var nodes = graph.Nodes.ToArray();
        var tensor = invalid switch
        {
            "type" => new TensorDescriptor(GraphElementType.Float16, [3]),
            "layout" => new TensorDescriptor(GraphElementType.Float32, [3], "token-major"),
            "overflow" => new TensorDescriptor(GraphElementType.Float32, [int.MaxValue, int.MaxValue, int.MaxValue]),
            _ => new TensorDescriptor(GraphElementType.Float32, [3]),
        };
        if (invalid == "overlap")
            nodes[1] = nodes[1] with { Resources = nodes[1].Resources.Select(binding => binding.Port == "output"
                ? binding with { View = new(0, tensor) } : binding).ToArray() };
        else
            nodes[0] = nodes[0] with { Resources = nodes[0].Resources.Select(binding => binding.Port == "left"
                ? binding with { View = new(invalid == "bounds" ? 40UL : invalid == "alignment" ? 1UL : 4UL, tensor) }
                : binding).ToArray() };
        Assert.Throws<InvalidDataException>(() => new LogicalGraph(
            graph.Identity with { IrVersion = invalid == "ir" ? 1 : 2 }, graph.Model, graph.Resources, graph.Regions,
            nodes, graph.Inputs, graph.Outputs, graph.GraphState));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PartialWritesRequireCompleteCoverageAndDependencies(bool missingDependency)
    {
        var builder = CopyBuilder()
            .AddNode("first", PrimitiveGraphOperations.Copy, "root",
                [View("input", "input", GraphResourceAccess.Read, 0, 2), View("output", "temporary", GraphResourceAccess.Write, 0, 2)]);
        if (missingDependency)
            builder.AddNode("second", PrimitiveGraphOperations.Copy, "root",
                [View("input", "input", GraphResourceAccess.Read, 8, 2), View("output", "temporary", GraphResourceAccess.Write, 8, 2)]);
        builder.AddNode("read", PrimitiveGraphOperations.Copy, "root",
            [GraphBindings.Read("input", "temporary"), GraphBindings.Write("output", "output")], ["first"]);
        var error = Assert.Throws<InvalidDataException>(() => builder.Build());
        Assert.Contains(missingDependency ? "unordered overlapping view accesses" : "uninitialized byte range", error.Message);
    }

    [Fact]
    public void DisjointWritesInitializeTemporaryStorageWithoutAssumingZeroFill()
    {
        var graph = CopyBuilder()
            .AddNode("first", PrimitiveGraphOperations.Copy, "root",
                [View("input", "input", GraphResourceAccess.Read, 0, 2), View("output", "temporary", GraphResourceAccess.Write, 0, 2)])
            .AddNode("second", PrimitiveGraphOperations.Copy, "root",
                [View("input", "input", GraphResourceAccess.Read, 8, 2), View("output", "temporary", GraphResourceAccess.Write, 8, 2)])
            .AddNode("read", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "temporary"), GraphBindings.Write("output", "output")], ["first", "second"])
            .Build();
        using var backend = CpuVmBackendFactory.Create();
        backend.Prepare(graph);
        using var session = backend.CreateGraphSession(new EmptyCatalog());
        var result = session.Execute(new Dictionary<ResourceId, ReadOnlyMemory<byte>>
        {
            [new("input")] = MemoryMarshal.AsBytes(new float[] { 4, 3, 2, 1 }.AsSpan()).ToArray(),
        });
        Assert.Equal([4f, 3f, 2f, 1f], MemoryMarshal.Cast<byte, float>(result[new("output")]).ToArray());
    }

    [Fact]
    public void HalfWeightViewKeepsStorageTypeAndMatrixShape()
    {
        var graph = new LogicalGraphBuilder(new("external", 2, "half-view"),
                new GraphModelSignature("matrix", "matrix.empty@1", new Dictionary<string, int>()))
            .AddRegion("root", GraphRegionTypes.Graph, "Root")
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("weight", "Weight", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                new TensorDescriptor(GraphElementType.Float16, [8]), "weight")
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphOutput: true)
            .AddNode("multiply", PrimitiveGraphOperations.MatVec, "root",
                [new("matrix", new("weight"), GraphResourceAccess.Read, View: new(8, new(GraphElementType.Float16, [2, 2]))),
                 GraphBindings.Read("input", "input"), GraphBindings.Write("output", "output")])
            .Build();
        using var backend = CpuVmBackendFactory.Create();
        backend.Prepare(graph);
        using var session = backend.CreateGraphSession(new HalfCatalog());
        var result = session.Execute(new Dictionary<ResourceId, ReadOnlyMemory<byte>>
        {
            [new("input")] = MemoryMarshal.AsBytes(new float[] { 1, 1 }.AsSpan()).ToArray(),
        });
        Assert.Equal([5f, 9f], MemoryMarshal.Cast<byte, float>(result[new("output")]).ToArray());
    }

    private static LogicalGraph CreateGraph() => new LogicalGraphBuilder(new("external", 2, "views.forward"),
            new GraphModelSignature("array", "array.empty@1", new Dictionary<string, int>()))
        .AddRegion("root", GraphRegionTypes.Graph, "Root")
        .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
            new TensorDescriptor(GraphElementType.Float32, [10]), graphInput: true)
        .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
            new TensorDescriptor(GraphElementType.Float32, [6]), graphOutput: true)
        .AddNode("first", PrimitiveGraphOperations.Add, "root",
            [View("left", "input", GraphResourceAccess.Read, 4, 3), View("right", "input", GraphResourceAccess.Read, 16, 3),
             View("output", "output", GraphResourceAccess.Write, 0, 3)])
        .AddNode("second", PrimitiveGraphOperations.Add, "root",
            [View("left", "input", GraphResourceAccess.Read, 16, 3), View("right", "input", GraphResourceAccess.Read, 28, 3),
             View("output", "output", GraphResourceAccess.Write, 12, 3)]).Build();

    private static LogicalGraphBuilder CopyBuilder() => new LogicalGraphBuilder(new("external", 2, "view-copy"),
            new GraphModelSignature("array", "array.empty@1", new Dictionary<string, int>()))
        .AddRegion("root", GraphRegionTypes.Graph, "Root")
        .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
            new TensorDescriptor(GraphElementType.Float32, [4]), graphInput: true)
        .AddResource("temporary", "Temporary", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
            new TensorDescriptor(GraphElementType.Float32, [4]))
        .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
            new TensorDescriptor(GraphElementType.Float32, [4]), graphOutput: true);

    private static NodeResourceBinding View(string port, string resource, GraphResourceAccess access, ulong offset, int count) =>
        new(port, new(resource), access, View: new(offset, new(GraphElementType.Float32, [count])));

    private sealed class EmptyCatalog : IModelTensorCatalog
    {
        public IReadOnlyCollection<string> Names => [];
        public bool TryGet(string name, out IModelTensor tensor) { tensor = null!; return false; }
        public IModelTensor GetRequired(string name) => throw new InvalidDataException(name);
    }

    private sealed class HalfCatalog : IModelTensorCatalog, IModelTensor
    {
        private readonly Half[] values = [(Half)0, (Half)0, (Half)0, (Half)0, (Half)2, (Half)3, (Half)4, (Half)5];
        public IReadOnlyCollection<string> Names => ["weight"];
        public string Name => "weight";
        public TensorDataType DataType => TensorDataType.Float16;
        public IReadOnlyList<int> Dimensions => [8];
        public ReadOnlySpan<float> FloatValues => throw new NotSupportedException();
        public ReadOnlySpan<Half> HalfValues => values;
        public bool TryGet(string name, out IModelTensor tensor) { tensor = this; return name == Name; }
        public IModelTensor GetRequired(string name) => name == Name ? this : throw new InvalidDataException(name);
    }
}
