using SharpInference.Graphs;
using TensorOps = SharpInference.Graphs.PortableTensorOperationContracts;

namespace SharpInference.Tests;

public sealed class PortableTensorOperationContractTests
{
    private const GraphElementType F32 = GraphElementType.Float32;
    private const GraphElementType F16 = GraphElementType.Float16;

    [Fact]
    public void ValidatesVersionedTypesAndWkvCompatibleShapes()
    {
        Assert.All(TensorOps.Contracts, description =>
            Assert.Equal(1, description.Operation.Version));
        Assert.Equal(TensorOps.Contracts.Count,
            TensorOps.Contracts.Select(description => description.Operation).Distinct().Count());
        Assert.Equal(["matrix", "vector"], TensorOps.Contracts.Single(contract =>
            contract.Operation == TensorOps.BatchedMatVec).InputPorts);

        Check(TensorOps.Fill, [], [1], attributes: new TensorFillValue(-0.606531f).ToAttributes());
        Check(TensorOps.Fill, [], [2, 4], attributes: new TensorFillValue(1e-5f).ToAttributes());
        Check(TensorOps.CastFp16ToFp32, [Input("input", F16, 8, 6)], [8, 6]);
        Check(TensorOps.Reshape, [Input("input", F32, 8, 6)], [6, 8]);
        Check(TensorOps.Slice, [Input("input", F32, 6, 8)], [1, 8],
            attributes: new TensorSlice(0, 2, 1).ToAttributes());
        Check(TensorOps.Reshape, [Input("input", F32, 1, 8)], [8]);
        Check(TensorOps.Broadcast, [Input("input", F32, 1)], [2, 4, 4]);
        Check(TensorOps.Broadcast, [Input("input", F32, 2, 1, 4)], [2, 4, 4]);
        Check(TensorOps.BatchedMatVec,
            [Input("matrix", F32, 2, 4, 4), Input("vector", F32, 2, 4)], [2, 4]);
        Check(TensorOps.ReduceLastSum, [Input("input", F32, 2, 4, 4)], [2, 4]);
        Check(TensorOps.ReduceLastMean, [Input("input", F32, 2, 4)], [2]);
        Check(TensorOps.Reshape, [Input("input", F32, 2)], [2, 1]);
        Check(TensorOps.Broadcast, [Input("input", F32, 2, 1)], [2, 4]);
        Check(TensorOps.HeadOuter,
            [Input("left", F32, 2, 4), Input("right", F32, 2, 4)], [2, 4, 4]);
        Check(TensorOps.HeadOuter,
            [Input("left", F32, 2, 3), Input("right", F32, 2, 4)], [2, 3, 4]);
    }

    [Fact]
    public void RejectsInvalidFillAndSliceAttributes()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorFillValue(float.NaN).ToAttributes());
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorSlice(-1, 0, 2).ToAttributes());
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorSlice(0, 0, 0).ToAttributes());
        foreach (var text in new[] { "NaN", "Infinity", "1e999", "1,5", " 1", "" })
            Reject(TensorOps.Fill, [], [1], attributes: new Dictionary<string, string> { ["value"] = text });
        Reject(TensorOps.Fill, [], [1], attributes: new Dictionary<string, string>
        {
            ["value"] = "1", ["unrecognized"] = "2",
        });
        Reject(TensorOps.Fill, [], [1]);
        Reject(TensorOps.Slice, [Input("input", F32, 6, 8)], [1, 8],
            attributes: new TensorSlice(0, 6, 1).ToAttributes());
        Reject(TensorOps.Slice, [Input("input", F32, 6, 8)], [1, 8],
            attributes: new TensorSlice(2, 0, 1).ToAttributes());
        Reject(TensorOps.Slice, [Input("input", F32, 6, 8)], [1, 8],
            attributes: new TensorSlice(0, int.MaxValue, int.MaxValue).ToAttributes());
        Reject(TensorOps.Slice, [Input("input", F32, 6, 8)], [2, 8],
            attributes: new TensorSlice(0, 1, 1).ToAttributes());
        Reject(TensorOps.Slice, [Input("input", F32, 6, 8)], [8],
            attributes: new TensorSlice(0, 1, 1).ToAttributes());
        Reject(TensorOps.Slice, [Input("input", F32, 6, 8)], [1, 8],
            attributes: new Dictionary<string, string>
            {
                ["axis"] = "-1", ["start"] = "0", ["length"] = "1",
            });
        Reject(TensorOps.Slice, [Input("input", F32, 6, 8)], [1, 8],
            attributes: new Dictionary<string, string>
            {
                ["axis"] = "0", ["start"] = "0", ["length"] = "1", ["step"] = "1",
            });
    }

    [Fact]
    public void RejectsMismatchedTypesShapesPortsAndVersions()
    {
        Reject(TensorOps.CastFp16ToFp32, [Input("input", F32, 8)], [8]);
        Reject(TensorOps.CastFp16ToFp32, [Input("input", F16, 8)], [4]);
        Reject(TensorOps.Reshape, [Input("input", F32, 8)], [7]);
        Reject(TensorOps.Broadcast, [Input("input", F32, 2, 4)], [2, 3, 4]);
        Reject(TensorOps.Broadcast, [Input("input", F32, 3)], [4]);
        Reject(TensorOps.BatchedMatVec,
            [Input("matrix", F32, 2, 3, 4), Input("vector", F32, 2, 5)], [2, 3]);
        Reject(TensorOps.BatchedMatVec,
            [Input("matrix", F32, 2, 3, 4), Input("vector", F32, 2, 4)], [2, 4]);
        Reject(TensorOps.ReduceLastSum, [Input("input", F32, 2, 4, 4)], [2, 4, 1]);
        Reject(TensorOps.ReduceLastMean, [Input("input", F32, 4)], [1]);
        Reject(TensorOps.HeadOuter,
            [Input("left", F32, 2, 4), Input("right", F32, 3, 4)], [2, 4, 4]);
        Reject(TensorOps.HeadOuter,
            [Input("left", F32, 2, 4), Input("right", F32, 2, 4)], [2, 4, 3]);
        Reject(TensorOps.HeadOuter,
            [Input("lhs", F32, 2, 4), Input("right", F32, 2, 4)], [2, 4, 4]);
        Reject(new GraphOperationId(TensorOps.HeadOuter.Name, 2),
            [Input("left", F32, 2, 4), Input("right", F32, 2, 4)], [2, 4, 4]);
        Reject(TensorOps.HeadOuter,
            [Input("left", F16, 2, 4), Input("right", F16, 2, 4)], [2, 4, 4]);
        Reject(TensorOps.Fill, [], [1], attributes: new TensorFillValue(1).ToAttributes(),
            outputLayout: "strided");
    }

    [Fact]
    public void GraphValidationIsExplicitNotImplicit()
    {
        var builder = new LogicalGraphBuilder(
            new GraphIdentity("test", 1, "tensor-contract-test"),
            TestGraphSignatures.Create(32, 4, 1, 1, 4, "state"));
        builder.AddRegion("graph", GraphRegionTypes.Graph, "Graph");
        builder.AddResource("filled", "Filled", GraphResourceKind.Temporary,
            GraphResourceLifetime.Invocation, new TensorDescriptor(F32, [4]));
        builder.AddNode("fill", PortableTensorOperationContracts.Fill, "graph",
            [GraphBindings.Write("output", "filled")],
            attributes: new TensorFillValue(2f).ToAttributes());
        var graph = builder.Build();
        PortableTensorOperationContracts.ValidateGraph(graph);
        Assert.Equal([4], graph.Resources.Single().Tensor.Dimensions);

        var invalid = new LogicalGraph(graph.Identity, graph.Model, graph.Resources, graph.Regions,
            [graph.Nodes.Single() with { Attributes = new Dictionary<string, string> { ["value"] = "NaN" } }],
            graph.Inputs, graph.Outputs);
        Assert.Throws<InvalidDataException>(() => PortableTensorOperationContracts.ValidateGraph(invalid));
        var invalidPort = graph.Nodes.Single() with
        {
            Resources = [GraphBindings.Read("output", "filled")],
        };
        Assert.Throws<InvalidDataException>(() =>
            TensorOps.ValidateNode(invalidPort, graph.Resources.ToDictionary(resource => resource.Id)));
        var readOnly = graph.Resources.Single() with { Kind = GraphResourceKind.Input };
        Assert.Throws<InvalidDataException>(() =>
            TensorOps.ValidateNode(graph.Nodes.Single(),
                new Dictionary<ResourceId, GraphResource> { [readOnly.Id] = readOnly }));
    }

    [Fact]
    public void WkvMatrixShapeOperationsComposeWithoutAnOpaqueUpdateNode()
    {
        var builder = new LogicalGraphBuilder(new GraphIdentity("rwkv-7", 1, "tensor-shapes"),
            TestGraphSignatures.Create(32, 8, 1, 2, 4, "rwkv-7.state.fp32@1"));
        builder.AddRegion("graph", GraphRegionTypes.Graph, "WKV tensor shapes");
        void Resource(string name, GraphResourceKind kind, params int[] dimensions) =>
            builder.AddResource(name, name, kind,
                kind == GraphResourceKind.Input ? GraphResourceLifetime.External :
                    GraphResourceLifetime.Invocation,
                new TensorDescriptor(F32, dimensions), graphInput: kind == GraphResourceKind.Input);
        Resource("state", GraphResourceKind.Input, 2, 4, 4);
        Resource("key", GraphResourceKind.Input, 2, 4);
        Resource("value", GraphResourceKind.Input, 2, 4);
        Resource("decay", GraphResourceKind.Input, 2, 4);
        Resource("decay.axes", GraphResourceKind.Temporary, 2, 1, 4);
        Resource("decay.matrix", GraphResourceKind.Temporary, 2, 4, 4);
        Resource("state.projection", GraphResourceKind.Temporary, 2, 4);
        Resource("value-key", GraphResourceKind.Temporary, 2, 4, 4);
        Resource("value-key.rows", GraphResourceKind.Temporary, 2, 4);
        builder.AddNode("reshape-decay", TensorOps.Reshape, "graph",
            [GraphBindings.Read("input", "decay"), GraphBindings.Write("output", "decay.axes")]);
        builder.AddNode("broadcast-decay", TensorOps.Broadcast, "graph",
            [GraphBindings.Read("input", "decay.axes"), GraphBindings.Write("output", "decay.matrix")]);
        builder.AddNode("project-state", TensorOps.BatchedMatVec, "graph",
            [
                GraphBindings.Read("matrix", "state"), GraphBindings.Read("vector", "key"),
                GraphBindings.Write("output", "state.projection"),
            ]);
        builder.AddNode("value-key-outer", TensorOps.HeadOuter, "graph",
            [
                GraphBindings.Read("left", "value"), GraphBindings.Read("right", "key"),
                GraphBindings.Write("output", "value-key"),
            ]);
        builder.AddNode("reduce-outer", TensorOps.ReduceLastSum, "graph",
            [GraphBindings.Read("input", "value-key"), GraphBindings.Write("output", "value-key.rows")]);
        var graph = builder.Build();
        TensorOps.ValidateGraph(graph);
        Assert.DoesNotContain(graph.Nodes, node => node.Operation.Name.StartsWith("rwkv7.", StringComparison.Ordinal));
        Assert.Equal([2, 4, 4], graph.Resources.Single(resource =>
            resource.Id.Value == "decay.matrix").Tensor.Dimensions);
    }

    private static (string Port, GraphElementType Type, int[] Dimensions) Input(
        string port, GraphElementType type, params int[] dimensions) => (port, type, dimensions);

    private static void Reject(
        GraphOperationId operation,
        (string Port, GraphElementType Type, int[] Dimensions)[] inputs,
        int[] dimensions,
        IReadOnlyDictionary<string, string>? attributes = null,
        string outputLayout = "dense") =>
        Assert.Throws<InvalidDataException>(() => Check(operation, inputs, dimensions,
            attributes: attributes, outputLayout: outputLayout));

    private static void Check(
        GraphOperationId operation,
        (string Port, GraphElementType Type, int[] Dimensions)[] inputs,
        int[] dimensions,
        IReadOnlyDictionary<string, string>? attributes = null,
        string outputLayout = "dense")
    {
        var resources = inputs.Select((item, index) =>
            new GraphResource(new ResourceId($"input.{index}"), item.Port,
                GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(item.Type, item.Dimensions)))
            .Append(new GraphResource(new ResourceId("output"), "Output",
                GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(F32, dimensions, outputLayout)))
            .ToDictionary(resource => resource.Id);
        var bindings = inputs.Select((item, index) =>
            GraphBindings.Read(item.Port, $"input.{index}"))
            .Append(GraphBindings.Write("output", "output")).ToArray();
        var node = new LogicalNode(new LogicalNodeId("node"), operation, new RegionId("graph"),
            bindings, [], attributes ?? new Dictionary<string, string>(),
            new PrecisionRequirement(F32, F32));
        PortableTensorOperationContracts.ValidateNode(node, resources);
    }
}
