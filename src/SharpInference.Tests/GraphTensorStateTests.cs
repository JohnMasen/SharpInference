using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class GraphTensorStateTests
{
    [Theory]
    [InlineData(GraphElementType.Byte, 1)]
    [InlineData(GraphElementType.Int32, 4)]
    [InlineData(GraphElementType.UInt32, 4)]
    [InlineData(GraphElementType.Float16, 2)]
    [InlineData(GraphElementType.Float32, 4)]
    public void TypedStatePreservesInitialValuesAndClonesIndependently(GraphElementType type, int elementSize)
    {
        var graph = CreateGraph(type);
        var initial = Enumerable.Range(1, elementSize * 3).Select(value => (byte)value).ToArray();
        var supplied = new Dictionary<string, ReadOnlyMemory<byte>> { ["counter"] = initial };
        var state = new GraphTensorState(graph, supplied);
        Assert.Equal("external-model", state.ArchitectureId);
        Assert.Equal("external.counter@3", state.StateAbiId);
        Assert.Equal(type, Assert.Single(state.Buffers).Tensor.ElementType);
        Assert.Equal(initial, state.Buffers[0].Data.ToArray());
        var expected = (byte[])initial.Clone();
        initial[0] = 99;
        Assert.Equal(expected, state.Buffers[0].Data.ToArray());
        var clone = state.Clone();
        clone.Buffers[0].Data.Span[0] = 42;
        Assert.Equal(expected, state.Buffers[0].Data.ToArray());
        clone.Reset();
        Assert.Equal(expected, clone.Buffers[0].Data.ToArray());
        state.Buffers[0].Data.Span.Clear();
        state.Reset();
        Assert.Equal(expected, state.Buffers[0].Data.ToArray());
    }

    [Fact]
    public void InitialStateRequiresExactCoverageAndByteLength()
    {
        var graph = CreateGraph(GraphElementType.Int32);
        Assert.Throws<InvalidDataException>(() => new GraphTensorState(graph,
            new Dictionary<string, ReadOnlyMemory<byte>>()));
        Assert.Throws<InvalidDataException>(() => new GraphTensorState(graph,
            new Dictionary<string, ReadOnlyMemory<byte>> { ["other"] = new byte[12] }));
        Assert.Throws<InvalidDataException>(() => new GraphTensorState(graph,
            new Dictionary<string, ReadOnlyMemory<byte>> { ["counter"] = new byte[3] }));
        Assert.Equal(new byte[12], new GraphTensorState(graph).Buffers[0].Data.ToArray());
    }

    [Fact]
    public void ModelNamesDoNotSelectSharedShapeValidation()
    {
        var graph = CreateGraph(GraphElementType.Int32, "rwkv");
        GraphValidator.Validate(graph);
        Assert.Equal(graph.Model, GraphXml.DeserializeLogical(GraphXml.Serialize(graph)).Model);
        Assert.Equal(graph.Model, GraphJson.DeserializeLogical(GraphJson.Serialize(graph)).Model);
        Assert.Equal(["counterWidth"], graph.Model.Dimensions.Keys);
    }

    private static LogicalGraph CreateGraph(GraphElementType type, string modelType = "counter") =>
        new LogicalGraphBuilder(new GraphIdentity("external-model", 1, "counter-state"),
                new GraphModelSignature(modelType, "external.counter@3",
                    new Dictionary<string, int> { ["counterWidth"] = 3 }))
            .AddRegion("graph", GraphRegionTypes.Graph, "Graph")
            .SetStateSchema(new StateSchema("Counter_State"))
            .AddResource("counter", "Counter", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                new TensorDescriptor(type, [3]))
            .AddStateSlot("counter", "counter")
            .Build();
}
