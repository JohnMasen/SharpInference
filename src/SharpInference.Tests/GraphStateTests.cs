using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class GraphStateTests
{
    private static LogicalGraph CreateGraph() =>
        new LogicalGraphBuilder(new GraphIdentity("rwkv6", 1, "state"),
            new GraphModelSignature(16, 8, 1, 2, 4, "rwkv6.state.v1"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("token", "token", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true)
            .AddResource("logits", "logits", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [16]), graphOutput: true)
            .AddResource("wkv", "wkv", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                new TensorDescriptor(GraphElementType.Float32, [2, 4, 4]))
            .AddStateSlot("state.0.wkv", "wkv")
            .Build();

    [Fact]
    public void Schema_IsNamedSeparatelyFromModelAbi()
    {
        var graph = CreateGraph();
        Assert.Equal("rwkv6.state.v1", graph.GraphState.Schema.Name);
        var named = new LogicalGraphBuilder(graph.Identity, graph.Model)
            .SetStateSchema(new StateSchema("RWKV6_State"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("wkv", "wkv", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                new TensorDescriptor(GraphElementType.Float32, [2, 4, 4]))
            .AddStateSlot("state.0.wkv", "wkv")
            .Build();
        Assert.Equal("RWKV6_State", named.GraphState.Schema.Name);
        Assert.Equal(graph.Model.StateAbiId, named.Model.StateAbiId);
        Assert.True(named.GraphState.Schema.IsCompatibleWith(new StateSchema("RWKV6_State")));
        Assert.False(named.GraphState.Schema.IsCompatibleWith(graph.GraphState.Schema));
    }

    [Fact]
    public void State_RoundTripsLogicalExecutionAndPreservesEndpoints()
    {
        var logical = CreateGraph();
        var execution = new GraphOptimizer().Optimize(logical);
        Assert.Equal(logical.GraphState.Schema, execution.GraphState.Schema);
        Assert.Equal(logical.GraphState.Entries, execution.GraphState.Entries);
        Assert.Equal([new ResourceId("token")], logical.Inputs);
        Assert.Equal([new ResourceId("logits")], logical.Outputs);
        Assert.Equal(logical.GraphState!.Schema,
            GraphXml.DeserializeLogical(GraphXml.Serialize(logical)).GraphState!.Schema);
        Assert.Equal(logical.GraphState.Slots,
            GraphJson.DeserializeLogical(GraphJson.Serialize(logical)).GraphState!.Slots);
        Assert.Equal(logical.GraphState.Schema,
            GraphXml.DeserializeExecution(GraphXml.Serialize(execution)).GraphState!.Schema);
        Assert.Equal(logical.GraphState.Slots,
            GraphJson.DeserializeExecution(GraphJson.Serialize(execution)).GraphState!.Slots);
        Assert.Equal(GraphXml.Serialize(execution),
            GraphXml.Serialize(GraphXml.DeserializeExecution(GraphXml.Serialize(execution))));
        Assert.Equal(GraphJson.Serialize(execution),
            GraphJson.Serialize(GraphJson.DeserializeExecution(GraphJson.Serialize(execution))));
    }

    [Fact]
    public void State_RejectsUnknownNonSessionOrMismatchedResources()
    {
        var graph = CreateGraph();
        LogicalGraph With(GraphState state) => new(graph.Identity, graph.Model, graph.Resources,
            graph.Regions, graph.Nodes, graph.Inputs, graph.Outputs, state);
        GraphState Rebind(string resource) => new(graph.GraphState!.Schema,
            [new GraphStateSlot("state.0.wkv", new ResourceId(resource))]);
        Assert.Throws<InvalidDataException>(() => With(Rebind("missing")));
        Assert.Throws<InvalidDataException>(() => With(Rebind("token")));
        Assert.Throws<InvalidDataException>(() => With(Rebind("logits")));
        Assert.Throws<ArgumentException>(() => new StateSchema(" "));
        Assert.Equal("other", With(new GraphState(new StateSchema("other"),
            graph.GraphState!.Slots)).GraphState.Schema.Name);
        Assert.Throws<InvalidDataException>(() => With(new GraphState(
            graph.GraphState!.Schema, [graph.GraphState.Slots[0], graph.GraphState.Slots[0]])));
    }

    [Fact]
    public void State_OmittedInLegacyXmlAndJson()
    {
        var graph = CreateGraph();
        var legacy = new LogicalGraph(graph.Identity, graph.Model, graph.Resources,
            graph.Regions, graph.Nodes, graph.Inputs, graph.Outputs);
        Assert.Empty(GraphXml.DeserializeLogical(GraphXml.Serialize(legacy)).GraphState.Entries);
        Assert.Empty(GraphJson.DeserializeLogical(GraphJson.Serialize(legacy)).GraphState.Entries);
        Assert.DoesNotContain("<GraphState", GraphXml.Serialize(legacy), StringComparison.Ordinal);
        Assert.DoesNotContain("\"GraphState\"", GraphJson.Serialize(legacy), StringComparison.Ordinal);
    }
}
