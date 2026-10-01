using SharpInference.Graphs;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class PortableGraphStateTests
{
    [Fact]
    public void State_IsAllocatedAndCopiedFromGraphSchema()
    {
        var state = new PortableGraphState(CreateGraph());
        Assert.Equal("test-model", state.ArchitectureId);
        Assert.Equal(4, state.ElementCount);
        Assert.Equal([1, 2, 2], Assert.Single(state.Views).Dimensions);
        Assert.Equal(0, state.Revision);
        state.Restore([1f, 2f, 3f, 4f]);
        Assert.Equal(1, state.Revision);
        var cloned = state.Clone();
        state.Reset();
        Assert.Equal(2, state.Revision);
        var copied = new float[4];
        cloned.CopyTo(copied);
        Assert.Equal([1f, 2f, 3f, 4f], copied);
        state.CopyTo(copied);
        Assert.Equal([0f, 0f, 0f, 0f], copied);
        Assert.Throws<ArgumentException>(() => state.Restore([1f]));
    }

    [Fact]
    public void State_SynchronizesDeviceChangesOnlyWhenRead()
    {
        var state = new PortableGraphState(CreateGraph());
        var reads = 0;
        state.AttachDeviceSynchronizer(views =>
        {
            reads++;
            views[0].Values[0] = 42f;
        });
        state.MarkDeviceModified();
        Assert.Equal(0, reads);
        var clone = state.Clone();
        Assert.Equal(1, reads);
        var copy = new float[4];
        clone.CopyTo(copy);
        Assert.Equal(42f, copy[0]);
        state.MarkDeviceModified();
        state.DetachDeviceSynchronizer();
        Assert.Equal(2, reads);
        Assert.Throws<InvalidOperationException>(() => state.MarkDeviceModified());
    }

    [Fact]
    public void GraphMetadata_DoesNotRequireAnArchitectureSpecificDetector()
    {
        var graph = CreateGraph();
        var logical = new LogicalGraph(
            graph.Identity, graph.Model, graph.Resources, graph.Regions,
            [], graph.Inputs, graph.Outputs, graph.GraphState);
        var metadata = new GraphArchitectureMetadataReader(logical).Read(new EmptyCatalog());
        Assert.Equal("test-model", metadata.ArchitectureId);
        Assert.Equal(4, metadata.VocabularySize);
    }

    [Fact]
    public void State_RejectsAnIncompleteSessionSchema()
    {
        var graph = CreateGraph();
        var missingSlot = new ExecutionGraph(graph.Identity, graph.Model,
            graph.Resources, graph.Regions, graph.Nodes, graph.Inputs, graph.Outputs);
        Assert.Throws<InvalidDataException>(() => new PortableGraphState(missingSlot));
    }

    private static ExecutionGraph CreateGraph()
    {
        var graph = new LogicalGraphBuilder(
                new GraphIdentity("test-model", 1, "test.forward"),
                new GraphModelSignature(4, 2, 1, 1, 2, "test-model.state.fp32@1"))
            .SetStateSchema(new StateSchema("Test_State"))
            .AddRegion("graph", GraphRegionTypes.Graph, "Graph")
            .AddResource("token", "Token", GraphResourceKind.Input,
                GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Int32, [1]),
                graphInput: true)
            .AddResource("logits", "Logits", GraphResourceKind.Output,
                GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Float32, [4]),
                graphOutput: true)
            .AddResource("state", "State", GraphResourceKind.SessionState,
                GraphResourceLifetime.Session, new TensorDescriptor(GraphElementType.Float32, [1, 2, 2]))
            .AddStateSlot("state", "state")
            .AddNode("output", PrimitiveGraphOperations.Copy, "graph",
                [GraphBindings.Read("input", "state"), GraphBindings.Write("output", "logits")])
            .Build();
        return new GraphOptimizer().Optimize(graph,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded));
    }

    private sealed class EmptyCatalog : IModelTensorCatalog
    {
        public int VocabularySize => 4;
        public int EmbeddingSize => 2;
        public int LayerCount => 1;
        public IReadOnlyCollection<string> Names => [];
        public bool TryGet(string name, out IModelTensor tensor)
        {
            tensor = null!;
            return false;
        }
        public IModelTensor GetRequired(string name) => throw new InvalidDataException(name);
    }
}
