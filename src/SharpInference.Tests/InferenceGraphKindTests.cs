using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Logical;
using SharpInference.Graphs.Execution;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Tests;

public sealed class InferenceGraphKindTests
{
    private static readonly TensorDescriptor Vector = new(ElementType.Float32, [4]);
    private sealed record CustomNode(GraphNodeId Id, string NodeKind) : IInferenceGraphNode;
    private sealed record OtherNode(GraphNodeId Id, string NodeKind, int Value) : IInferenceGraphNode;
    private sealed record CustomEdge(GraphEdgeId Id, GraphNodeId Source, GraphNodeId Target, string EdgeKind) : IInferenceGraphEdge;

    [Fact]
    public void MinimalContractsSupportUserTypesAndOpenCategories()
    {
        IInferenceGraphNode[] nodes =
        [
            new CustomNode(new("a"), "user-stage"), new CustomNode(new("b"), "other-stage"),
            new OtherNode(new("c"), "user-stage", 42),
        ];
        IInferenceGraphEdge[] edges =
        [
            new CustomEdge(new("ab"), new("a"), new("b"), "user-link"),
            new CustomEdge(new("bc"), new("b"), new("c"), "other-link"),
        ];
        var graph = new Graph<IInferenceGraphNode, IInferenceGraphEdge>(nodes, edges);
        Assert.Equal(3, graph.Nodes.Count);
        Assert.Equal(["a", "c"], graph.Filter(node => node.NodeKind == "user-stage").Nodes.Select(node => node.Id.Value));
        Assert.Single(graph.Filter(node => node is OtherNode { Value: 42 }).Nodes);
        Assert.Equal("b", graph.GetNode(graph.EnumerateEdgesFromNode(new("a")).Single().Target).Id.Value);
        Assert.Single(graph.Filter(edge: edge => edge.EdgeKind == "user-link").Edges);
        Assert.Equal(["NodeKind"], typeof(IInferenceGraphNode).GetProperties().Select(property => property.Name));
        Assert.Equal(["EdgeKind"], typeof(IInferenceGraphEdge).GetProperties().Select(property => property.Name));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("resource")]
    [InlineData("call")]
    public void LogicalAndExecutionValidationRejectInvalidOperationCategories(string? category)
    {
        SharpInference.Graphs.Graph[] graphs = [new LogicalGraph { Id = "logical" }, new ExecutionGraph { Id = "execution" }];
        foreach (var graph in graphs)
        {
            graph.Operation("op", Operations.Copy).Category = category!;
            Assert.Contains(graph.Validate(), diagnostic => diagnostic.Code == "G0006" &&
                diagnostic.Node == "op" && diagnostic.Graph == graph.Id);
        }
    }

    [Fact]
    public void CustomCategoriesSurviveSerializationConversionAndBusinessQueries()
    {
        var graph = new LogicalGraph { Id = "main" };
        var input = graph.Resource("input", Vector, GraphIO.In);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        input.Category = "custom.input";
        output.Category = "custom.output";
        var operation = graph.Operation("copy", Operations.Copy);
        operation.Category = "custom.copy";
        operation.Bind("input", input, ResourceAccess.Read).Bind("output", output, ResourceAccess.Write);
        var loaded = GraphPackageXml.Deserialize<LogicalGraph>(GraphPackageXml.Serialize(new GraphPackage<LogicalGraph>(graph)));
        Assert.Same(loaded.MainGraph!.Actions[0], Assert.Single(loaded.MainGraph.Readers(loaded.MainGraph.Resources[0])));
        Assert.Same(loaded.MainGraph.Actions[0], Assert.Single(loaded.MainGraph.Writers(loaded.MainGraph.Resources[1])));
        var execution = new DefaultExecutionGraphConverter([new CpuFloat32InstructionCollection()])
            .Convert(loaded, new("test", "cpu-vm"));
        Assert.Equal("custom.copy", execution.MainGraph!.Actions[0].Category);
        Assert.Equal(["custom.input", "custom.output"], execution.MainGraph.Resources.Select(resource => resource.Category));
        Assert.Equal("custom.copy", Assert.Single(execution.MainGraph.DependencyGraph().Nodes).NodeKind);
        var xml = GraphPackageXml.Serialize(execution);
        Assert.Equal(xml, GraphPackageXml.Serialize(GraphPackageXml.Deserialize<ExecutionGraph>(xml)));
    }

    [Fact]
    public void UnknownActionTypesAreRejectedRatherThanInterpretedByTheirCategory()
    {
        var graph = new LogicalGraph { Id = "main" };
        graph.Actions.Add(new UnknownAction(graph));
        Assert.Contains(graph.Validate(), diagnostic => diagnostic.Code == "G0010");
        Assert.Throws<NotSupportedException>(() => GraphPackageXml.Serialize(new GraphPackage<LogicalGraph>(graph)));
    }

    [Fact]
    public void ReadWriteQueriesUseBindingsRatherThanCategoryNames()
    {
        var graph = new LogicalGraph { Id = "main" };
        var state = graph.Resource("state", Vector, GraphIO.InOut, isState: true);
        state.Category = "custom.state";
        var operation = graph.Operation("update", "custom.update");
        operation.Category = "custom.step";
        operation.Bind("state", state, ResourceAccess.ReadWrite);
        Assert.Same(operation, Assert.Single(graph.Readers(state)));
        Assert.Same(operation, Assert.Single(graph.Writers(state)));
        Assert.Empty(graph.Validate());
    }

    [Fact]
    public void InliningRetainsCustomResourceAndOperationCategories()
    {
        var child = new ExecutionGraph { Id = "child" };
        var scratch = child.Resource("scratch", Vector);
        scratch.Category = "custom.scratch";
        var operation = child.Operation("write", Operations.Copy);
        operation.Category = "custom.write";
        operation.Bind("output", scratch, ResourceAccess.Write);
        var main = new ExecutionGraph { Id = "main" };
        main.Call("invoke", child).Category = "custom.call";
        var inlined = ExecutionGraphRewriter.InlineGraphCall(new GraphPackage<ExecutionGraph>(main, child), "main", "invoke");
        Assert.Equal("custom.scratch", Assert.Single(inlined.MainGraph!.Resources).Category);
        Assert.Equal("custom.write", Assert.Single(inlined.MainGraph.Actions).Category);
        var diagnostics = inlined.Validate();
        Assert.DoesNotContain(diagnostics, diagnostic => diagnostic.Severity == GraphDiagnosticSeverity.Error);
        var unused = Assert.Single(diagnostics);
        Assert.Equal("P0002", unused.Code);
        Assert.Equal(GraphDiagnosticSeverity.Warning, unused.Severity);
        Assert.Equal("child", unused.Graph);
    }

    private sealed class UnknownAction(SharpInference.Graphs.Graph graph) : GraphAction(graph, "unknown")
    {
        public override string Category { get; set; } = "operation";
        protected override GraphAction CopyAction(SharpInference.Graphs.Graph target) => throw new NotSupportedException();
    }
}
