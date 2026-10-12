using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class ExecutionGraphConverterTests
{
    private sealed class Tensor;
    private static readonly TensorDescriptor F = new(GraphElementType.Float32, [4]);

    private sealed class UnaryPorts : IGraphPortContract<UnaryPorts>
    {
        private UnaryPorts(IGraphPortFactory ports)
        {
            Input = ports.AddInput<Tensor>("input", F);
            Output = ports.AddOutput<Tensor>("output", F);
        }

        public GraphInput<Tensor> Input { get; }
        public GraphOutput<Tensor> Output { get; }
        public static UnaryPorts Create(IGraphPortFactory ports) => new(ports);
    }

    [Fact]
    public void ConverterPreservesDefinitionsCallsAndConcreteOperations()
    {
        var child = CopyGraph("attention", "attention.copy");
        var root = LogicalGraphDefinition.Define<UnaryPorts>("model", builder =>
        {
            var call = builder.AddNode("attention-0", child, bindings =>
                bindings.Bind(bindings.Ports.Input, builder.Ports.Input),
                new GraphAnnotation("Attention 0", "attention-instance"));
            builder.AddNode("finish", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", call.Output);
                node.AddOutput("output", builder.Ports.Output);
            });
        }, annotation: new GraphAnnotation("Model", "model")).Definition;

        var program = new DefaultExecutionGraphConverter().Convert(root, Context());

        Assert.Equal(new ExecutionGraphId("model"), program.RootGraph);
        Assert.Equal(["model", "attention"], program.Graphs.Select(graph => graph.Id.Value));
        var model = program.Graphs.Single(graph => graph.Id.Value == "model");
        var call = Assert.IsType<ExecutionGraphCallNode>(model.Nodes[0]);
        Assert.Equal(new ExecutionGraphId("attention"), call.Target);
        Assert.Equal("attention-instance", call.Architecture!.Role);
        var operation = Assert.IsType<ExecutionOperationNode>(model.Nodes[1]);
        Assert.Equal(PrimitiveGraphOperations.Copy, operation.Operation);
        var attention = program.Graphs.Single(graph => graph.Id.Value == "attention");
        Assert.Equal("attention", attention.Architecture!.Role);
        Assert.Equal("attention.copy",
            Assert.IsType<ExecutionOperationNode>(Assert.Single(attention.Nodes)).Operation.Name);
    }

    [Fact]
    public void ReusedDefinitionProducesOneExecutionGraph()
    {
        var child = CopyGraph("child");
        var root = LogicalGraphDefinition.Define<UnaryPorts>("root", builder =>
        {
            var first = builder.AddNode("first", child, bindings =>
                bindings.Bind(bindings.Ports.Input, builder.Ports.Input));
            var second = builder.AddNode("second", child, bindings =>
                bindings.Bind(bindings.Ports.Input, builder.Ports.Input));
            builder.AddNode("finish", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", second.Output);
                node.AddOutput("output", builder.Ports.Output);
            });
        }).Definition;

        var program = new DefaultExecutionGraphConverter().Convert(root, Context());

        Assert.Equal(2, program.Graphs.Count);
        var calls = program.Graphs.Single(graph => graph.Id.Value == "root")
            .Nodes.OfType<ExecutionGraphCallNode>().ToArray();
        Assert.Equal(2, calls.Length);
        Assert.All(calls, call => Assert.Equal(new ExecutionGraphId("child"), call.Target));
    }

    [Fact]
    public void ProgramRejectsGraphCallCycles()
    {
        var graphA = Graph(
            "a",
            new ExecutionGraphCallNode(
                new("call-b"), new("b"), [], [], GraphCompileBoundary.Auto,
                new ExecutionSourceMap([])));
        var graphB = Graph(
            "b",
            new ExecutionGraphCallNode(
                new("call-a"), new("a"), [], [], GraphCompileBoundary.Auto,
                new ExecutionSourceMap([])));

        Assert.Throws<InvalidDataException>(() =>
            new ExecutionGraphProgram([graphA, graphB]));
    }

    [Fact]
    public void ProgramRejectsMultipleRoots()
    {
        Assert.Throws<InvalidDataException>(() =>
            new ExecutionGraphProgram([Graph("left"), Graph("right")]));
    }

    [Fact]
    public void PostprocessorsReceiveProgramAndContextInOrder()
    {
        var logical = CopyGraph("root").Definition;
        var calls = new List<string>();
        var context = new ExecutionGraphCompilationContext(
            "phi4", "d3d12",
            new Dictionary<string, string> { ["layers"] = "32" },
            new Dictionary<string, string> { ["shaderModel"] = "6.8" });
        var program = new ExecutionGraphCompiler(
            [new RecordingProcessor("first", calls), new RecordingProcessor("second", calls)])
            .Compile(logical, context);

        Assert.Equal(["first:0:phi4:d3d12", "second:1:phi4:d3d12"], calls);
        var operation = Assert.IsType<ExecutionOperationNode>(
            Assert.Single(program.Graphs.Single().Nodes));
        Assert.Equal("true", operation.Attributes["processed:first"]);
        Assert.Equal("true", operation.Attributes["processed:second"]);
    }

    [Fact]
    public void RewriterExtractsAndInlinesOperationGraph()
    {
        var resources = new[]
        {
            Resource("input", GraphDefinitionResourceKind.Input),
            Resource("temporary", GraphDefinitionResourceKind.Temporary),
            Resource("output", GraphDefinitionResourceKind.Output),
        };
        var copy = Operation("copy", PrimitiveGraphOperations.Copy,
            [Binding("input", "input", GraphResourceAccess.Read),
             Binding("output", "temporary", GraphResourceAccess.Write)]);
        var relu = Operation("relu", PrimitiveGraphOperations.Relu,
            [Binding("input", "temporary", GraphResourceAccess.Read),
             Binding("output", "output", GraphResourceAccess.Write)],
            [new ExecutionNodeId("copy")]);
        var program = new ExecutionGraphProgram(
            [new ExecutionGraph(new("root"), resources, [copy, relu])]);

        var extracted = ExecutionGraphRewriter.ExtractNodesToGraph(
            program,
            new("root"),
            [new ExecutionNodeId("copy"), new ExecutionNodeId("relu")],
            new("kernel"),
            new("kernel-call"),
            GraphCompileBoundary.Atomic,
            new GraphArchitecture("Kernel", "backend-kernel"));

        Assert.Equal(2, extracted.Graphs.Count);
        Assert.IsType<ExecutionGraphCallNode>(
            Assert.Single(extracted.Graphs.Single(graph => graph.Id.Value == "root").Nodes));
        Assert.Equal(2,
            extracted.Graphs.Single(graph => graph.Id.Value == "kernel").Nodes.Count);

        var inlined = ExecutionGraphRewriter.InlineGraphCall(
            extracted,
            new("root"),
            new("kernel-call"));

        Assert.Single(inlined.Graphs);
        Assert.Equal(
            ["kernel-call/copy", "kernel-call/relu"],
            inlined.Graphs.Single().Nodes.Select(node => node.Id.Value));
    }

    [Fact]
    public void RewriterFusesNodesAndReplacesGraphCall()
    {
        var child = new ExecutionGraph(
            new("child"),
            [Resource("input", GraphDefinitionResourceKind.Input),
             Resource("output", GraphDefinitionResourceKind.Output)],
            [Operation("copy", PrimitiveGraphOperations.Copy,
                [Binding("input", "input", GraphResourceAccess.Read),
                 Binding("output", "output", GraphResourceAccess.Write)])]);
        var root = new ExecutionGraph(
            new("root"),
            [Resource("input", GraphDefinitionResourceKind.Input),
             Resource("temporary", GraphDefinitionResourceKind.Temporary),
             Resource("output", GraphDefinitionResourceKind.Output)],
            [Operation("first", PrimitiveGraphOperations.Copy,
                [Binding("input", "input", GraphResourceAccess.Read),
                 Binding("output", "temporary", GraphResourceAccess.Write)]),
             Operation("second", PrimitiveGraphOperations.Relu,
                [Binding("input", "temporary", GraphResourceAccess.Read),
                 Binding("output", "output", GraphResourceAccess.Write)],
                [new ExecutionNodeId("first")]),
             new ExecutionGraphCallNode(
                 new("child-call"), new("child"),
                 [Binding("input", "input", GraphResourceAccess.Read),
                  Binding("output", "output", GraphResourceAccess.Write)],
                 [], GraphCompileBoundary.Auto, new ExecutionSourceMap([]))]);
        var program = new ExecutionGraphProgram([root, child]);

        var fused = ExecutionGraphRewriter.FuseNodesToOperation(
            program,
            new("root"),
            [new ExecutionNodeId("first"), new ExecutionNodeId("second")],
            new("fused"),
            new("test.fused"),
            [Binding("input", "input", GraphResourceAccess.Read),
             Binding("output", "output", GraphResourceAccess.Write)]);
        Assert.Contains(fused.Graphs.Single(graph => graph.Id.Value == "root").Nodes,
            node => node is ExecutionOperationNode operation &&
                operation.Operation.Name == "test.fused");

        var replaced = ExecutionGraphRewriter.ReplaceGraphCallWithOperation(
            fused,
            new("root"),
            new("child-call"),
            new("test.specialized-child"));
        Assert.DoesNotContain(
            replaced.Graphs.Single(graph => graph.Id.Value == "root").Nodes,
            node => node is ExecutionGraphCallNode);
    }

    [Fact]
    public void ExecutionProgramXmlRoundTripsGraphsAndArchitecture()
    {
        var child = new ExecutionGraph(
            new("child"),
            [Resource("input", GraphDefinitionResourceKind.Input),
             Resource("output", GraphDefinitionResourceKind.Output)],
            [Operation("copy", PrimitiveGraphOperations.Copy,
                [Binding("input", "input", GraphResourceAccess.Read),
                 Binding("output", "output", GraphResourceAccess.Write)])],
            GraphCompileBoundary.Atomic,
            new("Attention", "attention", "Reusable attention body",
                new Dictionary<string, string> { ["layer"] = "1" }),
            new(new("logical-child")));
        var root = new ExecutionGraph(
            new("root"),
            [Resource("input", GraphDefinitionResourceKind.Input),
             Resource("output", GraphDefinitionResourceKind.Output)],
            [new ExecutionGraphCallNode(
                new("attention-1"),
                new("child"),
                [Binding("input", "input", GraphResourceAccess.Read),
                 Binding("output", "output", GraphResourceAccess.Write)],
                [],
                GraphCompileBoundary.Atomic,
                new([new("logical-call")], "preserve-call"),
                new("Layer 1", "decoder-layer"))],
            source: new(new("logical-root")));
        var original = new ExecutionGraphProgram([root, child]);

        var xml = GraphXml.Serialize(original);
        var restored = GraphXml.DeserializeProgram(xml);

        Assert.Equal(new ExecutionGraphId("root"), restored.RootGraph);
        Assert.Equal(2, restored.Graphs.Count);
        var restoredChild = restored.Graphs.Single(graph => graph.Id.Value == "child");
        Assert.Equal("attention", restoredChild.Architecture!.Role);
        Assert.Equal("1", restoredChild.Architecture.Attributes!["layer"]);
        var restoredCall = Assert.IsType<ExecutionGraphCallNode>(
            Assert.Single(restored.Graphs.Single(graph => graph.Id.Value == "root").Nodes));
        Assert.Equal("preserve-call", restoredCall.Source.Transformation);
        Assert.Equal("decoder-layer", restoredCall.Architecture!.Role);
    }

    private static ExecutionGraph Graph(string id, params ExecutionGraphNode[] nodes) =>
        new(new ExecutionGraphId(id), [], nodes);

    private static ExecutionGraphResource Resource(
        string id,
        GraphDefinitionResourceKind kind) =>
        new(new ResourceId(id), id, kind, F);

    private static ExecutionGraphBinding Binding(
        string port,
        string resource,
        GraphResourceAccess access) =>
        new(port, new ResourceId(resource), access);

    private static ExecutionOperationNode Operation(
        string id,
        GraphOperationId operation,
        IReadOnlyList<ExecutionGraphBinding> bindings,
        IReadOnlyList<ExecutionNodeId>? dependencies = null) =>
        new(
            new ExecutionNodeId(id),
            operation,
            bindings,
            dependencies ?? [],
            new Dictionary<string, string>(),
            new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32),
            new ExecutionSourceMap([]));

    private static GraphDefinition<UnaryPorts> CopyGraph(
        string id,
        string operation = "core.copy") =>
        LogicalGraphDefinition.Define<UnaryPorts>(
            id,
            builder => builder.AddNode("copy", new GraphOperationId(operation), node =>
            {
                node.AddInput("input", builder.Ports.Input);
                node.AddOutput("output", builder.Ports.Output);
            }),
            annotation: new GraphAnnotation(id, id == "attention" ? "attention" : null));

    private static ExecutionGraphCompilationContext Context() =>
        new("test-model", "test-backend");

    private sealed class RecordingProcessor(
        string name,
        ICollection<string> calls) : IExecutionGraphPostprocessor
    {
        public string Name => name;

        public ExecutionGraphProgram Process(
            ExecutionGraphProgram program,
            ExecutionGraphPostprocessContext context)
        {
            calls.Add($"{Name}:{context.ProcessorIndex}:{context.Compilation.ModelType}:{context.Compilation.Backend}");
            return new ExecutionGraphProgram(program.Graphs.Select(graph =>
                new ExecutionGraph(
                    graph.Id,
                    graph.Resources,
                    graph.Nodes.Select(node => node is ExecutionOperationNode operation
                        ? operation with
                        {
                            Attributes = new Dictionary<string, string>(operation.Attributes)
                            {
                                [$"processed:{Name}"] = "true",
                            },
                        }
                        : node),
                    graph.DefaultCompileBoundary,
                    graph.Architecture,
                    graph.Source)));
        }
    }
}
