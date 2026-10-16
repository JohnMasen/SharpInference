using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Vm.Optimization;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;

namespace SharpInference.Tests;

public sealed class ExecutionGraphConverterTests
{
    private static readonly TensorDescriptor F = new(GraphElementType.Float32, [4]);

    [Fact]
    public void ConverterPreservesDefinitionsCallsAndConcreteOperations()
    {
        var child = CopyGraph("attention", "attention.copy");
        var root = LogicalGraphDefinition.Define("model", builder =>
        {
            var input = builder.AddInput("input", F);
            var output = builder.AddOutput("output", F);
            var childOutput = builder.AddResource("attention-output", "Attention output", F);
            builder.AddCall("attention-0", child, bindings =>
            {
                bindings.AddInput("input", input);
                bindings.AddOutput("output", childOutput);
            }, new GraphAnnotation("Attention 0", "attention-instance"));
            builder.AddNode("finish", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", childOutput);
                node.AddOutput("output", output);
            });
        }, annotation: new GraphAnnotation("Model", "model"));

        var program = new DefaultExecutionGraphConverter().Convert(root, Context());

        Assert.Equal(new ExecutionGraphId("model"), program.RootGraph);
        Assert.Equal(["model", "attention"], program.Graphs.Select(graph => graph.Id.Value));
        var model = program.Graphs.Single(graph => graph.Id.Value == "model");
        var call = Assert.IsType<ExecutionGraphCallNode>(model.Nodes[0]);
        Assert.Equal(new ExecutionGraphId("attention"), call.Target);
        Assert.Equal("attention-instance", call.Architecture!.Role);
        Assert.Equal(["attention-0"], call.Source.LogicalNodes.Select(node => node.Value));
        Assert.Equal(new GraphDefinitionId("model"), model.Source!.Definition);
        var operation = Assert.IsType<ExecutionOperationNode>(model.Nodes[1]);
        Assert.Equal(PrimitiveGraphOperations.Copy, operation.Operation);
        Assert.Equal(["finish"], operation.Source.LogicalNodes.Select(node => node.Value));
        var attention = program.Graphs.Single(graph => graph.Id.Value == "attention");
        Assert.Equal("attention", attention.Architecture!.Role);
        Assert.Equal("attention.copy",
            Assert.IsType<ExecutionOperationNode>(Assert.Single(attention.Nodes)).Operation.Name);
    }

    [Fact]
    public void ReusedDefinitionProducesOneExecutionGraph()
    {
        var child = CopyGraph("child");
        var root = LogicalGraphDefinition.Define("root", builder =>
        {
            var input = builder.AddInput("input", F);
            var output = builder.AddOutput("output", F);
            var firstOutput = builder.AddResource("first-output", "First output", F);
            var secondOutput = builder.AddResource("second-output", "Second output", F);
            builder.AddCall("first", child, bindings =>
            {
                bindings.AddInput("input", input);
                bindings.AddOutput("output", firstOutput);
            });
            builder.AddCall("second", child, bindings =>
            {
                bindings.AddInput("input", input);
                bindings.AddOutput("output", secondOutput);
            });
            builder.AddNode("finish", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", secondOutput);
                node.AddOutput("output", output);
            });
        });

        var program = new DefaultExecutionGraphConverter().Convert(root, Context());

        Assert.Equal(2, program.Graphs.Count);
        var calls = program.Graphs.Single(graph => graph.Id.Value == "root")
            .Nodes.OfType<ExecutionGraphCallNode>().ToArray();
        Assert.Equal(2, calls.Length);
        Assert.All(calls, call => Assert.Equal(new ExecutionGraphId("child"), call.Target));
        Assert.Equal(2, program.Callers(new("child")).Count);
        Assert.Equal(2, program.Callees(new("root")).Count);
        Assert.False(program.IsExclusiveCall(new("root"), new("first")));
    }

    [Fact]
    public void ProgramRejectsGraphCallCycles()
    {
        var graphA = Graph(
            "a",
            Call("call-b", "b"));
        var graphB = Graph(
            "b",
            Call("call-a", "a"));

        Assert.Throws<InvalidDataException>(() =>
            new ExecutionGraphProgram([graphA, graphB]));
    }

    [Fact]
    public void ProgramRejectsMultipleRoots()
    {
        Assert.Throws<InvalidDataException>(() =>
            new ExecutionGraphProgram([Graph("left"), Graph("right")]));
    }

    [Theory]
    [InlineData("missing-input")]
    [InlineData("missing-output")]
    [InlineData("unknown-port")]
    [InlineData("wrong-input-access")]
    [InlineData("wrong-output-access")]
    [InlineData("incompatible-tensor")]
    public void ProgramRejectsIncompatibleCallInterfaces(string scenario)
    {
        var child = Graph("child",
            [Resource("input", GraphDefinitionResourceKind.Input),
             Resource("output", GraphDefinitionResourceKind.Output)],
            Operation("copy", PrimitiveGraphOperations.Copy,
                [Binding("input", "input", GraphResourceAccess.Read),
                 Binding("output", "output", GraphResourceAccess.Write)]));
        var bindings = new List<ModelResourceBinding>();
        if (scenario != "missing-input")
        {
            bindings.Add(Binding("input", "input", scenario == "wrong-input-access"
                ? GraphResourceAccess.Write : GraphResourceAccess.Read));
        }

        if (scenario != "missing-output")
        {
            bindings.Add(Binding("output", "output", scenario == "wrong-output-access"
                ? GraphResourceAccess.Read : GraphResourceAccess.Write));
        }

        if (scenario == "unknown-port")
        {
            bindings.Add(Binding("unknown", "input", GraphResourceAccess.Read));
        }

        var input = scenario == "incompatible-tensor"
            ? new ExecutionGraphResource(new("input"), "input", GraphDefinitionResourceKind.Input,
                new(GraphElementType.Float32, [8]))
            : Resource("input", GraphDefinitionResourceKind.Input);
        var root = Graph("root",
            [input, Resource("output", GraphDefinitionResourceKind.Output)],
            Call("child-call", "child", bindings));

        Assert.Throws<InvalidDataException>(() => new ExecutionGraphProgram([root, child]));
    }

    [Fact]
    public void PostprocessorsReceiveProgramAndContextInOrder()
    {
        var logical = CopyGraph("root");
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
            [Graph("root", resources, copy, relu)]);

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
        var child = Graph(
            "child",
            [Resource("input", GraphDefinitionResourceKind.Input),
             Resource("output", GraphDefinitionResourceKind.Output)],
            Operation("copy", PrimitiveGraphOperations.Copy,
                [Binding("input", "input", GraphResourceAccess.Read),
                 Binding("output", "output", GraphResourceAccess.Write)]));
        var root = Graph(
            "root",
            [Resource("input", GraphDefinitionResourceKind.Input),
             Resource("temporary", GraphDefinitionResourceKind.Temporary),
             Resource("output", GraphDefinitionResourceKind.Output)],
            Operation("first", PrimitiveGraphOperations.Copy,
                [Binding("input", "input", GraphResourceAccess.Read),
                 Binding("output", "temporary", GraphResourceAccess.Write)]),
            Operation("second", PrimitiveGraphOperations.Relu,
                [Binding("input", "temporary", GraphResourceAccess.Read),
                 Binding("output", "output", GraphResourceAccess.Write)],
                [new ExecutionNodeId("first")]),
            Call(
                "child-call",
                "child",
                [Binding("input", "input", GraphResourceAccess.Read),
                 Binding("output", "output", GraphResourceAccess.Write)]));
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
    public void XmlPostprocessorReplacesProgramForMatchingCompilation()
    {
        var source = CopyGraph("source");
        var replacement = new ExecutionGraphProgram(
            [Graph(
                "source",
                [Resource("input", GraphDefinitionResourceKind.Input)],
                Operation(
                    "replacement",
                    PrimitiveGraphOperations.Copy,
                    [Binding("input", "input", GraphResourceAccess.Read)]),
                source: new(source.Id))]);
        var path = Path.Combine(
            Directory.GetCurrentDirectory(),
            $"sharp-inference-execution-{Guid.NewGuid():N}.xml");
        try
        {
            File.WriteAllText(path, ExecutionGraphXml.Serialize(replacement));
            var processor = new XmlExecutionGraphProgramPostprocessor(
                path, "test-model", "test-backend");
            var result = processor.Process(
                new([Graph("original")]),
                new(source, Context(), 0, processor.Name));

            Assert.Equal(
                "replacement",
                Assert.Single(Assert.Single(result.Graphs).Nodes).Id.Value);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void XmlPostprocessorRejectsMismatchedCompilation()
    {
        var processor = new XmlExecutionGraphProgramPostprocessor(
            "unused.xml", "rwkv-7", "cpu-vm");

        var exception = Assert.Throws<InvalidOperationException>(() =>
            processor.Process(
                new([Graph("original")]),
                new(
                    CopyGraph("source"),
                    new("rwkv-6", "cpu-vm"),
                    0,
                    processor.Name)));

        Assert.Contains("rwkv-7", exception.Message);
        Assert.Contains("rwkv-6", exception.Message);
    }

    [Fact]
    public void ExecutionProgramXmlRoundTripsGraphsAndArchitecture()
    {
        var child = Graph(
            "child",
            [Resource("input", GraphDefinitionResourceKind.Input),
             Resource("output", GraphDefinitionResourceKind.Output)],
            Operation("copy", PrimitiveGraphOperations.Copy,
                [Binding("input", "input", GraphResourceAccess.Read),
                 Binding("output", "output", GraphResourceAccess.Write)]),
            GraphCompileBoundary.Atomic,
            new("Attention", "attention", "Reusable attention body",
                new Dictionary<string, string> { ["layer"] = "1" }),
            new(new("logical-child")));
        var root = Graph(
            "root",
            [Resource("input", GraphDefinitionResourceKind.Input),
             Resource("output", GraphDefinitionResourceKind.Output)],
            Call(
                "attention-1",
                "child",
                [Binding("input", "input", GraphResourceAccess.Read),
                 Binding("output", "output", GraphResourceAccess.Write)],
                GraphCompileBoundary.Atomic,
                new([new("logical-call")], "preserve-call"),
                new("Layer 1", "decoder-layer")),
            source: new(new("logical-root")));
        var original = new ExecutionGraphProgram([root, child]);

        var xml = ExecutionGraphXml.Serialize(original);
        var restored = ExecutionGraphXml.DeserializeProgram(xml);

        Assert.Contains("version=\"3\"", xml, StringComparison.Ordinal);
        Assert.Contains("<Bindings", xml, StringComparison.Ordinal);
        Assert.Contains("<Dependencies", xml, StringComparison.Ordinal);
        Assert.Equal(new ExecutionGraphId("root"), restored.RootGraph);
        Assert.Equal(2, restored.Graphs.Count);
        var restoredChild = restored.Graphs.Single(graph => graph.Id.Value == "child");
        Assert.Equal("attention", restoredChild.Architecture!.Role);
        Assert.Equal("1", restoredChild.Architecture.Attributes!["layer"]);
        var restoredCall = Assert.IsType<ExecutionGraphCallNode>(
            Assert.Single(restored.Graphs.Single(graph => graph.Id.Value == "root").Nodes));
        Assert.Equal("preserve-call", restoredCall.Source.Transformation);
        Assert.Equal("decoder-layer", restoredCall.Architecture!.Role);
        Assert.Equal(GraphCompileBoundary.Atomic, restoredCall.CompileBoundary);
        Assert.Equal(GraphCompileBoundary.Atomic, restoredChild.DefaultCompileBoundary);
        Assert.Equal(new GraphDefinitionId("logical-root"),
            restored.Graphs.Single(graph => graph.Id.Value == "root").Source!.Definition);
        Assert.Equal(new GraphDefinitionId("logical-child"), restoredChild.Source!.Definition);
        Assert.Equal(["logical-call"], restoredCall.Source.LogicalNodes.Select(node => node.Value));
        Assert.Equal(xml, ExecutionGraphXml.Serialize(restored));
    }

    [Theory]
    [InlineData("1")]
    [InlineData("2")]
    public void ExecutionProgramXmlRejectsLegacyVersions(string version)
    {
        var xml = ExecutionGraphXml.Serialize(new ExecutionGraphProgram([Graph("root")]))
            .Replace("version=\"3\"", $"version=\"{version}\"", StringComparison.Ordinal);

        Assert.Throws<InvalidDataException>(() => ExecutionGraphXml.DeserializeProgram(xml));
    }

    private static ExecutionGraph Graph(string id, params NodeSpec[] nodes) =>
        Graph(id, [], nodes);

    private static ExecutionGraph Graph(
        string id,
        IEnumerable<ExecutionGraphResource> resources,
        params NodeSpec[] nodes) =>
        Graph(id, resources, nodes, GraphCompileBoundary.Auto, null, null);

    private static ExecutionGraph Graph(
        string id,
        IEnumerable<ExecutionGraphResource> resources,
        NodeSpec node,
        GraphCompileBoundary boundary = GraphCompileBoundary.Auto,
        GraphArchitecture? architecture = null,
        ExecutionGraphSource? source = null) =>
        Graph(id, resources, [node], boundary, architecture, source);

    private static ExecutionGraph Graph(
        string id,
        IEnumerable<ExecutionGraphResource> resources,
        IEnumerable<NodeSpec> nodes,
        GraphCompileBoundary boundary = GraphCompileBoundary.Auto,
        GraphArchitecture? architecture = null,
        ExecutionGraphSource? source = null)
    {
        var builder = new ExecutionGraphBuilder(
            new(id), boundary, architecture, source);
        foreach (var resource in resources)
        {
            builder.AddResource(resource);
        }

        foreach (var node in nodes)
        {
            if (node.Node is ExecutionOperationNode operation)
            {
                builder.AddOperation(operation, node.Bindings, node.Dependencies);
            }
            else
            {
                builder.AddCall(
                    (ExecutionGraphCallNode)node.Node,
                    node.Bindings,
                    node.Dependencies);
            }
        }

        return builder.Build();
    }

    private static ExecutionGraphResource Resource(
        string id,
        GraphDefinitionResourceKind kind) =>
        new(new ResourceId(id), id, kind, F);

    private static ModelResourceBinding Binding(
        string port,
        string resource,
        GraphResourceAccess access,
        string? node = null) =>
        new(
            port,
            new(GraphBindingOwner.Parent, new(resource)),
            access,
            $"{node ?? "binding"}:{port}:{resource}");

    private static NodeSpec Operation(
        string id,
        GraphOperationId operation,
        IReadOnlyList<ModelResourceBinding> bindings,
        IReadOnlyList<ExecutionNodeId>? dependencies = null) =>
        new(
            new ExecutionOperationNode(
                new(id),
                operation,
                new Dictionary<string, string>(),
                new PrecisionRequirement(
                    GraphElementType.Float32, GraphElementType.Float32),
                new ExecutionSourceMap([])),
            bindings.Select(binding => binding with
            {
                BindingId = $"{id}:{binding.Port}",
            }).ToArray(),
            dependencies ?? []);

    private static NodeSpec Call(
        string id,
        string target,
        IReadOnlyList<ModelResourceBinding>? bindings = null,
        GraphCompileBoundary boundary = GraphCompileBoundary.Auto,
        ExecutionSourceMap? source = null,
        GraphArchitecture? architecture = null) =>
        new(
            new ExecutionGraphCallNode(
                new(id),
                new(target),
                boundary,
                source ?? new ExecutionSourceMap([]),
                architecture),
            (bindings ?? []).Select(binding => binding with
            {
                BindingId = $"{id}:{binding.Port}",
            }).ToArray(),
            []);

    private sealed record NodeSpec(
        ExecutionGraphNode Node,
        IReadOnlyList<ModelResourceBinding> Bindings,
        IReadOnlyList<ExecutionNodeId> Dependencies);

    private static LogicalGraphDefinition CopyGraph(
        string id,
        string operation = "core.copy") =>
        LogicalGraphDefinition.Define(
            id,
            builder =>
            {
                var input = builder.AddInput("input", F);
                var output = builder.AddOutput("output", F);
                builder.AddNode("copy", new GraphOperationId(operation), node =>
                {
                    node.AddInput("input", input);
                    node.AddOutput("output", output);
                });
            },
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
                    new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(
                        graph.Structure.Nodes.Select(node =>
                            node is ExecutionOperationNode operation
                        ? operation with
                        {
                            Attributes = new Dictionary<string, string>(operation.Attributes)
                            {
                                [$"processed:{Name}"] = "true",
                            },
                        }
                        : node),
                        graph.Structure.Edges),
                    graph.DefaultCompileBoundary,
                    graph.Architecture,
                    graph.Source)));
        }
    }
}
