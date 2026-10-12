using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class LogicalGraphDefinitionTests
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

    private sealed class BranchPorts : IGraphPortContract<BranchPorts>
    {
        private BranchPorts(IGraphPortFactory ports)
        {
            Input = ports.AddInput<Tensor>("input", F);
            Left = ports.AddOutput<Tensor>("left", F);
            Right = ports.AddOutput<Tensor>("right", F);
        }

        public GraphInput<Tensor> Input { get; }
        public GraphOutput<Tensor> Left { get; }
        public GraphOutput<Tensor> Right { get; }

        public static BranchPorts Create(IGraphPortFactory ports) => new(ports);
    }

    private sealed class StatefulPorts : IGraphPortContract<StatefulPorts>
    {
        private StatefulPorts(IGraphPortFactory ports)
        {
            Input = ports.AddInput<Tensor>("input", F);
            State = ports.AddInOut<Tensor>("state", F);
            Output = ports.AddOutput<Tensor>("output", F);
        }

        public GraphInput<Tensor> Input { get; }
        public GraphInOut<Tensor> State { get; }
        public GraphOutput<Tensor> Output { get; }

        public static StatefulPorts Create(IGraphPortFactory ports) => new(ports);
    }

    private sealed class AttentionPorts : IGraphPortContract<AttentionPorts>
    {
        private AttentionPorts(IGraphPortFactory ports)
        {
            HiddenStates = ports.AddInput<Tensor>("hidden_states", F);
            Position = ports.AddInput<Tensor>("position", F);
            KeyCache = ports.AddInOut<Tensor>("key_cache", F);
            ValueCache = ports.AddInOut<Tensor>("value_cache", F);
            Output = ports.AddOutput<Tensor>("output", F);
        }

        public GraphInput<Tensor> HiddenStates { get; }
        public GraphInput<Tensor> Position { get; }
        public GraphInOut<Tensor> KeyCache { get; }
        public GraphInOut<Tensor> ValueCache { get; }
        public GraphOutput<Tensor> Output { get; }

        public static AttentionPorts Create(IGraphPortFactory ports) => new(ports);
    }

    [Fact]
    public void OperationDataFlowInfersDagAndCompletionSink()
    {
        var graph = LogicalGraphDefinition.Define<UnaryPorts>(
            "unary",
            builder =>
            {
                var temporary = builder.AddResource<Tensor>("temporary", "Temporary", F);
                builder.AddNode("first", PrimitiveGraphOperations.Copy, node =>
                {
                    node.AddInput("input", builder.Ports.Input);
                    return node.AddOutput("output", temporary);
                });
                builder.AddNode("second", PrimitiveGraphOperations.Copy, node =>
                {
                    node.AddInput("input", temporary);
                    return node.AddOutput("output", builder.Ports.Output);
                });
            },
            GraphCompileBoundary.Atomic,
            new GraphAnnotation("Unary", "test", "A two-operation graph.")).Definition;

        Assert.Equal(GraphCompileBoundary.Atomic, graph.CompileBoundary);
        Assert.Equal(["first"], graph.Nodes[1].Dependencies.Select(id => id.Value));
        Assert.Equal(["second"], graph.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["second"], graph.TerminalWriters[new("output")].Select(id => id.Value));
        Assert.Empty(graph.Diagnostics);
    }

    [Fact]
    public void GraphCallBindsInputsAndReturnsStronglyTypedOutputs()
    {
        var child = CreateCopyGraph("copy");
        var parent = LogicalGraphDefinition.Define<UnaryPorts>(
            "parent",
            builder =>
            {
                var called = builder.AddNode("child", child, bindings =>
                    bindings.Bind(bindings.Ports.Input, builder.Ports.Input),
                    new GraphAnnotation("Child call"));
                builder.AddNode("finish", PrimitiveGraphOperations.Copy, node =>
                {
                    node.AddInput("input", called.Output);
                    return node.AddOutput("output", builder.Ports.Output);
                });
            },
            GraphCompileBoundary.Transparent).Definition;

        var call = Assert.IsType<GraphCallNode>(parent.Nodes[0]);
        Assert.Equal("copy", call.Definition.Value);
        Assert.Contains(call.Bindings, binding =>
            binding.Port == "input" && binding.Resource == new ResourceId("input"));
        Assert.Contains(call.Bindings, binding =>
            binding.Port == "output" && binding.Resource == new ResourceId("child/output"));
        Assert.Equal(["child"], parent.Nodes[1].Dependencies.Select(id => id.Value));
        Assert.Single(parent.Definitions);
    }

    [Fact]
    public void TransparentWrapperCanMapChildOutputDirectlyToParentOutput()
    {
        var child = CreateCopyGraph("copy");
        var parent = LogicalGraphDefinition.Define<UnaryPorts>(
            "wrapper",
            builder =>
            {
                var called = builder.AddNode("child", child, bindings =>
                {
                    bindings.Bind(bindings.Ports.Input, builder.Ports.Input);
                    bindings.Bind(bindings.Ports.Output, builder.Ports.Output);
                });
                Assert.Equal(builder.Ports.Output.Id, called.Output.Id);
            },
            GraphCompileBoundary.Transparent).Definition;

        var call = Assert.IsType<GraphCallNode>(Assert.Single(parent.Nodes));
        Assert.Contains(call.Bindings, binding =>
            binding.Port == "output" &&
            binding.Resource == parent.Resources.Single(resource =>
                resource.Kind == GraphDefinitionResourceKind.Output).Id);
        Assert.Equal(["child"], parent.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["child"], parent.TerminalWriters[new("output")].Select(id => id.Value));
    }

    [Fact]
    public void GraphCallSupportsStronglyTypedInOutPort()
    {
        var stateful = LogicalGraphDefinition.Define<StatefulPorts>("stateful", builder =>
        {
            builder.AddNode("update", new GraphOperationId("test.update"), node =>
            {
                node.AddInput("input", builder.Ports.Input);
                node.AddInOut("state", builder.Ports.State);
                node.AddOutput("output", builder.Ports.Output);
            });
        });
        var parent = LogicalGraphDefinition.Define<StatefulPorts>("parent", builder =>
        {
            var called = builder.AddNode("stateful", stateful, bindings =>
            {
                bindings.Bind(bindings.Ports.Input, builder.Ports.Input);
                bindings.Bind(bindings.Ports.State, builder.Ports.State);
                bindings.Bind(bindings.Ports.Output, builder.Ports.Output);
            });
            Assert.Equal(builder.Ports.State.Id, called.State.Id);
        }).Definition;

        var call = Assert.IsType<GraphCallNode>(Assert.Single(parent.Nodes));
        Assert.Equal(GraphResourceAccess.ReadWrite,
            call.Bindings.Single(binding => binding.Port == "state").Access);
        Assert.Equal(["stateful"], parent.TerminalWriters[new("state")].Select(id => id.Value));
    }

    [Fact]
    public void MissingGraphCallInputIsRejected()
    {
        var child = CreateCopyGraph("copy");
        Assert.Throws<InvalidOperationException>(() =>
            LogicalGraphDefinition.Define<UnaryPorts>("parent", builder =>
                builder.AddNode("child", child, _ => { })));
    }

    [Fact]
    public void OrderedOutputWritersHaveOneTerminalWriter()
    {
        var graph = LogicalGraphDefinition.Define<UnaryPorts>("ordered", builder =>
        {
            builder.AddNode("first", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", builder.Ports.Input);
                node.AddOutput("output", builder.Ports.Output);
            });
            builder.AddNode("second", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", builder.Ports.Input);
                node.AddOutput("output", builder.Ports.Output);
                node.DependsOn(new GraphDefinitionNodeId("first"));
            });
        }).Definition;

        Assert.Equal(["second"], graph.TerminalWriters[new("output")].Select(id => id.Value));
        Assert.DoesNotContain(graph.Diagnostics, diagnostic => diagnostic.Code == "LGW002");
    }

    [Fact]
    public void UnorderedOutputWritersProduceWarningWithoutInventingDependency()
    {
        var graph = LogicalGraphDefinition.Define<UnaryPorts>("unordered", builder =>
        {
            builder.AddNode("left", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", builder.Ports.Input);
                node.AddOutput("output", builder.Ports.Output);
            });
            builder.AddNode("right", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", builder.Ports.Input);
                node.AddOutput("output", builder.Ports.Output);
            });
        }).Definition;

        Assert.All(graph.Nodes, node => Assert.Empty(node.Dependencies));
        Assert.Equal(["left", "right"],
            graph.TerminalWriters[new("output")].Select(id => id.Value));
        Assert.Contains(graph.Diagnostics, diagnostic =>
            diagnostic.Code == "LGW002" &&
            diagnostic.Severity == LogicalGraphDiagnosticSeverity.Warning);
    }

    [Fact]
    public void OperationCannotDirectlyCaptureAnotherDefinitionResource()
    {
        var source = CreateCopyGraph("source");
        Assert.Throws<ArgumentException>(() =>
            LogicalGraphDefinition.Define<UnaryPorts>("debug", builder =>
            {
                builder.AddNode("capture", PrimitiveGraphOperations.Copy, node =>
                {
                    node.AddInput("input", source.Ports.Output);
                    node.AddOutput("output", builder.Ports.Output);
                });
            }));
    }

    [Fact]
    public void BindingDefaultsToParentButCanReferenceCallInstanceResource()
    {
        var producer = LogicalGraphDefinition.Define<UnaryPorts>("producer", builder =>
        {
            var internalValue = builder.AddResource<Tensor>("internal", "Internal value", F);
            builder.AddNode("produce-internal", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", builder.Ports.Input);
                node.AddOutput("output", internalValue);
            });
            builder.AddNode("produce-output", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", internalValue);
                node.AddOutput("output", builder.Ports.Output);
            });
        });
        var consumer = CreateCopyGraph("consumer");

        var graph = LogicalGraphDefinition.Define<UnaryPorts>("parent", builder =>
        {
            var producerInstance = builder.AddCall("producer-call", producer, bindings =>
                bindings.Bind(bindings.Ports.Input, builder.Ports.Input));
            var internalValue = producerInstance.Resource<Tensor>("internal");
            builder.AddCall("consumer-call", consumer, bindings =>
            {
                bindings.Bind(bindings.Ports.Input, internalValue, producerInstance);
                bindings.Bind(bindings.Ports.Output, builder.Ports.Output);
            });
        }).Definition;

        var producerCall = Assert.IsType<GraphCallNode>(graph.Nodes[0]);
        var producerInput = producerCall.Bindings.Single(binding => binding.Port == "input");
        Assert.True(producerInput.ResourceOwner.IsParent);
        Assert.Equal(new ResourceId("input"), producerInput.Resource);

        var consumerCall = Assert.IsType<GraphCallNode>(graph.Nodes[1]);
        var consumerInput = consumerCall.Bindings.Single(binding => binding.Port == "input");
        Assert.Equal(new GraphDefinitionNodeId("producer-call"), consumerInput.ResourceOwner.Call);
        Assert.Equal(new ResourceId("internal"), consumerInput.Resource);
        Assert.Contains(new GraphDefinitionNodeId("producer-call"), consumerCall.Dependencies);

        var xml = GraphXml.Serialize(graph);
        Assert.Contains("resource=\"internal\"", xml, StringComparison.Ordinal);
        Assert.Contains("instance=\"producer-call\"", xml, StringComparison.Ordinal);
        var restored = GraphXml.DeserializeDefinition(xml);
        Assert.Equal(
            graph.Nodes.SelectMany(node => node.Bindings),
            restored.Nodes.SelectMany(node => node.Bindings));
        Assert.Equal(xml, GraphXml.Serialize(restored));
    }

    [Fact]
    public void WritingAnotherInstanceResourceIsAllowedWithWarning()
    {
        var producer = LogicalGraphDefinition.Define<UnaryPorts>("producer", builder =>
        {
            var internalValue = builder.AddResource<Tensor>("internal", "Internal value", F);
            builder.AddNode("produce", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", builder.Ports.Input);
                node.AddOutput("internal", internalValue);
                node.AddOutput("output", builder.Ports.Output);
            });
        });
        var writer = CreateCopyGraph("writer");

        var graph = LogicalGraphDefinition.Define<UnaryPorts>("parent", builder =>
        {
            var producerInstance = builder.AddCall("producer-call", producer, bindings =>
            {
                bindings.Bind(bindings.Ports.Input, builder.Ports.Input);
                bindings.Bind(bindings.Ports.Output, builder.Ports.Output);
            });
            var internalValue = producerInstance.Resource<Tensor>("internal");
            builder.AddCall("writer-call", writer, bindings =>
            {
                bindings.Bind(bindings.Ports.Input, builder.Ports.Input);
                bindings.Bind(bindings.Ports.Output, internalValue, producerInstance);
            });
        }).Definition;

        var binding = graph.Nodes[1].Bindings.Single(value => value.Port == "output");
        Assert.Equal(GraphResourceAccess.Write, binding.Access);
        Assert.Equal(new GraphDefinitionNodeId("producer-call"), binding.ResourceOwner.Call);
        Assert.Contains(graph.Diagnostics, diagnostic =>
            diagnostic.Code == "LGW003" &&
            diagnostic.Severity == LogicalGraphDiagnosticSeverity.Warning &&
            diagnostic.Node == new GraphDefinitionNodeId("writer-call") &&
            diagnostic.Resource == new ResourceId("internal"));
    }

    [Fact]
    public void XmlRoundTripsNestedDefinitionsAnnotationsAndBoundary()
    {
        var child = CreateCopyGraph("copy");
        var original = LogicalGraphDefinition.Define<UnaryPorts>(
            "root",
            builder =>
            {
                var called = builder.AddNode("child", child, bindings =>
                    bindings.Bind(bindings.Ports.Input, builder.Ports.Input));
                builder.AddNode("output", PrimitiveGraphOperations.Copy, node =>
                {
                    node.AddInput("input", called.Output);
                    node.AddOutput("output", builder.Ports.Output);
                });
            },
            GraphCompileBoundary.Transparent,
            new GraphAnnotation(
                "Root architecture",
                "architecture",
                "A transparent graph.",
                new Dictionary<string, string> { ["group"] = "test" })).Definition;

        var xml = GraphXml.Serialize(original);
        var restored = GraphXml.DeserializeDefinition(xml);

        Assert.Contains("kind=\"LogicalGraphDefinition\"", xml, StringComparison.Ordinal);
        Assert.DoesNotContain("Region", xml, StringComparison.Ordinal);
        Assert.Equal(GraphCompileBoundary.Transparent, restored.CompileBoundary);
        Assert.Equal("Root architecture", restored.Annotation!.Name);
        Assert.Equal("test", restored.Annotation.Attributes!["group"]);
        Assert.Single(restored.Definitions);
        Assert.Equal(original.Nodes.Select(node => (node.Id, node.Bindings.Count, node.Dependencies.Count)),
            restored.Nodes.Select(node => (node.Id, node.Bindings.Count, node.Dependencies.Count)));
        Assert.Equal(original.Nodes.SelectMany(node => node.Bindings),
            restored.Nodes.SelectMany(node => node.Bindings));
        Assert.Equal(xml, GraphXml.Serialize(restored));
    }

    [Fact]
    public void CreateListGraphBuildsContinuousChain()
    {
        var graph = LogicalGraphDefinition.CreateListGraph<UnaryPorts>(
            "list",
            [
                PrimitiveGraphOperations.Copy,
                PrimitiveGraphOperations.Relu,
                PrimitiveGraphOperations.Tanh,
            ]).Definition;

        Assert.Equal(3, graph.Nodes.Count);
        Assert.Equal(["step-0"], graph.Nodes[1].Dependencies.Select(id => id.Value));
        Assert.Equal(["step-1"], graph.Nodes[2].Dependencies.Select(id => id.Value));
        Assert.Equal(["step-2"], graph.CompletionSinks.Select(id => id.Value));
    }

    [Fact]
    public void MultipleOutputsRemainIndependentLogicalResources()
    {
        var graph = LogicalGraphDefinition.Define<BranchPorts>("branch", builder =>
        {
            builder.AddNode("left", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", builder.Ports.Input);
                node.AddOutput("output", builder.Ports.Left);
            });
            builder.AddNode("right", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", builder.Ports.Input);
                node.AddOutput("output", builder.Ports.Right);
            });
        }).Definition;

        Assert.Equal(["left", "right"], graph.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["left"], graph.TerminalWriters[new("left")].Select(id => id.Value));
        Assert.Equal(["right"], graph.TerminalWriters[new("right")].Select(id => id.Value));
    }

    [Fact]
    public void AttentionShapeSupportsForkMergeAndStateWithoutRegions()
    {
        var graph = LogicalGraphDefinition.Define<AttentionPorts>(
            "attention",
            builder =>
            {
                var query = builder.AddResource<Tensor>("query", "Query", F);
                var scores = builder.AddResource<Tensor>("scores", "Scores", F);
                builder.AddNode("qkv", new GraphOperationId("attention.qkv"), node =>
                {
                    node.AddInput("hidden_states", builder.Ports.HiddenStates);
                    node.AddOutput("query", query);
                    node.AddInOut("key_cache", builder.Ports.KeyCache);
                    node.AddInOut("value_cache", builder.Ports.ValueCache);
                });
                builder.AddNode("scores", new GraphOperationId("attention.scores"), node =>
                {
                    node.AddInput("query", query);
                    node.AddInput("position", builder.Ports.Position);
                    node.AddInput("key_cache", builder.Ports.KeyCache);
                    node.AddOutput("scores", scores);
                });
                builder.AddNode("values", new GraphOperationId("attention.values"), node =>
                {
                    node.AddInput("scores", scores);
                    node.AddInput("value_cache", builder.Ports.ValueCache);
                    node.AddOutput("output", builder.Ports.Output);
                });
            },
            GraphCompileBoundary.Atomic,
            new GraphAnnotation("Self Attention", "attention")).Definition;

        Assert.DoesNotContain(graph.Resources, resource =>
            resource.GetType().Name.Contains("Region", StringComparison.Ordinal));
        Assert.Equal(["qkv"], graph.Nodes[1].Dependencies.Select(id => id.Value));
        Assert.Equal(["qkv", "scores"],
            graph.Nodes[2].Dependencies.Select(id => id.Value).OrderBy(id => id));
        Assert.Equal(["values"], graph.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["qkv"], graph.TerminalWriters[new("key_cache")].Select(id => id.Value));
        Assert.Equal(["qkv"], graph.TerminalWriters[new("value_cache")].Select(id => id.Value));
        Assert.Equal(["values"], graph.TerminalWriters[new("output")].Select(id => id.Value));
    }

    private static GraphDefinition<UnaryPorts> CreateCopyGraph(string id) =>
        LogicalGraphDefinition.Define<UnaryPorts>(
            id,
            builder => builder.AddNode("copy", PrimitiveGraphOperations.Copy, node =>
            {
                node.AddInput("input", builder.Ports.Input);
                node.AddOutput("output", builder.Ports.Output);
            }),
            GraphCompileBoundary.Atomic,
            new GraphAnnotation("Copy graph", "operation"));
}
