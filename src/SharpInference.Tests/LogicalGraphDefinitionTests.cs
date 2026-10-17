using SharpInference.Instructions;
using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

namespace SharpInference.Tests;

public sealed class LogicalGraphDefinitionTests
{
    private static readonly TensorDescriptor F = new(ElementType.Float32, [4]);

    [Fact]
    public void OperationDataFlowInfersDagAndCompletionSink()
    {
        var graph = LogicalGraphDefinition.Define(
            "unary",
            builder =>
            {
                var input = builder.AddInput("input", F);
                var output = builder.AddOutput("output", F);
                var temporary = builder.AddResource("temporary", "Temporary", F);
                var result = builder.AddNode("first", PrimitiveOperations.Copy, node =>
                {
                    node.AddInput("input", input);
                    return node.AddOutput("output", temporary);
                });
                Assert.Equal(temporary, result);
                builder.AddNode("second", PrimitiveOperations.Copy, node =>
                {
                    node.AddInput("input", temporary);
                    return node.AddOutput("output", output);
                });
            },
            GraphCompileBoundary.Atomic,
            new GraphAnnotation("Unary", "test", "A two-operation graph."));

        Assert.Equal(GraphCompileBoundary.Atomic, graph.CompileBoundary);
        Assert.Equal(["first"], graph.Dependencies(graph.Nodes[1]).Select(id => id.Value));
        Assert.Equal(["second"], graph.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["second"], graph.TerminalWriters[new("output")].Select(id => id.Value));
        Assert.Empty(graph.Diagnostics);
    }

    [Fact]
    public void GraphCallExplicitlyBindsInputsAndOutputs()
    {
        var child = CreateCopyGraph("copy");
        var parent = LogicalGraphDefinition.Define(
            "parent",
            builder =>
            {
                var input = builder.AddInput("input", F);
                var output = builder.AddOutput("output", F);
                var childOutput = builder.AddResource("child-output", "Child output", F);
                var call = builder.AddCall("child", child, bindings =>
                {
                    bindings.AddInput("input", input);
                    Assert.Equal(childOutput, bindings.AddOutput("output", childOutput));
                }, new GraphAnnotation("Child call"));
                Assert.Equal(new GraphDefinitionNodeId("child"), call);
                builder.AddNode("finish", PrimitiveOperations.Copy, node =>
                {
                    node.AddInput("input", childOutput);
                    node.AddOutput("output", output);
                });
            },
            GraphCompileBoundary.Transparent);

        var call = Assert.IsType<GraphCallNode>(parent.Nodes[0]);
        Assert.Equal("copy", call.Definition.Value);
        Assert.Equal("Child call", call.Annotation!.Name);
        Assert.Contains(parent.Bindings(call), binding =>
            binding.Port == "input" &&
            binding.Resource.Resource == new ResourceId("input"));
        Assert.Contains(parent.Bindings(call), binding =>
            binding.Port == "output" &&
            binding.Resource.Resource == new ResourceId("child-output"));
        Assert.Equal(["child"], parent.Dependencies(parent.Nodes[1]).Select(id => id.Value));
        Assert.Single(parent.Definitions);
    }

    [Fact]
    public void TransparentWrapperCanMapChildOutputDirectlyToParentOutput()
    {
        var child = CreateCopyGraph("copy");
        var parent = LogicalGraphDefinition.Define(
            "wrapper",
            builder =>
            {
                var input = builder.AddInput("input", F);
                var output = builder.AddOutput("output", F);
                builder.AddCall("child", child, bindings =>
                {
                    bindings.AddInput("input", input);
                    Assert.Equal(output, bindings.AddOutput("output", output));
                });
            },
            GraphCompileBoundary.Transparent);

        var call = Assert.IsType<GraphCallNode>(Assert.Single(parent.Nodes));
        Assert.Contains(parent.Bindings(call), binding =>
            binding.Port == "output" &&
            binding.Resource.Resource == parent.Resources.Single(resource =>
                resource.Kind == GraphDefinitionResourceKind.Output).Id);
        Assert.Equal(["child"], parent.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["child"], parent.TerminalWriters[new("output")].Select(id => id.Value));
    }

    [Fact]
    public void GraphCallSupportsExplicitInOutPort()
    {
        var stateful = LogicalGraphDefinition.Define("stateful", builder =>
        {
            var input = builder.AddInput("input", F);
            var state = builder.AddInOut("state", F);
            var output = builder.AddOutput("output", F);
            builder.AddNode("update", new OperationId("test.update"), node =>
            {
                node.AddInput("input", input);
                Assert.Equal(state, node.AddInOut("state", state));
                node.AddOutput("output", output);
            });
        });
        var parent = LogicalGraphDefinition.Define("parent", builder =>
        {
            var input = builder.AddInput("input", F);
            var state = builder.AddInOut("state", F);
            var output = builder.AddOutput("output", F);
            builder.AddCall("stateful", stateful, bindings =>
            {
                bindings.AddInput("input", input);
                Assert.Equal(state, bindings.AddInOut("state", state));
                bindings.AddOutput("output", output);
            });
        });

        var call = Assert.IsType<GraphCallNode>(Assert.Single(parent.Nodes));
        Assert.Equal(ResourceAccess.ReadWrite,
            parent.Bindings(call).Single(binding => binding.Port == "state").Access);
        Assert.Equal(["stateful"], parent.TerminalWriters[new("state")].Select(id => id.Value));
    }

    [Theory]
    [InlineData("input")]
    [InlineData("output")]
    public void MissingGraphCallPortIsRejected(string missingPort)
    {
        var child = CreateCopyGraph("copy");
        var builder = new GraphDefinitionBuilder("parent");
        var input = builder.AddInput("input", F);
        var output = builder.AddOutput("output", F);
        builder.AddCall("child", child, bindings =>
        {
            if (missingPort != "input")
            {
                bindings.AddInput("input", input);
            }

            if (missingPort != "output")
            {
                bindings.AddOutput("output", output);
            }
        });

        Assert.Contains(builder.Validate(), diagnostic =>
            diagnostic.Severity == GraphDiagnosticSeverity.Error);
        Assert.Throws<InvalidDataException>(() => builder.Build());
    }

    [Fact]
    public void OrderedOutputWritersHaveOneTerminalWriter()
    {
        var graph = LogicalGraphDefinition.Define("ordered", builder =>
        {
            var input = builder.AddInput("input", F);
            var output = builder.AddOutput("output", F);
            builder.AddNode("first", PrimitiveOperations.Copy, node =>
            {
                node.AddInput("input", input);
                node.AddOutput("output", output);
            });
            builder.AddNode("second", PrimitiveOperations.Copy, node =>
            {
                node.AddInput("input", input);
                node.AddOutput("output", output);
                node.DependsOn(new GraphDefinitionNodeId("first"));
            });
        });

        Assert.Equal(["second"], graph.TerminalWriters[new("output")].Select(id => id.Value));
        Assert.DoesNotContain(graph.Diagnostics, diagnostic => diagnostic.Code == "LGW002");
        var output = graph.Structure.NodesOfType<LogicalResourceNode>()
            .Single(node => node.Resource.Id == new ResourceId("output"));
        var conflict = Assert.Single(graph.Structure.ResourceConflicts(output));
        Assert.True(conflict.Ordered);
        Assert.Empty(graph.Structure.UnorderedResourceConflicts(output));
    }

    [Fact]
    public void ResourceConflictQueryFindsUnorderedReadWriteHazards()
    {
        var graph = LogicalGraphDefinition.Define("conflicts", builder =>
        {
            var input = builder.AddInput("input", F);
            var state = builder.AddInOut("state", F);
            var output = builder.AddOutput("output", F);
            builder.AddNode("first", new("test.update"), node =>
            {
                node.AddInput("input", input);
                node.AddInOut("state", state);
            });
            builder.AddNode("second", new("test.update"), node =>
            {
                node.AddInput("input", input);
                node.AddInOut("state", state);
                node.AddOutput("output", output);
            });
        });
        var state = graph.Structure.NodesOfType<LogicalResourceNode>()
            .Single(node =>
                node.ResourceNodeId.Owner.IsParent &&
                node.Resource.Id == new ResourceId("state"));

        var conflict = Assert.Single(graph.Structure.UnorderedResourceConflicts(state));
        var first = graph.Nodes.Single(node => node.Id.Value == "first");
        var boundary = graph.Structure.BoundaryResources([first.GraphId]);

        Assert.Equal("first", Assert.IsType<GraphOperationNode>(conflict.First).Id.Value);
        Assert.Equal("second", Assert.IsType<GraphOperationNode>(conflict.Second).Id.Value);
        Assert.True(conflict.Kind.HasFlag(ModelResourceConflictKind.ReadAfterWrite));
        Assert.True(conflict.Kind.HasFlag(ModelResourceConflictKind.WriteAfterRead));
        Assert.True(conflict.Kind.HasFlag(ModelResourceConflictKind.WriteAfterWrite));
        Assert.Equal(["first", "second"],
            graph.Structure.Readers(state).Cast<GraphOperationNode>().Select(node => node.Id.Value));
        Assert.Equal(["first", "second"],
            graph.Structure.Writers(state).Cast<GraphOperationNode>().Select(node => node.Id.Value));
        Assert.Equal(["input", "state"],
            boundary.Inputs.Cast<LogicalResourceNode>().Select(node => node.Resource.Id.Value));
        Assert.Equal(["state"],
            boundary.Outputs.Cast<LogicalResourceNode>().Select(node => node.Resource.Id.Value));
    }

    [Fact]
    public void UnorderedOutputWritersProduceWarningWithoutInventingDependency()
    {
        var graph = LogicalGraphDefinition.Define("unordered", builder =>
        {
            var input = builder.AddInput("input", F);
            var output = builder.AddOutput("output", F);
            foreach (var id in new[] { "left", "right" })
            {
                builder.AddNode(id, PrimitiveOperations.Copy, node =>
                {
                    node.AddInput("input", input);
                    node.AddOutput("output", output);
                });
            }
        });

        Assert.All(graph.Nodes, node => Assert.Empty(graph.Dependencies(node)));
        Assert.Equal(["left", "right"],
            graph.TerminalWriters[new("output")].Select(id => id.Value));
        Assert.Contains(graph.Diagnostics, diagnostic =>
            diagnostic.Code == "LGW002" &&
            diagnostic.Severity == GraphDiagnosticSeverity.Warning);
        Assert.Equal(graph.Diagnostics, graph.Validate());
    }

    [Fact]
    public void UnknownResourceIsRejectedInsteadOfResolvingAnotherDefinitionsPort()
    {
        var source = CreateCopyGraph("source");
        var builder = new GraphDefinitionBuilder("debug");
        builder.RegisterDefinition(source);
        var output = builder.AddOutput("output", F);
        builder.AddNode("capture", PrimitiveOperations.Copy, node =>
        {
            node.AddInput("input", new ResourceId("source/output"));
            node.AddOutput("output", output);
        });

        Assert.Contains(builder.Validate(), diagnostic =>
            diagnostic.Severity == GraphDiagnosticSeverity.Error);
        Assert.Throws<InvalidDataException>(() => builder.Build());
    }

    [Fact]
    public void BindingDefaultsToParentButCanReferenceCallInstanceResource()
    {
        var producer = LogicalGraphDefinition.Define("producer", builder =>
        {
            var input = builder.AddInput("input", F);
            var output = builder.AddOutput("output", F);
            var internalValue = builder.AddResource("internal", "Internal value", F);
            builder.AddNode("produce-internal", PrimitiveOperations.Copy, node =>
            {
                node.AddInput("input", input);
                node.AddOutput("output", internalValue);
            });
            builder.AddNode("produce-output", PrimitiveOperations.Copy, node =>
            {
                node.AddInput("input", internalValue);
                node.AddOutput("output", output);
            });
        });
        var consumer = CreateCopyGraph("consumer");

        var graph = LogicalGraphDefinition.Define("parent", builder =>
        {
            var input = builder.AddInput("input", F);
            var output = builder.AddOutput("output", F);
            var produced = builder.AddResource("produced", "Produced", F);
            builder.AddCall("producer-call", producer, bindings =>
            {
                bindings.AddInput("input", input);
                bindings.AddOutput("output", produced);
            });
            var internalValue = new ModelResourceNodeId(
                GraphBindingOwner.Instance(new GraphDefinitionNodeId("producer-call")),
                new ResourceId("internal"));
            builder.AddCall("consumer-call", consumer, bindings =>
            {
                bindings.AddInput("input", internalValue);
                bindings.AddOutput("output", output);
            });
        });

        var producerCall = Assert.IsType<GraphCallNode>(graph.Nodes[0]);
        var producerInput = graph.Bindings(producerCall).Single(binding => binding.Port == "input");
        Assert.True(producerInput.Resource.Owner.IsParent);
        Assert.Equal(new ResourceId("input"), producerInput.Resource.Resource);

        var consumerCall = Assert.IsType<GraphCallNode>(graph.Nodes[1]);
        var consumerInput = graph.Bindings(consumerCall).Single(binding => binding.Port == "input");
        Assert.Equal(new GraphDefinitionNodeId("producer-call"), consumerInput.Resource.Owner.Call);
        Assert.Equal(new ResourceId("internal"), consumerInput.Resource.Resource);
        Assert.Contains(new GraphDefinitionNodeId("producer-call"), graph.Dependencies(consumerCall));

        var xml = LogicalGraphXml.Serialize(graph);
        Assert.Contains("id=\"internal\"", xml, StringComparison.Ordinal);
        Assert.Contains("instance=\"producer-call\"", xml, StringComparison.Ordinal);
        var restored = LogicalGraphXml.DeserializeDefinition(xml);
        Assert.Equal(graph.Nodes.SelectMany(graph.Bindings), restored.Nodes.SelectMany(restored.Bindings));
        Assert.Equal(xml, LogicalGraphXml.Serialize(restored));
    }

    [Fact]
    public void WritingAnotherInstanceResourceIsAllowedWithWarning()
    {
        var producer = LogicalGraphDefinition.Define("producer", builder =>
        {
            var input = builder.AddInput("input", F);
            var output = builder.AddOutput("output", F);
            var internalValue = builder.AddResource("internal", "Internal value", F);
            builder.AddNode("produce", PrimitiveOperations.Copy, node =>
            {
                node.AddInput("input", input);
                node.AddOutput("internal", internalValue);
                node.AddOutput("output", output);
            });
        });
        var writer = CreateCopyGraph("writer");

        var graph = LogicalGraphDefinition.Define("parent", builder =>
        {
            var input = builder.AddInput("input", F);
            var output = builder.AddOutput("output", F);
            builder.AddCall("producer-call", producer, bindings =>
            {
                bindings.AddInput("input", input);
                bindings.AddOutput("output", output);
            });
            var internalValue = new ModelResourceNodeId(
                GraphBindingOwner.Instance(new GraphDefinitionNodeId("producer-call")),
                new ResourceId("internal"));
            builder.AddCall("writer-call", writer, bindings =>
            {
                bindings.AddInput("input", input);
                bindings.AddOutput("output", internalValue);
            });
        });

        var binding = graph.Bindings(graph.Nodes[1]).Single(value => value.Port == "output");
        Assert.Equal(ResourceAccess.Write, binding.Access);
        Assert.Equal(new GraphDefinitionNodeId("producer-call"), binding.Resource.Owner.Call);
        Assert.Contains(graph.Diagnostics, diagnostic =>
            diagnostic.Code == "LGW003" &&
            diagnostic.Severity == GraphDiagnosticSeverity.Warning &&
            diagnostic.Node == "writer-call" &&
            diagnostic.Resource == new ResourceId("internal"));
    }

    [Fact]
    public void XmlRoundTripsNestedDefinitionsAnnotationsAndBoundary()
    {
        var child = CreateCopyGraph("copy");
        var original = LogicalGraphDefinition.Define(
            "root",
            builder =>
            {
                var input = builder.AddInput("input", F);
                var output = builder.AddOutput("output", F);
                var childOutput = builder.AddResource("child-output", "Child output", F);
                builder.AddCall("child", child, bindings =>
                {
                    bindings.AddInput("input", input);
                    bindings.AddOutput("output", childOutput);
                });
                builder.AddNode("output", PrimitiveOperations.Copy, node =>
                {
                    node.AddInput("input", childOutput);
                    node.AddOutput("output", output);
                });
            },
            GraphCompileBoundary.Transparent,
            new GraphAnnotation(
                "Root architecture", "architecture", "A transparent graph.",
                new Dictionary<string, string> { ["group"] = "test" }));

        var xml = LogicalGraphXml.Serialize(original);
        var restored = LogicalGraphXml.DeserializeDefinition(xml);

        Assert.Contains("kind=\"LogicalGraphDefinition\"", xml, StringComparison.Ordinal);
        Assert.Contains("version=\"3\"", xml, StringComparison.Ordinal);
        Assert.Contains("<Bindings", xml, StringComparison.Ordinal);
        Assert.Contains("<Dependencies", xml, StringComparison.Ordinal);
        Assert.DoesNotContain("Region", xml, StringComparison.Ordinal);
        Assert.Equal(GraphCompileBoundary.Transparent, restored.CompileBoundary);
        Assert.Equal("Root architecture", restored.Annotation!.Name);
        Assert.Equal("test", restored.Annotation.Attributes!["group"]);
        Assert.Single(restored.Definitions);
        Assert.Equal(
            original.Nodes.Select(node =>
                (node.Id, original.Bindings(node).Count, original.Dependencies(node).Count)),
            restored.Nodes.Select(node =>
                (node.Id, restored.Bindings(node).Count, restored.Dependencies(node).Count)));
        Assert.Equal(original.Nodes.SelectMany(original.Bindings), restored.Nodes.SelectMany(restored.Bindings));
        Assert.Equal(xml, LogicalGraphXml.Serialize(restored));
    }

    [Theory]
    [InlineData("1")]
    [InlineData("2")]
    public void XmlRejectsLegacyVersions(string version)
    {
        var xml = LogicalGraphXml.Serialize(CreateCopyGraph("copy"))
            .Replace("version=\"3\"", $"version=\"{version}\"", StringComparison.Ordinal);

        Assert.Throws<InvalidDataException>(() => LogicalGraphXml.DeserializeDefinition(xml));
    }

    [Fact]
    public void CreateListGraphBuildsContinuousChain()
    {
        var graph = LogicalGraphDefinition.CreateListGraph(
            "list", F,
            [PrimitiveOperations.Copy, PrimitiveOperations.Relu, PrimitiveOperations.Tanh],
            GraphCompileBoundary.Atomic, new GraphAnnotation("List graph"));

        Assert.Equal(3, graph.Nodes.Count);
        Assert.Equal(GraphCompileBoundary.Atomic, graph.CompileBoundary);
        Assert.Equal("List graph", graph.Annotation!.Name);
        Assert.Equal(new ResourceId("input"),
            graph.Resources.Single(resource => resource.Kind == GraphDefinitionResourceKind.Input).Id);
        Assert.Equal(new ResourceId("output"),
            graph.Resources.Single(resource => resource.Kind == GraphDefinitionResourceKind.Output).Id);
        Assert.Equal(["step-0"], graph.Dependencies(graph.Nodes[1]).Select(id => id.Value));
        Assert.Equal(["step-1"], graph.Dependencies(graph.Nodes[2]).Select(id => id.Value));
        Assert.Equal(["step-2"], graph.CompletionSinks.Select(id => id.Value));
    }

    [Fact]
    public void MultipleOutputsRemainIndependentLogicalResources()
    {
        var graph = LogicalGraphDefinition.Define("branch", builder =>
        {
            var input = builder.AddInput("input", F);
            var left = builder.AddOutput("left", F);
            var right = builder.AddOutput("right", F);
            builder.AddNode("left", PrimitiveOperations.Copy, node =>
            {
                node.AddInput("input", input);
                node.AddOutput("output", left);
            });
            builder.AddNode("right", PrimitiveOperations.Copy, node =>
            {
                node.AddInput("input", input);
                node.AddOutput("output", right);
            });
        });

        Assert.Equal(["left", "right"], graph.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["left"], graph.TerminalWriters[new("left")].Select(id => id.Value));
        Assert.Equal(["right"], graph.TerminalWriters[new("right")].Select(id => id.Value));
    }

    [Fact]
    public void AttentionShapeSupportsForkMergeAndStateWithoutRegions()
    {
        var graph = LogicalGraphDefinition.Define(
            "attention",
            builder =>
            {
                var hiddenStates = builder.AddInput("hidden_states", F);
                var position = builder.AddInput("position", F);
                var keyCache = builder.AddInOut("key_cache", F);
                var valueCache = builder.AddInOut("value_cache", F);
                var output = builder.AddOutput("output", F);
                var query = builder.AddResource("query", "Query", F);
                var scores = builder.AddResource("scores", "Scores", F);
                builder.AddNode("qkv", new OperationId("attention.qkv"), node =>
                {
                    node.AddInput("hidden_states", hiddenStates);
                    node.AddOutput("query", query);
                    node.AddInOut("key_cache", keyCache);
                    node.AddInOut("value_cache", valueCache);
                });
                builder.AddNode("scores", new OperationId("attention.scores"), node =>
                {
                    node.AddInput("query", query);
                    node.AddInput("position", position);
                    node.AddInput("key_cache", keyCache);
                    node.AddOutput("scores", scores);
                });
                builder.AddNode("values", new OperationId("attention.values"), node =>
                {
                    node.AddInput("scores", scores);
                    node.AddInput("value_cache", valueCache);
                    node.AddOutput("output", output);
                });
            },
            GraphCompileBoundary.Atomic,
            new GraphAnnotation("Self Attention", "attention"));

        Assert.DoesNotContain(graph.Resources, resource =>
            resource.GetType().Name.Contains("Region", StringComparison.Ordinal));
        Assert.Equal(["qkv"], graph.Dependencies(graph.Nodes[1]).Select(id => id.Value));
        Assert.Equal(["qkv", "scores"],
            graph.Dependencies(graph.Nodes[2]).Select(id => id.Value).OrderBy(id => id));
        Assert.Equal(["values"], graph.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["qkv"], graph.TerminalWriters[new("key_cache")].Select(id => id.Value));
        Assert.Equal(["qkv"], graph.TerminalWriters[new("value_cache")].Select(id => id.Value));
        Assert.Equal(["values"], graph.TerminalWriters[new("output")].Select(id => id.Value));
    }

    private static LogicalGraphDefinition CreateCopyGraph(string id) =>
        LogicalGraphDefinition.Define(
            id,
            builder =>
            {
                var input = builder.AddInput("input", F);
                var output = builder.AddOutput("output", F);
                builder.AddNode("copy", PrimitiveOperations.Copy, node =>
                {
                    node.AddInput("input", input);
                    node.AddOutput("output", output);
                });
            },
            GraphCompileBoundary.Atomic,
            new GraphAnnotation("Copy graph", "operation"));
}
