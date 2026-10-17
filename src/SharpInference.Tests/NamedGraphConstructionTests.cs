using SharpInference.Instructions;
using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

namespace SharpInference.Tests;

public sealed class NamedGraphConstructionTests
{
    private static readonly TensorDescriptor F = new(ElementType.Float32, [4]);

    [Fact]
    public void ForwardResourceAndDependencyReferencesResolveAtBuild()
    {
        var builder = new GraphDefinitionBuilder("forward");
        builder.AddNode("consumer", PrimitiveOperations.Copy, node =>
        {
            node.AddInput("input", new ResourceId("temporary"));
            node.AddOutput("output", new ResourceId("output"));
            node.DependsOn(new GraphDefinitionNodeId("producer"));
        });
        builder.AddNode("producer", PrimitiveOperations.Copy, node =>
        {
            node.AddInput("input", new ResourceId("input"));
            node.AddOutput("output", new ResourceId("temporary"));
        });
        builder.AddOutput("output", F);
        builder.AddResource("temporary", "Temporary", F);
        builder.AddInput("input", F);

        Assert.Empty(builder.Validate());
        var graph = builder.Build();

        Assert.Empty(graph.Validate());
        Assert.Equal(["consumer", "producer"], graph.Nodes.Select(node => node.Id.Value));
        Assert.Equal(["producer"], graph.Dependencies(graph.Nodes[0]).Select(id => id.Value));
        Assert.Equal(["consumer"], graph.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["consumer"], graph.TerminalWriters[new("output")].Select(id => id.Value));
        Assert.Empty(builder.Validate());
        var rebuilt = builder.Build();
        Assert.Equal(graph.Structure.Edges, rebuilt.Structure.Edges);
        Assert.Equal(LogicalGraphXml.Serialize(graph), LogicalGraphXml.Serialize(rebuilt));
    }

    [Fact]
    public void ForwardCallTargetCanBeRegisteredAfterItsReferences()
    {
        var builder = new GraphDefinitionBuilder("parent");
        builder.AddCall("child-call", new GraphDefinitionId("child"), bind =>
        {
            bind.AddInput("input", new ResourceId("input"));
            bind.AddOutput("output", new ResourceId("output"));
        });
        builder.AddInput("input", F);
        builder.AddOutput("output", F);
        Assert.Contains(builder.Validate(), diagnostic =>
            diagnostic.Severity == GraphDiagnosticSeverity.Error);
        builder.RegisterDefinition(CopyGraph("child"));

        Assert.Empty(builder.Validate());
        var graph = builder.Build();
        var call = Assert.IsType<GraphCallNode>(Assert.Single(graph.Nodes));
        Assert.Equal(new GraphDefinitionId("child"), call.Definition);
        Assert.Single(graph.Definitions);
        Assert.Equal(["child-call"], graph.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["child-call"], graph.TerminalWriters[new("output")].Select(id => id.Value));
        Assert.Equal(2, graph.Bindings(call).Count);
    }

    [Fact]
    public void ValidateAndBuildCanBeRepeatedWithoutChangingDiagnosticsOrEdges()
    {
        var builder = new GraphDefinitionBuilder("repeat");
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

        var firstDiagnostics = builder.Validate();
        Assert.Contains(firstDiagnostics, diagnostic =>
            diagnostic.Code == "LGW002" && diagnostic.Severity == GraphDiagnosticSeverity.Warning);
        Assert.Equal(firstDiagnostics, builder.Validate());
        var first = builder.Build();
        var second = builder.Build();

        Assert.Equal(firstDiagnostics, first.Diagnostics);
        Assert.Equal(first.Diagnostics, first.Validate());
        Assert.Equal(first.Validate(), second.Validate());
        Assert.Equal(first.Structure.Edges, second.Structure.Edges);
        Assert.Equal(LogicalGraphXml.Serialize(first), LogicalGraphXml.Serialize(second));
        Assert.All(second.Nodes, node => Assert.Empty(second.Dependencies(node)));
        Assert.Equal(["left", "right"],
            second.TerminalWriters[output].Select(id => id.Value));
    }

    [Fact]
    public void BuilderChangesDoNotMutatePreviouslyBuiltDefinitions()
    {
        var builder = new GraphDefinitionBuilder("snapshots");
        var input = builder.AddInput("input", F);
        var output = builder.AddOutput("output", F);
        builder.AddNode("first", PrimitiveOperations.Copy, node =>
        {
            node.AddInput("input", input);
            node.AddOutput("output", output);
            node.AddAttribute("mode", "first");
            node.RequirePrecision(new(ElementType.Float16, ElementType.Float32));
        });
        var first = builder.Build();
        var firstXml = LogicalGraphXml.Serialize(first);
        builder.AddResource("later", "Later temporary", F);
        builder.AddNode("second", PrimitiveOperations.Copy, node =>
        {
            node.AddInput("input", input);
            node.AddOutput("output", output);
            node.DependsOn(new GraphDefinitionNodeId("first"));
        });
        var second = builder.Build();

        Assert.Equal(firstXml, LogicalGraphXml.Serialize(first));
        Assert.Single(first.Nodes);
        Assert.Equal(2, first.Resources.Count);
        Assert.Equal(["first"], first.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["first"], first.TerminalWriters[output].Select(id => id.Value));
        Assert.Equal(2, second.Nodes.Count);
        Assert.Equal(3, second.Resources.Count);
        Assert.Equal(["second"], second.CompletionSinks.Select(id => id.Value));
        Assert.Equal(["second"], second.TerminalWriters[output].Select(id => id.Value));
        var operation = Assert.IsType<GraphOperationNode>(first.Nodes[0]);
        Assert.Equal("first", operation.Attributes["mode"]);
        Assert.Equal(new PrecisionRequirement(ElementType.Float16, ElementType.Float32),
            operation.Requirements);
        Assert.Empty(first.Validate());
        Assert.Empty(second.Validate());
    }

    [Fact]
    public void BuilderCanRecoverAfterFailedBuildByDeclaringMissingResource()
    {
        var builder = new GraphDefinitionBuilder("repair");
        builder.AddOutput("output", F);
        builder.AddNode("copy", PrimitiveOperations.Copy, node =>
        {
            node.AddInput("input", new ResourceId("input"));
            node.AddOutput("output", new ResourceId("output"));
        });

        Assert.Contains(builder.Validate(), diagnostic =>
            diagnostic.Severity == GraphDiagnosticSeverity.Error &&
            diagnostic.Message.Contains("input", StringComparison.Ordinal));
        Assert.Throws<InvalidDataException>(() => builder.Build());
        builder.AddInput("input", F);

        Assert.Empty(builder.Validate());
        Assert.Empty(builder.Build().Validate());
    }

    [Fact]
    public void ValidateCollectsIndependentResourceDependencyAndTargetErrors()
    {
        var builder = new GraphDefinitionBuilder("invalid");
        var output = builder.AddOutput("output", F);
        builder.AddNode("bad-resource", PrimitiveOperations.Copy, node =>
        {
            node.AddInput("input", new ResourceId("missing-resource"));
            node.AddOutput("output", output);
            node.DependsOn(new GraphDefinitionNodeId("missing-dependency"));
        });
        builder.AddCall("bad-call", new GraphDefinitionId("missing-definition"), _ => { });
        builder.AddNode("bad-scope", PrimitiveOperations.Copy, node =>
        {
            node.AddInput("input", new ModelResourceNodeId(
                GraphBindingOwner.Instance(new GraphDefinitionNodeId("missing-call")),
                new ResourceId("missing-internal")));
            node.AddOutput("output", output);
        });

        var diagnostics = builder.Validate();
        var errors = diagnostics.Where(diagnostic =>
            diagnostic.Severity == GraphDiagnosticSeverity.Error).ToArray();
        Assert.True(errors.Length >= 4, string.Join(Environment.NewLine, errors.Select(error => error.Message)));
        Assert.Contains(errors, diagnostic => diagnostic.Message.Contains("missing-resource", StringComparison.Ordinal));
        Assert.Contains(errors, diagnostic => diagnostic.Message.Contains("missing-dependency", StringComparison.Ordinal));
        Assert.Contains(errors, diagnostic => diagnostic.Message.Contains("missing-definition", StringComparison.Ordinal));
        Assert.Contains(errors, diagnostic => diagnostic.Message.Contains("missing-internal", StringComparison.Ordinal));
        Assert.Equal(diagnostics, builder.Validate());
        Assert.Throws<InvalidDataException>(() => builder.Build());
    }

    [Theory]
    [InlineData(GraphDefinitionResourceKind.Input)]
    [InlineData(GraphDefinitionResourceKind.Output)]
    [InlineData(GraphDefinitionResourceKind.InOut)]
    [InlineData(GraphDefinitionResourceKind.Temporary)]
    [InlineData(GraphDefinitionResourceKind.State)]
    [InlineData(GraphDefinitionResourceKind.Weight)]
    [InlineData(GraphDefinitionResourceKind.Constant)]
    public void AddResourceSupportsEveryKind(GraphDefinitionResourceKind kind)
    {
        var builder = new GraphDefinitionBuilder("resources");
        var id = builder.AddResource("value", "Display name", F, kind, "external-value");
        if (kind is GraphDefinitionResourceKind.Output or GraphDefinitionResourceKind.InOut)
        {
            builder.AddNode("initialize", new OperationId("test.initialize"), node =>
            {
                if (kind == GraphDefinitionResourceKind.InOut)
                {
                    node.AddInOut("value", id);
                }
                else
                {
                    node.AddOutput("value", id);
                }
            });
        }
        var graph = builder.Build();
        var resource = Assert.Single(graph.Resources);

        Assert.Equal(new ResourceId("value"), id);
        Assert.Equal(id, resource.Id);
        Assert.Equal("Display name", resource.Name);
        Assert.Equal(F, resource.Tensor);
        Assert.Equal(kind, resource.Kind);
        Assert.Equal("external-value", resource.BindingKey);
        Assert.DoesNotContain(graph.Validate(), diagnostic => diagnostic.Severity == GraphDiagnosticSeverity.Error);
    }

    [Fact]
    public void ResourceIdsUseValueSemanticsAndDoNotResolveDisplayNames()
    {
        var graph = LogicalGraphDefinition.Define("names", builder =>
        {
            builder.AddResource("input", "Same display name", F, GraphDefinitionResourceKind.Input);
            builder.AddResource("output", "Same display name", F, GraphDefinitionResourceKind.Output);
            builder.AddNode("copy", PrimitiveOperations.Copy, node =>
            {
                node.AddInput("input", new ResourceId("input"));
                node.AddOutput("output", new ResourceId("output"));
            });
        });
        Assert.Empty(graph.Validate());
        Assert.Equal(new ResourceId("input"),
            graph.Bindings(graph.Nodes[0]).Single(binding => binding.Port == "input").Resource.Resource);

        var invalid = new GraphDefinitionBuilder("display-is-not-id");
        invalid.AddResource("actual", "display", F, GraphDefinitionResourceKind.Input);
        var output = invalid.AddOutput("output", F);
        invalid.AddNode("copy", PrimitiveOperations.Copy, node =>
        {
            node.AddInput("input", new ResourceId("display"));
            node.AddOutput("output", output);
        });
        Assert.Contains(invalid.Validate(), diagnostic =>
            diagnostic.Severity == GraphDiagnosticSeverity.Error &&
            diagnostic.Message.Contains("display", StringComparison.Ordinal));
        Assert.Throws<InvalidDataException>(() => invalid.Build());
    }

    [Fact]
    public void ParentAndReusedCallInstancesResolveSameResourceIdInTheirOwnScope()
    {
        var child = CopyGraph("child", includeInternal: true);
        var graph = LogicalGraphDefinition.Define("scopes", builder =>
        {
            var input = builder.AddInput("input", F);
            var parentValue = builder.AddResource("internal", "Internal", F, GraphDefinitionResourceKind.Input);
            foreach (var id in new[] { "first", "second" })
            {
                var childOutput = builder.AddResource($"{id}-output", "Child output", F);
                builder.AddCall(id, child, bind =>
                {
                    bind.AddInput("input", input);
                    bind.AddOutput("output", childOutput);
                });
                var output = builder.AddOutput($"{id}-result", F);
                builder.AddNode($"read-{id}", PrimitiveOperations.Copy, node =>
                {
                    node.AddInput("input", new ModelResourceNodeId(
                        GraphBindingOwner.Instance(new GraphDefinitionNodeId(id)),
                        new ResourceId("internal")));
                    node.AddOutput("output", output);
                });
            }

            var parentOutput = builder.AddOutput("parent-result", F);
            builder.AddNode("read-parent", PrimitiveOperations.Copy, node =>
            {
                node.AddInput("input", parentValue);
                node.AddOutput("output", parentOutput);
            });
        });

        Assert.Empty(graph.Validate());
        Assert.Single(graph.Definitions);
        foreach (var id in new[] { "first", "second" })
        {
            var reader = graph.Nodes.Single(node => node.Id.Value == $"read-{id}");
            var binding = graph.Bindings(reader).Single(value => value.Port == "input");
            Assert.Equal(new ResourceId("internal"), binding.Resource.Resource);
            Assert.Equal(new GraphDefinitionNodeId(id), binding.Resource.Owner.Call);
            Assert.Equal([id], graph.Dependencies(reader).Select(node => node.Value));
        }

        var parentReader = graph.Nodes.Single(node => node.Id.Value == "read-parent");
        Assert.True(graph.Bindings(parentReader).Single(value => value.Port == "input").Resource.Owner.IsParent);
        Assert.Empty(graph.Dependencies(parentReader));
    }

    [Theory]
    [InlineData("unknown-call", "internal")]
    [InlineData("child-call", "unknown-resource")]
    public void InstanceScopeRejectsUnknownOwnerOrResource(string owner, string resource)
    {
        var builder = new GraphDefinitionBuilder("scope-errors");
        var input = builder.AddInput("input", F);
        var output = builder.AddOutput("output", F);
        var childOutput = builder.AddResource("child-output", "Child output", F);
        builder.AddCall("child-call", CopyGraph("child", includeInternal: true), bind =>
        {
            bind.AddInput("input", input);
            bind.AddOutput("output", childOutput);
        });
        builder.AddNode("read", PrimitiveOperations.Copy, node =>
        {
            node.AddInput("input", new ModelResourceNodeId(
                GraphBindingOwner.Instance(new GraphDefinitionNodeId(owner)), new ResourceId(resource)));
            node.AddOutput("output", output);
        });

        Assert.Contains(builder.Validate(), diagnostic =>
            diagnostic.Severity == GraphDiagnosticSeverity.Error &&
            diagnostic.Message.Contains(resource, StringComparison.Ordinal));
        Assert.Throws<InvalidDataException>(() => builder.Build());
    }

    [Fact]
    public void ParentResourceReferenceDoesNotImplicitlySearchCallInstances()
    {
        var builder = new GraphDefinitionBuilder("parent-scope");
        var input = builder.AddInput("input", F);
        var output = builder.AddOutput("output", F);
        var childOutput = builder.AddResource("child-output", "Child output", F);
        builder.AddCall("child-call", CopyGraph("child", includeInternal: true), bind =>
        {
            bind.AddInput("input", input);
            bind.AddOutput("output", childOutput);
        });
        builder.AddNode("read", PrimitiveOperations.Copy, node =>
        {
            node.AddInput("input", new ResourceId("internal"));
            node.AddOutput("output", output);
        });

        Assert.Contains(builder.Validate(), diagnostic =>
            diagnostic.Severity == GraphDiagnosticSeverity.Error &&
            diagnostic.Message.Contains("internal", StringComparison.Ordinal));
        Assert.Throws<InvalidDataException>(() => builder.Build());
    }

    [Fact]
    public void ExplicitCallDependencyPreservesOrderingWithoutDataFlow()
    {
        var graph = LogicalGraphDefinition.Define("call-order", builder =>
        {
            var input = builder.AddInput("input", F);
            var firstOutput = builder.AddOutput("first-output", F);
            var secondOutput = builder.AddOutput("second-output", F);
            var child = CopyGraph("child");
            var first = builder.AddCall("first", child, bind =>
            {
                bind.AddInput("input", input);
                bind.AddOutput("output", firstOutput);
            });
            builder.AddCall("second", child, bind =>
            {
                bind.AddInput("input", input);
                bind.AddOutput("output", secondOutput);
                bind.DependsOn(first);
            });
        });

        Assert.Equal(["first"], graph.Dependencies(graph.Nodes[1]).Select(node => node.Value));
        Assert.Equal(["second"], graph.CompletionSinks.Select(node => node.Value));
        Assert.Equal(["first"], graph.TerminalWriters[new("first-output")].Select(node => node.Value));
        Assert.Equal(["second"], graph.TerminalWriters[new("second-output")].Select(node => node.Value));
        Assert.Single(graph.Definitions);
    }

    private static LogicalGraphDefinition CopyGraph(string id, bool includeInternal = false) =>
        LogicalGraphDefinition.Define(id, builder =>
        {
            var input = builder.AddInput("input", F);
            var output = builder.AddOutput("output", F);
            var internalValue = includeInternal
                ? builder.AddResource("internal", "Internal", F)
                : output;
            builder.AddNode("copy", PrimitiveOperations.Copy, node =>
            {
                node.AddInput("input", input);
                node.AddOutput("output", internalValue);
            });
            if (includeInternal)
            {
                builder.AddNode("finish", PrimitiveOperations.Copy, node =>
                {
                    node.AddInput("input", internalValue);
                    node.AddOutput("output", output);
                });
            }
        }, GraphCompileBoundary.Atomic);
}
