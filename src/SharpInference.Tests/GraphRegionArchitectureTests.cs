using System.Xml.Linq;
using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class GraphRegionArchitectureTests
{
    private static readonly TensorDescriptor Tensor = new(GraphElementType.Float32, [2]);

    private static LogicalGraphBuilder Builder() =>
        new LogicalGraphBuilder(new("test", 1, "architecture"),
            new("test", "test.empty@1", new Dictionary<string, int>()))
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External, Tensor, graphInput: true)
            .AddResource("mid", "Intermediate", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, Tensor)
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External, Tensor, graphOutput: true);

    private static GraphRegionArchitecture Metadata() => new(
        "projection", "First line\nSecond line < & >", "y = W x\nz = y + x",
        DefaultCollapsed: false, RepeatGroup: true, DefaultView: GraphArchitectureView.Architecture,
        Step: GraphArchitectureStep.Token)
    {
        Ports =
        [
            new(new("input"), GraphArchitecturePortDirection.In, "Input signal"),
            new(new("output"), GraphArchitecturePortDirection.Out, "Output signal"),
        ],
    };

    private static LogicalGraph Graph() => Builder().WithRegion("root", GraphRegionTypes.Graph, "Root", root =>
    {
        root.WithRegion("stage", GraphRegionTypes.Stage, "Stage", stage =>
        {
            stage.WithRegion("mix", GraphRegionTypes.Architecture, "Mix",
                mix => mix.Copy("first", "input", "mid"),
                architecture: new(Description: "A real computation group."));
            stage.Copy("second", "mid", "output");
        });
    }, architecture: Metadata()).BuildSequential();

    [Fact]
    public void NestedScopesInferParentsAndRestoreNodeOwnership()
    {
        var graph = Graph();
        Assert.Equal(["first", "second"], graph.Nodes.Select(n => n.Id.Value));
        Assert.Equal(["mix", "stage"], graph.Nodes.Select(n => n.Region.Value));
        Assert.Equal("root", graph.Regions.Single(r => r.Id.Value == "stage").ParentId!.Value.Value);
        Assert.Equal("stage", graph.Regions.Single(r => r.Id.Value == "mix").ParentId!.Value.Value);
        Assert.Equal([graph.Nodes[0].Id], graph.Nodes[1].Dependencies);
    }

    [Fact]
    public void CallbackResultsAndExceptionsRestoreThePreviousScope()
    {
        var builder = Builder();
        builder.WithRegion("root", GraphRegionTypes.Graph, "Root", root =>
        {
            var resource = root.WithRegion("child", GraphRegionTypes.Architecture, "Child", child =>
            {
                child.Copy("first", "input", "mid");
                return "mid";
            });
            var exception = Assert.Throws<InvalidOperationException>(() =>
                root.WithRegion("failure", GraphRegionTypes.Architecture, "Failure",
                    (Action<LogicalGraphBuilder>)(_ => throw new InvalidOperationException("test failure"))));
            Assert.Equal("test failure", exception.Message);
            root.Copy("second", resource, "output");
        });
        Assert.Throws<InvalidOperationException>(() => builder.Copy("outside", "input", "output"));
        Assert.Equal(["child", "root"], builder.Build().Nodes.Select(n => n.Region.Value));
        Assert.Throws<ArgumentNullException>(() => builder.WithRegion(
            "invalid", GraphRegionTypes.Stage, "Invalid", (Action<LogicalGraphBuilder>)null!));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void XmlAndJsonPreserveStructuredAndFlatMetadata(bool flat)
    {
        var graph = Graph();
        if (flat) graph = new(graph.Identity, graph.Model, graph.Resources, graph.Regions,
            graph.Nodes, graph.Inputs, graph.Outputs, graph.GraphState);
        var xml = GraphXml.Serialize(graph);
        var json = GraphJson.Serialize(graph);
        foreach (var restored in new[] { GraphXml.DeserializeLogical(xml), GraphJson.DeserializeLogical(json) })
        {
            GraphValidator.Validate(restored);
            var metadata = restored.Regions.Single(r => r.Id.Value == "root").Architecture!;
            Assert.Equal(Metadata().Description, metadata.Description);
            Assert.Equal(Metadata().Formula, metadata.Formula);
            Assert.Equal(Metadata().Ports, metadata.Ports);
            Assert.Equal(xml, GraphXml.Serialize(restored));
            Assert.Equal(json, GraphJson.Serialize(restored));
        }
        var execution = new GraphOptimizer().Optimize(graph, new(OptimizationBoundary.Off,
            DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded));
        var executionXml = GraphXml.Serialize(execution);
        var executionJson = GraphJson.Serialize(execution);
        Assert.Equal(executionXml, GraphXml.Serialize(GraphXml.DeserializeExecution(executionXml)));
        Assert.Equal(executionJson, GraphJson.Serialize(GraphJson.DeserializeExecution(executionJson)));
    }

    [Fact]
    public void MetadataAndLegacyAttributesAreBothOptional()
    {
        var document = XDocument.Parse(GraphXml.Serialize(Graph()));
        foreach (var region in document.Descendants("Region"))
        {
            region.Element("Architecture")?.Remove();
            region.Element("Attributes")?.Remove();
        }
        var graph = GraphXml.DeserializeLogical(document.ToString());
        Assert.All(graph.Regions, r => Assert.Null(r.Architecture));
        Assert.Equal(["first", "second"], graph.Nodes.Select(n => n.Id.Value));
    }

    [Theory]
    [InlineData("duplicate")]
    [InlineData("late")]
    [InlineData("unknown-child")]
    [InlineData("unknown-attribute")]
    [InlineData("invalid-flag")]
    [InlineData("invalid-enum")]
    [InlineData("numeric-enum")]
    [InlineData("nested-description")]
    [InlineData("duplicate-port")]
    [InlineData("unknown-resource")]
    [InlineData("wrong-direction")]
    public void InvalidMetadataIsRejected(string invalid)
    {
        var document = XDocument.Parse(GraphXml.Serialize(Graph()));
        var region = document.Descendants("Region").First();
        var architecture = region.Element("Architecture")!;
        var port = architecture.Descendants("Port").First();
        switch (invalid)
        {
            case "duplicate": region.AddFirst(new XElement(architecture)); break;
            case "late": architecture.Remove(); region.Add(architecture); break;
            case "unknown-child": architecture.Add(new XElement("Unknown")); break;
            case "unknown-attribute": architecture.SetAttributeValue("unknown", "true"); break;
            case "invalid-flag": architecture.SetAttributeValue("repeatGroup", "True"); break;
            case "invalid-enum": architecture.SetAttributeValue("step", "batch"); break;
            case "numeric-enum": architecture.SetAttributeValue("defaultView", "0"); break;
            case "nested-description": architecture.Element("Description")!.Add(new XElement("Nested")); break;
            case "duplicate-port": port.Parent!.Add(new XElement(port)); break;
            case "unknown-resource": port.SetAttributeValue("resource", "missing"); break;
            case "wrong-direction": port.SetAttributeValue("direction", "out"); break;
        }
        Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeLogical(document.ToString()));
    }

    [Fact]
    public void ReuseAndExpansionMapArchitecturePortResources()
    {
        var graph = GraphLayerReuse.Extract(Graph(), r => r.Id.Value == "stage" ? "Test.Layer" : null);
        var expandedPort = graph.Regions.Single(r => r.Id.Value == "root").Architecture!.Ports[0];
        Assert.Equal(new ResourceId("input"), expandedPort.Resource);
        var definition = Assert.Single(graph.Structure!.LayerDefinitions);
        var body = definition.Body with
        {
            Region = definition.Body.Region with
            {
                Architecture = new()
                {
                    Ports = definition.Ports.Select(p => new GraphArchitecturePort(new(p.Id),
                        p.Access == GraphResourceAccess.Read ? GraphArchitecturePortDirection.In : GraphArchitecturePortDirection.Out,
                        p.Id)).ToArray(),
                },
            },
        };
        var withPorts = new LogicalGraph(graph.Identity, graph.Model, graph.DeclaredResources,
            graph.Structure with { LayerDefinitions = [definition with { Body = body }] },
            graph.Inputs, graph.Outputs, graph.GraphState);
        var calledRegion = withPorts.Regions.Single(r => r.Id.Value == "stage");
        Assert.Contains(calledRegion.Architecture!.Ports, p => p.Resource == new ResourceId("input"));
        Assert.Contains(calledRegion.Architecture.Ports, p => p.Resource == new ResourceId("output"));
        Assert.Equal(GraphXml.Serialize(withPorts), GraphXml.Serialize(GraphXml.DeserializeLogical(GraphXml.Serialize(withPorts))));
    }

    [Fact]
    public void ExtractionMapsMetadataForBothBoundaryAndLocalResources()
    {
        var original = Graph();
        var stage = (GraphRegionBody)original.Structure!.Root.Children[0];
        var ports = new GraphRegionArchitecture
        {
            Ports =
            [
                new(new("input"), GraphArchitecturePortDirection.In, "Input"),
                new(new("mid"), GraphArchitecturePortDirection.Out, "Intermediate"),
                new(new("output"), GraphArchitecturePortDirection.Out, "Output"),
            ],
        };
        var structured = new LogicalGraph(original.Identity, original.Model, original.DeclaredResources,
            original.Structure with
            {
                Root = original.Structure.Root with
                {
                    Children = [stage with { Region = stage.Region with { Architecture = ports } }],
                },
            }, original.Inputs, original.Outputs);
        var reused = GraphLayerReuse.Extract(structured, r => r.Id.Value == "stage" ? "Test.Layer" : null);
        var metadata = reused.Regions.Single(r => r.Id.Value == "stage").Architecture!;
        Assert.Contains(metadata.Ports, p => p.Resource == new ResourceId("input"));
        Assert.Contains(metadata.Ports, p => p.Resource.Value == "stage::n0_output");
        Assert.Contains(metadata.Ports, p => p.Resource == new ResourceId("output"));
        Assert.Equal(GraphXml.Serialize(reused), GraphXml.Serialize(GraphXml.DeserializeLogical(GraphXml.Serialize(reused))));
    }

    [Fact]
    public void InvalidMetadataInUnusedDefinitionIsRejected()
    {
        var graph = Graph();
        var region = new GraphRegion(new("body"), null, GraphRegionTypes.Layer, null, "Unused",
            new Dictionary<string, string>())
        { Architecture = new() { Ports = [new(new("missing"), GraphArchitecturePortDirection.In, "Missing")] } };
        var definition = new GraphLayerDefinition("Unused", [], [], new(region, []));
        Assert.Throws<InvalidDataException>(() => new LogicalGraph(graph.Identity, graph.Model,
            graph.DeclaredResources, graph.Structure! with { LayerDefinitions = [definition] },
            graph.Inputs, graph.Outputs, graph.GraphState));
    }

    [Fact]
    public void UnusedDefinitionsMustHaveRealArchitecturePortBindings()
    {
        var graph = Graph();
        var region = new GraphRegion(new("body"), null, GraphRegionTypes.Layer, null, "Unused",
            new Dictionary<string, string>())
        {
            Architecture = new() { Ports = [new(new("input"), GraphArchitecturePortDirection.In, "Input")] },
        };
        var definition = new GraphLayerDefinition("Unused", [new("input", GraphResourceAccess.Read, Tensor)],
            [], new(region, []));
        var error = Assert.Throws<InvalidDataException>(() => new LogicalGraph(graph.Identity, graph.Model,
            graph.DeclaredResources, graph.Structure! with { LayerDefinitions = [definition] },
            graph.Inputs, graph.Outputs, graph.GraphState));
        Assert.Contains("does not match", error.Message);
    }

    [Fact]
    public void NullPortsFailExplicitlyBeforeResourceMapping()
    {
        Assert.Throws<InvalidDataException>(() => Builder().WithRegion("root", GraphRegionTypes.Graph, "Root",
            root => root.Copy("copy", "input", "output"), architecture: new() { Ports = null! }).Build());
        Assert.Throws<InvalidDataException>(() => new GraphRegionArchitecture { Ports = [null!] }
            .MapResources(resource => resource));
    }

    [Theory]
    [InlineData(OptimizationBoundary.WithinStage)]
    [InlineData(OptimizationBoundary.WithinLayer)]
    public void ArchitectureScopesDoNotAddFusionBoundaries(OptimizationBoundary boundary)
    {
        var builder = Builder();
        builder.WithRegion("root", GraphRegionTypes.Graph, "Root", root =>
            root.WithRegion("layer", GraphRegionTypes.Layer, "Layer", layer =>
                layer.WithRegion("stage", GraphRegionTypes.Stage, "Stage", stage =>
                    stage.WithRegion("semantic", GraphRegionTypes.Architecture, "Semantic", semantic =>
                    {
                        semantic.Copy("first", "input", "mid");
                        semantic.Copy("second", "mid", "output");
                    }))));
        var graph = builder.Build();
        var execution = new GraphOptimizer().Optimize(graph, new(boundary),
            [new SequenceFusionRule("copy-pair", [PrimitiveGraphOperations.Copy, PrimitiveGraphOperations.Copy], new("test.copy-pair"))]);
        Assert.Equal(new GraphOperationId("test.copy-pair"), Assert.Single(execution.Nodes).Operation);
    }
}
