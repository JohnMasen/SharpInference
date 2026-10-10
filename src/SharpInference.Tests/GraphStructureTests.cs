using System.Xml.Linq;
using SharpInference.Graphs;
using SharpInference.Backends.Cpu;

namespace SharpInference.Tests;

public sealed class GraphStructureTests
{
    private static readonly TensorDescriptor F = new(GraphElementType.Float32, [2]);
    private static readonly IReadOnlyDictionary<string, string> Empty = new Dictionary<string, string>();
    private static readonly PrecisionRequirement Precision = new(GraphElementType.Float32, GraphElementType.Float32);
    private static GraphRegionBody Region(string id, params GraphElement[] children) =>
        new(new(new(id), null, id == "root" ? "graph" : "stage", null, id, Empty), children);
    private static GraphNodeElement Copy(string id, string input, string output) =>
        new(new(new(id), PrimitiveGraphOperations.Copy, new("body"),
            [GraphBindings.Read("input", input), GraphBindings.Write("output", output)], [], Empty, Precision));
    private static GraphResource Resource(string id, GraphResourceKind kind = GraphResourceKind.Temporary) =>
        new(new(id), id, kind, kind is GraphResourceKind.Input or GraphResourceKind.Output ? GraphResourceLifetime.External : GraphResourceLifetime.Invocation, F);
    private static GraphLayerDefinition Definition() => new("CopyTwice",
        [new("input", GraphResourceAccess.Read, F), new("output", GraphResourceAccess.Write, F)],
        [Resource("n0_output")], Region("body", Copy("n0", "input", "n0_output"), Copy("n1", "n0_output", "output")));
    private static GraphCall Call(string id, string input, string output, string definition = "CopyTwice") =>
        new(id, definition, [GraphBindings.Read("input", input), GraphBindings.Write("output", output)]);
    private static LogicalGraph Graph(GraphStructure structure, params GraphResource[] extra) =>
        new(new("test", 1, "structured"), new("test", "empty@1", new Dictionary<string, int>()),
            new[] { Resource("input", GraphResourceKind.Input), Resource("mid"), Resource("output", GraphResourceKind.Output) }.Concat(extra),
            structure, [new("input")], [new("output")], extra.Where(r => r.Kind == GraphResourceKind.SessionState).Select(r => new GraphStateSlot(r.Id.Value, r.Id)));
    private static GraphStructure Structure() => new([Definition()], Region("root", Call("first", "input", "mid"), Call("second", "mid", "output")));

    [Fact]
    public void CallsRoundTripWithoutLosingStructureOrScopes()
    {
        var graph = Graph(Structure());
        Assert.Equal(4, graph.Nodes.Count);
        Assert.Contains(graph.Resources, r => r.Id.Value == "first::n0_output");
        Assert.Contains(graph.Resources, r => r.Id.Value == "second::n0_output");
        var xml = GraphXml.Serialize(graph);
        Assert.Contains("kind=\"LogicalGraph\" version=\"1\"", xml);
        var json = System.Text.Json.Nodes.JsonNode.Parse(GraphJson.Serialize(graph))!;
        Assert.Equal("LogicalGraph", json["Kind"]!.GetValue<string>());
        Assert.Equal(1, json["FormatVersion"]!.GetValue<int>());
        Assert.DoesNotContain("<Nodes>", xml);
        Assert.DoesNotContain("<Dependencies>", xml);
        Assert.DoesNotContain(" region=", xml);
        Assert.Equal(2, XDocument.Parse(xml).Descendants("Call").Count());
        foreach (var restored in new[] { GraphXml.DeserializeLogical(xml), GraphJson.DeserializeLogical(GraphJson.Serialize(graph)) })
        {
            Assert.Single(restored.Structure!.LayerDefinitions);
            Assert.Equal(graph.Nodes.Select(n => n.Id), restored.Nodes.Select(n => n.Id));
            Assert.Equal(xml, GraphXml.Serialize(restored));
            Assert.Equal(graph.Resources.Select(r => r.Id), restored.Resources.Select(r => r.Id));
        }
        var result = new CpuPrimitiveGraphExecutor(graph, new EmptyCatalog()).Execute(
            new Dictionary<ResourceId, Array> { [new("input")] = new float[] { 2, 3 } });
        Assert.Equal(new float[] { 2, 3 }, Assert.IsType<float[]>(result.Outputs[new("output")]));
    }

    [Theory]
    [InlineData("consumer")]
    [InlineData("consumer_output")]
    [InlineData("odd::local")]
    public void OutputAliasesCannotClaimDeclaredCounterpartIds(string counterpart)
    {
        var writer = Copy("writer", "input", counterpart + "_output");
        var reader = Copy(counterpart, counterpart + "_output", "output") with { DependsOn = [writer.Id] };
        var definition = new GraphLayerDefinition("Arbitrary", Definition().Ports,
            [Resource(counterpart + "_output")], Region("body", Region("left", writer), Region("right", reader)));
        var graph = Graph(new([definition], Region("root", Call("run", "input", "output", "Arbitrary"))));
        var restored = GraphXml.DeserializeLogical(GraphXml.Serialize(graph));
        Assert.Equal(["run::writer", "run::" + counterpart], restored.Nodes.Select(node => node.Id.Value));
        Assert.Equal([restored.Nodes[0].Id], restored.Nodes[1].Dependencies);
        Assert.Equal(writer.Id, ((GraphNodeElement)((GraphRegionBody)restored.Structure!.LayerDefinitions[0].Body.Children[0]).Children[0]).Id);
        Assert.Equal(GraphXml.Serialize(graph), GraphXml.Serialize(restored));
        var result = new CpuPrimitiveGraphExecutor(restored, new EmptyCatalog()).Execute(
            new Dictionary<ResourceId, Array> { [new("input")] = new float[] { 17, 19 } });
        Assert.Equal(new float[] { 17, 19 }, Assert.IsType<float[]>(result.Outputs[new("output")]));
        AssertCpuVmOutputs(restored, new Dictionary<ResourceId, float[]> { [new("output")] = [17, 19] });
    }

    [Fact]
    public void OutputAliasesCannotClaimRegionIds()
    {
        var definition = new GraphLayerDefinition("Arbitrary", Definition().Ports, [Resource("consumer_output")],
            Region("body", Copy("writer", "input", "consumer_output"),
                Region("consumer", Copy("finish", "consumer_output", "output"))));
        var graph = Graph(new([definition], Region("root", Call("run", "input", "output", "Arbitrary"))));
        Assert.Equal("run::writer", graph.Nodes[0].Id.Value);
        Assert.Contains(graph.Regions, region => region.Id.Value == "run::consumer");
    }

    [Theory]
    [InlineData("empty", false)]
    [InlineData("empty", true)]
    [InlineData("partial", false)]
    [InlineData("partial", true)]
    [InlineData("overlap", false)]
    [InlineData("overlap", true)]
    [InlineData("nested-empty", false)]
    [InlineData("nested-empty", true)]
    [InlineData("nested-false-write", false)]
    [InlineData("nested-false-write", true)]
    public void DefinitionsMustProveEveryDeclaredWrite(string invalid, bool unused)
    {
        var half = new TensorDescriptor(GraphElementType.Float32, [1]);
        GraphNodeElement Partial(string id) => new(Copy(id, "input", "output").Node with
        {
            Resources = [new("input", new("input"), GraphResourceAccess.Read, View: new(0, half)),
                new("output", new("output"), GraphResourceAccess.Write, View: new(0, half))],
        });
        var definitions = new List<GraphLayerDefinition>();
        GraphElement[] body = invalid switch
        {
            "partial" => [Partial("half")],
            "overlap" => [Partial("first"), Partial("second")],
            _ => [],
        };
        if (invalid.StartsWith("nested", StringComparison.Ordinal))
        {
            var ports = invalid == "nested-false-write" ? Definition().Ports : [Definition().Ports[0]];
            definitions.Add(new("Empty", ports, [], Region("body")));
            body = [new GraphCall("nested", "Empty", invalid == "nested-false-write"
                ? Call("nested", "input", "output").Bindings : [GraphBindings.Read("input", "input")])];
        }
        definitions.Add(new("Incomplete", Definition().Ports, [], Region("body", body)));
        definitions.Add(Definition());
        var error = Assert.Throws<InvalidDataException>(() => new LogicalGraph(
            new("test", 2, "incomplete"), new("test", "empty@1", new Dictionary<string, int>()),
            [Resource("input", GraphResourceKind.Input), Resource("output", GraphResourceKind.Output)],
            new GraphStructure(definitions, Region("root", Call("run", "input", "output", unused ? "CopyTwice" : "Incomplete"))),
            [new("input")], [new("output")]));
        Assert.Equal("Output 'output' has an uninitialized byte range.", error.Message);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompleteViewAndMultiOutputWritesAllowPreservedReadWritePorts(bool views)
    {
        var ports = Definition().Ports.Concat(new[] { new GraphDefinitionPort("second", GraphResourceAccess.Write, F),
            new GraphDefinitionPort("preserved", GraphResourceAccess.ReadWrite, F) }).ToArray();
        var half = new TensorDescriptor(GraphElementType.Float32, [1]);
        GraphNodeElement Part(string id, ulong offset) => new(Copy(id, "input", "output").Node with
        {
            Resources = [new("input", new("input"), GraphResourceAccess.Read, View: new(offset, half)),
                new("output", new("output"), GraphResourceAccess.Write, View: new(offset, half))],
        });
        GraphElement[] body = views ? [Part("first", 0), Part("last", 4), Copy("second", "input", "second")]
            : [Copy("first", "input", "output"), Copy("second", "input", "second")];
        var inner = new GraphLayerDefinition("Complete", ports, [Resource("constant", GraphResourceKind.Constant)], Region("body", body));
        NodeResourceBinding[] bindings = [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "output"),
            GraphBindings.Write("second", "second"), new("preserved", new("preserved"), GraphResourceAccess.ReadWrite)];
        var outer = new GraphLayerDefinition("Outer", ports, [], Region("body", new GraphCall("nested", "Complete", bindings)));
        var structure = new GraphStructure([inner, outer], Region("root", new GraphCall("run", "Outer",
            bindings.Select(binding => binding.Port == "preserved" ? binding with { Resource = new("input") } : binding).ToArray())));
        var graph = new LogicalGraph(new("test", 2, "complete"), new("test", "empty@1", new Dictionary<string, int>()),
            [Resource("input", GraphResourceKind.Input), Resource("output", GraphResourceKind.Output), Resource("second", GraphResourceKind.Output)],
            structure, [new("input")], [new("output"), new("second")]);
        var restored = GraphXml.DeserializeLogical(GraphXml.Serialize(graph));
        AssertCpuVmOutputs(restored, new Dictionary<ResourceId, float[]> { [new("output")] = [17, 19], [new("second")] = [17, 19] });
    }

    [Fact]
    public void ConstantReadsCanInitializeDeclaredOutputs()
    {
        var definition = new GraphLayerDefinition("ConstantOutput", [Definition().Ports[1]],
            [Resource("constant", GraphResourceKind.Constant)], Region("body", Copy("copy", "constant", "output")));
        var structure = new GraphStructure([definition], Region("root",
            new GraphCall("run", "ConstantOutput", [GraphBindings.Write("output", "output")])));
        var restored = GraphXml.DeserializeLogical(GraphXml.Serialize(Graph(structure)));
        Assert.Equal("run::constant", restored.Nodes[0].Resources[0].Resource.Value);
        Assert.Equal(GraphResourceKind.Constant, restored.Resources.Single(resource => resource.Id.Value == "run::constant").Kind);
    }

    private static void AssertCpuVmOutputs(LogicalGraph graph, IReadOnlyDictionary<ResourceId, float[]> expected)
    {
        using var backend = SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create();
        backend.Prepare(graph);
        using var session = backend.CreateGraphSession(new EmptyCatalog());
        var result = session.Execute(new Dictionary<ResourceId, ReadOnlyMemory<byte>>
        {
            [new("input")] = System.Runtime.InteropServices.MemoryMarshal.AsBytes(new float[] { 17, 19 }.AsSpan()).ToArray(),
        });
        foreach (var (id, values) in expected)
            Assert.Equal(values, System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(result[id]).ToArray());
    }

    [Fact]
    public void CompositeDefinitionsInstantiateNestedCalls()
    {
        var inner = Definition();
        var outer = new GraphLayerDefinition("Composite", inner.Ports, [], Region("body", Call("nested", "input", "output")));
        var graph = Graph(new([inner, outer], Region("root", Call("run", "input", "output", "Composite"))));
        Assert.Equal(2, graph.Nodes.Count);
        Assert.Contains(graph.Regions, r => r.Id.Value == "run::nested");
        Assert.Contains(graph.Resources, r => r.Id.Value == "run::nested::n0_output");
    }

    [Theory]
    [InlineData("missing-definition")]
    [InlineData("duplicate-definition")]
    [InlineData("duplicate-call")]
    [InlineData("missing-binding")]
    [InlineData("duplicate-binding")]
    [InlineData("missing-resource")]
    [InlineData("recursion")]
    [InlineData("scope-leak")]
    [InlineData("port-access")]
    [InlineData("shape")]
    [InlineData("type")]
    [InlineData("duplicate-local")]
    [InlineData("duplicate-port")]
    [InlineData("unused-missing-resource")]
    public void RejectsInvalidComputationalContracts(string error)
    {
        var definition = Definition();
        var call = Call("run", "input", "output");
        var definitions = new List<GraphLayerDefinition> { definition };
        var children = new List<GraphElement> { call };
        switch (error)
        {
            case "missing-definition": children[0] = call with { Definition = "absent" }; break;
            case "duplicate-definition": definitions.Add(definition); break;
            case "duplicate-call": children.Add(call); break;
            case "missing-binding": children[0] = call with { Bindings = [call.Bindings[0]] }; break;
            case "duplicate-binding": children[0] = call with { Bindings = [call.Bindings[0], call.Bindings[0]] }; break;
            case "missing-resource": children[0] = Call("run", "absent", "output"); break;
            case "recursion": definitions[0] = definition with { Body = Region("body", Call("self", "input", "output")) }; break;
            case "scope-leak": children.Add(Copy("escape", "run::n0_output", "output")); break;
            case "port-access": children[0] = call with { Bindings = [GraphBindings.Write("input", "input"), call.Bindings[1]] }; break;
            case "shape": definitions[0] = definition with { Ports = [definition.Ports[0] with { Tensor = new(GraphElementType.Float32, [3]) }, definition.Ports[1]] }; break;
            case "type": definitions[0] = definition with { Ports = [definition.Ports[0] with { Tensor = new(GraphElementType.Float16, [2]) }, definition.Ports[1]] }; break;
            case "duplicate-local": definitions[0] = definition with { Resources = [definition.Resources[0], definition.Resources[0]] }; break;
            case "duplicate-port": definitions[0] = definition with { Ports = [definition.Ports[0], definition.Ports[0]] }; break;
            case "unused-missing-resource": definitions.Add(definition with { Id = "unused", Body = Region("body", Copy("broken", "secret", "output")) }); break;
        }
        Assert.Throws<InvalidDataException>(() => Graph(new(definitions, Region("root", children.ToArray()))));
    }

    [Fact]
    public void SameRegionDependenciesAreForbiddenAndCrossRegionOrderCannotBeReversed()
    {
        var same = Structure();
        var a = same.Root.Children[0];
        var b = same.Root.Children[1] with { DependsOn = [a.Id] };
        Assert.Throws<InvalidDataException>(() => Graph(same with { Root = Region("root", a, b) }));
        var first = Copy("a", "input", "mid");
        var second = Copy("b", "mid", "output") with { DependsOn = ["a"] };
        var graph = Graph(new([], Region("root", Region("left", first), Region("right", second))));
        Assert.Equal([graph.Nodes[0].Id], graph.Nodes[1].Dependencies);
        first = first with { DependsOn = ["b"] };
        Assert.Throws<InvalidDataException>(() => Graph(new([], Region("root", Region("left", first), Region("right", second)))));
    }

    [Fact]
    public void NestedRegionCompletesBeforeSiblingStateRead()
    {
        var state = Resource("state") with { Kind = GraphResourceKind.SessionState, Lifetime = GraphResourceLifetime.Session };
        var root = Region("root", Region("update", Copy("write", "input", "state")), Copy("read", "state", "output"));
        var graph = Graph(new([], root), state);
        Assert.Equal([graph.Nodes[0].Id], graph.Nodes[1].Dependencies);
        var result = new CpuPrimitiveGraphExecutor(graph, new EmptyCatalog()).Execute(
            new Dictionary<ResourceId, Array> { [new("input")] = new float[] { 5, 7 } },
            [new GraphStateValue("state", [2], [0, 0])]);
        Assert.Equal(new float[] { 5, 7 }, Assert.IsType<float[]>(result.Outputs[new("output")]));
    }

    [Fact]
    public void StructuredXmlRejectsLegacyDependencyChainsAndPrototypeMetadata()
    {
        var xml = GraphXml.Serialize(Graph(Structure()));
        var document = XDocument.Parse(xml);
        document.Descendants("Node").First().Add(new XElement("Dependencies", new XElement("Dependency", new XAttribute("id", "n0"))));
        Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeLogical(document.ToString()));
        document = XDocument.Parse(xml);
        document.Descendants("Call").First().SetAttributeValue("resourceOrdinalBase", "0");
        Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeLogical(document.ToString()));
    }

    [Theory]
    [InlineData("port-local-collision")]
    [InlineData("local-weight")]
    [InlineData("invalid-access")]
    [InlineData("unused-port-access")]
    [InlineData("unused-view")]
    [InlineData("unused-operation")]
    [InlineData("nested-shape")]
    [InlineData("nested-access")]
    [InlineData("kind")]
    [InlineData("duplicate-dependency")]
    [InlineData("unknown-dependency")]
    [InlineData("expanded-collision")]
    public void RejectsAdditionalInvalidScopesAndUnusedContracts(string error)
    {
        var definition = Definition();
        var definitions = new List<GraphLayerDefinition> { definition };
        GraphElement call = Call("run", "input", "output");
        GraphResource[] extra = [];
        switch (error)
        {
            case "port-local-collision": definitions[0] = definition with { Resources = [Resource("input")] }; break;
            case "local-weight": definitions[0] = definition with { Resources = [Resource("n0_output") with { Kind = GraphResourceKind.Weight }] }; break;
            case "invalid-access": definitions[0] = definition with { Ports = [definition.Ports[0] with { Access = (GraphResourceAccess)99 }, definition.Ports[1]] }; break;
            case "unused-port-access": definitions.Add(definition with { Id = "unused", Body = Region("body", Copy("bad", "output", "input")) }); break;
            case "unused-view":
                var node = Copy("bad", "input", "output");
                definitions.Add(definition with { Id = "unused", Body = Region("body", new GraphNodeElement(node.Node with
                { Resources = [node.Node.Resources[0] with { View = new(8, F) }, node.Node.Resources[1]] })) });
                break;
            case "unused-operation":
                definitions.Add(definition with { Id = "unused", Body = Region("body", new GraphNodeElement(Copy("bad", "input", "output").Node with
                { Operation = new("", 1) })) });
                break;
            case "nested-shape":
            case "nested-access":
                var ports = error == "nested-shape"
                    ? new[] { definition.Ports[0] with { Tensor = new(GraphElementType.Float32, [3]) }, definition.Ports[1] }
                    : new[] { definition.Ports[0] with { Access = GraphResourceAccess.Write }, definition.Ports[1] };
                definitions.Add(new("outer", ports, [], Region("body", Call("nested", "input", "output"))));
                break;
            case "kind": definitions[0] = definition with { Ports = [definition.Ports[0] with { ResourceKind = GraphResourceKind.Weight }, definition.Ports[1]] }; break;
            case "duplicate-dependency": call = call with { DependsOn = ["absent", "absent"] }; break;
            case "unknown-dependency": call = call with { DependsOn = ["absent"] }; break;
            case "expanded-collision": extra = [Resource("run::n0_output")]; break;
        }
        Assert.Throws<InvalidDataException>(() => Graph(new(definitions, Region("root", call)), extra));
    }

    [Fact]
    public void ViewsAndMultipleOutputsRoundTripThroughStructuredVersionTwo()
    {
        var half = new TensorDescriptor(GraphElementType.Float32, [1]);
        var node = new GraphNodeElement(new(new("split"), new("custom.split"), new("body"),
            [new("input", new("input"), GraphResourceAccess.Read, View: new(0, half)),
             new("left", new("left_output"), GraphResourceAccess.Write),
             new("right", new("right_output"), GraphResourceAccess.Write)], [], Empty,
            new(GraphElementType.Float16, GraphElementType.Float32)));
        var definition = new GraphLayerDefinition("Split", [new("input", GraphResourceAccess.Read, F)],
            [Resource("left_output") with { Tensor = half }, Resource("right_output") with { Tensor = half }], Region("body", node));
        var structure = new GraphStructure([definition], Region("root",
            new GraphCall("first", "Split", [GraphBindings.Read("input", "input")]),
            new GraphCall("second", "Split", [GraphBindings.Read("input", "input")])));
        var graph = new LogicalGraph(new("test", 2, "views"), new("test", "empty@1", new Dictionary<string, int>()),
            [Resource("input", GraphResourceKind.Input)], structure, [new("input")], []);
        var xml = GraphXml.Serialize(graph);
        Assert.Contains("kind=\"LogicalGraph\" version=\"2\"", xml);
        var json = System.Text.Json.Nodes.JsonNode.Parse(GraphJson.Serialize(graph))!;
        Assert.Equal("LogicalGraphV2", json["Kind"]!.GetValue<string>());
        Assert.Equal(2, json["FormatVersion"]!.GetValue<int>());
        foreach (var restored in new[] { GraphXml.DeserializeLogical(xml), GraphJson.DeserializeLogical(json.ToJsonString()) })
        {
            Assert.Equal(["first::split", "second::split"], restored.Nodes.Select(n => n.Id.Value));
            Assert.Equal(5, restored.Resources.Count);
            Assert.Equal(2, restored.Nodes[0].Resources.Count(b => b.Access == GraphResourceAccess.Write));
            Assert.Equal(half.Dimensions, restored.Nodes[0].Resources[0].View!.Tensor.Dimensions);
            Assert.Equal(node.Node.Requirements, restored.Nodes[0].Requirements);
            Assert.Equal(xml, GraphXml.Serialize(restored));
        }
        Assert.Throws<InvalidDataException>(() => new LogicalGraph(new("test", 1, "views"), graph.Model,
            graph.DeclaredResources, structure, graph.Inputs, graph.Outputs));
    }

    [Theory]
    [InlineData("root")]
    [InlineData("used-definition")]
    [InlineData("unused-definition")]
    [InlineData("nested-unused-definition")]
    public void StructuredViewsRequireVersionTwoEvenInUnusedDefinitions(string location)
    {
        var node = Copy("view", "input", "output");
        node = new(node.Node with
        {
            Resources = [node.Node.Resources[0] with { View = new(0, F) }, node.Node.Resources[1]],
        });
        var definition = new GraphLayerDefinition("ViewCopy", Definition().Ports, [], Region("body", Region("nested", node)));
        var definitions = new List<GraphLayerDefinition> { Definition() };
        GraphElement child = Call("run", "input", "output");
        if (location == "root") child = Region("nested", node);
        else
        {
            definitions.Add(definition);
            if (location == "used-definition") child = Call("run", "input", "output", "ViewCopy");
            if (location == "nested-unused-definition")
                definitions.Add(new("UnusedWrapper", Definition().Ports, [], Region("body", Call("nested", "input", "output", "ViewCopy"))));
        }
        var graph = new LogicalGraph(new("test", 2, "views"), new("test", "empty@1", new Dictionary<string, int>()),
            [Resource("input", GraphResourceKind.Input), Resource("output", GraphResourceKind.Output)],
            new GraphStructure(definitions, Region("root", child)), [new("input")], [new("output")]);
        var json = GraphJson.Serialize(graph);
        var parsed = System.Text.Json.Nodes.JsonNode.Parse(json)!;
        Assert.Equal("LogicalGraphV2", parsed["Kind"]!.GetValue<string>());
        Assert.Equal(2, parsed["FormatVersion"]!.GetValue<int>());
        Assert.Equal(json, GraphJson.Serialize(GraphJson.DeserializeLogical(json)));
        if (location.Contains("unused", StringComparison.Ordinal))
            Assert.All(graph.Nodes, expanded => Assert.All(expanded.Resources, binding => Assert.Null(binding.View)));
        foreach (var (kind, version) in new (string, int?)[]
        {
            ("LogicalGraph", 1), ("LogicalGraph", null), ("LogicalGraph", 2),
            ("LogicalGraphV2", 1), ("LogicalGraphV2", null),
        })
        {
            parsed = System.Text.Json.Nodes.JsonNode.Parse(json)!;
            parsed["Kind"] = kind;
            if (version is { } value) parsed["FormatVersion"] = value;
            else parsed.AsObject().Remove("FormatVersion");
            Assert.Throws<InvalidDataException>(() => GraphJson.DeserializeLogical(parsed.ToJsonString()));
        }
        parsed = System.Text.Json.Nodes.JsonNode.Parse(json)!;
        parsed["Identity"]!["IrVersion"] = 1;
        Assert.Throws<InvalidDataException>(() => GraphJson.DeserializeLogical(parsed.ToJsonString()));
        var xml = XElement.Parse(GraphXml.Serialize(graph));
        Assert.Equal("Graph", xml.Name.LocalName);
        Assert.Equal("LogicalGraph", (string?)xml.Attribute("kind"));
        Assert.Equal("2", (string?)xml.Attribute("version"));
        Assert.Equal(GraphXml.Serialize(graph), GraphXml.Serialize(GraphXml.DeserializeLogical(xml.ToString())));
        xml.SetAttributeValue("version", 1);
        Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeLogical(xml.ToString()));
    }

    [Fact]
    public void ExtractionSharesActualBodiesAndDistinguishesComputationalChanges()
    {
        var resources = new[] { Resource("input", GraphResourceKind.Input), Resource("mid"), Resource("output", GraphResourceKind.Output),
            Resource("a_output"), Resource("c_output") };
        var left = Region("left", Copy("a", "input", "a_output"), Copy("b", "a_output", "mid"));
        var right = Region("right", Copy("c", "mid", "c_output"), Copy("d", "c_output", "output"));
        LogicalGraph Create(GraphRegionBody second) => new(new("test", 1, "reuse"), new("test", "empty@1", new Dictionary<string, int>()),
            resources, new GraphStructure([], Region("root", left, second)), [new("input")], [new("output")]);
        var graph = GraphLayerReuse.Extract(Create(right), region => region.Id.Value == "root" ? null : "Layer");
        Assert.Single(graph.Structure!.LayerDefinitions);
        Assert.Equal(2, graph.Structure.Root.Children.OfType<GraphCall>().Count());
        Assert.Equal(4, graph.Nodes.Count);
        var changed = right with { Children = [right.Children[0], new GraphNodeElement(((GraphNodeElement)right.Children[1]).Node with
            { Requirements = new(GraphElementType.Float16, GraphElementType.Float32) })] };
        Assert.Equal(2, GraphLayerReuse.Extract(Create(changed), region => region.Id.Value == "root" ? null : "Layer").Structure!.LayerDefinitions.Count);
        var result = new CpuPrimitiveGraphExecutor(GraphXml.DeserializeLogical(GraphXml.Serialize(graph)), new EmptyCatalog()).Execute(
            new Dictionary<ResourceId, Array> { [new("input")] = new float[] { 11, 13 } });
        Assert.Equal(new float[] { 11, 13 }, Assert.IsType<float[]>(result.Outputs[new("output")]));
    }

    [Fact]
    public void MalformedStructuredNodeBindingsAreRejectedPrecisely()
    {
        var document = XDocument.Parse(GraphXml.Serialize(Graph(Structure())));
        document.Descendants("Node").First().Element("Bindings")!.Remove();
        Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeLogical(document.ToString()));
    }

    [Fact]
    public void StructuredJsonRejectsPrototypeMetadataAndFlatShadowNodes()
    {
        var json = System.Text.Json.Nodes.JsonNode.Parse(GraphJson.Serialize(Graph(Structure())))!;
        json["Structure"]!["Root"]!["Children"]![0]!["resourceOrdinalBase"] = 0;
        Assert.Throws<InvalidDataException>(() => GraphJson.DeserializeLogical(json.ToJsonString()));
        json = System.Text.Json.Nodes.JsonNode.Parse(GraphJson.Serialize(Graph(Structure())))!;
        json["Nodes"] = new System.Text.Json.Nodes.JsonArray();
        Assert.Throws<InvalidDataException>(() => GraphJson.DeserializeLogical(json.ToJsonString()));
    }

    [Fact]
    public void SequentialMigrationRejectsDisconnectedAndInterleavedRegions()
    {
        var root = Region("root").Region;
        var orphan = Region("orphan").Region with { ParentId = new RegionId("missing") };
        Assert.Throws<InvalidDataException>(() => GraphStructure.FromExpanded([root, orphan], []));
        var cyclic = orphan with { ParentId = orphan.Id };
        Assert.Throws<InvalidDataException>(() => GraphStructure.FromExpanded([root, cyclic], []));
        var left = Region("left").Region with { ParentId = root.Id };
        var right = Region("right").Region with { ParentId = root.Id };
        var nodes = new[] { Copy("a", "input", "mid").Node with { Region = left.Id },
            Copy("b", "mid", "mid").Node with { Region = right.Id },
            Copy("c", "mid", "output").Node with { Region = left.Id } };
        Assert.Throws<InvalidDataException>(() => GraphStructure.FromExpanded([root, left, right], nodes));
    }

    [Fact]
    public void StructuredIdentitiesMustAgreeWithTheirPayloads()
    {
        var invalid = Call("run", "input", "output") with { Id = "shadow" };
        Assert.Throws<InvalidDataException>(() => Graph(new([Definition()], Region("root", invalid))));
        var node = Copy("copy", "input", "output") with { Id = "shadow" };
        Assert.Throws<InvalidDataException>(() => Graph(new([], Region("root", node))));
    }

    private sealed class EmptyCatalog : IModelTensorCatalog
    {
        public IReadOnlyCollection<string> Names => [];
        public bool TryGet(string name, out IModelTensor tensor) { tensor = null!; return false; }
        public IModelTensor GetRequired(string name) => throw new KeyNotFoundException(name);
    }
}
