using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class GraphXmlTests
{
    [Fact]
    public void UserSuppliedNodeAndResourceIdsAreNotRenamed()
    {
        var graph = new LogicalGraphBuilder(new("custom", 1, "shared-id"),
                new GraphModelSignature("custom", "custom.empty@1", new Dictionary<string, int>()))
            .AddRegion("root", GraphRegionTypes.Graph, "Root")
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [3]), graphInput: true)
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [3]), graphOutput: true)
            .AddNode("output", PrimitiveGraphOperations.Copy, "root",
                [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "output")])
            .Build();
        Assert.Same(graph, GraphOperationLowering.Apply(graph, null));
        foreach (var restored in new[]
        {
            GraphXml.DeserializeLogical(GraphXml.Serialize(graph)),
            GraphJson.DeserializeLogical(GraphJson.Serialize(graph)),
        })
        {
            GraphValidator.Validate(restored);
            Assert.Equal("output", Assert.Single(restored.Nodes).Id.Value);
            Assert.Equal("output", Assert.Single(restored.Outputs).Value);
            Assert.Equal("output", restored.Nodes[0].Resources.Single(binding => binding.Port == "output").Resource.Value);
        }
    }

    [Fact]
    public void LogicalGraph_RoundTripsAllGraphFields()
    {
        var original = CreateLogical();
        var xml = GraphXml.Serialize(original);
        var restored = GraphXml.DeserializeLogical(xml);

        Assert.Contains("kind=\"LogicalGraph\" version=\"1\"", xml, StringComparison.Ordinal);
        Assert.Equal(original.Identity, restored.Identity);
        Assert.Equal(original.Model, restored.Model);
        Assert.Equal(original.Resources.Select(r => (r.Id, r.Name, r.Kind, r.Lifetime, r.BindingKey, r.Tensor.ElementType)),
            restored.Resources.Select(r => (r.Id, r.Name, r.Kind, r.Lifetime, r.BindingKey, r.Tensor.ElementType)));
        Assert.Equal(original.Regions.Select(r => (r.Id, r.ParentId, r.Type, r.Role, r.Name)),
            restored.Regions.Select(r => (r.Id, r.ParentId, r.Type, r.Role, r.Name)));
        Assert.Equal(original.Regions[1].Attributes, restored.Regions[1].Attributes);
        Assert.Equal(original.Inputs, restored.Inputs);
        Assert.Equal(original.Outputs, restored.Outputs);
        Assert.Equal(original.Nodes.Select(n => (n.Id, n.Operation, n.Region)),
            restored.Nodes.Select(n => (n.Id, n.Operation, n.Region)));
        Assert.Equal(original.Nodes[1].Resources, restored.Nodes[1].Resources);
        Assert.Equal("tensor.weight", restored.Resources[1].BindingKey);
        Assert.Contains(restored.Nodes[1].Resources,
            binding => binding.Port == "weight" && binding.Resource == new ResourceId("weight"));
        Assert.Equal(original.Nodes[1].Dependencies, restored.Nodes[1].Dependencies);
        Assert.Equal(original.Nodes[1].Attributes, restored.Nodes[1].Attributes);
        Assert.Equal(original.Nodes[1].Requirements, restored.Nodes[1].Requirements);
        Assert.Equal(original.Resources[1].Tensor.Dimensions, restored.Resources[1].Tensor.Dimensions);
        Assert.Equal(original.Resources[1].Tensor.Layout, restored.Resources[1].Tensor.Layout);
        Assert.Equal(xml, GraphXml.Serialize(restored));
    }

    [Fact]
    public void ExecutionGraph_RoundTripsSourceAndPorts()
    {
        var logical = CreateLogical();
        var scratch = new GraphResource(new ResourceId("scratch"), "fused scratch", GraphResourceKind.Temporary,
            GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float16, [2, 3]));
        var unproved = new GraphResource(new ResourceId("unproved"), "unproved scratch", GraphResourceKind.Temporary,
            GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float16, [2, 3]));
        var publicBindings = logical.Nodes[1].Resources
            .Where(binding => binding.Port != "out")
            .Concat(
            [
                new NodeResourceBinding("alias.read", new ResourceId("output"), GraphResourceAccess.Read),
                new NodeResourceBinding("alias.write", new ResourceId("output"), GraphResourceAccess.Write),
            ]).ToArray();
        var original = new ExecutionGraph(logical.Identity, logical.Model,
            logical.Resources.Concat([scratch, unproved]), logical.Regions,
        [
            new ExecutionNode(new ExecutionNodeId("run"), new GraphOperationId("custom.fused", 3),
                new RegionId("stage"), publicBindings, [],
                logical.Nodes[1].Attributes, logical.Nodes[1].Requirements,
                new ExecutionSourceMap([new LogicalNodeId("first"), new LogicalNodeId("second")], "fuse-rule"))
            {
                InternalResources =
                [
                    new NodeResourceBinding("scratch.scratch", scratch.Id, GraphResourceAccess.ReadWrite, true),
                    new NodeResourceBinding("scratch.unproved", unproved.Id, GraphResourceAccess.ReadWrite),
                ],
            },
        ], logical.Inputs, logical.Outputs);

        var xml = GraphXml.Serialize(original);
        var restored = GraphXml.DeserializeExecution(xml);

        var node = Assert.Single(restored.Nodes);
        Assert.Equal(original.Nodes[0].Operation, node.Operation);
        Assert.Equal(original.Nodes[0].Resources, node.Resources);
        Assert.Equal(
            [("alias.read", GraphResourceAccess.Read), ("alias.write", GraphResourceAccess.Write)],
            node.Resources.Where(binding => binding.Resource == new ResourceId("output"))
                .Select(binding => (binding.Port, binding.Access)));
        Assert.Equal(original.Nodes[0].InternalResources, node.InternalResources);
        Assert.True(node.InternalResources[0].InitializedBeforeRead);
        Assert.False(node.InternalResources[1].InitializedBeforeRead);
        Assert.Equal(1, xml.Split("initializedBeforeRead=\"true\"", StringSplitOptions.None).Length - 1);
        Assert.DoesNotContain("initializedBeforeRead=\"false\"", xml, StringComparison.Ordinal);
        Assert.DoesNotContain(node.Resources, binding => binding.Resource == scratch.Id);
        Assert.DoesNotContain(node.InternalResources, binding => binding.Resource == new ResourceId("weight"));
        Assert.Equal(node.Id, restored.InternalResourceOwners[scratch.Id]);
        Assert.DoesNotContain("InternalResourceOwners", xml, StringComparison.Ordinal);
        Assert.Equal(["first", "second"], node.Source.LogicalNodes.Select(id => id.Value));
        Assert.Equal("fuse-rule", node.Source.AppliedRuleId);
        Assert.Equal(xml, GraphXml.Serialize(restored));
    }

    [Fact]
    public void DeviceAndScopeRoundTripAndRejectContradictoryScope()
    {
        var logical = CreateLogical();
        var resources = logical.Resources.Select(resource =>
            resource with { DeviceId = resource.Id == new ResourceId("weight") ? "gpu0" : null }).ToArray();
        var graph = new LogicalGraph(logical.Identity, logical.Model, resources, logical.Regions,
            logical.Nodes, logical.Inputs, logical.Outputs);
        var xml = GraphXml.Serialize(graph);

        Assert.Contains("scope=\"Global\"", xml, StringComparison.Ordinal);
        Assert.Contains("deviceId=\"gpu0\"", xml, StringComparison.Ordinal);
        Assert.Equal(xml, GraphXml.Serialize(GraphXml.DeserializeLogical(xml)));
        Assert.Equal("gpu0", GraphXml.DeserializeLogical(xml).Resources.Single(r => r.Kind == GraphResourceKind.Weight).DeviceId);
        Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeLogical(
            xml.Replace("scope=\"Global\"", "scope=\"Local\"", StringComparison.Ordinal)));
        var legacy = xml.Replace(" scope=\"Global\"", "", StringComparison.Ordinal)
            .Replace(" scope=\"Session\"", "", StringComparison.Ordinal)
            .Replace(" scope=\"Local\"", "", StringComparison.Ordinal);
        Assert.Equal(graph.Resources.Select(resource => resource.Scope),
            GraphXml.DeserializeLogical(legacy).Resources.Select(resource => resource.Scope));
    }

    [Fact]
    public void ExternalInputAndOutputAreLocalBoundariesNotSharedGlobalStorage()
    {
        var logical = CreateLogical();
        var input = logical.Resources.Single(resource => resource.Kind == GraphResourceKind.Input);
        var output = logical.Resources.Single(resource => resource.Kind == GraphResourceKind.Output);
        Assert.Equal(GraphResourceLifetime.External, input.Lifetime);
        Assert.Equal(GraphResourceScope.Local, input.Scope);
        Assert.Equal(GraphResourceScope.Local, output.Scope);
        var execution = new GraphOptimizer().Optimize(logical);
        var xmlOnly = new ExecutionGraph(execution.Identity, execution.Model, execution.Resources,
            execution.Regions, execution.Nodes.Select(node => node with { Source = ExecutionSourceMap.XmlOnly }),
            execution.Inputs, execution.Outputs);
        var restored = GraphXml.DeserializeExecution(GraphXml.Serialize(xmlOnly));
        Assert.All(restored.Nodes, node => Assert.Empty(node.Source.LogicalNodes));
        Assert.Equal(GraphResourceScope.Local, restored.Resources.Single(r => r.Id == input.Id).Scope);
        Assert.Equal(GraphResourceScope.Local, restored.Resources.Single(r => r.Id == output.Id).Scope);
        var plan = GraphResourceAllocator.PlanLocal(restored, _ => true);
        Assert.False(plan.SlotByResource.ContainsKey(input.Id));
        Assert.False(plan.SlotByResource.ContainsKey(output.Id));
    }

    [Theory]
    [InlineData("true")]
    [InlineData("false")]
    [InlineData("TRUE")]
    [InlineData("1")]
    [InlineData("")]
    public void Import_RejectsPublicBindingInitializationMarker(string value)
    {
        var xml = GraphXml.Serialize(CreateLogical())
            .Replace("port=\"in\"", $"port=\"in\" initializedBeforeRead=\"{value}\"", StringComparison.Ordinal);
        Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeLogical(xml));
    }

    [Theory]
    [InlineData("TRUE")]
    [InlineData("1")]
    [InlineData("")]
    public void Import_RejectsInvalidPrivateInitializationMarker(string value)
    {
        var logical = CreateLogical();
        var scratch = new GraphResource(new ResourceId("scratch"), "scratch", GraphResourceKind.Temporary,
            GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float16, [2, 3]));
        var graph = new ExecutionGraph(logical.Identity, logical.Model, logical.Resources.Append(scratch), logical.Regions,
        [
            new ExecutionNode(new ExecutionNodeId("run"), new GraphOperationId("custom.write"),
                new RegionId("stage"), logical.Nodes[1].Resources, [], logical.Nodes[1].Attributes,
                logical.Nodes[1].Requirements, ExecutionSourceMap.XmlOnly)
            {
                InternalResources = [new NodeResourceBinding("scratch", scratch.Id, GraphResourceAccess.Write, true)],
            },
        ], logical.Inputs, logical.Outputs);
        var xml = GraphXml.Serialize(graph).Replace("initializedBeforeRead=\"true\"",
            $"initializedBeforeRead=\"{value}\"", StringComparison.Ordinal);
        Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeExecution(xml));
    }

    [Fact]
    public void Import_AcceptsExplicitFalsePrivateInitializationMarker()
    {
        var logical = CreateLogical();
        var scratch = new GraphResource(new ResourceId("scratch"), "scratch", GraphResourceKind.Temporary,
            GraphResourceLifetime.Invocation, new TensorDescriptor(GraphElementType.Float16, [2, 3]));
        var graph = new ExecutionGraph(logical.Identity, logical.Model, logical.Resources.Append(scratch), logical.Regions,
        [
            new ExecutionNode(new ExecutionNodeId("run"), new GraphOperationId("custom.write"),
                new RegionId("stage"), logical.Nodes[1].Resources, [], logical.Nodes[1].Attributes,
                logical.Nodes[1].Requirements, ExecutionSourceMap.XmlOnly)
            {
                InternalResources = [new NodeResourceBinding("scratch", scratch.Id, GraphResourceAccess.Write)],
            },
        ], logical.Inputs, logical.Outputs);

        var xml = GraphXml.Serialize(graph).Replace("port=\"scratch\" resource=\"scratch\" access=\"Write\" />",
            "port=\"scratch\" resource=\"scratch\" access=\"Write\" initializedBeforeRead=\"false\" />",
            StringComparison.Ordinal);
        Assert.Contains("initializedBeforeRead=\"false\"", xml, StringComparison.Ordinal);
        var restored = GraphXml.DeserializeExecution(xml);
        Assert.False(Assert.Single(Assert.Single(restored.Nodes).InternalResources).InitializedBeforeRead);
        Assert.DoesNotContain("initializedBeforeRead", GraphXml.Serialize(restored), StringComparison.Ordinal);

        var invalidRead = xml.Replace("port=\"scratch\" resource=\"scratch\" access=\"Write\" initializedBeforeRead=\"false\"",
            "port=\"scratch\" resource=\"scratch\" access=\"Read\" initializedBeforeRead=\"true\"",
            StringComparison.Ordinal);
        Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeExecution(invalidRead));
    }

    [Fact]
    public void ExecutionGraph_RoundTripsAbsentSource()
    {
        var logical = CreateLogical();
        var original = new ExecutionGraph(logical.Identity, logical.Model, logical.Resources, logical.Regions,
        [
            new ExecutionNode(new ExecutionNodeId("xml-only"), new GraphOperationId("custom.write"),
                new RegionId("stage"), logical.Nodes[1].Resources, [],
                logical.Nodes[1].Attributes, logical.Nodes[1].Requirements, ExecutionSourceMap.XmlOnly),
        ], logical.Inputs, logical.Outputs);

        var xml = GraphXml.Serialize(original);
        Assert.DoesNotContain("<Source", xml, StringComparison.Ordinal);
        Assert.DoesNotContain("<LogicalGraph", xml, StringComparison.Ordinal);
        var restored = GraphXml.DeserializeExecution(xml);
        Assert.Empty(Assert.Single(restored.Nodes).Source.LogicalNodes);
        Assert.Null(restored.Nodes[0].Source.AppliedRuleId);
        Assert.Equal(xml, GraphXml.Serialize(restored));

        var explicitEmptySource = xml.Replace("</Node>", "<Source /></Node>", StringComparison.Ordinal);
        Assert.Empty(Assert.Single(GraphXml.DeserializeExecution(explicitEmptySource).Nodes).Source.LogicalNodes);
        var emptySourceWithRule = xml.Replace("</Node>", "<Source appliedRuleId=\"rule\" /></Node>", StringComparison.Ordinal);
        var withRule = Assert.Single(GraphXml.DeserializeExecution(emptySourceWithRule).Nodes);
        Assert.Empty(withRule.Source.LogicalNodes);
        Assert.Equal("rule", withRule.Source.AppliedRuleId);
        Assert.Equal(emptySourceWithRule, GraphXml.Serialize(GraphXml.DeserializeExecution(emptySourceWithRule)));
    }

    [Theory]
    [InlineData("LogicalGraph", "ExecutionGraph")]
    [InlineData("LogicalGraph", "OtherGraph")]
    public void Import_RejectsWrongKind(string expected, string actual)
    {
        var xml = GraphXml.Serialize(CreateLogical()).Replace("kind=\"LogicalGraph\"", $"kind=\"{actual}\"");
        Assert.Throws<InvalidDataException>(() =>
        {
            if (expected == "LogicalGraph") GraphXml.DeserializeLogical(xml);
            else GraphXml.DeserializeExecution(xml);
        });
    }

    [Fact]
    public void Import_RejectsDtdAndOversizedInput()
    {
        var xml = GraphXml.Serialize(CreateLogical());
        Assert.Throws<InvalidDataException>(() =>
            GraphXml.DeserializeLogical("<!DOCTYPE Graph [<!ENTITY x 'bad'>]>" + xml));
        Assert.Throws<InvalidDataException>(() =>
            GraphXml.DeserializeLogical(new string('x', 16 * 1024 * 1024 + 1)));
    }

    [Theory]
    [InlineData("version=\"1\"", "version=\"99\"")]
    [InlineData("kind=\"LogicalGraph\"", "kind=\"logicalgraph\"")]
    [InlineData("access=\"Read\"", "access=\"Invalid\"")]
    [InlineData("resource=\"input\"", "resource=\"missing\"")]
    [InlineData("operationVersion=\"2\"", "operationVersion=\"nope\"")]
    [InlineData("elementType=\"Float16\"", "elementType=\"Float99\"")]
    [InlineData("port=\"in\"", "port=\"in\" unexpected=\"true\"")]
    [InlineData("<Inputs>", "<Inputs extra=\"1\">")]
    [InlineData("<Attributes>", "<Unexpected>")]
    public void Import_RejectsInvalidSchemaAndReferences(string before, string after)
    {
        var xml = GraphXml.Serialize(CreateLogical()).Replace(before, after, StringComparison.Ordinal);
        Assert.NotEqual(GraphXml.Serialize(CreateLogical()), xml);
        Assert.Throws<InvalidDataException>(() => GraphXml.DeserializeLogical(xml));
    }

    private static LogicalGraph CreateLogical()
    {
        var resources = new[]
        {
            new GraphResource(new ResourceId("input"), "Input < & \"", GraphResourceKind.Input,
                GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Float16, [2, 3])),
            new GraphResource(new ResourceId("weight"), "weight", GraphResourceKind.Weight,
                GraphResourceLifetime.Model, new TensorDescriptor(GraphElementType.Float32, [3], "custom-layout"), "tensor.weight"),
            new GraphResource(new ResourceId("output"), "result", GraphResourceKind.Output,
                GraphResourceLifetime.External, new TensorDescriptor(GraphElementType.Float16, [2, 3])),
        };
        var regions = new[]
        {
            new GraphRegion(new RegionId("root"), null, GraphRegionTypes.Graph, null, "graph", new Dictionary<string, string>()),
            new GraphRegion(new RegionId("stage"), new RegionId("root"), GraphRegionTypes.Stage, "compute",
                "test stage", new Dictionary<string, string> { ["label"] = "<xml> & \"value\"" }),
        };
        var nodes = new[]
        {
            new LogicalNode(new LogicalNodeId("first"), new GraphOperationId("custom.read", 2), new RegionId("stage"),
                [new NodeResourceBinding("in", new ResourceId("input"), GraphResourceAccess.Read)],
                [], new Dictionary<string, string>(), new PrecisionRequirement(GraphElementType.Float16, GraphElementType.Float32)),
            new LogicalNode(new LogicalNodeId("second"), new GraphOperationId("custom.write"), new RegionId("stage"),
                [
                    new NodeResourceBinding("in", new ResourceId("input"), GraphResourceAccess.Read),
                    new NodeResourceBinding("weight", new ResourceId("weight"), GraphResourceAccess.Read),
                    new NodeResourceBinding("out", new ResourceId("output"), GraphResourceAccess.Write),
                ],
                [new LogicalNodeId("first")], new Dictionary<string, string> { ["activation"] = "a&b" },
                new PrecisionRequirement(GraphElementType.Float16, GraphElementType.Float32)),
        };
        return new LogicalGraph(new GraphIdentity("test", 2, "special < & \" graph"),
            TestGraphSignatures.Create(16, 8, 1, 2, 4, "state.abi"), resources, regions, nodes,
            [new ResourceId("input")], [new ResourceId("output")]);
    }
}
