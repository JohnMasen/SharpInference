using System.Xml.Linq;
using System.Runtime.InteropServices;
using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;

namespace SharpInference.Tests;

public sealed class GraphSerializationTests
{
    private static readonly TensorDescriptor Vector = new(GraphElementType.Float32, [4]);

    private const string LogicalFixture = """
        <Graph kind="LogicalGraphDefinition" version="3">
          <Root definition="copy" />
          <Definitions>
            <Definition id="copy" compileBoundary="Atomic">
              <Bindings>
                <Binding id="copy:input" node="copy" port="input" resource="input"
                         access="Read" readEdge="read-custom" readOrder="1" />
                <Binding id="copy:output" node="copy" port="output" resource="output"
                         access="Write" writeEdge="write-custom" writeOrder="0" />
              </Bindings>
              <Dependencies />
              <Nodes>
                <Operation id="copy" operation="core.copy" operationVersion="1">
                  <Attributes /><Precision arithmetic="Float32" accumulator="Float32" />
                </Operation>
                <Resource id="output" name="Result" kind="Output" owner="copy">
                  <Tensor elementType="Float32" layout="dense"><Dimension value="4" /></Tensor>
                </Resource>
                <Resource id="input" name="Argument" kind="Input" owner="copy" bindingKey="input-key">
                  <Tensor elementType="Float32" layout="dense"><Dimension value="4" /></Tensor>
                </Resource>
              </Nodes>
            </Definition>
          </Definitions>
        </Graph>
        """;

    [Fact]
    public void IndependentFixtureResolvesForwardReferencesAndPreservesEdgeIdentityAndOrder()
    {
        var graph = LogicalGraphXml.DeserializeDefinition(LogicalFixture);
        Assert.Equal("copy", graph.Id.Value);
        Assert.Equal(GraphCompileBoundary.Atomic, graph.CompileBoundary);
        Assert.Equal(["write-custom", "read-custom"], graph.Structure.Edges.Select(edge => edge.Id.Value));
        Assert.Equal("input-key", graph.Resources.Single(resource => resource.Id.Value == "input").BindingKey);
        var node = Assert.IsType<GraphOperationNode>(Assert.Single(graph.Nodes));
        Assert.Equal(PrimitiveGraphOperations.Copy, node.Operation);
        Assert.Equal(["output", "input"], graph.Bindings(node).Select(binding => binding.Port));
        Assert.Equal(["copy"], graph.CompletionSinks.Select(id => id.Value));
        Assert.Empty(graph.Validate());
        var xml = LogicalGraphXml.Serialize(graph);
        Assert.Contains("<Bindings>", xml, StringComparison.Ordinal);
        Assert.DoesNotContain("<Edges>", xml, StringComparison.Ordinal);
        Assert.Equal(xml, LogicalGraphXml.Serialize(LogicalGraphXml.DeserializeDefinition(xml)));
    }

    [Fact]
    public void IndependentExecutionFixturePreservesSourcesAndExplicitDependencies()
    {
        const string xml = """
            <Graph kind="ExecutionGraphProgram" version="3">
              <Root graph="root" />
              <Graphs>
                <ExecutionGraph id="root" compileBoundary="Transparent">
                  <Source definition="logical-root" />
                  <Architecture name="Root" role="model"><Attribute name="layer" value="0" /></Architecture>
                  <Nodes>
                    <Operation id="first" operation="core.copy" operationVersion="1">
                      <SourceMap transformation="fixture"><LogicalNode id="origin" /></SourceMap>
                      <Attributes /><Precision arithmetic="Float32" accumulator="Float32" />
                    </Operation>
                    <Operation id="second" operation="core.relu" operationVersion="1">
                      <SourceMap><LogicalNode id="origin-2" /></SourceMap>
                      <Attributes /><Precision arithmetic="Float32" accumulator="Float32" />
                    </Operation>
                    <Resource id="x" name="x" kind="Input">
                      <Tensor elementType="Float32" layout="dense"><Dimension value="4" /></Tensor>
                    </Resource>
                    <Resource id="y" name="y" kind="Output">
                      <Tensor elementType="Float32" layout="dense"><Dimension value="4" /></Tensor>
                    </Resource>
                  </Nodes>
                  <Bindings>
                    <Binding id="f:i" node="first" port="input" resource="x" access="Read"
                             readEdge="fi" readOrder="0" />
                    <Binding id="s:o" node="second" port="output" resource="y" access="Write"
                             writeEdge="so" writeOrder="2" />
                  </Bindings>
                  <Dependencies><Dependency id="custom-dependency" source="first" target="second" order="1" /></Dependencies>
                </ExecutionGraph>
              </Graphs>
            </Graph>
            """;
        var program = ExecutionGraphXml.DeserializeProgram(xml);
        var graph = Assert.Single(program.Graphs);
        Assert.Equal(["fi", "custom-dependency", "so"], graph.Structure.Edges.Select(edge => edge.Id.Value));
        Assert.Equal("logical-root", graph.Source!.Definition.Value);
        Assert.Equal("model", graph.Architecture!.Role);
        Assert.Equal("0", graph.Architecture.Attributes!["layer"]);
        Assert.Equal(GraphCompileBoundary.Transparent, graph.DefaultCompileBoundary);
        Assert.Equal("fixture", graph.Nodes[0].Source.Transformation);
        Assert.Equal(["origin"], graph.Nodes[0].Source.LogicalNodes.Select(node => node.Value));
        Assert.Equal(["first"], graph.Dependencies(graph.Nodes[1]).Select(node => node.Value));
        var serialized = ExecutionGraphXml.Serialize(program);
        Assert.Equal(serialized, ExecutionGraphXml.Serialize(ExecutionGraphXml.DeserializeProgram(serialized)));
        Assert.Empty(program.Validate());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void OldAndUnknownXmlVersionsAreRejected(int version)
    {
        var xml = LogicalFixture.Replace("version=\"3\"", $"version=\"{version}\"", StringComparison.Ordinal);
        Assert.Contains("Unsupported", Assert.Throws<InvalidDataException>(
            () => LogicalGraphXml.DeserializeDefinition(xml)).Message, StringComparison.Ordinal);
        var execution = $"<Graph kind=\"ExecutionGraphProgram\" version=\"{version}\" />";
        Assert.Contains("Unsupported", Assert.Throws<InvalidDataException>(
            () => ExecutionGraphXml.DeserializeProgram(execution)).Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("resource=\"input\"", "resource=\"missing\"")]
    [InlineData("access=\"Read\"", "access=\"Bogus\"")]
    [InlineData("readOrder=\"1\"", "readOrder=\"0\"")]
    [InlineData("node=\"copy\"", "node=\"missing\"")]
    [InlineData("kind=\"Input\"", "kind=\"Bogus\"")]
    public void MalformedReferencesAndRelationsAreRejected(string before, string after)
    {
        var xml = LogicalFixture.Replace(before, after, StringComparison.Ordinal);
        Assert.Throws<InvalidDataException>(() => LogicalGraphXml.DeserializeDefinition(xml));
    }

    [Fact]
    public void DuplicateDefinitionAndUnsupportedContainersAreRejected()
    {
        var document = XDocument.Parse(LogicalFixture);
        var definitions = document.Root!.Element("Definitions")!;
        definitions.Add(new XElement(definitions.Element("Definition")!));
        Assert.Throws<InvalidDataException>(() => LogicalGraphXml.DeserializeDefinition(document.ToString()));
        document = XDocument.Parse(LogicalFixture);
        document.Root!.Element("Definitions")!.Element("Definition")!.Add(new XElement("Edges"));
        Assert.Throws<InvalidDataException>(() => LogicalGraphXml.DeserializeDefinition(document.ToString()));
    }

    [Fact]
    public void DtdAndExcessiveDepthAreRejected()
    {
        var dtd = "<!DOCTYPE Graph [<!ENTITY x 'copy'>]>" + LogicalFixture;
        Assert.Throws<InvalidDataException>(() => LogicalGraphXml.DeserializeDefinition(dtd));
        var deep = LogicalFixture.Replace("<Dependencies />",
            string.Concat(Enumerable.Repeat("<Nested>", 70)) +
            string.Concat(Enumerable.Repeat("</Nested>", 70)), StringComparison.Ordinal);
        Assert.Throws<InvalidDataException>(() => LogicalGraphXml.DeserializeDefinition(deep));
    }

    [Fact]
    public void XmlCharacterAndElementLimitsRemainEnforced()
    {
        var oversized = new string(' ', 16 * 1024 * 1024 + 1);
        Assert.Contains("size limit", Assert.Throws<InvalidDataException>(
            () => LogicalGraphXml.DeserializeDefinition(oversized)).Message, StringComparison.Ordinal);
        var tooManyElements = "<Graph kind=\"LogicalGraphDefinition\" version=\"3\">" +
            string.Concat(Enumerable.Repeat("<Element />", 100_001)) + "</Graph>";
        Assert.Contains("structural limit", Assert.Throws<InvalidDataException>(
            () => LogicalGraphXml.DeserializeDefinition(tooManyElements)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ReadWriteBindingIsSerializedOnceAndSharedCallsRemainDistinct()
    {
        var child = LogicalGraphDefinition.Define("stateful", builder =>
        {
            var state = builder.AddInOut("state", Vector);
            builder.AddNode("update", new GraphOperationId("test.update"),
                node => { node.AddInOut("state", state); });
        });
        var parent = LogicalGraphDefinition.Define("parent", builder =>
        {
            var first = builder.AddResource("first", "first", Vector, GraphDefinitionResourceKind.State, "first-key");
            var second = builder.AddResource("second", "second", Vector, GraphDefinitionResourceKind.State, "second-key");
            builder.AddCall("first-call", child, node => { node.AddInOut("state", first); });
            builder.AddCall("second-call", child, node => { node.AddInOut("state", second); });
        });
        var xml = LogicalGraphXml.Serialize(parent);
        Assert.Equal(3, XDocument.Parse(xml).Descendants("Binding")
            .Count(element => (string?)element.Attribute("access") == "ReadWrite"));
        var restored = LogicalGraphXml.DeserializeDefinition(xml);
        Assert.Single(restored.Definitions);
        Assert.Equal(["first", "second"], restored.Nodes
            .Select(node => Assert.Single(restored.Bindings(node)).Resource.Resource.Value));
        Assert.Equal(xml, LogicalGraphXml.Serialize(restored));
        var execution = new ExecutionGraphCompiler().Compile(restored, new("test", "cpu"));
        var serialized = ExecutionGraphXml.Serialize(execution);
        Assert.Equal(serialized, ExecutionGraphXml.Serialize(ExecutionGraphXml.DeserializeProgram(serialized)));
    }

    [Fact]
    public void CodeAndXmlGraphsProduceTheSameVmCpuSource()
    {
        var code = LogicalGraphDefinition.CreateListGraph("chain", Vector,
            [PrimitiveGraphOperations.Copy, PrimitiveGraphOperations.Relu]);
        var restored = LogicalGraphXml.DeserializeDefinition(LogicalGraphXml.Serialize(code));
        var context = new ExecutionGraphCompilationContext("test", "cpu");
        var first = new DefaultExecutionGraphConverter().Convert(code, context);
        var second = new DefaultExecutionGraphConverter().Convert(restored, context);
        Assert.Equal(ExecutionGraphXml.Serialize(first), ExecutionGraphXml.Serialize(second));
        SharpInference.Instructions.IInstructionCollectionProvider[] providers =
            [new SharpInference.Instructions.Cpu.CpuFloat32InstructionCollection()];
        var emitter = new SharpInference.Vm.Optimization.VmExecutionGraphEmitter(SharpInference.Vm.VmTarget.Cpu, providers);
        var compiler = new SharpInference.Backends.CpuVm.CpuVmCompiler(providers);
        Assert.Equal(compiler.GenerateSource(emitter.Emit(first)).Source,
            compiler.GenerateSource(emitter.Emit(second)).Source);
    }

    [Fact]
    public void NamedResidualGraphRunsOnCpuAfterXmlRoundTrip()
    {
        var logical = LogicalGraphDefinition.Define("residual", builder =>
        {
            builder.AddInput("input", Vector);
            builder.AddOutput("output", Vector);
            builder.AddResource("positive", "Positive", Vector);
            builder.AddNode("add", PrimitiveGraphOperations.Add, node =>
            {
                node.AddInput("left", new ResourceId("positive"));
                node.AddInput("right", new ResourceId("input"));
                node.AddOutput("output", new ResourceId("output"));
            });
            builder.AddNode("relu", PrimitiveGraphOperations.Relu, node =>
            {
                node.AddInput("input", new ResourceId("input"));
                node.AddOutput("output", new ResourceId("positive"));
            });
        });
        logical = LogicalGraphXml.DeserializeDefinition(LogicalGraphXml.Serialize(logical));
        var execution = new ExecutionGraphCompiler().Compile(logical, new("test", "cpu"));
        execution = ExecutionGraphXml.DeserializeProgram(ExecutionGraphXml.Serialize(execution));
        SharpInference.Instructions.IInstructionCollectionProvider[] providers =
            [new SharpInference.Instructions.Cpu.CpuFloat32InstructionCollection()];
        var program = new SharpInference.Vm.Optimization.VmExecutionGraphEmitter(
            SharpInference.Vm.VmTarget.Cpu, providers).Emit(execution);
        var artifact = new SharpInference.Backends.CpuVm.CpuVmCompiler(providers).Compile(program);
        using var executor = artifact.CreateExecutor();
        var buffers = program.Slots.Select(slot => new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();
        var input = program.Slots.ToList().FindIndex(slot => slot.Id == "input");
        var output = program.Slots.ToList().FindIndex(slot => slot.Id == "output");
        new float[] { -2, -1, 0, 3 }.AsSpan().CopyTo(MemoryMarshal.Cast<byte, float>(buffers[input].AsSpan()));
        executor.Execute("forward", buffers);
        Assert.Equal([-2f, -1f, 0f, 6f], MemoryMarshal.Cast<byte, float>(buffers[output]).ToArray());
    }

    [Fact]
    public void ExecutionBuilderReportsMissingReferencesAndIsRepeatable()
    {
        var builder = new ExecutionGraphBuilder(new("root"));
        var operation = new ExecutionOperationNode(new("copy"), PrimitiveGraphOperations.Copy,
            new Dictionary<string, string>(), new(GraphElementType.Float32, GraphElementType.Float32), new([]));
        builder.AddOperation(operation,
        [
            new("input", new(GraphBindingOwner.Parent, new("input")), GraphResourceAccess.Read, "copy:input"),
            new("output", new(GraphBindingOwner.Parent, new("output")), GraphResourceAccess.Write, "copy:output"),
        ]);
        Assert.Equal(2, builder.Validate().Count);
        Assert.Throws<InvalidDataException>(() => builder.Build());
        builder.AddResource(new(new("input"), "input", GraphDefinitionResourceKind.Input, Vector));
        builder.AddResource(new(new("output"), "output", GraphDefinitionResourceKind.Output, Vector));
        Assert.Empty(builder.Validate());
        var graph = builder.Build();
        Assert.Equal(2, graph.Structure.Edges.Count);
        Assert.Equal(graph.Structure.Edges, builder.Build().Structure.Edges);
        builder.AddResource(new(new("later"), "later", GraphDefinitionResourceKind.Temporary, Vector));
        Assert.Equal(2, graph.Resources.Count);
        Assert.Equal(3, builder.Build().Resources.Count);
    }

    [Fact]
    public void RecursiveTargetNamesAreRejectedBeforeCoreMaterialization()
    {
        var builder = new GraphDefinitionBuilder("self");
        builder.AddCall("recursive", new GraphDefinitionId("self"), _ => { });
        Assert.Contains(builder.Validate(), diagnostic =>
            diagnostic.Code == "LG0030" && diagnostic.Node == "recursive");
        Assert.Throws<InvalidDataException>(() => builder.Build());
    }
}
