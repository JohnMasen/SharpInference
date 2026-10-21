using System.Runtime.InteropServices;
using System.Xml.Linq;
using SharpInference.Backends.CpuVm;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Tests;

public sealed class GraphSerializationTests
{
    private static readonly TensorDescriptor Vector = new(ElementType.Float32, [4]);
    private const string Fixture = """
        <GraphPackage version="1" stage="logical" main="main">
          <Graphs>
            <Graph id="main" name="Copy">
              <Resources>
                <Resource id="input" category="custom.input" graphIO="In">
                  <Tensor elementType="Float32" layout="dense"><Dimension value="4" /></Tensor>
                </Resource>
                <Resource id="output" category="resource" graphIO="Out">
                  <Tensor elementType="Float32" layout="dense"><Dimension value="4" /></Tensor>
                </Resource>
              </Resources>
              <Actions>
                <Operation id="copy" category="custom.copy" operation="core.copy" version="1">
                  <Attributes />
                  <Binding property="input" targetKind="Graph" targetProperty="input" access="Read" />
                  <Binding property="output" targetKind="Graph" targetProperty="output" access="Write" />
                </Operation>
              </Actions>
              <GraphFlow />
            </Graph>
          </Graphs>
        </GraphPackage>
        """;

    [Fact]
    public void IndependentPackageFixturePreservesCategoriesBindingsAndCanonicalRoundTrip()
    {
        var package = GraphPackageXml.Deserialize<LogicalGraph>(Fixture);
        Assert.Empty(package.Validate());
        Assert.Equal("main", package.MainGraphId);
        Assert.Equal("Copy", package.MainGraph!.Name);
        Assert.Equal("custom.input", package.MainGraph.Resources[0].Category);
        var operation = Assert.IsType<OperationAction>(Assert.Single(package.MainGraph.Actions));
        Assert.Equal("custom.copy", operation.Category);
        Assert.Equal(["input", "output"], operation.Bindings.Select(binding => binding.SourceProperty));
        var xml = GraphPackageXml.Serialize(package);
        Assert.Equal(xml, GraphPackageXml.Serialize(GraphPackageXml.Deserialize<LogicalGraph>(xml)));
    }

    [Fact]
    public void ExecutionRoundTripRetainsSelectedImplementationStorageSourcesAndFlow()
    {
        var package = Convert(Residual());
        var operation = Assert.IsType<ExecutionOperation>(package.MainGraph!.Actions[0]);
        operation.Source("fixture", "original", "add");
        package.MainGraph.Metadata["role"] = "residual";
        var xml = GraphPackageXml.Serialize(package);
        var loaded = GraphPackageXml.Deserialize<ExecutionGraph>(xml);
        Assert.Empty(loaded.Validate());
        Assert.Equal("cpu-vm", loaded.GetTarget());
        Assert.Equal(operation.SelectedImplementation, Assert.IsType<ExecutionOperation>(loaded.MainGraph!.Actions[0]).SelectedImplementation);
        Assert.Equal(operation.Sources, Assert.IsType<ExecutionOperation>(loaded.MainGraph.Actions[0]).Sources);
        Assert.Equal(package.MainGraph.GraphFlow, loaded.MainGraph.GraphFlow);
        Assert.Equal(package.MainGraph.Storage.OrderBy(pair => pair.Key), loaded.MainGraph.Storage.OrderBy(pair => pair.Key));
        Assert.Equal(xml, GraphPackageXml.Serialize(loaded));
    }

    [Theory]
    [InlineData("version=\"1\" stage=\"logical\"", "version=\"99\" stage=\"logical\"")]
    [InlineData("stage=\"logical\"", "stage=\"execution\"")]
    [InlineData("access=\"Read\"", "access=\"Bogus\"")]
    [InlineData("targetKind=\"Graph\"", "targetKind=\"Bogus\"")]
    [InlineData("category=\"custom.copy\"", "unexpected=\"value\"")]
    public void UnsupportedOrMalformedXmlIsRejected(string before, string after)
    {
        Assert.Throws<InvalidDataException>(() =>
            GraphPackageXml.Deserialize<LogicalGraph>(Fixture.Replace(before, after, StringComparison.Ordinal)));
    }

    [Fact]
    public void MissingReferencesAreReportedByPackageValidation()
    {
        var package = GraphPackageXml.Deserialize<LogicalGraph>(
            Fixture.Replace("targetProperty=\"input\"", "targetProperty=\"missing\"", StringComparison.Ordinal));
        Assert.Contains(package.Validate(), diagnostic => diagnostic.Code == "G0012");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void XmlSafetyLimitsRejectDtdOversizedDeepAndNumerousElements(bool execution)
    {
        void Read(string xml)
        {
            if (execution)
                GraphPackageXml.Deserialize<ExecutionGraph>(xml);
            else
                GraphPackageXml.Deserialize<LogicalGraph>(xml);
        }
        var xml = execution ? GraphPackageXml.Serialize(Convert(Residual())) : Fixture;
        Assert.Throws<InvalidDataException>(() => Read("<!DOCTYPE GraphPackage [<!ENTITY x 'copy'>]>" + xml));
        Assert.Contains("size limit", Assert.Throws<InvalidDataException>(() =>
            Read(new string(' ', 16 * 1024 * 1024 + 1))).Message, StringComparison.Ordinal);
        var deep = "<GraphPackage>" + string.Concat(Enumerable.Repeat("<Nested>", 70)) +
            string.Concat(Enumerable.Repeat("</Nested>", 70)) + "</GraphPackage>";
        Assert.Contains("structural limit", Assert.Throws<InvalidDataException>(() => Read(deep)).Message, StringComparison.Ordinal);
        var numerous = "<GraphPackage>" + string.Concat(Enumerable.Repeat("<Element />", 100_001)) + "</GraphPackage>";
        Assert.Contains("structural limit", Assert.Throws<InvalidDataException>(() => Read(numerous)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ReadWriteBindingsAndSharedCallsRemainDistinctAfterRoundTrip()
    {
        var child = new LogicalGraph { Id = "child" };
        var state = child.Resource("state", Vector, GraphIO.InOut, isState: true);
        child.Operation("update", "custom.update").Bind("state", state, ResourceAccess.ReadWrite);
        var root = new LogicalGraph { Id = "main" };
        foreach (var id in new[] { "first", "second" })
        {
            var resource = root.Resource(id, Vector, isState: true, bindingKey: id);
            root.Call(id, child).Bind("state", resource, ResourceAccess.ReadWrite);
        }
        var xml = GraphPackageXml.Serialize(new GraphPackage<LogicalGraph>(root, child));
        Assert.Equal(3, XDocument.Parse(xml).Descendants("Binding").Count());
        var loaded = GraphPackageXml.Deserialize<LogicalGraph>(xml);
        Assert.Equal(["first", "second"], loaded.MainGraph!.Actions.Select(action => Assert.Single(action.Bindings).Target.TargetProperty));
        Assert.Empty(loaded.Validate());
        Assert.Equal(xml, GraphPackageXml.Serialize(loaded));
    }

    [Fact]
    public void CodeAndXmlProduceIdenticalCpuSourceAndExecuteResidualCorrectly()
    {
        var source = Residual();
        var restored = GraphPackageXml.Deserialize<LogicalGraph>(GraphPackageXml.Serialize(source));
        IInstructionCollectionProvider[] providers = [new CpuFloat32InstructionCollection()];
        var emitter = new VmExecutionGraphEmitter(VmTarget.Cpu, providers);
        var compiler = new CpuVmCompiler(providers);
        var first = emitter.Emit(Convert(source));
        var execution = Convert(restored);
        execution = GraphPackageXml.Deserialize<ExecutionGraph>(GraphPackageXml.Serialize(execution));
        var program = emitter.Emit(execution);
        Assert.Equal(compiler.GenerateSource(first).Source, compiler.GenerateSource(program).Source);
        using var executor = compiler.Compile(program).CreateExecutor();
        var buffers = program.Slots.Select(slot => new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();
        var input = program.Slots.ToList().FindIndex(slot => slot.Id == "input");
        var output = program.Slots.ToList().FindIndex(slot => slot.Id == "output");
        new float[] { -2, -1, 0, 3 }.AsSpan().CopyTo(MemoryMarshal.Cast<byte, float>(buffers[input].AsSpan()));
        executor.Execute("forward", buffers);
        Assert.Equal([-2f, -1f, 0f, 6f], MemoryMarshal.Cast<byte, float>(buffers[output].AsSpan()).ToArray());
    }

    private static GraphPackage<LogicalGraph> Residual()
    {
        var graph = new LogicalGraph { Id = "main" };
        var input = graph.Resource("input", Vector, GraphIO.In);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        var positive = graph.Resource("positive", Vector);
        graph.Operation("add", Operations.Add).Bind("left", positive, ResourceAccess.Read)
            .Bind("right", input, ResourceAccess.Read).Bind("output", output, ResourceAccess.Write);
        graph.Operation("relu", "core.relu").Bind("input", input, ResourceAccess.Read)
            .Bind("output", positive, ResourceAccess.Write);
        return new(graph);
    }

    private static GraphPackage<ExecutionGraph> Convert(GraphPackage<LogicalGraph> source) =>
        new DefaultExecutionGraphConverter([new CpuFloat32InstructionCollection()]).Convert(source, new("test", "cpu-vm"));
}
