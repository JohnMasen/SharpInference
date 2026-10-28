using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Tests;

public sealed class ExecutionGraphConverterTests
{
    private static readonly TensorDescriptor Vector = new(ElementType.Float32, [4]);

    [Fact]
    public void ConversionPreservesSharedDefinitionsMetadataAndSourcesWithoutMutatingSource()
    {
        var child = CopyGraph("child");
        child.Name = "Shared body";
        child.CompileBoundary = GraphCompileBoundary.Atomic;
        child.Metadata["role"] = "attention";
        var root = new LogicalGraph { Id = "main" };
        var input = root.Resource("input", Vector, GraphIO.In);
        var middle = root.Resource("middle", Vector);
        var output = root.Resource("output", Vector, GraphIO.Out);
        var first = root.Call("first", child);
        first.Metadata["role"] = "first-instance";
        first.Bind("input", input, ResourceAccess.Read).Bind("output", middle, ResourceAccess.Write);
        var second = root.Call("second", child);
        second.Bind("input", middle, ResourceAccess.Read).Bind("output", output, ResourceAccess.Write);
        var source = new LogicalGraphPackage(root, child);
        var before = LogicalGraphPackageXml.Serialize(source);

        var result = Converter().Convert(source, new("test", "cpu-vm"));

        Assert.Equal("main", result.MainGraphId);
        Assert.Equal(["main", "child"], result.Select(graph => graph.Id));
        Assert.All(result.MainGraph!.Actions, action => Assert.Equal("child", Assert.IsType<ExecutionCall>(action).Target));
        Assert.Equal("first-instance", result.MainGraph.Actions[0].Metadata["role"]);
        Assert.Equal("attention", result[1].Metadata["role"]);
        Assert.Equal("Shared body", result[1].Name);
        var operation = Assert.IsType<ExecutionOperation>(Assert.Single(result[1].Actions));
        Assert.Equal(Operations.Copy, operation.Operation.Name);
        Assert.Equal("copy", Assert.Single(operation.Sources).Node);
        Assert.NotNull(operation.SelectedImplementation);
        Assert.Equal(before, LogicalGraphPackageXml.Serialize(source));
        Assert.Single(result.MainGraph.GraphFlow);
    }

    [Fact]
    public void CompilerRunsProcessorsInOrderWithOriginalSourceAndContext()
    {
        var source = new LogicalGraphPackage(CopyGraph("main"));
        var records = new List<string>();
        var context = new ExecutionGraphCompilationContext("test", "cpu-vm");
        var compiler = new ExecutionGraphCompiler([new CpuFloat32InstructionCollection()],
            [new RecordingProcessor("first", records), new RecordingProcessor("second", records)],
            converter: Converter());
        var result = compiler.Compile(source, context);
        Assert.Equal(["first:0:test:cpu-vm", "second:1:test:cpu-vm"], records);
        Assert.Equal("yes", result.MainGraph!.Metadata["first"]);
        Assert.Equal("yes", result.MainGraph.Metadata["second"]);
        Assert.Empty(source.MainGraph!.Metadata);
    }

    [Fact]
    public void RewriterExtractsAndInlinesAnOperationChain()
    {
        var package = Chain();
        var before = ExecutionGraphPackageXml.Serialize(package);
        var extracted = ExecutionGraphRewriter.ExtractNodesToGraph(package, "main", ["copy", "relu"], "kernel", "invoke");
        Assert.Equal(2, extracted.Count);
        Assert.IsType<ExecutionCall>(Assert.Single(extracted.MainGraph!.Actions));
        Assert.Equal(2, extracted.Single(graph => graph.Id == "kernel").Actions.Count);
        var inlined = ExecutionGraphRewriter.InlineGraphCall(extracted, "main", "invoke");
        Assert.Equal(["main", "kernel"], inlined.Select(graph => graph.Id));
        Assert.Equal(ExecutionGraphPackageXml.Serialize(new ExecutionGraphPackage(extracted.Single(graph => graph.Id == "kernel"))),
            ExecutionGraphPackageXml.Serialize(new ExecutionGraphPackage(inlined.Single(graph => graph.Id == "kernel"))));
        Assert.Equal(["invoke/copy", "invoke/relu"], inlined.MainGraph!.Actions.Select(action => action.Id));
        var diagnostics = inlined.Validate();
        Assert.DoesNotContain(diagnostics, diagnostic => diagnostic.Severity == GraphDiagnosticSeverity.Error);
        var unused = Assert.Single(diagnostics);
        Assert.Equal("P0002", unused.Code);
        Assert.Equal(GraphDiagnosticSeverity.Warning, unused.Severity);
        Assert.Equal("kernel", unused.Graph);
        Assert.Equal(before, ExecutionGraphPackageXml.Serialize(package));
    }

    [Fact]
    public void RewriterFusesNodesAndReplacesCallsWhilePreservingBindings()
    {
        var package = Chain();
        var fused = ExecutionGraphRewriter.FuseNodesToOperation(package, "main", ["copy", "relu"],
            "fused", new("test.fused"),
            [new("input", ResourceRef.Local("input"), ResourceAccess.Read),
             new("output", ResourceRef.Local("output"), ResourceAccess.Write)]);
        Assert.Equal("test.fused", Assert.IsType<ExecutionOperation>(Assert.Single(fused.MainGraph!.Actions)).Operation.Name);
        var extracted = ExecutionGraphRewriter.ExtractNodesToGraph(package, "main", ["copy", "relu"], "kernel", "invoke");
        var replaced = ExecutionGraphRewriter.ReplaceGraphCallWithOperation(extracted, "main", "invoke", new("test.specialized"));
        var operation = Assert.IsType<ExecutionOperation>(Assert.Single(replaced.MainGraph!.Actions));
        Assert.Equal(2, operation.Bindings.Count);
        Assert.Equal("test.specialized", operation.Operation.Name);
        Assert.DoesNotContain(replaced.Validate(), diagnostic => diagnostic.Severity == GraphDiagnosticSeverity.Error);
    }

    [Fact]
    public void FusionRejectsMissingBoundaryBindingsAndInterveningActions()
    {
        var package = Chain();
        Assert.Throws<InvalidDataException>(() => ExecutionGraphRewriter.FuseNodesToOperation(
            package, "main", ["copy", "relu"], "fused", new("test.fused"), []));
        var graph = package.MainGraph!;
        var between = graph.Operation("between", Operations.Copy);
        graph.GraphFlow.Clear();
        graph.Actions[0].Then(between);
        between.Then(graph.Actions[1]);
        Assert.Throws<InvalidDataException>(() => ExecutionGraphRewriter.FuseNodesToOperation(
            package, "main", ["copy", "relu"], "fused", new("test.fused"),
            [new("input", ResourceRef.Local("input"), ResourceAccess.Read),
             new("output", ResourceRef.Local("output"), ResourceAccess.Write)]));
    }

    [Fact]
    public void XmlProcessorReplacesMatchingPackageAndRejectsMismatchedTargetOrMain()
    {
        var source = new LogicalGraphPackage(CopyGraph("main"));
        var package = Converter().Convert(source, new("test", "cpu-vm"));
        var path = Path.Combine(Directory.GetCurrentDirectory(), $"execution-package-{Guid.NewGuid():N}.xml");
        try
        {
            var replacement = package.Snapshot();
            replacement.MainGraph!.Metadata["loaded"] = "yes";
            File.WriteAllText(path, ExecutionGraphPackageXml.Serialize(replacement));
            var processor = new XmlExecutionPackagePostprocessor(path, "test", "cpu-vm");
            var context = new ExecutionPassContext(source, new("test", "cpu-vm"), 0, processor.Name);
            Assert.Equal("yes", processor.Process(package, context).MainGraph!.Metadata["loaded"]);
            Assert.Throws<InvalidOperationException>(() => processor.Process(package,
                context with
                {
                    Compilation = new("other-model", "cpu-vm")
                }));
            replacement.SetTarget("d3d12-vm");
            File.WriteAllText(path, ExecutionGraphPackageXml.Serialize(replacement));
            Assert.Throws<InvalidDataException>(() => processor.Process(package, context));
            replacement = Converter().Convert(new(CopyGraph("other-main")), new("test", "cpu-vm"));
            File.WriteAllText(path, ExecutionGraphPackageXml.Serialize(replacement));
            Assert.Throws<InvalidDataException>(() => processor.Process(package, context));
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static ExecutionGraphPackage Chain()
    {
        var graph = CopyGraph("main");
        graph.Resources.Single(resource => resource.Id == "output").GraphIO = GraphIO.Internal;
        graph.Resources.Single(resource => resource.Id == "output").Id = "middle";
        graph.Actions[0].Bindings[1] = new("output", ResourceRef.Local("middle"), ResourceAccess.Write);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        var relu = graph.Operation("relu", "core.relu");
        relu.Bind("input", "middle", ResourceAccess.Read).Bind("output", output, ResourceAccess.Write);
        return Converter().Convert(new(graph), new("test", "cpu-vm"));
    }

    private static LogicalGraph CopyGraph(string id)
    {
        var graph = new LogicalGraph { Id = id };
        var input = graph.Resource("input", Vector, GraphIO.In);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        graph.Operation("copy", Operations.Copy)
            .Bind("input", input, ResourceAccess.Read).Bind("output", output, ResourceAccess.Write);
        return graph;
    }

    private static DefaultExecutionGraphConverter Converter() => new([new CpuFloat32InstructionCollection()]);

    private sealed class RecordingProcessor(string name, List<string> records) : IExecutionGraphPostprocessor
    {
        public string Name => name;
        public ExecutionGraphPackage Process(ExecutionGraphPackage package, ExecutionPassContext context)
        {
            Assert.Equal("main", Assert.IsType<LogicalGraphPackage>(context.Source).MainGraphId);
            records.Add($"{Name}:{context.ProcessorIndex}:{context.Compilation.ModelType}:{context.Compilation.Backend}");
            var result = package.Snapshot();
            result.MainGraph!.Metadata[Name] = "yes";
            return result;
        }
    }
}
