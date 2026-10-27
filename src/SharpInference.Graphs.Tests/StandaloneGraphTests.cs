using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;

namespace SharpInference.Graphs.Tests;

public sealed class StandaloneGraphTests
{
    private static readonly TensorDescriptor Vector = new(ElementType.Float32, [4]);

    [Fact]
    public void FoundationWorksWithoutInstructionOrBackendAssemblies()
    {
        var graph = new Graph { Id = "main" };
        var value = graph.Resource("value", Vector, GraphIO.InOut);
        graph.Operation("update", "custom.update").Bind("value", value, ResourceAccess.ReadWrite);
        var package = new GraphPackage<Graph>(graph);
        Assert.Empty(package.Validate());
        var loaded = GraphPackageXml.Deserialize<Graph>(GraphPackageXml.Serialize(package));
        Assert.Empty(loaded.Validate());
        var loadedGraph = Assert.IsType<Graph>(loaded.MainGraph);
        Assert.Single(loadedGraph.Readers(loadedGraph.Resources[0]));
        Assert.NotSame(graph, package.Snapshot().MainGraph);
        Assert.All(typeof(Graph).Assembly.GetReferencedAssemblies().Where(reference =>
            reference.Name!.StartsWith("SharpInference.", StringComparison.Ordinal)),
            reference => Assert.Equal("SharpInference.Graphs.Core", reference.Name));
    }

    [Fact]
    public void BaseReferencesAndPackageTraversalUseStageValidation()
    {
        var logical = new LogicalGraph { Id = "logical", CompileBoundary = (GraphCompileBoundary)99 };
        Graph reference = logical;
        Assert.Contains(reference.Validate(), diagnostic => diagnostic.Code == "G0003");
        Assert.Contains(new GraphPackage<LogicalGraph>(logical).Validate(), diagnostic => diagnostic.Code == "G0003");

        var execution = new ExecutionGraph { Id = "execution" };
        execution.Actions.Add(new OperationAction(execution, "bad", new("pending")));
        reference = execution;
        Assert.Contains(reference.Validate(), diagnostic => diagnostic.Code == "G0017");
        Assert.Contains(new GraphPackage<ExecutionGraph>(execution).Validate(), diagnostic => diagnostic.Code == "G0017");
    }

    [Fact]
    public void LogicalPackageDerivesDependenciesWithoutChangingSource()
    {
        var graph = new LogicalGraph { Id = "main" };
        var input = graph.Resource("input", Vector, GraphIO.In);
        var scratch = graph.Resource("scratch", Vector);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        graph.Operation("write", Operations.Copy).Bind("input", input, ResourceAccess.Read).Bind("output", scratch, ResourceAccess.Write);
        graph.Operation("read", Operations.Copy).Bind("input", scratch, ResourceAccess.Read).Bind("output", output, ResourceAccess.Write);
        var package = new LogicalGraphPackage(graph);
        var before = LogicalGraphPackageXml.Serialize(package);
        Assert.Empty(package.Validate());
        Assert.Single(package.Analyze(graph).Dependencies);
        Assert.Empty(graph.GraphFlow);
        Assert.Equal(before, LogicalGraphPackageXml.Serialize(package));
        Assert.IsType<LogicalGraphPackage>(package.Snapshot());
        graph.CompileBoundary = GraphCompileBoundary.Atomic;
        var saved = LogicalGraphPackageXml.Deserialize(LogicalGraphPackageXml.Serialize(package));
        Assert.Equal(GraphCompileBoundary.Atomic, saved.MainGraph!.CompileBoundary);
    }

    [Fact]
    public void LogicalBusinessValidationChecksValuesAtPackageBoundary()
    {
        var graph = new LogicalGraph { Id = "main" };
        graph.Resource("output", Vector, GraphIO.Out);
        var package = new LogicalGraphPackage(graph);
        Assert.Empty(graph.Validate());
        Assert.Empty(package.ValidateStructure());
        Assert.Contains(package.Validate(), diagnostic => diagnostic.Code == "A0004");
        Assert.Contains(package.Analyze(graph).Diagnostics, diagnostic => diagnostic.Code == "A0004");
        Assert.Contains(new LogicalGraphPackage().Validate(), diagnostic => diagnostic.Code == "P0001");
        Assert.Contains(new ExecutionGraphPackage().Validate(), diagnostic => diagnostic.Code == "P0001");
    }

    [Fact]
    public void ExecutionPlanCanBeCheckedAndRewrittenWithoutOptimizerOrInstructions()
    {
        var package = ExecutionCopy();
        Assert.Empty(package.Validate());
        Assert.Empty(ExecutionPlanValidation.ValidatePlan(package));
        var before = ExecutionGraphPackageXml.Serialize(package);
        var extracted = ExecutionGraphRewriter.ExtractNodesToGraph(package, "main", ["copy"], "helper", "invoke");
        Assert.Empty(extracted.Validate());
        Assert.Single(extracted.MainGraph!.Actions.OfType<ExecutionCall>());
        var inlined = ExecutionGraphRewriter.InlineGraphCall(extracted, "main", "invoke");
        Assert.DoesNotContain(inlined.Validate(), diagnostic => diagnostic.Severity == GraphDiagnosticSeverity.Error);
        Assert.Equal(before, ExecutionGraphPackageXml.Serialize(package));
        var copy = package.Snapshot();
        Assert.IsType<ExecutionGraphPackage>(copy);
        copy.MainGraph!.Storage["output"] = new(ResourceStorageKind.LocalBuffer);
        Assert.NotEqual(copy.MainGraph.Storage["output"], package.MainGraph!.Storage["output"]);
        var loaded = ExecutionGraphPackageXml.Deserialize(before);
        Assert.Equal(before, ExecutionGraphPackageXml.Serialize(loaded));
        Assert.Empty(ExecutionPlanValidation.ValidatePlan(loaded));
    }

    [Fact]
    public void ExecutionStorageFailuresAreExplicitReadOnlyDiagnostics()
    {
        var package = ExecutionCopy();
        package.MainGraph!.Storage.Remove("output");
        var before = ExecutionGraphPackageXml.Serialize(package);
        Assert.Contains(ExecutionPlanValidation.ValidatePlan(package), diagnostic => diagnostic.Code == "E0002");
        Assert.Equal(before, ExecutionGraphPackageXml.Serialize(package));
        package.MainGraph.Storage["output"] = new(ResourceStorageKind.LocalBuffer, "input");
        package.MainGraph.Storage["input"] = new(ResourceStorageKind.LocalBuffer, "output");
        Assert.Contains(ExecutionPlanValidation.ValidatePlan(package), diagnostic => diagnostic.Code == "E0002");
    }

    [Fact]
    public void EditableExecutionSourceAndCompletePlanHaveDifferentRequirements()
    {
        var package = ExecutionCopy();
        var operation = Assert.IsType<ExecutionOperation>(package.MainGraph!.Actions[0]);
        operation.SelectedImplementation = null;
        Assert.Empty(package.Validate());
        Assert.Contains(ExecutionPlanValidation.ValidatePlan(package), diagnostic => diagnostic.Code == "E0006");
        operation.Implementation(new(Guid.Empty, "", default, new(ElementType.Float32, ElementType.Float32)));
        Graph graph = package.MainGraph;
        Assert.Contains(graph.Validate(), diagnostic => diagnostic.Code == "G0021");
    }

    [Theory]
    [InlineData("")]
    [InlineData(null)]
    public void InvalidEditableIdentitiesProduceDiagnosticsInsteadOfAnalysisCrashes(string? id)
    {
        var package = new LogicalGraphPackage(new LogicalGraph { Id = id! });
        Assert.Contains(package.Validate(), diagnostic => diagnostic.Code == "G0001");
        Assert.Throws<InvalidDataException>(() => package.Analyze());
    }

    [Fact]
    public void StageXmlRejectsOtherStagesAndForeignFields()
    {
        var business = GraphPackageXml.Serialize(new GraphPackage<Graph>(new Graph { Id = "main" }));
        Assert.Throws<NotSupportedException>(() => GraphPackageXml.Deserialize<LogicalGraph>(business));
        Assert.Throws<NotSupportedException>(() => GraphPackageXml.Deserialize<ExecutionGraph>(business));
        var package = ExecutionCopy();
        var xml = ExecutionGraphPackageXml.Serialize(package);
        Assert.Throws<InvalidDataException>(() => LogicalGraphPackageXml.Deserialize(xml));
        Assert.Throws<InvalidDataException>(() => GraphPackageXml.Deserialize<Graph>(xml));
        Assert.Throws<InvalidDataException>(() => ExecutionGraphPackageXml.Deserialize(
            xml.Replace("<Graph id=", "<Graph compileBoundary=\"Atomic\" id=", StringComparison.Ordinal)));
        var logical = LogicalGraphPackageXml.Serialize(new(new LogicalGraph { Id = "main" }));
        Assert.Throws<InvalidDataException>(() => LogicalGraphPackageXml.Deserialize(
            logical.Replace("<GraphFlow", "<Storage /><GraphFlow", StringComparison.Ordinal)));
    }

    private static ExecutionGraphPackage ExecutionCopy()
    {
        var graph = new ExecutionGraph { Id = "main" };
        var input = graph.Resource("input", Vector, new(ResourceStorageKind.ExternalArgument), GraphIO.In);
        var output = graph.Resource("output", Vector, new(ResourceStorageKind.ExternalArgument), GraphIO.Out);
        graph.Operation("copy", Operations.Copy)
            .Implementation(new(Guid.Empty, "copy", ExecutionTarget.Cpu, new(ElementType.Float32, ElementType.Float32)))
            .Source("test", "source", "copy")
            .Bind("input", input, ResourceAccess.Read).Bind("output", output, ResourceAccess.Write);
        var package = new ExecutionGraphPackage(graph);
        package.SetTarget("cpu-vm");
        return package;
    }
}
