using System.Text;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;
using SharpInference.Runtime;
using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class ProcessorPipelineTests
{
    [Fact]
    public void Builder_ReportsOrderedStepsAndCleansUpStartedStepsInReverse()
    {
        var failure = new InvalidOperationException("Build step failed.");
        var cleanup = new List<string>();
        var builder = new ProcessorPipelineBuilder("virtual-model")
            .AddStep("first", _ => { }, _ => cleanup.Add("first"))
            .AddStep("second", _ => throw failure, _ => cleanup.Add("second"))
            .AddStep("unstarted", _ => { }, _ => cleanup.Add("unstarted"));
        Assert.Equal("virtual-model", builder.Path);
        Assert.Equal(["first", "second", "unstarted"], builder.StepNames);
        Assert.Same(failure, Assert.Throws<InvalidOperationException>(() => builder.Build()));
        Assert.Equal(["second", "first"], cleanup);
    }

    [Fact]
    public void RuntimeSurface_ExposesOnlyCompiledVmExecutionContracts()
    {
        Assert.Equal(typeof(VmCompiledPlan), typeof(Processor).GetProperty("PreparedPlan")!.PropertyType);
        Assert.Equal(typeof(VmProgram), typeof(Processor).GetProperty("InferenceProgram")!.PropertyType);
        Assert.Equal(typeof(VmProgram), typeof(Processor).GetProperty("PrefillProgram")!.PropertyType);
        Assert.Equal(typeof(VmGraphBackend), typeof(ProcessorBuildContext).GetProperty("Backend")!.PropertyType);
        Assert.Equal(typeof(VmCompiledPlan), typeof(ProcessorBuildContext).GetProperty("PreparedPlan")!.PropertyType);
        Assert.Null(typeof(ProcessorPipelineBuilder).GetProperty("Debug"));
        Assert.Null(typeof(ProcessorBuildContext).GetProperty("Debug"));
        foreach (var name in new[] { "IsDebugMode", "ExecutionGraph", "InferenceExecutionGraph",
                     "InferencePreparedPlan", "PrefillExecutionGraph", "PrefillPreparedPlan" })
            Assert.Null(typeof(Processor).GetProperty(name));
        foreach (var name in new[] { "ExecutionGraph", "InferenceExecutionGraph",
                     "InferencePreparedPlan", "PrefillExecutionGraph", "PrefillPreparedPlan" })
            Assert.Null(typeof(ProcessorBuildContext).GetProperty(name));
        foreach (var name in new[] { "IsDebugMode", "DebugSession", "LayerTrace" })
            Assert.Null(typeof(ProcessorSession).GetProperty(name));
        Assert.Empty(typeof(ProcessorSession).GetEvents());
        Assert.DoesNotContain(typeof(ProcessorSession).GetMethods(), method => method.Name is "Resume" or "Stop");
        Assert.DoesNotContain(typeof(ProcessorPipelineBuilder).Assembly.GetExportedTypes()
                .Where(type => type.IsSealed && type.IsAbstract).SelectMany(type => type.GetMethods()),
            method => method.Name is "UseExecutionGraph" or "UseExecutionXml" or "UseXmlExecutionGraph"
                or "UsePrefillOptimizers");
        Assert.Single(typeof(GraphArchitectureMetadataReader).GetConstructors());
        Assert.Equal(typeof(LogicalGraph),
            typeof(GraphArchitectureMetadataReader).GetConstructors()[0].GetParameters().Single().ParameterType);
        Assert.Equal(typeof(string),
            typeof(XmlArchitectureMetadataReader).GetConstructors().Single().GetParameters().Single().ParameterType);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphExports_WriteUtf8AtCurrentPositionAndLeaveStreamsOpen(bool rwkv7)
    {
        using var processor = LoadPortable(rwkv7);
        Assert.NotNull(processor.LogicalGraph);
        Assert.Same(processor.InferenceProgram, processor.PreparedPlan.Program);
        Check(processor.ExportLogicalGraph, GraphJson.Serialize(processor.LogicalGraph));
        Check(stream => processor.ExportExecutionGraph(stream),
            VmProgramXml.Serialize(processor.InferenceProgram));
        Check(stream => processor.ExportExecutionGraph(stream, ProcessorExecutionGraphKind.Inference),
            VmProgramXml.Serialize(processor.InferenceProgram));
        Check(stream => processor.ExportExecutionGraph(stream, ProcessorExecutionGraphKind.Prefill),
            VmProgramXml.Serialize(processor.PrefillProgram));

        using var invalid = new MemoryStream();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            processor.ExportExecutionGraph(invalid, (ProcessorExecutionGraphKind)999));
        processor.Dispose();
        Assert.Throws<ObjectDisposedException>(() => processor.ExportLogicalGraph(invalid));
        Assert.Throws<ObjectDisposedException>(() => processor.ExportExecutionGraph(invalid));

        static void Check(Action<Stream> export, string expected)
        {
            using var stream = new MemoryStream();
            stream.WriteByte(0x7F);
            export(stream);
            Assert.True(stream.CanWrite);
            Assert.Equal(0x7F, stream.GetBuffer()[0]);
            Assert.Equal(stream.Length, stream.Position);
            Assert.Equal(Encoding.UTF8.GetBytes(expected), stream.ToArray().AsSpan(1).ToArray());
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ProcessorLoad_AndSuppliedPortableGraphAgreeAcrossPrefillForkAndReset(bool rwkv7)
    {
        var model = rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6;
        using var automatic = RwkvRuntimeFactory.Load(TestModelLoader.GetPath(model), SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create());
        using var supplied = LoadPortable(rwkv7);
        using var first = automatic.CreateSession();
        using var second = supplied.CreateSession();
        Assert.Equal(rwkv7 ? "rwkv-7" : "rwkv-6", automatic.Metadata.ArchitectureId);
        Assert.Same(automatic.InferenceProgram, automatic.PreparedPlan.Program);
        Assert.Equal(first.Prefill([0, 1, 2]).ToArray(), second.Prefill([0, 1, 2]).ToArray());
        StateSnapshotAssertions.Equal(StateSnapshotAssertions.Capture(first),
            StateSnapshotAssertions.Capture(second));

        using var fork = second.Fork();
        Assert.Equal(first.ForwardToken(3).ToArray(), second.ForwardToken(3).ToArray());
        Assert.Equal(second.ForwardToken(4).ToArray(), fork.Prefill([3, 4]).ToArray());
        second.Reset();
        using var fresh = supplied.CreateSession();
        Assert.Equal(fresh.ForwardToken(0).ToArray(), second.ForwardToken(0).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void VmProgramImporter_RejectsLegacyLogicalAndExecutionXml(bool rwkv7)
    {
        using var processor = LoadPortable(rwkv7);
        Assert.Throws<InvalidDataException>(() =>
            VmProgramXml.Deserialize(GraphXml.Serialize(processor.LogicalGraph!)));
        Assert.Throws<InvalidDataException>(() =>
            VmProgramXml.Deserialize(GraphXml.Serialize(new GraphOptimizer().Optimize(processor.LogicalGraph!))));
        var imported = VmProgramXml.Deserialize(VmProgramXml.Serialize(processor.InferenceProgram));
        Assert.Equal(processor.InferenceProgram.Abi, imported.Abi);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeProgramPaths_RejectLegacyExecutionXml(bool prefill)
    {
        using var processor = LoadPortable(rwkv7: false);
        var path = Path.Combine(Path.GetTempPath(), $"legacy-execution-{Guid.NewGuid():N}.xml");
        try
        {
            File.WriteAllText(path, GraphXml.Serialize(new GraphOptimizer().Optimize(processor.LogicalGraph!)));
            var configuration = prefill
                ? new VmRuntimeConfig { PrefillProgramPath = path }
                : new VmRuntimeConfig { ProgramPath = path };
            Assert.Throws<InvalidDataException>(() => SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create(configuration));
            using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv6);
            Assert.Throws<InvalidDataException>(() => new XmlArchitectureMetadataReader(path).Read(catalog));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphReader_RejectsIncompatibleWeightBeforeBackendPreparation(bool rwkv7)
    {
        var model = rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6;
        using var catalog = TestModelLoader.OpenCatalog(model);
        ILogicalGraphProvider provider = rwkv7
            ? new PortableRwkv7GraphProvider() : new PortableRwkv6GraphProvider();
        var graph = provider.Build(catalog);
        var head = graph.Resources.Single(resource =>
            resource.Kind == GraphResourceKind.Weight && resource.BindingKey == "head.weight");
        var corrupted = new LogicalGraph(graph.Identity, graph.Model,
            graph.Resources.Select(resource => resource.Id == head.Id
                ? resource with { BindingKey = "nonexistent.weight" }
                : resource), graph.Regions, graph.Nodes,
            graph.Inputs, graph.Outputs, graph.GraphState);
        var prepared = false;
        using var backend = new VmGraphBackend(VmTarget.Cpu, _ =>
        {
            prepared = true;
            throw new InvalidOperationException("Invalid weights must fail before compilation.");
        });
        Assert.Throws<InvalidDataException>(() =>
            Processor.LoadGraph(TestModelLoader.GetPath(model), corrupted, backend));
        Assert.False(prepared);
    }

    private static Processor LoadPortable(bool rwkv7) =>
        Processor.LoadGraph(TestModelLoader.GetPath(rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6),
            rwkv7 ? new PortableRwkv7GraphProvider() : new PortableRwkv6GraphProvider(),
            SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create());
}
