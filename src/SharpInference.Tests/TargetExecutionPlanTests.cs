using System.Runtime.InteropServices;
using System.Xml.Linq;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime.Cpu;
using SharpInference.Runtime;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class TargetExecutionPlanTests
{
    [Fact]
    public void EditedIlNodesAreTheOnlyInputToTargetPlanning()
    {
        var original = new IlExecutionPlan(CopyGraph());
        var edited = original.WithNodes(original.Resources, []);
        var generator = new VmExecutionGraphGenerator(InstructionTarget.Cpu, CpuInstructionCollections.Create());
        foreach (var il in new[] { edited, IlExecutionPlan.Deserialize(edited.Serialize()) })
        {
            Assert.Empty(il.Nodes);
            var plan = generator.GeneratePlan(il);
            Assert.Empty(plan.Definitions.Single(definition => definition.Id == "forward").Nodes);
            Assert.DoesNotContain(plan.Definitions.SelectMany(definition => definition.Nodes),
                node => node.Instruction is ExecutionPlanOperator);
        }
    }

    [Fact]
    public void IlAndTargetReloadCompileWithoutOriginalGraph()
    {
        var il = new IlExecutionPlan(CopyGraph());
        var loadedIl = IlExecutionPlan.Deserialize(il.Serialize());
        Assert.Equal(il.Serialize(), loadedIl.Serialize());
        var generator = new VmExecutionGraphGenerator(InstructionTarget.Cpu, CpuInstructionCollections.Create());
        var selected = generator.GeneratePlan(loadedIl);
        var xml = ExecutionPlanXml.Serialize(selected);
        var loaded = ExecutionPlanXml.DeserializeTarget(xml);
        Assert.Equal(xml, ExecutionPlanXml.Serialize(loaded));
        Assert.Equal(VmProgramXml.Serialize(ExecutionPlanAssembler.Assemble(selected)),
            VmProgramXml.Serialize(ExecutionPlanAssembler.Assemble(loaded)));
        Assert.NotNull(loaded.Bindings);
        Assert.Empty(loaded.Bindings.Nodes);
        Assert.Single(loaded.Bindings.Regions);

        using var backend = CpuVmBackendFactory.Create();
        var prepared = backend.Prepare(loaded);
        Assert.Same(loaded, prepared.ExecutionPlan);
        using var session = backend.CreateGraphSession(new EmptyCatalog());
        float[] input = [1, 2, 3, 4];
        var result = session.Execute(new Dictionary<ResourceId, ReadOnlyMemory<byte>>
        {
            [new("input")] = MemoryMarshal.AsBytes(input.AsSpan()).ToArray(),
        });
        Assert.Equal(input, MemoryMarshal.Cast<byte, float>(result[new("output")]).ToArray());
    }

    [Fact]
    public void AssemblyDoesNotRestoreRemovedStepsOrAddEntries()
    {
        var selected = VmGraphOptimizer.BuildPlan(CopyGraph(), VmTarget.Cpu).Plan;
        var root = XElement.Parse(ExecutionPlanXml.Serialize(selected));
        root.Element("Definitions")!.Elements("Definition").Single(d => (string?)d.Attribute("id") == "forward")
            .Element("Body")!.RemoveNodes();
        var loaded = ExecutionPlanXml.DeserializeTarget(root.ToString());
        var program = ExecutionPlanAssembler.Assemble(loaded);
        Assert.Empty(program.Definitions.Single(d => d.Id == "forward").Nodes);
        Assert.Single(program.Entries);
        Assert.Equal("forward", program.Entries[0].Name);
    }

    [Fact]
    public void ChangingAnEntryBindingChangesTheActualComputation()
    {
        var root = XElement.Parse(ExecutionPlanXml.Serialize(
            VmGraphOptimizer.BuildPlan(CopyGraph(), VmTarget.Cpu).Plan));
        var alternate = new XElement(root.Element("Slots")!.Elements().Single(slot => (string?)slot.Attribute("id") == "input"));
        alternate.SetAttributeValue("id", "alternate");
        root.Element("Slots")!.Add(alternate);
        root.Element("EntryPoints")!.Descendants("Argument").Single(argument => (string?)argument.Attribute("source") == "input")
            .SetAttributeValue("source", "alternate");
        var program = ExecutionPlanAssembler.Assemble(ExecutionPlanXml.DeserializeTarget(root.ToString()));
        Assert.Equal("alternate", program.Entries.Single().Arguments.Single(argument => argument.Parameter == "input").Source);
        using var executor = new CpuVmCompiler(CpuInstructionCollections.Create()).Compile(program).LoadExecutable();
        var storage = program.Slots.Select(slot => slot.Id switch
        {
            "input" => MemoryMarshal.AsBytes(new float[] { 1, 2, 3, 4 }.AsSpan()).ToArray(),
            "alternate" => MemoryMarshal.AsBytes(new float[] { 5, 6, 7, 8 }.AsSpan()).ToArray(),
            _ => new byte[(int)slot.Tensor.ByteLength],
        }).ToArray();
        executor.Execute("forward", storage);
        Assert.Equal([5f, 6f, 7f, 8f], MemoryMarshal.Cast<byte, float>(
            storage[program.Slots.Select(slot => slot.Id).ToList().IndexOf("output")]).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OfflinePlanAndArtifactDriveTheRuntimeWithoutRegeneration(bool gpu)
    {
        var directory = Path.Combine(Path.GetTempPath(), $"sharp-execution-plan-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            var modelPath = TestModelLoader.GetPath(TestModel.Rwkv7Fp16);
            using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv7Fp16);
            var logical = RwkvRuntimeFactory.CreateGraphProvider("rwkv-7").Build(catalog);
            var selected = new VmExecutionGraphGenerator(gpu ? InstructionTarget.Direct3D12 : InstructionTarget.Cpu,
                gpu ? SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create() : CpuInstructionCollections.Create())
                .GeneratePlan(new(logical), new(PrefillCapacity: 2));
            Assert.Contains(selected.Definitions, definition =>
                definition.Id.StartsWith("region.", StringComparison.Ordinal) &&
                definition.Kind == ExecutionPlanDefinitionKind.Orchestration);
            var layerCalls = selected.Definitions.Single(definition => definition.Id == "forward").Nodes
                .Select(node => Assert.IsType<ExecutionPlanCall>(node.Instruction)).ToArray();
            Assert.True(layerCalls.Select(call => call.Definition).Distinct().Count() < layerCalls.Length,
                "Repeated layers must reuse a normalized definition, with concrete slots bound at each call.");
            Assert.Contains(selected.Definitions.Where(definition => definition.Id.StartsWith("region.", StringComparison.Ordinal)),
                definition => definition.Parameters.Any(parameter => parameter.Name.StartsWith("slot.", StringComparison.Ordinal)));
            var planPath = Path.Combine(directory, "plan.xml");
            File.WriteAllText(planPath, ExecutionPlanXml.Serialize(selected));
            var artifacts = Path.Combine(directory, "artifact");
            VmGraphBackend Backend(VmRuntimeConfig config) => gpu
                ? SharpInference.Runtime.D3D12.D3D12VmBackendFactory.Create(config) : CpuVmBackendFactory.Create(config);
            using var processor = RwkvRuntimeFactory.Load(modelPath, Backend(new()
            {
                ExecutionPlanPath = planPath, PrefillCapacity = 8, WeightViews = false, TensorViews = false,
                PrefillInstances = 1, InferenceInstances = 1,
            }));
            Assert.Equal(ExecutionPlanXml.Serialize(selected), ExecutionPlanXml.Serialize(processor.PreparedPlan.ExecutionPlan!));
            Assert.DoesNotContain(processor.InferenceProgram.Entries, entry => entry.Name == "prefill.8");
            using var exportedPlan = new MemoryStream();
            processor.ExportExecutionPlan(exportedPlan);
            Assert.Equal(ExecutionPlanXml.Serialize(selected), System.Text.Encoding.UTF8.GetString(exportedPlan.ToArray()));
            processor.ExportCompiledArtifact(artifacts);
            Assert.True(File.Exists(Path.Combine(artifacts, "execution-plan.xml")));
            using var reloaded = RwkvRuntimeFactory.Load(modelPath, Backend(new()
            {
                ArtifactDirectory = artifacts, PrefillInstances = 1, InferenceInstances = 1,
            }));
            Assert.NotNull(reloaded.PreparedPlan.ExecutionPlan);
            using var original = processor.CreateSession();
            using var restored = reloaded.CreateSession();
            foreach (var token in new[] { 1, 2, 3 })
                Assert.Equal(original.ForwardToken(token).ToArray(), restored.ForwardToken(token).ToArray());
            var root = XElement.Parse(File.ReadAllText(Path.Combine(artifacts, "execution-plan.xml")));
            root.Descendants("Operation").First(operation => operation.Attribute("implementation") is null)
                .SetAttributeValue("arithmetic", "Float16");
            File.WriteAllText(Path.Combine(artifacts, "execution-plan.xml"), root.ToString());
            Assert.Throws<InvalidDataException>(() => Backend(new() { ArtifactDirectory = artifacts }));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void GpuPlanRejectsMissingBarriersBeforeCompilation()
    {
        var root = XElement.Parse(ExecutionPlanXml.Serialize(
            VmGraphOptimizer.BuildPlan(ReshapeGraph(overwrite: false), VmTarget.Direct3D12,
                new(ReuseLocalStorage: false, TensorViews: false)).Plan));
        foreach (var node in root.Descendants("Node").Where(node => node.Element("Barrier") is not null).ToArray())
        {
            var id = (string)node.Attribute("id")!;
            foreach (var dependency in root.Descendants("DependsOn").Elements("NodeRef")
                .Where(reference => (string?)reference.Attribute("id") == id).ToArray())
                dependency.Remove();
            node.Remove();
        }
        Assert.Throws<InvalidDataException>(() => ExecutionPlanXml.DeserializeTarget(root.ToString()));
    }

    [Fact]
    public void TargetBindingRejectsActualWritableAliasBeforeCompilation()
    {
        var root = XElement.Parse(ExecutionPlanXml.Serialize(
            VmGraphOptimizer.BuildPlan(CopyGraph(), VmTarget.Cpu).Plan));
        root.Element("Slots")!.Elements().Single(slot => (string?)slot.Attribute("id") == "input")
            .SetAttributeValue("access", "ReadWrite");
        root.Element("Definitions")!.Elements().Single(definition => (string?)definition.Attribute("id") == "forward")
            .Element("Parameters")!.Elements().Single(parameter => (string?)parameter.Attribute("name") == "input")
            .SetAttributeValue("access", "ReadWrite");
        root.Element("EntryPoints")!.Descendants("Argument")
            .Single(argument => (string?)argument.Attribute("parameter") == "output").SetAttributeValue("source", "input");
        var plan = ExecutionPlanXml.DeserializeTarget(root.ToString());
        Assert.Throws<InvalidDataException>(() => ExecutionPlanAssembler.Bind(plan, new(CpuInstructionCollections.Create())));
    }

    [Fact]
    public void TargetReloadRejectsAnUninitializedTemporaryRead()
    {
        var generator = new VmExecutionGraphGenerator(InstructionTarget.Cpu, CpuInstructionCollections.Create());
        var root = XElement.Parse(ExecutionPlanXml.Serialize(generator.GeneratePlan(new(ReshapeGraph(false)),
            new(ReuseLocalStorage: false, TensorViews: false))));
        var body = root.Element("Definitions")!.Elements().Single(definition => (string?)definition.Attribute("id") == "forward")
            .Element("Body")!;
        var removed = body.Elements().First();
        var id = (string)removed.Attribute("id")!;
        foreach (var dependency in body.Descendants("NodeRef").Where(reference => (string?)reference.Attribute("id") == id).ToArray())
            dependency.Remove();
        removed.Remove();
        Assert.Throws<InvalidDataException>(() => ExecutionPlanXml.DeserializeTarget(root.ToString()));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ContinuousViewsPreserveValuesAndDoNotAliasAcrossAnOverwrite(bool overwrite)
    {
        var graph = ReshapeGraph(overwrite);
        float[] Execute(bool views)
        {
            var selected = VmGraphOptimizer.BuildPlan(graph, VmTarget.Cpu,
                new(ReuseLocalStorage: true, WeightViews: false, TensorViews: views)).Plan;
            var view = selected.ResourceBindings.Single(binding => binding.Resource == "reshaped");
            Assert.Equal(views && !overwrite, view.IsView);
            var program = ExecutionPlanAssembler.Assemble(selected);
            using var executor = new CpuVmCompiler(CpuInstructionCollections.Create()).Compile(program).LoadExecutable();
            var storage = program.Slots.Select(slot => slot.Id == "input"
                ? MemoryMarshal.AsBytes(new float[] { 1, 2, 3, 4 }.AsSpan()).ToArray() : new byte[(int)slot.Tensor.ByteLength]).ToArray();
            executor.Execute("forward", storage);
            return MemoryMarshal.Cast<byte, float>(
                storage[program.Slots.Select(slot => slot.Id).ToList().IndexOf("output")]).ToArray();
        }
        Assert.Equal([1f, 2f, 3f, 4f], Execute(false));
        Assert.Equal(Execute(false), Execute(true));
    }

    [Fact]
    public void ExplicitOffsetViewsSurviveAssemblyAndExecute()
    {
        var small = new TensorDescriptor(GraphElementType.Float32, [2]);
        var large = new TensorDescriptor(GraphElementType.Float32, [4]);
        ExecutionPlanParameter[] parameters = [new("input", GraphResourceAccess.Read, small),
            new("output", GraphResourceAccess.ReadWrite, small)];
        var copy = new ExecutionPlanDefinition("copy", ExecutionPlanDefinitionKind.Function, parameters,
            [new("body", new ExecutionPlanOperator(new(GraphElementType.Float32, GraphElementType.Float32),
                InstructionCollectionIds.TierZeroFloat32, "core.copy", [new("input", "input"), new("output", "output")]))]);
        var forward = new ExecutionPlanDefinition("forward", ExecutionPlanDefinitionKind.Orchestration,
            [new("input", GraphResourceAccess.Read, large), new("output", GraphResourceAccess.ReadWrite, large)],
            [new("a", new ExecutionPlanCall("copy", [new("input", "input", 8), new("output", "output")])),
             new("b", new ExecutionPlanCall("copy", [new("input", "input"), new("output", "output", 8)]), ["a"])]);
        var plan = new TargetExecutionPlan("views", "views", ExecutionPlanTarget.Cpu,
            [new("input", GraphResourceScope.Local, GraphResourceAccess.Read, large),
             new("output", GraphResourceScope.Local, GraphResourceAccess.ReadWrite, large)],
            [copy, forward], [new("forward", "forward", [new("input", "input"), new("output", "output")])], "none", 1, []);
        plan = ExecutionPlanXml.DeserializeTarget(ExecutionPlanXml.Serialize(plan));
        var program = ExecutionPlanAssembler.Assemble(plan);
        Assert.Equal(2, program.Slots.Count);
        using var executor = new CpuVmCompiler(CpuInstructionCollections.Create()).Compile(program).LoadExecutable();
        byte[][] buffers = [MemoryMarshal.AsBytes(new float[] { 1, 2, 3, 4 }.AsSpan()).ToArray(), new byte[16]];
        executor.Execute("forward", buffers);
        Assert.Equal([3f, 4f, 1f, 2f], MemoryMarshal.Cast<byte, float>(buffers[1]).ToArray());
    }

    [Theory]
    [InlineData("0")]
    [InlineData("2")]
    [InlineData("32")]
    [InlineData("63")]
    [InlineData("65")]
    [InlineData("96")]
    [InlineData("128")]
    [InlineData("1024")]
    [InlineData("-1")]
    [InlineData("1.5")]
    [InlineData(null)]
    public void UnsupportedCooperativeThreadsAreRejected(string? value)
    {
        var plan = VmGraphOptimizer.BuildPlan(MatVecGraph(3), VmTarget.Direct3D12).Plan;
        var root = XElement.Parse(ExecutionPlanXml.Serialize(plan));
        root.Descendants("Operation").Single().SetAttributeValue("cooperativeThreads", value);
        Assert.Throws<InvalidDataException>(() => ExecutionPlanXml.DeserializeTarget(root.ToString()));
    }

    [Theory]
    [InlineData("unknown")]
    [InlineData("duplicate")]
    [InlineData("bounds")]
    [InlineData("type")]
    [InlineData("threads")]
    [InlineData("grid")]
    [InlineData("operator-offset")]
    public void InvalidContractsAreRejected(string change)
    {
        var root = XElement.Parse(ExecutionPlanXml.Serialize(
            VmGraphOptimizer.BuildPlan(MatVecGraph(3), VmTarget.Direct3D12).Plan));
        switch (change)
        {
            case "unknown": root.Descendants("Operation").Single().SetAttributeValue("hiddenFallback", "true"); break;
            case "duplicate": root.Add(new XElement(root.Element("Slots")!)); break;
            case "bounds": root.Descendants("Dispatch").Single().Descendants("Argument").First().SetAttributeValue("byteOffset", 1024); break;
            case "type": root.Element("Slots")!.Elements().First().SetAttributeValue("elementType", "Int32"); break;
            case "threads": root.Descendants("Threads").Single().SetAttributeValue("x", 32); break;
            case "grid": root.Descendants("Groups").Single().SetAttributeValue("x", 1); break;
            case "operator-offset": root.Descendants("Operation").Single().Descendants("Argument").First()
                .SetAttributeValue("byteOffset", 4); break;
        }
        Assert.Throws<InvalidDataException>(() =>
            ExecutionPlanAssembler.Assemble(ExecutionPlanXml.DeserializeTarget(root.ToString())));
    }

    [Theory]
    [InlineData(1, 3)]
    [InlineData(64, 3)]
    [InlineData(64, 64)]
    [InlineData(64, 65)]
    public void GpuPlanUsesExactlyTheSelectedMatVec(int cooperativeThreads, int columns)
    {
        var generator = new VmExecutionGraphGenerator(InstructionTarget.Direct3D12,
            SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create());
        var plan = generator.GeneratePlan(new(MatVecGraph(columns)), new(GpuMatVec:
            new(cooperativeThreads == 1 ? GpuMatVecMode.Serial : GpuMatVecMode.Cooperative)));
        var xml = ExecutionPlanXml.Serialize(plan);
        Assert.Contains($"cooperativeThreads=\"{cooperativeThreads}\"", xml);
        var program = ExecutionPlanAssembler.Assemble(ExecutionPlanXml.DeserializeTarget(xml));
        var operation = program.Definitions.SelectMany(d => d.Nodes).Select(n => n.Instruction).OfType<VmOperator>().Single();
        Assert.Equal(cooperativeThreads == 1 ? GpuMatVecExecution.Serial : GpuMatVecExecution.Cooperative,
            operation.ExecutionConfiguration);
        using var executor = new D3D12VmCompiler(SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create())
            .Compile(program).CreateExecutor();
        executor.Upload("matrix", MemoryMarshal.AsBytes(Enumerable.Repeat(1f, 2 * columns).ToArray().AsSpan()).ToArray());
        executor.Upload("input", MemoryMarshal.AsBytes(Enumerable.Repeat(2f, columns).ToArray().AsSpan()).ToArray());
        executor.Execute("forward");
        Assert.Equal([(float)(2 * columns), (float)(2 * columns)],
            MemoryMarshal.Cast<byte, float>(executor.Readback("output")).ToArray());
    }

    private static LogicalGraph CopyGraph() => new LogicalGraphBuilder(new("external", 1, "copy"),
            new("array", "array@1", new Dictionary<string, int>()))
        .Root("root", GraphRegionTypes.Graph, "Root")
        .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
            new(GraphElementType.Float32, [4]), graphInput: true)
        .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
            new(GraphElementType.Float32, [4]), graphOutput: true)
        .Node("copy", PrimitiveGraphOperations.Copy,
            [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "output")]).Build();

    private static LogicalGraph ReshapeGraph(bool overwrite)
    {
        var builder = new LogicalGraphBuilder(new("external", 1, "reshape"),
                new("array", "array@1", new Dictionary<string, int>()))
            .Root("root", GraphRegionTypes.Graph, "Root")
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new(GraphElementType.Float32, [4]), graphInput: true)
            .AddResource("base", "Base", GraphResourceKind.Temporary, GraphResourceLifetime.Token,
                new(GraphElementType.Float32, [4]))
            .AddResource("reshaped", "Reshaped", GraphResourceKind.Temporary, GraphResourceLifetime.Token,
                new(GraphElementType.Float32, [2, 2]))
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new(GraphElementType.Float32, [2, 2]), graphOutput: true)
            .Node("initialize", PrimitiveGraphOperations.Copy,
                [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "base")])
            .Node("reshape", PortableTensorOperationContracts.Reshape,
                [GraphBindings.Read("input", "base"), GraphBindings.Write("output", "reshaped")], ["initialize"]);
        if (overwrite)
            builder.Node("overwrite", PortableTensorOperationContracts.Fill,
                [GraphBindings.Write("output", "base")], ["reshape"], new Dictionary<string, string> { ["value"] = "0" });
        return builder.Node("copy", PrimitiveGraphOperations.Copy,
            [GraphBindings.Read("input", "reshaped"), GraphBindings.Write("output", "output")],
            [overwrite ? "overwrite" : "reshape"]).Build();
    }

    private static LogicalGraph MatVecGraph(int columns) => new LogicalGraphBuilder(new("external", 1, "matvec"),
            new("array", "array@1", new Dictionary<string, int>()))
        .Root("root", GraphRegionTypes.Graph, "Root")
        .AddResource("matrix", "Matrix", GraphResourceKind.Input, GraphResourceLifetime.External,
            new(GraphElementType.Float32, [2, columns]), graphInput: true)
        .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
            new(GraphElementType.Float32, [columns]), graphInput: true)
        .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
            new(GraphElementType.Float32, [2]), graphOutput: true)
        .Node("matvec", PrimitiveGraphOperations.MatVec,
            [GraphBindings.Read("matrix", "matrix"), GraphBindings.Read("input", "input"),
             GraphBindings.Write("output", "output")]).Build();

    private sealed class EmptyCatalog : IModelTensorCatalog
    {
        public IReadOnlyCollection<string> Names => [];
        public bool TryGet(string name, out IModelTensor tensor) { tensor = null!; return false; }
        public IModelTensor GetRequired(string name) => throw new InvalidDataException(name);
    }
}
