using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;

namespace SharpInference.Tests;

public sealed class ExecutionPackageTests
{
    private static readonly TensorDescriptor Vector = new(ElementType.Float32, [8]);

    [Fact]
    public void ConversionSelectsInstalledPrecisionWithoutAddingSourceRequirements()
    {
        var source = Pass();
        var execution = Convert(source);
        var operation = Assert.IsType<ExecutionOperation>(Assert.Single(execution.MainGraph!.Actions));
        Assert.Null(operation.Precision);
        Assert.Null(Assert.IsType<OperationAction>(source.MainGraph!.Actions[0]).Precision);
        Assert.Equal(ElementType.Float32, operation.SelectedImplementation!.Precision.ArithmeticType);
        Assert.Equal(new SourceOrigin("test", "main", "copy"), Assert.Single(operation.Sources));
        var xml = ExecutionGraphPackageXml.Serialize(execution);
        var loaded = ExecutionGraphPackageXml.Deserialize(xml);
        Assert.Equal(xml, ExecutionGraphPackageXml.Serialize(loaded));
        Assert.Equal("cpu-vm", loaded.GetTarget());
        Assert.NotSame(operation, loaded.MainGraph!.Actions[0]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Float16ArithmeticIsSelectedOnlyWhenRequirementsPermitIt(bool requireFloat32)
    {
        var source = Pass(ElementType.Float16);
        if (requireFloat32)
        {
            Assert.IsType<OperationAction>(source.MainGraph!.Actions[0])
                .RequirePrecision(ElementType.Float32, ElementType.Float32);
        }

        var converter = new DefaultExecutionGraphConverter([new HalfArithmeticInstruction()]);
        if (requireFloat32)
        {
            Assert.Throws<NotSupportedException>(() => converter.Convert(source, new("test", "cpu-vm")));
        }
        else
        {
            var execution = converter.Convert(source, new("test", "cpu-vm"));
            Assert.Equal(ElementType.Float16, Assert.IsType<ExecutionOperation>(execution.MainGraph!.Actions[0])
                .SelectedImplementation!.Precision.ArithmeticType);
        }
    }

    [Fact]
    public void UnknownContractVersionDoesNotSelectTheVersionOneInstruction()
    {
        var source = Pass();
        Assert.IsType<OperationAction>(source.MainGraph!.Actions[0]).Operation = new(Operations.Copy, 2);
        var error = Assert.Throws<NotSupportedException>(() => Convert(source));
        Assert.Contains("core.copy@2", error.Message);
    }

    [Fact]
    public void CpuEmitterConsumesTheSelectedImplementationAndPreservesScope()
    {
        var execution = Convert(Pass());
        var program = new VmExecutionGraphEmitter(VmTarget.Cpu,
            [new CpuFloat32InstructionCollection()]).Emit(execution);
        Assert.All(program.Slots, slot => Assert.Equal(VmSlotScope.Local, slot.Scope));
        Assert.Empty(program.State.Entries);
        Assert.NotEmpty(new CpuVmCompiler([new CpuFloat32InstructionCollection()]).GenerateSource(program).Source);
    }

    [Fact]
    public void MissingImplementationIsRejectedByEmitterRatherThanSelectedThere()
    {
        var execution = Convert(Pass());
        Assert.IsType<ExecutionOperation>(execution.MainGraph!.Actions[0]).SelectedImplementation = null;
        Assert.Throws<InvalidDataException>(() => new VmExecutionGraphEmitter(VmTarget.Cpu,
            [new CpuFloat32InstructionCollection()]).Emit(execution));
    }

    [Fact]
    public void GraphCallsKeepVmCallsAndDistinctInternalSlots()
    {
        var child = Pass().MainGraph!;
        child.Id = "child";
        foreach (var action in child.Actions)
        {
            action.Bindings.Clear();
            action.Bind("input", "input", ResourceAccess.Read);
            action.Bind("output", "output", ResourceAccess.Write);
        }
        child.Resource("scratch", Vector);
        var main = new LogicalGraph { Id = "main" };
        var input = main.Resource("input", Vector, GraphIO.In);
        var middle = main.Resource("middle", Vector);
        var output = main.Resource("output", Vector, GraphIO.Out);
        var first = main.Call("first", child);
        first.Bind("input", input, ResourceAccess.Read);
        first.Bind("output", middle, ResourceAccess.Write);
        var second = main.Call("second", child);
        second.Bind("input", middle, ResourceAccess.Read);
        second.Bind("output", output, ResourceAccess.Write);
        var execution = Convert(new(main, child));
        var program = new VmExecutionGraphEmitter(VmTarget.Cpu,
            [new CpuFloat32InstructionCollection()]).Emit(execution);
        Assert.Equal(2, program.Definitions.Count);
        var calls = program.Definitions.Single(definition => definition.Id == "graph.main").Nodes;
        Assert.All(calls, node => Assert.IsType<VmCall>(node.Instruction));
        Assert.Contains(program.Slots, slot => slot.Id == "first/scratch");
        Assert.Contains(program.Slots, slot => slot.Id == "second/scratch");
        Assert.NotEmpty(new CpuVmCompiler([new CpuFloat32InstructionCollection()]).GenerateSource(program).Source);
    }

    [Fact]
    public void GpuEmissionHasSelectedDispatchAndCompletionBarrier()
    {
        IInstructionCollectionProvider[] providers = [new GpuFloat32InstructionCollection()];
        var execution = new DefaultExecutionGraphConverter(providers).Convert(Pass(), new("test", "d3d12-vm"));
        var program = new VmExecutionGraphEmitter(VmTarget.Direct3D12, providers).Emit(execution);
        var orchestration = Assert.Single(program.Definitions, definition => definition.Kind == VmDefinitionKind.Orchestration);
        Assert.IsType<VmDispatch>(orchestration.Nodes[0].Instruction);
        Assert.IsType<VmBarrier>(orchestration.Nodes[1].Instruction);
        Assert.NotEmpty(new D3D12VmCompiler(providers).GenerateSources(program));
    }

    [Fact]
    public void ExecutionCanBeBuiltDirectlyAndSourcesCanBeEmptyOrMultiple()
    {
        var graph = InferenceGraphBuilder.Builder<ExecutionGraph>("main", g =>
        {
            var input = g.Resource("input", Vector, GraphIO.In);
            var output = g.Resource("output", Vector, GraphIO.Out);
            var operation = g.Operation("copy", Operations.Copy);
            operation.Bind("input", input, ResourceAccess.Read);
            operation.Bind("output", output, ResourceAccess.Write);
            operation.Source("doc", "left", "a").Source("doc", "right", "b");
        });
        foreach (var resource in graph.Resources)
        {
            graph.Storage.Add(resource.Id, new(ResourceStorageKind.ExternalArgument));
        }

        var package = new ExecutionGraphPackage(graph);
        new ExecutionImplementationSelector([new CpuFloat32InstructionCollection()]).Select(package, "cpu-vm");
        var loaded = ExecutionGraphPackageXml.Deserialize(ExecutionGraphPackageXml.Serialize(package));
        Assert.Equal(2, Assert.IsType<ExecutionOperation>(loaded.MainGraph!.Actions[0]).Sources.Count);
        Assert.IsType<ExecutionOperation>(graph.Actions[0]).Sources.Clear();
        var saved = package.Snapshot();
        Assert.Empty(Assert.IsType<ExecutionOperation>(saved.MainGraph!.Actions[0]).Sources);
    }

    private static LogicalGraphPackage Pass(ElementType elementType = ElementType.Float32)
    {
        var graph = new LogicalGraph { Id = "main" };
        var tensor = new TensorDescriptor(elementType, [8]);
        var input = graph.Resource("input", tensor, GraphIO.In);
        var output = graph.Resource("output", tensor, GraphIO.Out);
        var operation = graph.Operation("copy", Operations.Copy);
        operation.Bind("input", input, ResourceAccess.Read);
        operation.Bind("output", output, ResourceAccess.Write);
        return new(graph);
    }

    private static ExecutionGraphPackage Convert(LogicalGraphPackage source) =>
        new DefaultExecutionGraphConverter([new CpuFloat32InstructionCollection()])
            .Convert(source, new("test", "cpu-vm"));

    private sealed class HalfArithmeticInstruction : Instruction, IInstructionCollectionProvider
    {
        public override Guid CollectionId => Guid.Empty;
        public override string Name => Operations.Copy;
        public override ExecutionTarget Target => ExecutionTarget.Cpu;
        public override IReadOnlyList<InstructionSignature> Signatures =>
            [new([new("input", ElementType.Float16, ResourceAccess.Read),
                new("output", ElementType.Float16, ResourceAccess.Write)], [],
                new(ElementType.Float16, ElementType.Float16))];
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
            [new(CollectionId, "test-half", 0, Target)];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
            collectionId == CollectionId && instructionName == Name ? [this] : [];
        protected override InstructionRecording Generate(InstructionParameter[] parameters) =>
            throw new NotSupportedException("This test contract is used only for implementation selection.");
    }
}
