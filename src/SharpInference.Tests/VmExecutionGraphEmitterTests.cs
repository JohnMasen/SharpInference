using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class VmExecutionGraphEmitterTests
{
    [Fact]
    public void CpuEmitterPreservesGraphCallAsVmCall()
    {
        var program = ProgramWithChild();
        var provider = new Provider(InstructionTarget.Cpu);
        var emitted = new VmExecutionGraphEmitter(VmTarget.Cpu, [provider])
            .Emit(program);

        Assert.Equal(VmTarget.Cpu, emitted.Target);
        Assert.Equal(2, emitted.Definitions.Count);
        var root = emitted.Definitions.Single(definition => definition.Id == "graph.root");
        var call = Assert.IsType<VmCall>(Assert.Single(root.Nodes).Instruction);
        Assert.Equal("graph.child", call.Definition);
        Assert.Equal(["input", "output"], call.Arguments.Select(argument => argument.Source));
        Assert.NotEmpty(new CpuVmCompiler([provider]).GenerateSource(emitted).Source);
    }

    [Fact]
    public void D3D12EmitterCreatesKernelDispatchAndBarrier()
    {
        var graph = new ExecutionGraph(
            new("root"),
            Resources(),
            [Operation("copy")]);
        var provider = new Provider(InstructionTarget.Direct3D12);
        var emitted = new VmExecutionGraphEmitter(
            VmTarget.Direct3D12,
            [provider]).Emit(
                new ExecutionGraphProgram([graph]));

        var kernel = Assert.Single(emitted.Definitions,
            definition => definition.Kind == VmDefinitionKind.Kernel);
        Assert.Equal(new VmThreadGroup(8), kernel.Threads);
        var root = emitted.Definitions.Single(
            definition => definition.Kind == VmDefinitionKind.Orchestration);
        Assert.IsType<VmDispatch>(root.Nodes[0].Instruction);
        Assert.IsType<VmBarrier>(root.Nodes[1].Instruction);
    }

    [Fact]
    public void ProductionCompilersConsumeEmittedProgram()
    {
        var graph = new ExecutionGraph(
            new("root"),
            Resources(),
            [Operation("copy") with { Operation = PrimitiveGraphOperations.Copy }]);
        var program = new ExecutionGraphProgram([graph]);
        IInstructionCollectionProvider[] cpuProviders = [new CpuFloat32InstructionCollection()];
        var cpuProgram = new VmExecutionGraphEmitter(VmTarget.Cpu, cpuProviders).Emit(program);
        var cpuArtifact = new CpuVmCompiler(cpuProviders).GenerateSource(cpuProgram);
        Assert.NotEmpty(cpuArtifact.Source);

        IInstructionCollectionProvider[] gpuProviders = [new GpuFloat32InstructionCollection()];
        var gpuProgram = new VmExecutionGraphEmitter(VmTarget.Direct3D12, gpuProviders).Emit(program);
        var sources = new D3D12VmCompiler(gpuProviders).GenerateSources(gpuProgram);
        Assert.Single(sources);
        Assert.Contains("RWByteAddressBuffer", sources.Values.Single(), StringComparison.Ordinal);
    }

    [Fact]
    public void CpuEmitterAllocatesChildInternalsAndUsesStableTopologicalOrder()
    {
        var child = new ExecutionGraph(
            new("child"),
            [
                .. Resources(),
                new(new("scratch"), "scratch", GraphDefinitionResourceKind.Temporary,
                    new(GraphElementType.Float32, [16])),
            ],
            [Operation("copy")]);
        var call = new ExecutionGraphCallNode(
            new("child-call"),
            child.Id,
            [Binding("input", "input", GraphResourceAccess.Read),
             Binding("output", "middle", GraphResourceAccess.Write)],
            [],
            GraphCompileBoundary.Atomic,
            new ExecutionSourceMap([]));
        var finish = Operation("finish") with
        {
            Bindings =
            [
                Binding("input", "middle", GraphResourceAccess.Read),
                Binding("output", "output", GraphResourceAccess.Write),
            ],
            Dependencies = [call.Id],
        };
        var root = new ExecutionGraph(
            new("root"),
            [
                .. Resources(),
                new(new("middle"), "middle", GraphDefinitionResourceKind.Temporary,
                    new(GraphElementType.Float32, [16])),
            ],
            [finish, call]);

        var emitted = new VmExecutionGraphEmitter(
            VmTarget.Cpu, [new Provider(InstructionTarget.Cpu)])
            .Emit(new ExecutionGraphProgram([root, child]));

        Assert.Contains(emitted.Slots, slot =>
            slot.Id == "child-call/scratch" && slot.Scope == VmSlotScope.Local);
        var definition = emitted.Definitions.Single(value => value.Id == "graph.root");
        Assert.Equal(["child-call", "finish"], definition.Nodes.Select(node => node.Id));
    }

    [Fact]
    public void UnsupportedOperationNamesTargetAndOperation()
    {
        var graph = new ExecutionGraph(new("root"), Resources(), [Operation("copy")]);

        var error = Assert.Throws<NotSupportedException>(() =>
            new VmExecutionGraphEmitter(VmTarget.Cpu, [new EmptyProvider()])
                .Emit(new ExecutionGraphProgram([graph])));

        Assert.Contains("cpu.managed", error.Message);
        Assert.Contains("test.copy", error.Message);
    }

    private static ExecutionGraphProgram ProgramWithChild()
    {
        var child = new ExecutionGraph(
            new("child"),
            Resources(),
            [Operation("copy")]);
        var root = new ExecutionGraph(
            new("root"),
            Resources(),
            [new ExecutionGraphCallNode(
                new("child-call"),
                new("child"),
                [Binding("input", "input", GraphResourceAccess.Read),
                 Binding("output", "output", GraphResourceAccess.Write)],
                [],
                GraphCompileBoundary.Auto,
                new ExecutionSourceMap([]))]);
        return new([root, child]);
    }

    private static ExecutionOperationNode Operation(string id) =>
        new(
            new(id),
            new("test.copy"),
            [Binding("input", "input", GraphResourceAccess.Read),
             Binding("output", "output", GraphResourceAccess.Write)],
            [],
            new Dictionary<string, string>(),
            new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32),
            new ExecutionSourceMap([]));

    private static ExecutionGraphBinding Binding(
        string port,
        string resource,
        GraphResourceAccess access) =>
        new(port, new ResourceId(resource), access);

    private static ExecutionGraphResource[] Resources() =>
    [
        new(new("input"), "input", GraphDefinitionResourceKind.Input,
            new(GraphElementType.Float32, [16])),
        new(new("output"), "output", GraphDefinitionResourceKind.Output,
            new(GraphElementType.Float32, [16])),
    ];

    private sealed class Provider(InstructionTarget target) :
        IInstructionCollectionProvider,
        IGraphInstructionProvider
    {
        private readonly Guid collectionId = Guid.NewGuid();

        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
            [new(collectionId, "Emitter test", 0, target)];

        public IReadOnlyList<Instruction> QueryInstruction(
            Guid requestedCollection,
            string instructionName) =>
            requestedCollection == collectionId && instructionName == "copy"
                ? [new TestInstruction(collectionId, target)]
                : [];

        public IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings() =>
            [new(
                new("test.copy"),
                collectionId,
                "copy",
                target,
                target == InstructionTarget.Direct3D12 ? _ => new(8, 1, 1, 2, 1, 1) : null)];
    }

    private sealed class EmptyProvider : IInstructionCollectionProvider
    {
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() => [];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) => [];
    }

    private sealed class TestInstruction(Guid collectionId, InstructionTarget target) : Instruction
    {
        public override Guid CollectionId => collectionId;
        public override string Name => "copy";
        public override InstructionTarget Target => target;
        public override IReadOnlyList<InstructionSignature> Signatures =>
            [new(
                [new("input", GraphElementType.Float32, GraphResourceAccess.Read),
                 new("output", GraphElementType.Float32, GraphResourceAccess.Write)],
                [],
                new(GraphElementType.Float32, GraphElementType.Float32))];

        protected override InstructionRecording Generate(InstructionParameter[] parameters) => new("");
    }
}
