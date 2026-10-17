using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

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
        var graph = BuildGraph("root", Resources(), Operation("copy"));
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
        var graph = BuildGraph(
            "root",
            Resources(),
            Operation("copy", PrimitiveOperations.Copy));
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
        var child = BuildGraph(
            "child",
            [
                .. Resources(),
                new(new("scratch"), "scratch", GraphDefinitionResourceKind.Temporary,
                    new(ElementType.Float32, [16])),
            ],
            Operation("copy"));
        var call = Call(
            "child-call",
            child.Id.Value,
            [Binding("input", "input", ResourceAccess.Read),
             Binding("output", "middle", ResourceAccess.Write)],
            GraphCompileBoundary.Atomic);
        var finish = Operation(
            "finish",
            new("test.copy"),
            [
                Binding("input", "middle", ResourceAccess.Read),
                Binding("output", "output", ResourceAccess.Write),
            ],
            [new ExecutionNodeId("child-call")]);
        var root = BuildGraph(
            "root",
            [
                .. Resources(),
                new(new("middle"), "middle", GraphDefinitionResourceKind.Temporary,
                    new(ElementType.Float32, [16])),
            ],
            finish,
            call);

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
        var graph = BuildGraph("root", Resources(), Operation("copy"));

        var error = Assert.Throws<NotSupportedException>(() =>
            new VmExecutionGraphEmitter(VmTarget.Cpu, [new EmptyProvider()])
                .Emit(new ExecutionGraphProgram([graph])));

        Assert.Contains("cpu.managed", error.Message);
        Assert.Contains("test.copy", error.Message);
    }

    private static ExecutionGraphProgram ProgramWithChild()
    {
        var child = BuildGraph("child", Resources(), Operation("copy"));
        var root = BuildGraph(
            "root",
            Resources(),
            Call(
                "child-call",
                "child",
                [Binding("input", "input", ResourceAccess.Read),
                 Binding("output", "output", ResourceAccess.Write)],
                GraphCompileBoundary.Auto));
        return new([root, child]);
    }

    private static NodeSpec Operation(
        string id,
        OperationId? operation = null,
        IReadOnlyList<ModelResourceBinding>? bindings = null,
        IReadOnlyList<ExecutionNodeId>? dependencies = null) =>
        new(
            new ExecutionOperationNode(
                new(id),
                operation ?? new("test.copy"),
                new Dictionary<string, string>(),
                new PrecisionRequirement(
                    ElementType.Float32, ElementType.Float32),
                new ExecutionSourceMap([])),
            (bindings ??
             [Binding("input", "input", ResourceAccess.Read),
              Binding("output", "output", ResourceAccess.Write)])
                .Select(binding => binding with
                {
                    BindingId = $"{id}:{binding.Port}",
                }).ToArray(),
            dependencies ?? []);

    private static ModelResourceBinding Binding(
        string port,
        string resource,
        ResourceAccess access) =>
        new(
            port,
            new(GraphBindingOwner.Parent, new(resource)),
            access,
            $"binding:{port}:{resource}");

    private static NodeSpec Call(
        string id,
        string target,
        IReadOnlyList<ModelResourceBinding> bindings,
        GraphCompileBoundary boundary) =>
        new(
            new ExecutionGraphCallNode(
                new(id),
                new(target),
                boundary,
                new ExecutionSourceMap([])),
            bindings.Select(binding => binding with
            {
                BindingId = $"{id}:{binding.Port}",
            }).ToArray(),
            []);

    private static ExecutionGraph BuildGraph(
        string id,
        IEnumerable<ExecutionGraphResource> resources,
        params NodeSpec[] nodes)
    {
        var builder = new ExecutionGraphBuilder(new(id));
        foreach (var resource in resources)
        {
            builder.AddResource(resource);
        }

        foreach (var node in nodes)
        {
            if (node.Node is ExecutionOperationNode operation)
            {
                builder.AddOperation(operation, node.Bindings, node.Dependencies);
            }
            else
            {
                builder.AddCall(
                    (ExecutionGraphCallNode)node.Node,
                    node.Bindings,
                    node.Dependencies);
            }
        }

        return builder.Build();
    }

    private sealed record NodeSpec(
        ExecutionGraphNode Node,
        IReadOnlyList<ModelResourceBinding> Bindings,
        IReadOnlyList<ExecutionNodeId> Dependencies);

    private static ExecutionGraphResource[] Resources() =>
    [
        new(new("input"), "input", GraphDefinitionResourceKind.Input,
            new(ElementType.Float32, [16])),
        new(new("output"), "output", GraphDefinitionResourceKind.Output,
            new(ElementType.Float32, [16])),
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
                [new("input", ElementType.Float32, ResourceAccess.Read),
                 new("output", ElementType.Float32, ResourceAccess.Write)],
                [],
                new(ElementType.Float32, ElementType.Float32))];

        protected override InstructionRecording Generate(InstructionParameter[] parameters) => new("");
    }
}
