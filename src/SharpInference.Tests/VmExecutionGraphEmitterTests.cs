using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Tests;

public sealed class VmExecutionGraphEmitterTests
{
    private static readonly TensorDescriptor Vector = new(ElementType.Float32, [16]);

    [Fact]
    public void CpuEmitterPreservesCallsAllocatesChildInternalsAndOrdersDependencies()
    {
        var child = CopyGraph("child");
        child.Resource("scratch", Vector);
        var root = new LogicalGraph { Id = "main" };
        var input = root.Resource("input", Vector, GraphIO.In);
        var middle = root.Resource("middle", Vector);
        var output = root.Resource("output", Vector, GraphIO.Out);
        var finish = root.Operation("finish", Operations.Copy);
        finish.Bind("input", middle, ResourceAccess.Read).Bind("output", output, ResourceAccess.Write);
        var call = root.Call("invoke", child);
        call.Bind("input", input, ResourceAccess.Read).Bind("output", middle, ResourceAccess.Write);
        IInstructionCollectionProvider[] providers = [new CpuFloat32InstructionCollection()];
        var package = new DefaultExecutionGraphConverter(providers).Convert(new(root, child), new("test", "cpu-vm"));
        var program = new VmExecutionGraphEmitter(VmTarget.Cpu, providers).Emit(package);
        var main = program.Definitions.Single(definition => definition.Id == "graph.main");
        Assert.Equal(["invoke", "finish"], main.Nodes.Select(node => node.Id));
        var vmCall = Assert.IsType<VmCall>(main.Nodes[0].Instruction);
        Assert.Equal("graph.child", vmCall.Definition);
        Assert.Equal(["input", "middle", "invoke/scratch"], vmCall.Arguments.Select(argument => argument.Source));
        Assert.Contains(program.Slots, slot => slot.Id == "invoke/scratch" && slot.Scope == VmSlotScope.Local);
        Assert.NotEmpty(new CpuVmCompiler(providers).GenerateSource(program).Source);
    }

    [Fact]
    public void D3D12EmitterProducesSelectedKernelDispatchBarrierAndCompilableSource()
    {
        IInstructionCollectionProvider[] providers = [new GpuFloat32InstructionCollection()];
        var package = new DefaultExecutionGraphConverter(providers).Convert(new(CopyGraph("main")), new("test", "d3d12-vm"));
        var program = new VmExecutionGraphEmitter(VmTarget.Direct3D12, providers).Emit(package);
        Assert.NotNull(Assert.Single(program.Definitions, definition => definition.Kind == VmDefinitionKind.Kernel).Threads);
        var main = program.Definitions.Single(definition => definition.Kind == VmDefinitionKind.Orchestration);
        Assert.IsType<VmDispatch>(main.Nodes[0].Instruction);
        Assert.IsType<VmBarrier>(main.Nodes[1].Instruction);
        Assert.Contains("RWByteAddressBuffer", new D3D12VmCompiler(providers).GenerateSources(program).Values.Single());
    }

    [Fact]
    public void EmitterRejectsMissingSelectedImplementationRatherThanSelectingIt()
    {
        IInstructionCollectionProvider[] providers = [new CpuFloat32InstructionCollection()];
        var package = new DefaultExecutionGraphConverter(providers).Convert(new(CopyGraph("main")), new("test", "cpu-vm"));
        Assert.IsType<ExecutionOperation>(package.MainGraph!.Actions[0]).SelectedImplementation = null;
        Assert.Throws<InvalidDataException>(() => new VmExecutionGraphEmitter(VmTarget.Cpu, providers).Emit(package));
    }

    private static LogicalGraph CopyGraph(string id)
    {
        var graph = new LogicalGraph { Id = id };
        var input = graph.Resource("input", Vector, GraphIO.In);
        var output = graph.Resource("output", Vector, GraphIO.Out);
        graph.Operation("copy", Operations.Copy).Bind("input", input, ResourceAccess.Read)
            .Bind("output", output, ResourceAccess.Write);
        return graph;
    }
}
