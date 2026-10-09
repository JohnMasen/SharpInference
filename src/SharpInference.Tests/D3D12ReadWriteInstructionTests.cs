using System.Runtime.InteropServices;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Vm;

namespace SharpInference.Tests;

[Collection(VmGraphGpuCollection.Name)]
public sealed class D3D12ReadWriteInstructionTests
{
    [Fact]
    public void MultipleReadWriteOutputsOfOneOperatorAreNotCrossThreadProducers()
    {
        var artifact = new D3D12VmCompiler([new Provider()]).Compile(Program(false));
        using var executor = artifact.CreateExecutor();
        executor.Upload("input", Bytes([1, 2, 3, 4]));
        executor.Upload("first", Bytes([10, 20, 30, 40]));
        executor.Upload("second", Bytes([100, 200, 300, 400]));
        executor.Execute("forward");
        Assert.Equal([11f, 22, 33, 44], Floats(executor.Readback("first")));
        Assert.Equal([101f, 202, 303, 404], Floats(executor.Readback("second")));
        Assert.Equal([112f, 224, 336, 448], Floats(executor.Readback("output")));
    }

    [Fact]
    public void SeparateReadWriteOperatorsStillRequireDispatchIsolation()
    {
        Assert.Throws<NotSupportedException>(() => new D3D12VmCompiler([new Provider()]).Compile(Program(true)));
    }

    private static VmProgram Program(bool duplicate)
    {
        var tensor = new VmTensor(VmElementType.Float32, [4]);
        VmParameter[] parameters = [new("input", VmAccess.ReadOnly, tensor), new("first", VmAccess.ReadWrite, tensor),
            new("second", VmAccess.ReadWrite, tensor), new("output", VmAccess.ReadWrite, tensor)];
        var operation = new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), Provider.Id, "test.multi-state",
            parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)));
        VmNode[] nodes = duplicate ? [new("first", operation), new("second", operation, ["first"])] : [new("first", operation)];
        var slots = parameters.Select(parameter => new VmSlot(parameter.Name,
            parameter.Name is "first" or "second" ? VmSlotScope.Session : VmSlotScope.Local, parameter.Access, tensor)).ToArray();
        return new("multi-state", "multi-state@1", VmTarget.Direct3D12, slots,
            [new("kernel", VmDefinitionKind.Kernel, parameters, nodes, new(64)),
             new("forward", VmDefinitionKind.Orchestration, parameters,
                [new("dispatch", new VmDispatch("kernel", parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)), new(1))),
                 new("barrier", new VmBarrier(["first", "second", "output"]), ["dispatch"])])],
            [new("forward", "forward", parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)))],
            new("multi-state@1", 1, [new("first", "first"), new("second", "second")]));
    }
    private static byte[] Bytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static float[] Floats(byte[] values) => MemoryMarshal.Cast<byte, float>(values).ToArray();
    private sealed class Provider : IInstructionCollectionProvider
    {
        public static readonly Guid Id = new("7c3a43b4-2b9d-4201-bd95-2d228573e4fe");
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() => [new(Id, "Multiple native state outputs", 1, InstructionTarget.Direct3D12)];
        public IReadOnlyList<Instruction> QueryInstruction(Guid id, string name) => id == Id && name == "test.multi-state" ? [new MultiState()] : [];
    }
    private sealed class MultiState : Instruction
    {
        public override Guid CollectionId => Provider.Id;
        public override string Name => "test.multi-state";
        public override InstructionTarget Target => InstructionTarget.Direct3D12;
        public override IReadOnlyList<InstructionSignature> Signatures =>
            [new([new("input", GraphElementType.Float32, GraphResourceAccess.Read),
                  new("first", GraphElementType.Float32, GraphResourceAccess.ReadWrite),
                  new("second", GraphElementType.Float32, GraphResourceAccess.ReadWrite),
                  new("output", GraphElementType.Float32, GraphResourceAccess.Write)], [], new(GraphElementType.Float32, GraphElementType.Float32))];
        protected override InstructionRecording Generate(InstructionParameter[] parameters)
        {
            var tensors = parameters.OfType<InstructionTensorParameter>().ToDictionary(parameter => parameter.Name);
            string Load(string port) => $"asfloat({tensors[port].Expression}.Load({tensors[port].OffsetExpression}+i*4u))";
            string Store(string port, string value) => $"{tensors[port].Expression}.Store({tensors[port].OffsetExpression}+i*4u,asuint({value}));";
            return new($$"""
                {
                    if(i<4u) {
                        float first={{Load("first")}}+{{Load("input")}};
                        float second={{Load("second")}}+{{Load("input")}};
                        {{Store("first", "first")}}
                        {{Store("second", "second")}}
                        {{Store("output", "first+second")}}
                    }
                }
                """);
        }
    }
}
