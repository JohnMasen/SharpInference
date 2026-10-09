using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime.Cpu;

namespace SharpInference.Tests;

public sealed class NativeGraphStateInstructionTests
{
    private static readonly GraphOperationId Accumulate = new("external.accumulate", 1);

    [Fact]
    public void ExplicitReadWriteMappingExecutesTypedStateAndRestoresAfterFailure()
    {
        var provider = new Provider(["state"]);
        var graph = Graph(GraphResourceAccess.ReadWrite);
        using var backend = CpuVmBackendFactory.Create(instructionCollections: [provider]);
        backend.Prepare(graph);
        var state = new GraphTensorState(graph, new Dictionary<string, ReadOnlyMemory<byte>> { ["counter"] = BitConverter.GetBytes(7) });
        using var session = backend.CreateGraphSession(new EmptyCatalog(), state);
        var inputs = new Dictionary<ResourceId, ReadOnlyMemory<byte>> { [new("delta")] = BitConverter.GetBytes(3) };
        Assert.Equal(10, BitConverter.ToInt32(session.Execute(inputs)[new("output")]));
        Assert.Equal(10, BitConverter.ToInt32(state.Buffers[0].Data.Span));
        inputs[new("delta")] = BitConverter.GetBytes(-1);
        Assert.Throws<InvalidOperationException>(() => session.Execute(inputs));
        Assert.Equal(10, BitConverter.ToInt32(state.Buffers[0].Data.Span));
        inputs[new("delta")] = BitConverter.GetBytes(2);
        Assert.Equal(12, BitConverter.ToInt32(session.Execute(inputs)[new("output")]));
        session.Reset();
        Assert.Equal(9, BitConverter.ToInt32(session.Execute(inputs)[new("output")]));
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("wrong")]
    [InlineData("duplicate")]
    public void RegistryRejectsMissingOrIncompatibleReadWriteDeclaration(string invalid)
    {
        IReadOnlyList<string>? ports = invalid switch
        {
            "missing" => null,
            "wrong" => ["output"],
            _ => ["state", "state"],
        };
        var registry = new InstructionRegistry([new Provider(ports)]);
        Assert.Throws<InvalidDataException>(() => registry.QueryGraphInstructionBindings());
    }

    [Fact]
    public void GraphCannotDowngradeDeclaredStateAccessToWriteOnly()
    {
        using var backend = CpuVmBackendFactory.Create(instructionCollections: [new Provider(["state"])]);
        Assert.Throws<NotSupportedException>(() => backend.Prepare(Graph(GraphResourceAccess.Write)));
    }

    private static LogicalGraph Graph(GraphResourceAccess access) =>
        new LogicalGraphBuilder(new("external.counter", 1, "forward"),
                new GraphModelSignature("external.counter", "external.counter.state@1", new Dictionary<string, int>()))
            .AddRegion("root", GraphRegionTypes.Graph, "Root")
            .SetStateSchema(new("external.counter.state@1"))
            .AddResource("delta", "Delta", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphInput: true)
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [1]), graphOutput: true)
            .AddResource("state", "Counter", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                new TensorDescriptor(GraphElementType.Int32, [1]))
            .AddStateSlot("counter", "state")
            .AddNode("accumulate", Accumulate, "root",
                [GraphBindings.Read("delta", "delta"), new("state", new("state"), access), GraphBindings.Write("output", "output")])
            .Build();

    private sealed class EmptyCatalog : IModelTensorCatalog
    {
        public IReadOnlyCollection<string> Names => [];
        public bool TryGet(string name, out IModelTensor tensor) { tensor = null!; return false; }
        public IModelTensor GetRequired(string name) => throw new InvalidDataException(name);
    }

    private sealed class Provider(IReadOnlyList<string>? ports) : IInstructionCollectionProvider, IGraphInstructionProvider
    {
        public static readonly Guid Id = new("1474d7b0-e83e-412e-89b2-bc694b84024d");
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
            [new(Id, "Stateful native test", 2, InstructionTarget.Cpu)];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string name) =>
            collectionId == Id && name == "test.accumulate" ? [new Accumulator()] : [];
        public IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings() =>
            [new(Accumulate, Id, "test.accumulate", InstructionTarget.Cpu, ReadWritePorts: ports)];
    }

    private sealed class Accumulator : Instruction
    {
        public override Guid CollectionId => Provider.Id;
        public override string Name => "test.accumulate";
        public override InstructionTarget Target => InstructionTarget.Cpu;
        public override IReadOnlyList<InstructionSignature> Signatures =>
            [new([new("delta", GraphElementType.Int32, GraphResourceAccess.Read),
                new("state", GraphElementType.Int32, GraphResourceAccess.ReadWrite), new("output", GraphElementType.Int32, GraphResourceAccess.Write)],
                [], new(GraphElementType.Float32, GraphElementType.Float32))];
        protected override InstructionRecording Generate(InstructionParameter[] parameters)
        {
            var tensors = parameters.OfType<InstructionTensorParameter>().ToDictionary(parameter => parameter.Name);
            return new($$"""
                {
                    var delta = System.Runtime.InteropServices.MemoryMarshal.Cast<byte,int>({{tensors["delta"].Expression}})[0];
                    var counter = System.Runtime.InteropServices.MemoryMarshal.Cast<byte,int>({{tensors["state"].Expression}});
                    counter[0] += delta;
                    if(delta<0) throw new System.InvalidOperationException("Synthetic failure after state write.");
                    System.Runtime.InteropServices.MemoryMarshal.Cast<byte,int>({{tensors["output"].Expression}})[0] = counter[0];
                }
                """);
        }
    }
}
