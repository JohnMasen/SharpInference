using System.Runtime.InteropServices;
using SharpInference.Backends.CpuVm;
using SharpInference.Graphs;
using SharpInference.Runtime;
using SharpInference.Runtime.Cpu;
using SharpInference.Runtime.D3D12;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

[Collection(VmGraphGpuCollection.Name)]
public sealed class VmTensorGraphSessionTests
{
    [Theory]
    [InlineData("core.matrix-multiply", false)]
    [InlineData("core.matrix-multiply", true)]
    [InlineData("core.affine", false)]
    [InlineData("core.affine", true)]
    [InlineData("core.bias-add", false)]
    [InlineData("core.bias-add", true)]
    public void CompleteStandardCatalogPreparesAndExecutesMatrixOperations(string operation, bool gpu)
    {
        var biasAdd = operation == "core.bias-add";
        var builder = new LogicalGraphBuilder(new("matrix", 1, "forward"),
                new GraphModelSignature("matrix", "matrix.empty@1", new Dictionary<string, int>()))
            .AddRegion("root", GraphRegionTypes.Graph, "Root")
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2, biasAdd ? 2 : 3]), graphInput: true)
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2, 2]), graphOutput: true);
        var bindings = new List<NodeResourceBinding> { GraphBindings.Read(biasAdd ? "input" : "left", "input") };
        var inputs = new Dictionary<ResourceId, ReadOnlyMemory<byte>>
        {
            [new("input")] = Bytes(biasAdd ? [4f, 8f, -2f, 2f] : [1f, 2f, 3f, -1f, 0f, 1f]),
        };
        if (!biasAdd)
        {
            builder.AddResource("matrix", "Matrix", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2, 3]), graphInput: true);
            bindings.Add(GraphBindings.Read("right", "matrix"));
            inputs[new("matrix")] = Bytes([2f, 1f, 0f, 0f, 1f, 2f]);
        }
        if (operation != "core.matrix-multiply")
        {
            builder.AddResource("bias", "Bias", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [2]), graphInput: true);
            bindings.Add(GraphBindings.Read("bias", "bias"));
            inputs[new("bias")] = Bytes([0.5f, -0.5f]);
        }
        bindings.Add(GraphBindings.Write("output", "output"));
        var graph = builder.AddNode("operation", new(operation), "root", bindings,
            attributes: biasAdd ? new Dictionary<string, string>() : new Dictionary<string, string>
            {
                ["transpose_left"] = "false", ["transpose_right"] = "true",
            }).Build();
        using var backend = gpu ? D3D12VmBackendFactory.Create() : CpuVmBackendFactory.Create();
        backend.Prepare(graph);
        using var session = backend.CreateGraphSession(new EmptyCatalog());
        var actual = MemoryMarshal.Cast<byte, float>(session.Execute(inputs)[new("output")]).ToArray();
        Assert.Equal(operation == "core.matrix-multiply" ? [4f, 8f, -2f, 2f] : [4.5f, 7.5f, -1.5f, 1.5f], actual);

        static byte[] Bytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    }

    [Fact]
    public void NonTokenGraphExecutesThroughCompiledCpuProgram()
    {
        var graph = new LogicalGraphBuilder(new("external", 1, "double-array"),
                new GraphModelSignature("array", "array.empty@1", new Dictionary<string, int>()))
            .AddRegion("root", GraphRegionTypes.Graph, "Root")
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [3]), graphInput: true)
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [3]), graphOutput: true)
            .AddNode("add", PrimitiveGraphOperations.Add, "root",
                [GraphBindings.Read("left", "input"), GraphBindings.Read("right", "input"), GraphBindings.Write("output", "output")])
            .Build();
        var collections = SharpInference.Runtime.Cpu.CpuInstructionCollections.Create();
        using var backend = new VmGraphBackend(VmTarget.Cpu, program =>
        {
            var artifact = new CpuVmCompiler(collections).Compile(program);
            return () => artifact.LoadExecutable();
        }, optimization: new(ReuseLocalStorage: false), generatorCollections: collections);
        backend.Prepare(graph);
        using var session = backend.CreateGraphSession(new EmptyCatalog());
        Assert.Throws<InvalidOperationException>(() => backend.Prepare(graph));
        var result = session.Execute(new Dictionary<ResourceId, ReadOnlyMemory<byte>>
        {
            [new("input")] = MemoryMarshal.AsBytes(new float[] { 1, -2, 3 }.AsSpan()).ToArray(),
        });
        Assert.Equal([2f, -4f, 6f], MemoryMarshal.Cast<byte, float>(result[new("output")]).ToArray());
        Assert.Throws<ArgumentException>(() => session.Execute(new Dictionary<ResourceId, ReadOnlyMemory<byte>>()));
        backend.Dispose();
        Assert.Throws<ObjectDisposedException>(() => session.Execute(new Dictionary<ResourceId, ReadOnlyMemory<byte>>()));
    }

    [Fact]
    public void TypedIntegerStateResetsAndRestoresAfterFailedExecution()
    {
        var graph = new LogicalGraphBuilder(new("counter", 1, "counter"),
                new GraphModelSignature("counter", "counter.state@1", new Dictionary<string, int>()))
            .AddRegion("root", GraphRegionTypes.Graph, "Root")
            .SetStateSchema(new("Counter_State"))
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [2]), graphInput: true)
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Int32, [2]), graphOutput: true)
            .AddResource("state", "State", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                new TensorDescriptor(GraphElementType.Int32, [1]))
            .AddStateSlot("position", "state").Build();
        var slots = new VmSlot[]
        {
            new("input", VmSlotScope.Local, VmAccess.ReadOnly, new(VmElementType.Int32, [2])),
            new("output", VmSlotScope.Local, VmAccess.ReadWrite, new(VmElementType.Int32, [2])),
            new("state", VmSlotScope.Session, VmAccess.ReadWrite, new(VmElementType.Int32, [1])),
        };
        var program = new VmProgram("counter", "vm:counter.state@1", VmTarget.Cpu, slots,
            [new VmDefinition("forward", VmDefinitionKind.Orchestration,
                slots.Select(slot => new VmParameter(slot.Id, slot.Access, slot.Tensor)), [])],
            [new VmEntry("forward", "forward", slots.Select(slot => new VmArgument(slot.Id, slot.Id)))],
            new VmState("Counter_State", 1, [new VmStateEntry("position", "state")]));
        CounterExecutable? executable = null;
        using var backend = new VmGraphBackend(VmTarget.Cpu, _ => () => executable = new CounterExecutable(program),
            suppliedProgram: program);
        backend.Prepare(graph);
        var state = new GraphTensorState(graph, new Dictionary<string, ReadOnlyMemory<byte>>
        {
            ["position"] = BitConverter.GetBytes(7),
        });
        using var session = backend.CreateGraphSession(new EmptyCatalog(), state);
        var inputs = new Dictionary<ResourceId, ReadOnlyMemory<byte>> { [new("input")] = new byte[8] };
        Assert.Equal(8, BitConverter.ToInt32(session.Execute(inputs)[new("output")]));
        Assert.Equal(8, BitConverter.ToInt32(state.Buffers[0].Data.Span));
        executable!.Fail = true;
        Assert.Throws<InvalidOperationException>(() => session.Execute(inputs));
        Assert.Equal(8, BitConverter.ToInt32(state.Buffers[0].Data.Span));
        executable.Fail = false;
        using var cancellation = new CancellationTokenSource();
        executable.AfterWrite = cancellation.Cancel;
        Assert.Throws<OperationCanceledException>(() => session.Execute(inputs, cancellation.Token));
        Assert.Equal(8, BitConverter.ToInt32(state.Buffers[0].Data.Span));
        executable.AfterWrite = null;
        Assert.Equal(9, BitConverter.ToInt32(session.Execute(inputs)[new("output")]));
        session.Reset();
        Assert.Equal(8, BitConverter.ToInt32(session.Execute(inputs)[new("output")]));
    }

    private sealed class EmptyCatalog : IModelTensorCatalog
    {
        public IReadOnlyCollection<string> Names => Array.Empty<string>();
        public bool TryGet(string name, out IModelTensor tensor) { tensor = null!; return false; }
        public IModelTensor GetRequired(string name) => throw new InvalidDataException(name);
    }

    private sealed class CounterExecutable(VmProgram program) : IVmTaskExecutable
    {
        public VmProgram Program => program;
        public bool Fail { get; set; }
        public Action? AfterWrite { get; set; }
        public void ExecuteTask(string entryName, VmExecutionLease lease, CancellationToken cancellation)
        {
            cancellation.ThrowIfCancellationRequested();
            var value = new byte[4];
            lease.Read("state", 0, value);
            value = BitConverter.GetBytes(BitConverter.ToInt32(value) + 1);
            lease.Write("state", 0, value);
            AfterWrite?.Invoke();
            if (Fail) throw new InvalidOperationException("Injected execution failure.");
            var output = new byte[8];
            value.CopyTo(output, 0);
            lease.Write("output", 0, output);
        }
        public void Execute(string entryName, byte[][] slots) => throw new NotSupportedException();
        public void Dispose() { }
    }
}
