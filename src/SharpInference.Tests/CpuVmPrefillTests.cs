using SharpInference.Graphs;
using System.Runtime.InteropServices;
using SharpInference.Backends.CpuVm;
using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class CpuVmPrefillTests
{
    [Fact]
    public void ExecutesForwardAndPowerOfTwoPrefillWithOffsetsAndFourIndependentStates()
    {
        var program = Program();
        var artifact = new CpuVmCompiler(SharpInference.Runtime.Cpu.CpuInstructionCollections.Create()).Compile(program);
        var weights = MemoryMarshal.AsBytes(Enumerable.Range(1, 64).Select(i => (float)i).ToArray().AsSpan()).ToArray();
        var instances = Enumerable.Range(0, 4).Select(_ => artifact.CreateExecutor()).ToArray();
        try
        {
            for (var instance = 0; instance < instances.Length; instance++)
            {
                IVmExecutable executor = instances[instance];
                Assert.Same(program, executor.Program);
                var slots = Buffers(weights);
                var tokens = MemoryMarshal.Cast<byte, int>(slots[0].AsSpan());
                for (var i = 0; i < 64; i++) tokens[i] = i;
                var counts = new[] { 1, 2, 4, 64 };
                var count = counts[instance];
                executor.Execute("prefill." + count, slots);
                Assert.Equal(count * (count + 1) / 2f, MemoryMarshal.Cast<byte, float>(slots[2])[0]);
                // Entry-level offset selects token 1 without changing any slot bindings.
                executor.Execute("forward.next", slots);
                Assert.Equal(count * (count + 1) / 2f + 2, MemoryMarshal.Cast<byte, float>(slots[2])[0]);
                executor.Execute("forward", slots);
                Assert.Equal(count * (count + 1) / 2f + 3, MemoryMarshal.Cast<byte, float>(slots[2])[0]);
                Assert.Same(weights, slots[1]);
            }
            Assert.Equal(Enumerable.Range(1, 64).Select(i => (float)i), MemoryMarshal.Cast<byte, float>(weights).ToArray());
        }
        finally { foreach (var instance in instances) instance.Dispose(); }
    }

    [Fact]
    public void ExecutesDirectlyThroughManagedExecutionLease()
    {
        var program = Program();
        using var executor = new CpuVmCompiler(SharpInference.Runtime.Cpu.CpuInstructionCollections.Create()).Compile(program).CreateExecutor();
        using var bindings = new VmBindings(program);
        var owners = new List<VmResource>();
        var handles = new List<VmResourceLease>();
        try
        {
            foreach (var slot in program.Slots)
            {
                var storage = new VmMemoryStorage((int)slot.Tensor.ByteLength);
                if (slot.Id == "weights")
                    storage.Write(0, MemoryMarshal.AsBytes(Enumerable.Range(1, 64).Select(i => (float)i).ToArray().AsSpan()));
                if (slot.Id == "tokens") storage.Write(0, MemoryMarshal.AsBytes(new int[] { 2, 3 }.AsSpan()));
                var owner = new VmResource(slot.Tensor, slot.Scope, slot.Access, storage);
                owners.Add(owner);
                var handle = owner.Acquire();
                handles.Add(handle);
                bindings.Bind(slot.Id, handle);
            }
            using var lease = bindings.BeginExecution();
            Assert.Same(program, lease.Program);
            var originalBuffers = lease.GetBuffers();
            executor.Execute("prefill.2", lease);
            Assert.Same(originalBuffers[2], lease.GetBuffers()[2]);
            Assert.Equal(7, MemoryMarshal.Cast<byte, float>(originalBuffers[2])[0]);
        }
        finally
        {
            foreach (var handle in handles) handle.Dispose();
            foreach (var owner in owners) owner.Dispose();
        }
    }

    private static byte[][] Buffers(byte[] weights) => [new byte[64 * sizeof(int)], weights, new byte[4], new byte[4], new byte[4]];

    private static VmProgram Program()
    {
        var scalar = new VmTensor(VmElementType.Float32, [1]);
        var token = new VmTensor(VmElementType.Int32, [1]);
        var tokenBuffer = new VmTensor(VmElementType.Int32, [64]);
        var table = new VmTensor(VmElementType.Float32, [64, 1]);
        var slots = new VmSlot[]
        {
            new("tokens", VmSlotScope.Local, VmAccess.ReadOnly, tokenBuffer),
            new("weights", VmSlotScope.Global, VmAccess.ReadOnly, table),
            new("state", VmSlotScope.Session, VmAccess.ReadWrite, scalar),
            new("scratch", VmSlotScope.Local, VmAccess.ReadWrite, scalar),
            new("sum", VmSlotScope.Local, VmAccess.ReadWrite, scalar),
        };
        var gather = new VmDefinition("gather", VmDefinitionKind.Function,
            [new("table", VmAccess.ReadOnly, table), new("index", VmAccess.ReadOnly, token), new("output", VmAccess.ReadWrite, scalar)],
            [new("gather", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), "core.gather-row", 1, [new("table", "table"), new("index", "index"), new("output", "output")]))]);
        var add = new VmDefinition("add", VmDefinitionKind.Function,
            [new("left", VmAccess.ReadOnly, scalar), new("right", VmAccess.ReadOnly, scalar), new("output", VmAccess.ReadWrite, scalar)],
            [new("add", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), "core.add", 1, [new("left", "left"), new("right", "right"), new("output", "output")]))]);
        var copy = new VmDefinition("copy", VmDefinitionKind.Function,
            [new("input", VmAccess.ReadOnly, scalar), new("output", VmAccess.ReadWrite, scalar)],
            [new("copy", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), "core.copy", 1, [new("input", "input"), new("output", "output")]))]);
        var forwardParameters = slots.Select(s => new VmParameter(s.Id, s.Access, s.Id == "tokens" ? token : s.Tensor)).ToArray();
        var forward = new VmDefinition("forward", VmDefinitionKind.Orchestration, forwardParameters,
            [new("gather", new VmCall("gather", [new("table", "weights"), new("index", "tokens"), new("output", "scratch")])),
             new("add", new VmCall("add", [new("left", "state"), new("right", "scratch"), new("output", "sum")]), ["gather"]),
             new("copy", new VmCall("copy", [new("input", "sum"), new("output", "state")]), ["add"])]);
        var definitions = new List<VmDefinition> { gather, add, copy, forward };
        var entries = new List<VmEntry>
        {
            new("forward", "forward", slots.Select(s => new VmArgument(s.Id, s.Id))),
            new("forward.next", "forward", slots.Select(s => new VmArgument(s.Id, s.Id, s.Id == "tokens" ? 4UL : 0))),
        };
        foreach (var count in new[] { 1, 2, 4, 8, 16, 32, 64 })
        {
            var name = "prefill." + count;
            var calls = Enumerable.Range(0, count).Select(i => new VmNode("token." + i, new VmCall("forward",
                slots.Select(s => new VmArgument(s.Id, s.Id, s.Id == "tokens" ? (ulong)(i * sizeof(int)) : 0))),
                i == 0 ? [] : ["token." + (i - 1)]));
            definitions.Add(new VmDefinition(name, VmDefinitionKind.Orchestration,
                slots.Select(s => new VmParameter(s.Id, s.Access, s.Tensor)), calls));
            entries.Add(new VmEntry(name, name, slots.Select(s => new VmArgument(s.Id, s.Id))));
        }
        return new VmProgram("prefill", "cpu", VmTarget.Cpu, slots, definitions, entries,
            new VmState("state", 1, [new("state", "state")]));
    }
}
