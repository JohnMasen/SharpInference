using System.Runtime.InteropServices;
using SharpInference.Graphs;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class VmInferenceTests
{
    [Fact]
    public void OptimizerExportsReusableDefinitionsAndUnrolledPrefill()
    {
        var program = VmGraphOptimizer.Optimize(Graph(), VmTarget.Cpu, new(PrefillCapacity: 8));
        Assert.Contains(program.Entries, entry => entry.Name == "prefill.8");
        Assert.Equal(8UL, program.Slots.Single(slot => slot.Id == "token").Tensor.ElementCount);
        var prefill = program.Definitions.Single(definition => definition.Id == "prefill.8");
        Assert.Equal(2, prefill.Nodes.Count);
        var last = Assert.IsType<VmCall>(prefill.Nodes[^1].Instruction);
        Assert.Equal("prefill.4", last.Definition);
        Assert.Equal(16UL, last.Arguments.Single(argument => argument.Parameter == "token").ByteOffset);
        Assert.Equal(VmProgramXml.Serialize(program), VmProgramXml.Serialize(VmProgramXml.Deserialize(VmProgramXml.Serialize(program))));
        Assert.All(program.State.Entries, entry =>
            Assert.Equal(VmSlotScope.Session, program.Slots.Single(slot => slot.Id == entry.Slot).Scope));
    }

    [Fact]
    public void GpuOptimizerKeepsDispatchAndBarrierDecisionsInXml()
    {
        var program = VmGraphOptimizer.Optimize(Graph(), VmTarget.Direct3D12, new(ThreadsPerGroup: 128));
        var forward = program.Definitions.Single(definition => definition.Id == "forward");
        Assert.Equal(Graph().Nodes.Count, forward.Nodes.Count(node => node.Instruction is VmDispatch));
        Assert.Equal(Graph().Nodes.Count, forward.Nodes.Count(node => node.Instruction is VmBarrier));
        Assert.All(program.Definitions.Where(definition => definition.Kind == VmDefinitionKind.Kernel),
            kernel => Assert.Equal(128U, kernel.Threads!.X));
    }

    [Fact]
    public async Task PrefillAndGenerationShareStateWhileSessionsRemainIsolated()
    {
        var program = VmGraphOptimizer.Optimize(Graph(), VmTarget.Cpu, new(PrefillCapacity: 8));
        var created = 0;
        var initializedWeights = 0;
        await using var engine = await VmInferenceEngine.CreateAsync(program, "token", "logits",
            (_, storage) => { initializedWeights++; storage.Write(0, new byte[(int)storage.ByteLength]); },
            () => { created++; return new TestExecutable(program); }, new(8, 8));
        await using var first = engine.CreateSession();
        await using var second = engine.CreateSession();
        Assert.Equal(4, created);
        Assert.Equal(1, initializedWeights);
        Assert.All(await first.PrefillAsync(new int[] { 1, 2, 3 }), value => Assert.Equal(6f, value));
        await using (var generation = await first.BeginGenerationAsync())
        {
            Assert.All(generation.ForwardToken(4), value => Assert.Equal(10f, value));
            Assert.All(generation.ForwardToken(3), value => Assert.Equal(13f, value));
        }
        Assert.All(await second.ForwardAsync(1), value => Assert.Equal(1f, value));
        using var snapshot = new MemoryStream();
        await first.ExportStateAsync(snapshot, "test-model");
        snapshot.Position = 0;
        await second.ImportStateAsync(snapshot, "test-model", 4);
        Assert.All(await second.ForwardAsync(1), value => Assert.Equal(14f, value));
        await first.ResetAsync();
        Assert.All(await first.ForwardAsync(2), value => Assert.Equal(2f, value));
        Assert.Equal(4, created);
    }

    [Fact]
    public async Task SessionCloseCancelsHeldGenerationAndEngineCanContinue()
    {
        var program = VmGraphOptimizer.Optimize(Graph(), VmTarget.Cpu, new(PrefillCapacity: 8));
        await using var engine = await VmInferenceEngine.CreateAsync(program, "token", "logits",
            (_, storage) => storage.Write(0, new byte[(int)storage.ByteLength]),
            () => new TestExecutable(program), new(8, 8, 1, 1));
        var first = engine.CreateSession();
        var generation = await first.BeginGenerationAsync();
        generation.ForwardToken(2);
        await first.DisposeAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(10));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => generation.DisposeAsync().AsTask());
        await using var next = engine.CreateSession();
        Assert.All(await next.ForwardAsync(3), value => Assert.Equal(3f, value));
    }

    [Fact]
    public async Task InvalidTokensAreRejectedBeforeStateChanges()
    {
        var program = VmGraphOptimizer.Optimize(Graph(), VmTarget.Cpu);
        await using var engine = await VmInferenceEngine.CreateAsync(program, "token", "logits",
            (_, storage) => storage.Write(0, new byte[(int)storage.ByteLength]),
            () => new TestExecutable(program), new(2, 2, MaximumPrefillTokens: 4));
        await using var session = engine.CreateSession();
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => session.PrefillAsync(new int[] { 8 }).AsTask());
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => session.PrefillAsync(new int[] { 1, 1, 1, 1, 1 }).AsTask());
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => session.ForwardAsync(-1).AsTask());
        await using (var generation = await session.BeginGenerationAsync())
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => generation.ForwardToken(8));
            Assert.All(generation.ForwardToken(1), value => Assert.Equal(1f, value));
        }
        await session.ResetAsync();
        Assert.All(await session.ForwardAsync(1), value => Assert.Equal(1f, value));
    }

    [Theory]
    [InlineData(VmTuningWorkload.Prefill)]
    [InlineData(VmTuningWorkload.Inference)]
    public async Task TuningRestoresStateAndRejectsIncorrectCandidates(VmTuningWorkload workload)
    {
        var program = VmGraphOptimizer.Optimize(Graph(), VmTarget.Cpu);
        IReadOnlyDictionary<string, byte[]> initial = new Dictionary<string, byte[]> { ["sum"] = BitConverter.GetBytes(0f) };
        var reference = new VmTuningReference(Enumerable.Repeat(3f, 8).ToArray(),
            new Dictionary<string, byte[]> { ["sum"] = BitConverter.GetBytes(3f) });
        var calls = 0;
        var result = await VmProgramTuner.TuneAsync(
            [new("wrong", program), new("correct", program)], "token", "logits", new int[] { 1, 2 },
            initial, reference, (candidate, _) =>
            {
                var wrong = calls++ == 0;
                return ValueTask.FromResult<Func<IVmExecutable>>(() => new TestExecutable(candidate, wrong));
            }, (_, storage) => storage.Write(0, new byte[(int)storage.ByteLength]),
            new("test-hardware", "test-driver", "test-backend", "test-compiler"),
            new(WarmupRuns: 2, MeasuredRuns: 3, Workload: workload));
        Assert.Equal("correct", result.Winner);
        Assert.False(result.Candidates[0].Accepted);
        Assert.Contains("differs", result.Candidates[0].Diagnostic);
        Assert.True(result.Candidates[1].Accepted);
        Assert.True(result.Candidates[1].MedianMilliseconds > 0);
        Assert.Equal(2, calls);
    }

    [Fact]
    public async Task StateWriteSnapshotsCallerBuffersBeforeWaitingForGeneration()
    {
        var program = VmGraphOptimizer.Optimize(Graph(), VmTarget.Cpu);
        await using var engine = await VmInferenceEngine.CreateAsync(program, "token", "logits",
            (_, storage) => storage.Write(0, new byte[(int)storage.ByteLength]),
            () => new TestExecutable(program), new(2, 2));
        await using var session = engine.CreateSession();
        var generation = await session.BeginGenerationAsync();
        var bytes = BitConverter.GetBytes(4f);
        var write = session.WriteStateAsync(new Dictionary<string, byte[]> { ["sum"] = bytes }).AsTask();
        BitConverter.GetBytes(100f).CopyTo(bytes, 0);
        await generation.DisposeAsync();
        await write;
        Assert.All(await session.ForwardAsync(1), value => Assert.Equal(5f, value));
    }

    internal static LogicalGraph Graph()
    {
        var builder = new LogicalGraphBuilder(new("test", 1, "accumulator"),
            TestGraphSignatures.Create(8, 1, 1, 1, 1, "test-state"))
            .SetStateSchema(new StateSchema("accumulator-state"));
        builder.AddRegion("graph", GraphRegionTypes.Graph, "graph");
        builder.AddResource("token", "token", GraphResourceKind.Input, GraphResourceLifetime.External,
            new(GraphElementType.Int32, [1]), graphInput: true);
        builder.AddResource("logits", "logits", GraphResourceKind.Output, GraphResourceLifetime.External,
            new(GraphElementType.Float32, [8]), graphOutput: true);
        builder.AddResource("weight", "weight", GraphResourceKind.Weight, GraphResourceLifetime.Model,
            new(GraphElementType.Float32, [8, 1]), "weight");
        builder.AddResource("state", "state", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
            new(GraphElementType.Float32, [1])).AddStateSlot("sum", "state");
        builder.AddResource("value", "value", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
            new(GraphElementType.Float32, [1]));
        builder.AddResource("next", "next", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
            new(GraphElementType.Float32, [1]));
        builder.AddNode("gather", PrimitiveGraphOperations.GatherRow, "graph",
            [GraphBindings.Read("table", "weight"), GraphBindings.Read("index", "token"), GraphBindings.Write("output", "value")]);
        builder.AddNode("add", PrimitiveGraphOperations.Add, "graph",
            [GraphBindings.Read("left", "state"), GraphBindings.Read("right", "value"), GraphBindings.Write("output", "next")], ["gather"]);
        builder.AddNode("copy", PrimitiveGraphOperations.Copy, "graph",
            [GraphBindings.Read("input", "next"), GraphBindings.Write("output", "state")], ["add"]);
        builder.AddNode("broadcast", PortableTensorOperationContracts.Broadcast, "graph",
            [GraphBindings.Read("input", "state"), GraphBindings.Write("output", "logits")], ["copy"]);
        return builder.Build();
    }

    [Fact]
    public async Task SeparateProgramsShareNamedPhysicalStateAndOneGlobalWeightAllocation()
    {
        var prefill = VmGraphOptimizer.Optimize(Graph(), VmTarget.Cpu, new(PrefillCapacity: 8));
        var original = VmGraphOptimizer.Optimize(Graph(), VmTarget.Cpu, new(PrefillCapacity: 1));
        var inference = new VmProgram("inference", original.Abi, original.Target,
            original.Slots.Select(slot => slot.Id == "state" ? slot with { Id = "inference-state" } : slot),
            original.Definitions, original.Entries.Where(entry => entry.Name == "forward").Select(entry =>
                new VmEntry(entry.Name, entry.Definition, entry.Arguments.Select(argument =>
                    argument.Source == "state" ? argument with { Source = "inference-state" } : argument))),
            new(original.State.Schema, original.State.Version, [new("sum", "inference-state")]));
        var initialized = 0;
        await using var engine = await VmInferenceEngine.CreateAsync(
            new(prefill, "token", "logits", () => new TestExecutable(prefill)),
            new(inference, "token", "logits", () => new TestExecutable(inference)),
            (_, storage) => { initialized++; storage.Write(0, new byte[(int)storage.ByteLength]); },
            new(2, 2, 1, 1));
        await using var session = engine.CreateSession();
        Assert.All(await session.PrefillAsync(new int[] { 1, 2, 3, 4, 3 }), value => Assert.Equal(13f, value));
        Assert.All(await session.ForwardAsync(2), value => Assert.Equal(15f, value));
        Assert.Equal(1, initialized);
        Assert.Same(inference, engine.Program);
        Assert.Same(prefill, engine.PrefillProgram);
    }

    [Fact]
    public async Task SparsePrefillVariantsUseOnlyAvailableEntries()
    {
        var original = VmGraphOptimizer.Optimize(Graph(), VmTarget.Cpu, new(PrefillCapacity: 8));
        var program = new VmProgram(original.Name, original.Abi, original.Target, original.Slots,
            original.Definitions, original.Entries.Where(entry => entry.Name is "forward" or "prefill.1" or "prefill.4"),
            original.State);
        await using var engine = await VmInferenceEngine.CreateAsync(program, "token", "logits",
            (_, storage) => storage.Write(0, new byte[(int)storage.ByteLength]), () => new TestExecutable(program), new(2, 2));
        await using var session = engine.CreateSession();
        Assert.All(await session.PrefillAsync(new int[] { 1, 2, 3, 4, 3, 2, 1 }), value => Assert.Equal(16f, value));
    }

    private sealed class TestExecutable(VmProgram program, bool incorrect = false) : IVmExecutable
    {
        public VmProgram Program => program;
        public void Execute(string entryName, byte[][] slots)
        {
            var input = Array.FindIndex(program.Slots.ToArray(), slot => slot.Id == "token");
            var state = Array.FindIndex(program.Slots.ToArray(), slot => slot.Id == program.State.Entries.Single().Slot);
            var output = Array.FindIndex(program.Slots.ToArray(), slot => slot.Id == "logits");
            var count = entryName == "forward" ? 1 : int.Parse(entryName.Split('.')[1]);
            var tokens = MemoryMarshal.Cast<byte, int>(slots[input].AsSpan());
            var values = MemoryMarshal.Cast<byte, float>(slots[state].AsSpan());
            for (var i = 0; i < count; i++) values[0] += tokens[i];
            MemoryMarshal.Cast<byte, float>(slots[output].AsSpan()).Fill(values[0] + (incorrect ? 1 : 0));
        }
        public void Dispose() { }
    }
}
