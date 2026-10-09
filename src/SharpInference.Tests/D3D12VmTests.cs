using System.IO.Compression;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using SharpInference.Graphs;
using SharpInference.Vm;
using Vortice.Direct3D12;
using Xunit.Abstractions;

namespace SharpInference.Tests;

/// <summary>Load the owned backend separately until the parent wires the Tests project reference.</summary>
public sealed class D3D12VmTests(ITestOutputHelper output)
{
    private static readonly Lazy<Assembly> Backend = new(() =>
    {
        var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
        var configuration = new DirectoryInfo(AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar)).Parent!.Name;
        var path = Path.Combine(root, "SharpInference.Backends.D3D12Vm", "bin", configuration,
            "net10.0-windows10.0.19041.0", "SharpInference.Backends.D3D12Vm.dll");
        if (!File.Exists(path))
            throw new FileNotFoundException("Build SharpInference.Backends.D3D12Vm before running these tests.", path);
        return Assembly.LoadFrom(path);
    });

    private static Type Type(string name) => Backend.Value.GetType("SharpInference.Backends.D3D12Vm." + name, true)!;
    private static object Invoke(MethodInfo method, object? target, params object?[] args)
    {
        try { return method.Invoke(target, args)!; }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(e.InnerException).Throw();
            throw;
        }
    }

    [Fact]
    public void LeaseBuffersExecuteMappedForwardAndPrefillWithEntryScopedTokenValidation()
    {
        var graph = new LogicalGraph(new("test", 1, "tokens"), TestGraphSignatures.Create(4, 4, 1, 1, 4, "test.state"),
            [
                new(new("table"), "table", GraphResourceKind.Weight, GraphResourceLifetime.Model,
                        new(GraphElementType.Float16, [4, 4]), "table"),
                    new(new("token"), "token", GraphResourceKind.Input, GraphResourceLifetime.External,
                        new(GraphElementType.Int32, [1])),
                    new(new("output"), "output", GraphResourceKind.Output, GraphResourceLifetime.External,
                        new(GraphElementType.Float32, [4])),
                ],
            [new(new("root"), null, "graph", null, "root", new Dictionary<string, string>())],
            [new(new("gather"), PrimitiveGraphOperations.GatherRow, new("root"),
                    [new("table", new("table"), GraphResourceAccess.Read), new("index", new("token"), GraphResourceAccess.Read),
                        new("output", new("output"), GraphResourceAccess.Write)],
                    [], new Dictionary<string, string>(), new(GraphElementType.Float32, GraphElementType.Float32))],
            [new("token")], [new("output")]);
        var optimizer = Assembly.LoadFrom(Path.Combine(AppContext.BaseDirectory, "SharpInference.Vm.Optimization.dll"))
            .GetType("SharpInference.Vm.Optimization.VmGraphOptimizer", true)!;
        var program = (VmProgram)Invoke(optimizer.GetMethod("Optimize")!, null, graph, VmTarget.Direct3D12, null);
        var artifact = Compile(program);
        var (device, _) = Device();
        using (device)
        using (var executor = (IVmExecutable)Executor(device, artifact))
        using (var manager = new VmResourceManager((_, storage) => storage.Write(0,
            HalfBytes(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))))
        using (var bindings = manager.CreateBindings(program))
        using (var lease = bindings.BeginExecution())
        {
            var context = lease.GetBuffers();
            var secondView = lease.GetBuffers();
            for (var i = 0; i < context.Length; i++) Assert.Same(context[i], secondView[i]);
            var inputIndex = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "token").i;
            var outputIndex = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "output").i;
            var tokens = MemoryMarshal.Cast<byte, int>(context[inputIndex].AsSpan());
            tokens.Fill(-1);
            tokens[0] = 2;
            tokens[1] = 1;
            executor.Execute("forward", context);
            Assert.Equal([9f, 10, 11, 12], Floats(context[outputIndex]));
            executor.Execute("prefill.2", context);
            Assert.Equal([5f, 6, 7, 8], Floats(context[outputIndex]));
            Assert.Throws<ArgumentOutOfRangeException>(() => executor.Execute("prefill.4", context));
            Assert.Equal([5f, 6, 7, 8], Floats(context[outputIndex]));
            Assert.Same(context[outputIndex], lease.GetBuffers()[outputIndex]);
            output.WriteLine("Actual VmExecutionLease.GetBuffers backing arrays: forward and mapped prefill.2 verified; " +
                "unused invalid tokens ignored and active invalid tokens rejected before GPU transfer.");
        }
    }
    [Fact]
    public void ArtifactParameterlessFactoryOwnsHardwareDevice()
    {
        var artifact = Compile(Program("core.copy", [P("input", F(4)), P("output", F(4))]));
        var executor = (IVmExecutable)Invoke(artifact.GetType().GetMethod("CreateExecutor", System.Type.EmptyTypes)!, artifact);
        using (executor)
        {
            Assert.True((bool)executor.GetType().GetProperty("OwnsDevice")!.GetValue(executor)!);
            byte[][] context = [Bytes([1, 2, 3, 4]), new byte[16]];
            executor.Execute("run", context);
            Assert.Equal([1f, 2, 3, 4], Floats(context[1]));
        }
        executor.Dispose();
        Assert.Throws<ObjectDisposedException>(() => executor.Execute("run", [Bytes([1, 2, 3, 4]), new byte[16]]));
    }

    [Fact]
    public async Task ImportedArtifactCreatesTwoPlusTwoIndependentExecutorsWithSharedCpuWeights()
    {
        var program = AccumulatingProgram();
        var artifact = Compile(program);
        using var archive = new MemoryStream();
        Export(artifact, archive);
        archive.Position = 0;
        artifact = Import(archive);
        Assert.Equal(3, ((System.Collections.ICollection)artifact.GetType().GetProperty("Kernels")!.GetValue(artifact)!).Count);
        var (device, _) = Device();
        using (device)
        using (var manager = new VmResourceManager((_, storage) => storage.Write(0, HalfBytes(1, 2, 3, 4))))
        {
            var executors = new List<IVmExecutable>();
            var bindings = new List<VmBindings>();
            var sessions = new List<VmSessionResources>();
            var leases = new List<VmExecutionLease>();
            var contexts = new List<byte[][]>();
            var table = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "table").i;
            var token = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "token").i;
            var state = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "state").i;
            var result = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "output").i;
            try
            {
                for (var i = 0; i < 4; i++)
                {
                    var executor = (IVmExecutable)Invoke(artifact.GetType().GetMethod("CreateExecutor", [typeof(ID3D12Device)])!,
                        artifact, device);
                    executors.Add(executor);
                    Assert.False((bool)executor.GetType().GetProperty("OwnsDevice")!.GetValue(executor)!);
                    Assert.Same(artifact, executor.GetType().GetProperty("Artifact")!.GetValue(executor));
                    sessions.Add(manager.CreateSession(program));
                    bindings.Add(manager.CreateBindings(program));
                    manager.BindSession(bindings[i], sessions[i]);
                    leases.Add(bindings[i].BeginExecution());
                    contexts.Add(leases[i].GetBuffers());
                    BitConverter.GetBytes(i).CopyTo(contexts[i][token], 0);
                    BitConverter.GetBytes(i).CopyTo(contexts[i][token], 4);
                    BitConverter.GetBytes(10f * i).CopyTo(contexts[i][state], 0);
                    if (i != 0)
                    {
                        Assert.Same(contexts[0][table], contexts[i][table]);
                        Assert.NotSame(contexts[0][state], contexts[i][state]);
                    }
                }
                await Task.WhenAll(Enumerable.Range(0, 4).Select(i => Task.Run(() =>
                    Invoke(executors[i].GetType().GetMethod("Execute", [typeof(string), typeof(VmExecutionLease)])!,
                        executors[i], "prefill.2", leases[i]))));
                for (var i = 0; i < 4; i++)
                {
                    var expected = 10f * i + 2 * (i + 1);
                    Assert.Equal([expected], Floats(contexts[i][state]));
                    Assert.Equal([expected], Floats(contexts[i][result]));
                    BitConverter.GetBytes((i + 1) % 4).CopyTo(contexts[i][token], 4);
                }
                await Task.WhenAll(Enumerable.Range(0, 4).Select(i => Task.Run(() =>
                    Invoke(executors[i].GetType().GetMethod("Execute", [typeof(string), typeof(VmExecutionLease)])!,
                        executors[i], "forward.second", leases[i]))));
                for (var i = 0; i < 4; i++)
                {
                    var expected = 10f * i + 2 * (i + 1) + (i + 1) % 4 + 1;
                    Assert.Equal([expected], Floats(contexts[i][state]));
                    Assert.Equal([expected], Floats(contexts[i][result]));
                }
                for (var i = 0; i < 64; i++)
                    BitConverter.GetBytes(i % 4).CopyTo(contexts[0][token], i * sizeof(int));
                var leaseExecute = executors[0].GetType().GetMethod("Execute", [typeof(string), typeof(VmExecutionLease)])!;
                Invoke(leaseExecute, executors[0], "prefill.4", leases[0]);
                Assert.Equal([14f], Floats(contexts[0][state]));
                Invoke(leaseExecute, executors[0], "prefill.64", leases[0]);
                Assert.Equal([174f], Floats(contexts[0][state]));
                Assert.Equal([174f], Floats(contexts[0][result]));
                Assert.Equal([17f], Floats(contexts[1][state]));
                output.WriteLine("One imported three-kernel DXIL artifact -> 2+2 independent concurrent executors; " +
                    "CPU weight backing shared, session state independent, repeated forward calls and entry byteOffset=4 verified. " +
                    "One lease held across prefill.4 + prefill.64 chunks; token byteOffset=252 and accumulated state verified.");
            }
            finally
            {
                foreach (var lease in leases) lease.Dispose();
                foreach (var binding in bindings) binding.Dispose();
                foreach (var session in sessions) session.Dispose();
                foreach (var executor in executors) executor.Dispose();
            }
            using var fence = device.CreateFence();
            device.DeviceRemovedReason.CheckError();
        }
    }

    [Fact]
    public void LeaseOverloadRejectsDifferentProgramBeforeTransfer()
    {
        var artifact = Compile(Program("core.copy", [P("input", F(4)), P("output", F(4))]));
        var wrong = Program("core.exp", [P("input", F(4)), P("output", F(4))]);
        using var manager = new VmResourceManager((_, storage) => storage.Write(0, Bytes([1, 2, 3, 4])));
        using var session = manager.CreateSession(wrong);
        using var bindings = manager.CreateBindings(wrong);
        manager.BindSession(bindings, session);
        using var lease = bindings.BeginExecution();
        var (device, _) = Device();
        using (device)
        {
            var executor = Executor(device, artifact);
            using ((IDisposable)executor)
                Assert.Throws<ArgumentException>(() =>
                    Invoke(executor.GetType().GetMethod("Execute", [typeof(string), typeof(VmExecutionLease)])!,
                        executor, "run", lease));
        }
    }

    [Fact]
    public void ResourceManagerRebindsOneCompiledGpuWorkerWithoutLeakingSessionState()
    {
        var program = AccumulatingProgram();
        var artifact = Compile(program);
        var (device, _) = Device();
        using (device)
        using (var executor = (IVmExecutable)Invoke(artifact.GetType().GetMethod("CreateExecutor", [typeof(ID3D12Device)])!,
            artifact, device))
        using (var manager = new VmResourceManager((_, storage) => storage.Write(0, HalfBytes(1, 2, 3, 4))))
        using (var bindings = manager.CreateBindings(program))
        using (var first = manager.CreateSession(program))
        using (var second = manager.CreateSession(program))
        {
            var tokenIndex = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "token").i;
            var stateIndex = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "state").i;
            var outputIndex = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "output").i;
            var tableIndex = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "table").i;
            byte[]? sharedWeights = null;
            byte[]? workerTokens = null;
            byte[]? firstState = null;
            byte[]? secondState = null;

            Run(first, "prefill.2", [0, 1], 3, isFirst: true);
            Run(second, "forward", [3], 4, isFirst: false);
            Run(first, "forward", [2], 6, isFirst: true);
            Run(second, "forward", [0], 5, isFirst: false);
            Assert.NotSame(firstState, secondState);
            Assert.Equal([6f], Floats(firstState!));
            Assert.Equal([5f], Floats(secondState!));
            output.WriteLine("One compiled GPU worker: idle UnbindSession/BindSession alternated two actual managed sessions; " +
                "shared CPU weights and fixed worker locals preserved, final independent states 6 and 5 verified.");

            void Run(VmSessionResources session, string entry, int[] tokens, float expected, bool isFirst)
            {
                manager.BindSession(bindings, session);
                try
                {
                    using var lease = bindings.BeginExecution();
                    var context = lease.GetBuffers();
                    if (sharedWeights is null) sharedWeights = context[tableIndex];
                    else Assert.Same(sharedWeights, context[tableIndex]);
                    if (workerTokens is null) workerTokens = context[tokenIndex];
                    else Assert.Same(workerTokens, context[tokenIndex]);
                    ref var savedState = ref (isFirst ? ref firstState : ref secondState);
                    if (savedState is null) savedState = context[stateIndex];
                    else Assert.Same(savedState, context[stateIndex]);
                    for (var i = 0; i < tokens.Length; i++)
                        BitConverter.GetBytes(tokens[i]).CopyTo(context[tokenIndex], i * sizeof(int));
                    Assert.Throws<InvalidOperationException>(() => bindings.UnbindSession());
                    executor.Execute(entry, context);
                    Assert.Equal([expected], Floats(context[stateIndex]));
                    Assert.Equal([expected], Floats(context[outputIndex]));
                }
                finally { bindings.UnbindSession(); }
            }
        }
    }

    [Fact]
    public void CpuAuthoritativeStateCopiesHonorMigrationAndInPlaceUpdatesWithoutStaleGpuState()
    {
        var program = AccumulatingProgram();
        var artifact = Compile(program);
        var (device, _) = Device();
        using (device)
        using (var first = (IVmExecutable)Invoke(artifact.GetType().GetMethod("CreateExecutor", [typeof(ID3D12Device)])!,
            artifact, device))
        using (var second = (IVmExecutable)Invoke(artifact.GetType().GetMethod("CreateExecutor", [typeof(ID3D12Device)])!,
            artifact, device))
        using (var manager = new VmResourceManager((_, storage) => storage.Write(0, HalfBytes(1, 2, 3, 4))))
        using (var firstBindings = manager.CreateBindings(program))
        using (var secondBindings = manager.CreateBindings(program))
        using (var session = manager.CreateSession(program))
        {
            var tokenIndex = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "token").i;
            var stateIndex = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "state").i;
            var outputIndex = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "output").i;
            byte[]? backingState = null;
            Run(first, firstBindings, 1, 2);
            Run(second, secondBindings, 2, 5);
            Run(first, firstBindings, 0, 101, cpuState: 100, exerciseExplicitTransfers: true);
            Run(second, secondBindings, 3, 12);
            Assert.Equal([12f], Floats(backingState!));
            output.WriteLine("Explicit CPU-authoritative state copies: the same session migrated between two GPU workers, " +
                "in-place CPU state=100 was observed despite stable array identity, GPU-only state changes required explicit " +
                "readback, and explicit CPU upload restored state=7 before resuming. Final migrated state=12 verified.");

            void Run(IVmExecutable executor, VmBindings bindings, int token, float expected,
                float? cpuState = null, bool exerciseExplicitTransfers = false)
            {
                manager.BindSession(bindings, session);
                try
                {
                    using var lease = bindings.BeginExecution();
                    var context = lease.GetBuffers();
                    if (backingState is null) backingState = context[stateIndex];
                    else Assert.Same(backingState, context[stateIndex]);
                    MemoryMarshal.Cast<byte, int>(context[tokenIndex].AsSpan())[0] = token;
                    if (cpuState is float value)
                        MemoryMarshal.Cast<byte, float>(context[stateIndex].AsSpan())[0] = value;
                    executor.Execute("forward", context);
                    Assert.Equal([expected], Floats(context[stateIndex]));
                    Assert.Equal([expected], Floats(context[outputIndex]));
                    if (!exerciseExplicitTransfers) return;
                    Execute(executor, "forward");
                    Assert.Equal([101f], Floats(context[stateIndex]));
                    var readback = executor.GetType().GetMethod("ReadbackSlots")!;
                    Invoke(readback, executor, context, new int[] { stateIndex, outputIndex });
                    Assert.Equal([102f], Floats(context[stateIndex]));
                    Assert.Equal([102f], Floats(context[outputIndex]));
                    MemoryMarshal.Cast<byte, float>(context[stateIndex].AsSpan())[0] = 7;
                    Invoke(executor.GetType().GetMethod("UploadSlots", [typeof(byte[][]), typeof(IReadOnlyList<int>)])!,
                        executor, context, new int[] { stateIndex });
                    Execute(executor, "forward");
                    Invoke(readback, executor, context, new int[] { stateIndex, outputIndex });
                    Assert.Equal([8f], Floats(context[stateIndex]));
                    Assert.Equal([8f], Floats(context[outputIndex]));
                    Assert.Same(backingState, context[stateIndex]);
                }
                finally { bindings.UnbindSession(); }
            }
        }
    }

    [Theory]
    [InlineData("core.tensor.reduce-last-sum")]
    [InlineData("core.tensor.reduce-last-mean")]
    public void ReduceLastPreservesHeadAndRowCoordinatesForRankThreeTensors(string operation)
    {
        var input = Enumerable.Range(0, 24).Select(i => (float)(i - 10)).ToArray();
        float[] expected = [-34, -18, -2, 14, 30, 46];
        if (operation.EndsWith("mean", StringComparison.Ordinal))
            expected = expected.Select(v => v / 4).ToArray();
        Check(Program(operation, [P("input", F(2, 3, 4)), P("output", F(2, 3))]),
            new() { ["input"] = Bytes(input) }, expected, reimport: true);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void BroadcastAlignsTrailingAxesAndRepeatsLeadingAndSingletonDimensions(int shape)
    {
        int[] inputDimensions;
        float[] input;
        float[] expected;
        if (shape is 0 or 1)
        {
            inputDimensions = shape == 0 ? [4] : [1, 4];
            input = [-2, 0, 3, 5];
            expected = Enumerable.Range(0, 24).Select(i => input[i % 4]).ToArray();
        }
        else
        {
            inputDimensions = shape == 2 ? [3, 1] : [1, 3, 1];
            input = [2, -3, 7];
            expected = Enumerable.Range(0, 24).Select(i => input[i / 4 % 3]).ToArray();
        }
        Check(Program("core.tensor.broadcast", [P("input", F(inputDimensions)), P("output", F(2, 3, 4))]),
            new() { ["input"] = Bytes(input) }, expected, reimport: true);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BatchedMatVecConsumesWeightsDirectlyWithFp32VectorsAndAccumulation(bool half)
    {
        float[] matrix = [2048, 1, 1, 1, 1.5f, -2, 3.25f, 0.5f, -2048, 1, 1, 1, 0.3333f, -2, 3.5f, -4.25f];
        float[] vectors = [1.0001f, 1.25f, -0.75f, 2.125f, 0.9999f, -1.5f, 2.25f, -0.125f];
        var program = Program("core.tensor.batched-mat-vec",
            [P("matrix", new(half ? VmElementType.Float16 : VmElementType.Float32, [2, 2, 4])),
                P("vector", F(2, 4)), P("output", F(2, 2))]);
        var source = Assert.Single(Sources(program)).Value;
        Assert.Contains(half ? "load16(p0, p0o" : "load32(p0, p0o", source, StringComparison.Ordinal);
        Assert.Contains("load32(p1, p1o", source, StringComparison.Ordinal);
        Assert.Contains("float sum=0.0f", source, StringComparison.Ordinal);
        Assert.Equal(3, program.Slots.Count);
        var expected = new float[4];
        for (var row = 0; row < expected.Length; row++)
            for (var column = 0; column < 4; column++)
            {
                var weight = matrix[row * 4 + column];
                if (half) weight = (float)(Half)weight;
                expected[row] += weight * vectors[row / 2 * 4 + column];
            }
        Check(program, new() { ["matrix"] = half ? HalfBytes(matrix) : Bytes(matrix), ["vector"] = Bytes(vectors) },
            expected, reimport: true);
        output.WriteLine($"Direct {(half ? "FP16" : "FP32")} matrix storage, FP32 vectors and accumulation; " +
            "large-plus-small sums and non-half-representable vector values verified without a cast slot or dispatch.");
    }

    [Fact]
    public void BatchedFp16MatrixExtensionStillRejectsUnsupportedTypesShapesAndAttributes()
    {
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => Sources(Program("core.tensor.batched-mat-vec",
            [P("matrix", new(VmElementType.Float16, [2, 2, 4])),
                P("vector", new(VmElementType.Float16, [2, 4])), P("output", F(2, 2))])));
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => Sources(Program("core.tensor.batched-mat-vec",
            [P("matrix", new(VmElementType.Float16, [2, 2, 4])),
                P("vector", F(2, 4)), P("output", new(VmElementType.Float16, [2, 2]))])));
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => Sources(Program("core.tensor.batched-mat-vec",
            [P("matrix", new(VmElementType.Float16, [4, 4])), P("vector", F(2, 4)), P("output", F(2, 2))])));
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => Sources(Program("core.tensor.batched-mat-vec",
            [P("matrix", new(VmElementType.Float16, [2, 2, 4])), P("vector", F(2, 4)), P("output", F(2, 2))],
            new Dictionary<string, string> { ["unsupported"] = "1" })));
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => Sources(Program("core.tensor.broadcast",
            [P("input", new(VmElementType.Float16, [1, 4])), P("output", F(2, 4))])));
    }

    internal static VmProgram AccumulatingProgram()
    {
        VmSlot[] slots =
        [
            new("table", VmSlotScope.Global, VmAccess.ReadOnly, new(VmElementType.Float16, [4, 1])),
            new("token", VmSlotScope.Local, VmAccess.ReadOnly, new(VmElementType.Int32, [64])),
            new("state", VmSlotScope.Session, VmAccess.ReadWrite, F(1)),
            new("temporary", VmSlotScope.Local, VmAccess.ReadWrite, F(1)),
            new("next", VmSlotScope.Local, VmAccess.ReadWrite, F(1)),
            new("output", VmSlotScope.Local, VmAccess.ReadWrite, F(1)),
        ];
        VmParameter[] gatherParameters = [P("table", slots[0].Tensor), P("index", new(VmElementType.Int32, [1])), P("output", F(1))];
        var gather = Kernel("gather", "core.gather-row", gatherParameters);
        var add = Kernel("add", "core.add", [P("left", F(1)), P("right", F(1)), P("output", F(1))]);
        var copy = Kernel("copy", "core.copy", [P("input", F(1)), P("output", F(1))]);
        var forwardParameters = slots.Select(s => new VmParameter(s.Id, s.Access,
            s.Id == "token" ? new VmTensor(VmElementType.Int32, [1]) : s.Tensor)).ToArray();
        var nodes = new List<VmNode>();
        void Dispatch(string id, string kernel, VmArgument[] arguments, string destination)
        {
            nodes.Add(new(id, new VmDispatch(kernel, arguments, new(1)), nodes.Count == 0 ? [] : [nodes[^1].Id]));
            nodes.Add(new(id + ".barrier", new VmBarrier([destination]), [id]));
        }
        Dispatch("gather", "gather", [new("table", "table"), new("index", "token"), new("output", "temporary")], "temporary");
        Dispatch("add", "add", [new("left", "state"), new("right", "temporary"), new("output", "next")], "next");
        Dispatch("state", "copy", [new("input", "next"), new("output", "state")], "state");
        Dispatch("result", "copy", [new("input", "state"), new("output", "output")], "output");
        var forward = new VmDefinition("forward", VmDefinitionKind.Orchestration, forwardParameters, nodes);
        var definitions = new List<VmDefinition> { gather, add, copy, forward };
        var entries = new List<VmEntry>
        {
            new("forward", "forward", slots.Select(s => new VmArgument(s.Id, s.Id))),
            new("forward.second", "forward", slots.Select(s => new VmArgument(s.Id, s.Id, s.Id == "token" ? 4UL : 0UL))),
        };
        for (var count = 1; count <= 64; count *= 2)
        {
            var id = $"prefill.{count}";
            definitions.Add(new(id, VmDefinitionKind.Orchestration,
                slots.Select(s => new VmParameter(s.Id, s.Access, s.Tensor)),
                Enumerable.Range(0, count).Select(i => new VmNode($"token.{i}",
                    new VmCall("forward", slots.Select(s => new VmArgument(s.Id, s.Id,
                        s.Id == "token" ? (ulong)i * 4 : 0))), i == 0 ? [] : [$"token.{i - 1}"]))));
            entries.Add(new(id, id, slots.Select(s => new VmArgument(s.Id, s.Id))));
        }
        return new("accumulator", "acc.v1", VmTarget.Direct3D12, slots, definitions, entries, new("acc.state", 1, [new("state", "state")]));

        static VmDefinition Kernel(string id, string operation, VmParameter[] parameters) =>
            new(id, VmDefinitionKind.Kernel, parameters,
                [new("body", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), operation, 1, parameters.Select(p => new VmArgument(p.Name, p.Name))))], new(64));
    }

    private static object Compile(VmProgram program) =>
        Invoke(Type("D3D12VmCompiler").GetMethod("Compile")!,
            Activator.CreateInstance(Type("D3D12VmCompiler"), [SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create()]), program, null);
    private static IReadOnlyDictionary<string, string> Sources(VmProgram program) =>
        (IReadOnlyDictionary<string, string>)Invoke(Type("D3D12VmCompiler").GetMethod("GenerateSources")!,
            Activator.CreateInstance(Type("D3D12VmCompiler"), [SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create()]), program);
    private static object Import(Stream stream) => Invoke(Type("D3D12VmArtifact").GetMethod("Import")!, null, stream);
    private static void Export(object artifact, Stream stream) => Invoke(artifact.GetType().GetMethod("Export")!, artifact, stream);
    private (ID3D12Device Device, string Name) Device()
    {
        var result = ((ID3D12Device Device, string Name))Invoke(Type("D3D12VmDeviceFactory").GetMethod("Create")!, null, 0);
        output.WriteLine("Real D3D12 hardware: " + result.Name);
        return result;
    }
    private static object Executor(ID3D12Device device, object artifact) =>
        Activator.CreateInstance(Type("D3D12VmExecutor"), device, artifact)!;
    private static void Upload(object executor, string slot, byte[] bytes) =>
        Invoke(executor.GetType().GetMethod("Upload", [typeof(string), typeof(byte[])])!, executor, slot, bytes);
    private static void Execute(object executor, string entry = "run") =>
        Invoke(executor.GetType().GetMethod("Execute", [typeof(string)])!, executor, entry);
    private static byte[] Readback(object executor, string slot) =>
        (byte[])Invoke(executor.GetType().GetMethod("Readback", [typeof(string)])!, executor, slot);
    private static byte[] Bytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static byte[] HalfBytes(params float[] values) =>
        MemoryMarshal.AsBytes(values.Select(v => (Half)v).ToArray().AsSpan()).ToArray();
    private static float[] Floats(byte[] bytes) => MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    private static VmTensor F(params int[] dimensions) => new(VmElementType.Float32, dimensions);
    private static VmParameter P(string name, VmTensor tensor) => new(name,
        name == "output" ? VmAccess.ReadWrite : VmAccess.ReadOnly, tensor);
    private static VmProgram Program(string operation, VmParameter[] parameters,
        IReadOnlyDictionary<string, string>? attributes = null, VmThreadGroup? threads = null, VmThreadGroup? groups = null)
    {
        var args = parameters.Select(p => new VmArgument(p.Name, p.Name)).ToArray();
        var kernel = new VmDefinition("kernel", VmDefinitionKind.Kernel, parameters,
            [new("op", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), operation, 1, args, attributes))], threads ?? new(8, 2));
        var orchestration = new VmDefinition("host", VmDefinitionKind.Orchestration, parameters,
            [new("dispatch", new VmDispatch("kernel", args, groups ?? new(1, 2)))]);
        return new("test", "test.v1", VmTarget.Direct3D12,
            parameters.Select(p => new VmSlot(p.Name,
                p.Name == "output" ? VmSlotScope.Session : VmSlotScope.Global, p.Access, p.Tensor)),
            [kernel, orchestration], [new("run", "host", args)], new("test.state", 1, [new("result", "output")]));
    }
    private void Check(VmProgram program, Dictionary<string, byte[]> inputs, float[] expected, bool reimport = false)
    {
        var artifact = Compile(program);
        if (reimport)
        {
            using var stream = new MemoryStream();
            Export(artifact, stream);
            stream.Position = 0;
            artifact = Import(stream);
        }
        var (device, _) = Device();
        using (device)
        {
            var executor = Executor(device, artifact);
            using ((IDisposable)executor)
            {
                foreach (var (slot, bytes) in inputs) Upload(executor, slot, bytes);
                Execute(executor);
                var actual = Floats(Readback(executor, "output"));
                Assert.Equal(expected.Length, actual.Length);
                for (var i = 0; i < actual.Length; i++)
                    Assert.True(float.IsFinite(actual[i]) && Math.Abs(actual[i] - expected[i]) <= 2e-5f * Math.Max(1, Math.Abs(expected[i])),
                        $"Element {i}: expected {expected[i]:R}, actual {actual[i]:R}.");
                output.WriteLine($"{program.Definitions[0].Nodes[0].Instruction.GetType().Name}: {actual.Length} FP32 values verified.");
            }
        }
    }

    [Theory]
    [InlineData("core.copy")]
    [InlineData("core.exp")]
    [InlineData("core.tanh")]
    [InlineData("core.sigmoid")]
    [InlineData("core.rsqrt")]
    [InlineData("core.square")]
    [InlineData("core.relu")]
    public void UnaryOperatorsOnHardware(string operation)
    {
        float[] input = [0.25f, 0.5f, 1, 2];
        var expected = input.Select(v => operation switch
        {
            "core.copy" => v,
            "core.exp" => MathF.Exp(v),
            "core.tanh" => MathF.Tanh(v),
            "core.sigmoid" => 1 / (1 + MathF.Exp(-v)),
            "core.rsqrt" => 1 / MathF.Sqrt(v),
            "core.square" => v * v,
            "core.relu" => Math.Max(0, v),
            _ => throw new InvalidOperationException(),
        }).ToArray();
        Check(Program(operation, [P("input", F(4)), P("output", F(4))]),
            new() { ["input"] = Bytes(input) }, expected);
    }

    [Theory]
    [InlineData("core.add")]
    [InlineData("core.subtract")]
    [InlineData("core.multiply")]
    [InlineData("core.divide")]
    [InlineData("core.maximum")]
    public void BinaryOperatorsOnHardware(string operation)
    {
        float[] left = [-2, 0, 1, 3], right = [2, -1, 4, 0.5f];
        var expected = left.Zip(right, (a, b) => operation switch
        {
            "core.add" => a + b,
            "core.subtract" => a - b,
            "core.multiply" => a * b,
            "core.divide" => a / b,
            "core.maximum" => Math.Max(a, b),
            _ => throw new InvalidOperationException(),
        }).ToArray();
        Check(Program(operation, [P("left", F(4)), P("right", F(4)), P("output", F(4))]),
            new() { ["left"] = Bytes(left), ["right"] = Bytes(right) }, expected);
    }

    [Theory]
    [InlineData("core.reduce-sum", 10)]
    [InlineData("core.reduce-mean", 2.5f)]
    public void GlobalReductionsOnHardware(string operation, float expected) =>
        Check(Program(operation, [P("input", F(4)), P("output", F(1))]),
            new() { ["input"] = Bytes([1, 2, 3, 4]) }, [expected]);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MatVecAndGatherWithFp32OrFp16Weights(bool half)
    {
        var weight = new VmTensor(half ? VmElementType.Float16 : VmElementType.Float32, [2, 3]);
        var bytes = half ? HalfBytes(1, 2, 3, 4, 5, 6) : Bytes([1, 2, 3, 4, 5, 6]);
        Check(Program("core.mat-vec", [P("weight", weight), P("input", F(3)), P("output", F(2))]),
            new() { ["weight"] = bytes, ["input"] = Bytes([2, -1, 3]) }, [9, 21], reimport: true);
        Check(Program("core.gather-row", [P("table", weight),
                P("index", new(VmElementType.Int32, [1])), P("output", F(3))]),
            new() { ["table"] = bytes, ["index"] = BitConverter.GetBytes(1) }, [4, 5, 6]);
    }

    [Theory]
    [InlineData("core.tensor.fill")]
    [InlineData("core.tensor.cast-f16-f32")]
    [InlineData("core.tensor.reshape")]
    [InlineData("core.tensor.slice")]
    [InlineData("core.tensor.broadcast")]
    [InlineData("core.tensor.batched-mat-vec")]
    [InlineData("core.tensor.reduce-last-sum")]
    [InlineData("core.tensor.reduce-last-mean")]
    [InlineData("core.tensor.head-outer")]
    public void PortableTensorContractsOnHardware(string operation)
    {
        VmParameter[] parameters;
        var inputs = new Dictionary<string, byte[]>();
        Dictionary<string, string>? attributes = null;
        float[] expected;
        switch (operation)
        {
            case "core.tensor.fill":
                parameters = [P("output", F(5))];
                attributes = new() { ["value"] = "-1.25" };
                expected = [-1.25f, -1.25f, -1.25f, -1.25f, -1.25f];
                break;
            case "core.tensor.cast-f16-f32":
                parameters = [P("input", new(VmElementType.Float16, [5])), P("output", F(5))];
                inputs["input"] = HalfBytes(-2, 0, 1, 1.5f, 3);
                expected = [-2, 0, 1, 1.5f, 3];
                break;
            case "core.tensor.reshape":
                parameters = [P("input", F(2, 3)), P("output", F(3, 2))];
                inputs["input"] = Bytes([1, 2, 3, 4, 5, 6]);
                expected = [1, 2, 3, 4, 5, 6];
                break;
            case "core.tensor.slice":
                parameters = [P("input", F(2, 3)), P("output", F(2, 2))];
                attributes = new() { ["axis"] = "1", ["start"] = "1", ["length"] = "2" };
                inputs["input"] = Bytes([1, 2, 3, 4, 5, 6]);
                expected = [2, 3, 5, 6];
                break;
            case "core.tensor.broadcast":
                parameters = [P("input", F(2, 1)), P("output", F(2, 3))];
                inputs["input"] = Bytes([2, 3]);
                expected = [2, 2, 2, 3, 3, 3];
                break;
            case "core.tensor.batched-mat-vec":
                parameters = [P("matrix", F(2, 2, 3)), P("vector", F(2, 3)), P("output", F(2, 2))];
                inputs["matrix"] = Bytes([1, 2, 3, 4, 5, 6, 1, 0, -1, 2, 3, 4]);
                inputs["vector"] = Bytes([2, -1, 3, 1, 2, 3]);
                expected = [9, 21, -2, 20];
                break;
            case "core.tensor.reduce-last-sum":
            case "core.tensor.reduce-last-mean":
                parameters = [P("input", F(2, 3)), P("output", F(2))];
                inputs["input"] = Bytes([1, 2, 3, 4, 5, 6]);
                expected = operation.EndsWith("mean", StringComparison.Ordinal) ? [2, 5] : [6, 15];
                break;
            case "core.tensor.head-outer":
                parameters = [P("left", F(2, 2)), P("right", F(2, 3)), P("output", F(2, 2, 3))];
                inputs["left"] = Bytes([1, 2, 3, 4]);
                inputs["right"] = Bytes([2, 3, 4, 5, 6, 7]);
                expected = [2, 3, 4, 4, 6, 8, 15, 18, 21, 20, 24, 28];
                break;
            default: throw new InvalidOperationException();
        }
        Check(Program(operation, parameters, attributes), inputs, expected, reimport: true);
    }

    private static VmProgram NestedProgram(bool barrier = true)
    {
        VmParameter[] parameters = [P("input", F(4)), P("output", F(4))];
        VmArgument[] args = [new("input", "input"), new("output", "output")];
        var leaf = new VmDefinition("leaf", VmDefinitionKind.Function, parameters,
            [new("square", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), "core.square", 1, args))]);
        var helper = new VmDefinition("helper", VmDefinitionKind.Function, parameters,
            [new("call", new VmCall("leaf", args))]);
        var kernel = new VmDefinition("first", VmDefinitionKind.Kernel, parameters,
            [new("call", new VmCall("helper", args))], new(2, 2));
        var second = new VmDefinition("second", VmDefinitionKind.Kernel, parameters,
            [new("exp", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), "core.exp", 1, args))], new(4));
        VmParameter[] hostParameters = [P("input", F(4)), new("temporary", VmAccess.ReadWrite, F(4)), P("output", F(4))];
        var nodes = new List<VmNode>
        {
            new("first", new VmDispatch("first", [new("input", "input"), new("output", "temporary")], new(1))),
        };
        if (barrier) nodes.Add(new("barrier", new VmBarrier(["temporary"]), ["first"]));
        nodes.Add(new("second", new VmDispatch("second", [new("input", "temporary"), new("output", "output")], new(1)),
            [barrier ? "barrier" : "first"]));
        var host = new VmDefinition("nestedHost", VmDefinitionKind.Orchestration, hostParameters, nodes);
        var outer = new VmDefinition("outer", VmDefinitionKind.Orchestration, hostParameters,
            [new("call", new VmCall("nestedHost", hostParameters.Select(p => new VmArgument(p.Name, p.Name))))]);
        return new("nested", "nested.v1", VmTarget.Direct3D12,
            hostParameters.Select(p => new VmSlot(p.Name, p.Name == "output" ? VmSlotScope.Session : VmSlotScope.Local, p.Access, p.Tensor)),
            [leaf, helper, kernel, second, host, outer],
            [new("run", "outer", hostParameters.Select(p => new VmArgument(p.Name, p.Name)))],
            new("nested.state", 1, [new("out", "output")]));
    }

    [Fact]
    public void NestedFunctionsAndOrchestrationsMultiKernelReimportAndReplay()
    {
        var program = NestedProgram();
        var artifact = Compile(program);
        using var archive = new MemoryStream();
        Export(artifact, archive);
        archive.Position = 0;
        artifact = Import(archive);
        var (device, _) = Device();
        using (device)
        {
            var executor = Executor(device, artifact);
            using ((IDisposable)executor)
            {
                Assert.Equal(2, Invoke(executor.GetType().GetMethod("DispatchCount")!, executor, "run"));
                for (var iteration = 0; iteration < 3; iteration++)
                {
                    float[] input = [0, 0.5f, 1, 1.5f];
                    Upload(executor, "input", Bytes(input));
                    Execute(executor);
                    var actual = Floats(Readback(executor, "output"));
                    for (var i = 0; i < input.Length; i++)
                        Assert.InRange(Math.Abs(actual[i] - MathF.Exp(input[i] * input[i])), 0, 2e-5f);
                }
                output.WriteLine("Imported two DXIL kernels, nested device/host helpers, explicit UAV barrier, three hardware replays verified.");
            }
        }
    }

    [Fact]
    public void RejectMissingBarrierUnsupportedPortsAttributesAndCrossThreadFusion()
    {
        Assert.Throws<InvalidDataException>(() => Sources(NestedProgram(barrier: false)));
        Assert.Throws<NotSupportedException>(() => Sources(Program("core.unknown", [P("input", F(4)), P("output", F(4))])));
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => Sources(Program("core.copy",
            [P("input", F(4)), P("extra", F(4)), P("output", F(4))])));
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => Sources(Program("core.copy", [P("input", F(4)), P("output", F(4))],
            new Dictionary<string, string> { ["unexpected"] = "1" })));
        VmParameter[] parameters = [P("input", F(4)), new("temp", VmAccess.ReadWrite, F(4)), P("output", F(1))];
        var kernel = new VmDefinition("k", VmDefinitionKind.Kernel, parameters,
        [
            new("copy", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), "core.copy", 1, [new("input", "input"), new("output", "temp")])),
            new("reduce", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), "core.reduce-sum", 1, [new("input", "temp"), new("output", "output")]), ["copy"]),
        ], new(4));
        var host = new VmDefinition("host", VmDefinitionKind.Orchestration, parameters,
            [new("dispatch", new VmDispatch("k", parameters.Select(p => new VmArgument(p.Name, p.Name)), new(1)))]);
        var program = new VmProgram("unsafe", "v1", VmTarget.Direct3D12,
            parameters.Select(p => new VmSlot(p.Name, VmSlotScope.Local, p.Access, p.Tensor)),
            [kernel, host], [new("run", "host", parameters.Select(p => new VmArgument(p.Name, p.Name)))],
            new("empty", 1, []));
        Assert.Throws<NotSupportedException>(() => Sources(program));
    }

    [Fact]
    public void ArtifactIntegrityRejectsChangedDxil()
    {
        var artifact = Compile(Program("core.copy", [P("input", F(4)), P("output", F(4))]));
        using var archive = new MemoryStream();
        Export(artifact, archive);
        archive.Position = 0;
        using (var zip = new ZipArchive(archive, ZipArchiveMode.Update, leaveOpen: true))
        {
            var entry = zip.GetEntry("kernels/0.dxil")!;
            entry.Delete();
            using var stream = zip.CreateEntry("kernels/0.dxil").Open();
            stream.Write("DXBCcorrupted"u8);
        }
        archive.Position = 0;
        Assert.Throws<InvalidDataException>(() => Import(archive));
    }

    [Fact]
    public void ByteOffsetsThroughDispatchAndNestedDeviceCallsWithThreeDimensionalThreads()
    {
        VmParameter[] leafParameters = [P("input", new(VmElementType.Float16, [5])), P("output", F(5))];
        VmArgument[] leafArgs = [new("input", "input"), new("output", "output")];
        var leaf = new VmDefinition("cast", VmDefinitionKind.Function, leafParameters,
            [new("cast", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), "core.tensor.cast-f16-f32", 1, leafArgs))]);
        var helper = new VmDefinition("helper", VmDefinitionKind.Function, leafParameters,
            [new("call", new VmCall("cast", leafArgs))]);
        VmParameter[] kernelParameters = [P("input", new(VmElementType.Float16, [6])), P("output", F(7))];
        var kernel = new VmDefinition("kernel", VmDefinitionKind.Kernel, kernelParameters,
            [new("view", new VmCall("helper", [new("input", "input", 2), new("output", "output", 4)]))], new(2, 2, 2));
        VmParameter[] hostParameters = [P("input", new(VmElementType.Float16, [8])), P("output", F(9))];
        var host = new VmDefinition("host", VmDefinitionKind.Orchestration, hostParameters,
            [new("view", new VmDispatch("kernel", [new("input", "input", 2), new("output", "output", 4)], new(1)))]);
        var program = new VmProgram("views", "v1", VmTarget.Direct3D12,
            hostParameters.Select(p => new VmSlot(p.Name, p.Name == "output" ? VmSlotScope.Session : VmSlotScope.Global, p.Access, p.Tensor)),
            [leaf, helper, kernel, host], [new("run", "host", leafArgs)],
            new("views.state", 1, [new("result", "output")]));
        Check(program, new() { ["input"] = HalfBytes(99, 98, -1.5f, 0, 0.5f, 2, 3, 97) },
            [0, 0, -1.5f, 0, 0.5f, 2, 3, 0, 0], reimport: true);
    }

    [Fact]
    public void JumboKernelReusesHelperForSameThreadProducerConsumerAcrossMultipleGroups()
    {
        const int count = 257;
        VmParameter[] helperParameters = [P("input", F(count)), P("output", F(count))];
        var helper = new VmDefinition("square", VmDefinitionKind.Function, helperParameters,
            [new("square", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), "core.square", 1, [new("input", "input"), new("output", "output")]))]);
        VmParameter[] parameters = [P("input", F(count)), new("temporary", VmAccess.ReadWrite, F(count)), P("output", F(count))];
        var kernel = new VmDefinition("jumbo", VmDefinitionKind.Kernel, parameters,
        [
            new("first", new VmCall("square", [new("input", "input"), new("output", "temporary")])),
            new("second", new VmCall("square", [new("input", "temporary"), new("output", "output")]), ["first"]),
        ], new(8, 2));
        var args = parameters.Select(p => new VmArgument(p.Name, p.Name)).ToArray();
        var host = new VmDefinition("host", VmDefinitionKind.Orchestration, parameters,
            [new("dispatch", new VmDispatch("jumbo", args, new(3, 3, 2)))]);
        var program = new VmProgram("jumbo", "v1", VmTarget.Direct3D12,
            parameters.Select(p => new VmSlot(p.Name, p.Name == "output" ? VmSlotScope.Session : VmSlotScope.Local, p.Access, p.Tensor)),
            [helper, kernel, host], [new("run", "host", args)], new("state", 1, [new("result", "output")]));
        var input = Enumerable.Range(0, count).Select(i => (i % 11 - 5) / 4f).ToArray();
        var sources = Sources(program);
        Assert.Single(sources);
        Assert.Equal(1, sources["jumbo"].Split("void d0(", StringSplitOptions.None).Length - 1);
        Check(program, new() { ["input"] = Bytes(input) }, input.Select(v => v * v * v * v).ToArray(), reimport: true);
    }

    [Fact]
    public void ControlPlaneContextPersistsResourcesAndRejectsInvalidInput()
    {
        var program = Program("core.gather-row", [P("table", new(VmElementType.Float16, [2, 3])),
            P("index", new(VmElementType.Int32, [1])), P("output", F(3))]);
        var artifact = Compile(program);
        var (device, _) = Device();
        using (device)
        {
            var executor = Executor(device, artifact);
            using ((IDisposable)executor)
            {
                Assert.Equal([0f, 0, 0], Floats(Readback(executor, "output")));
                Invoke(executor.GetType().GetMethod("UploadSlots", [typeof(byte[][])])!, executor,
                    (object)new byte[]?[] { HalfBytes(1, 2, 3, 4, 5, 6), BitConverter.GetBytes(0), null });
                Execute(executor);
                Assert.Equal([1f, 2, 3], Floats(Readback(executor, "output")));
                Invoke(executor.GetType().GetMethod("UploadSlots", [typeof(byte[][])])!, executor,
                    (object)new byte[]?[] { null, BitConverter.GetBytes(1), null });
                Execute(executor);
                Assert.Equal([4f, 5, 6], Floats(Readback(executor, "output")));
                Assert.Throws<ArgumentOutOfRangeException>(() => Upload(executor, "index", BitConverter.GetBytes(-1)));
                Assert.Throws<ArgumentOutOfRangeException>(() => Upload(executor, "index", BitConverter.GetBytes(2)));
                Assert.Throws<ArgumentException>(() => Upload(executor, "table", [0]));
                Assert.Throws<ArgumentException>(() => Execute(executor, "missing"));
                Execute(executor);
                Assert.Equal([4f, 5, 6], Floats(Readback(executor, "output")));
            }
            Assert.Throws<ObjectDisposedException>(() => Execute(executor));
        }
    }

    [Theory]
    [InlineData("program.xml")]
    [InlineData("kernels/0.hlsl")]
    [InlineData("manifest.json")]
    public void ArtifactIntegrityRejectsChangedProgramSourceOrManifest(string path)
    {
        var artifact = Compile(Program("core.copy", [P("input", F(4)), P("output", F(4))]));
        using var archive = new MemoryStream();
        Export(artifact, archive);
        archive.Position = 0;
        using (var zip = new ZipArchive(archive, ZipArchiveMode.Update, leaveOpen: true))
        {
            var entry = zip.GetEntry(path)!;
            var bytes = new byte[checked((int)entry.Length)];
            using (var stream = entry.Open())
            {
                stream.ReadExactly(bytes);
            }
            // Keep manifest JSON well formed while changing the embedded target.
            if (path == "manifest.json")
                bytes = System.Text.Encoding.UTF8.GetBytes(System.Text.Encoding.UTF8.GetString(bytes).Replace("Direct3D12", "Direct3D11", StringComparison.Ordinal));
            else bytes[bytes.Length / 2] ^= 1;
            entry.Delete();
            using var outputStream = zip.CreateEntry(path).Open();
            outputStream.Write(bytes);
        }
        archive.Position = 0;
        Assert.Throws<InvalidDataException>(() => Import(archive));
    }

    [Fact]
    public void FillPreservesSignedZeroAndExactFp32Value()
    {
        var program = Program("core.tensor.fill", [P("output", F(3))],
            new Dictionary<string, string> { ["value"] = "-0" });
        var artifact = Compile(program);
        var (device, _) = Device();
        using (device)
        {
            var executor = Executor(device, artifact);
            using ((IDisposable)executor)
            {
                Execute(executor);
                Assert.All(Floats(Readback(executor, "output")),
                    v => Assert.Equal(0x80000000u, BitConverter.SingleToUInt32Bits(v)));
            }
        }
    }

    [Fact]
    public void ManagedExecutableUsesExistingBuffersAndSwitchesSessionState()
    {
        var original = Program("core.gather-row", [P("table", new(VmElementType.Float16, [2, 3])),
                P("index", new(VmElementType.Int32, [1])), P("output", F(3))]);
        var program = new VmProgram(original.Name, original.Abi, original.Target,
            original.Slots.Select(s => s.Id == "index" ? s with { Scope = VmSlotScope.Local } : s),
            original.Definitions, original.Entries, original.State);
        var artifact = Compile(program);
        var (device, _) = Device();
        using (device)
        using (var executable = (IVmExecutable)Executor(device, artifact))
        {
            Assert.Same(program, executable.Program);
            var weights = HalfBytes(1, 2, 3, 4, 5, 6);
            byte[][] first = [weights, BitConverter.GetBytes(0), new byte[12]];
            byte[][] second = [weights, BitConverter.GetBytes(1), new byte[12]];
            var firstOutput = first[2];
            var secondOutput = second[2];
            executable.Execute("run", first);
            Assert.Equal([1f, 2, 3], Floats(first[2]));
            executable.Execute("run", second);
            Assert.Equal([4f, 5, 6], Floats(second[2]));
            Assert.Same(firstOutput, first[2]);
            Assert.Same(secondOutput, second[2]);
            Assert.Equal([1f, 2, 3], Floats(first[2]));
            executable.Execute("run", first);
            Assert.Equal([1f, 2, 3], Floats(first[2]));
            byte[][] replacement = [(byte[])weights.Clone(), BitConverter.GetBytes(0), new byte[12]];
            Assert.Throws<ArgumentException>(() => executable.Execute("run", replacement));
            Assert.Throws<ArgumentException>(() => executable.Execute("run", [weights]));
            first[1] = BitConverter.GetBytes(-1);
            Assert.Throws<ArgumentOutOfRangeException>(() => executable.Execute("run", first));
            output.WriteLine("IVmExecutable bridge reused existing output arrays and immutable GPU weights across distinct session contexts.");
        }
    }

    [Fact]
    public void SelectiveTransfersReadIntoCallerBackingArrays()
    {
        var program = Program("core.copy", [P("input", F(4)), P("output", F(4))]);
        var artifact = Compile(program);
        var (device, _) = Device();
        using (device)
        {
            var executor = Executor(device, artifact);
            using ((IDisposable)executor)
            {
                byte[][] context = [Bytes([1, 2, 3, 4]), new byte[16]];
                var destination = context[1];
                Invoke(executor.GetType().GetMethod("UploadSlots", [typeof(byte[][]), typeof(IReadOnlyList<int>)])!,
                    executor, context, new int[] { 0 });
                Execute(executor);
                Invoke(executor.GetType().GetMethod("ReadbackSlots")!, executor, context, new int[] { 1 });
                Assert.Same(destination, context[1]);
                Assert.Equal([1f, 2, 3, 4], Floats(destination));
                Array.Clear(destination);
                Invoke(executor.GetType().GetMethod("Readback", [typeof(string), typeof(byte[])])!,
                    executor, "output", destination);
                Assert.Equal([1f, 2, 3, 4], Floats(destination));
                Assert.Throws<ArgumentException>(() =>
                    Invoke(executor.GetType().GetMethod("Readback", [typeof(string), typeof(byte[])])!,
                        executor, "output", new byte[4]));
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MatVecMatrixPortSupportsFp32AndFp16(bool half)
    {
        Check(Program("core.mat-vec", [P("matrix", new(half ? VmElementType.Float16 : VmElementType.Float32, [2, 3])),
                    P("input", F(3)), P("output", F(2))], threads: new(64), groups: new(1)),
            new()
            {
                ["matrix"] = half ? HalfBytes(1, 2, 3, 4, 5, 6) : Bytes([1, 2, 3, 4, 5, 6]),
                ["input"] = Bytes([2, -1, 3])
            }, [9, 21]);
    }

    [Fact]
    public void ParentMapperDeduplicatesDefinitionsAndExplicitlyReusesLocalSlotsOnHardware()
    {
        var resources = new[]
        {
                Resource("input", SharpInference.Graphs.GraphResourceKind.Input),
                Resource("a", SharpInference.Graphs.GraphResourceKind.TokenTransient),
                Resource("b", SharpInference.Graphs.GraphResourceKind.TokenTransient),
                Resource("c", SharpInference.Graphs.GraphResourceKind.TokenTransient),
                Resource("output", SharpInference.Graphs.GraphResourceKind.Output),
            };
        var graph = new SharpInference.Graphs.LogicalGraph(new("test", 1, "mapped"),
            TestGraphSignatures.Create(4, 4, 1, 1, 4, "test.state"), resources,
            [new(new("root"), null, "graph", null, "root", new Dictionary<string, string>())],
            [Node("copy1", "core.copy", "input", "a"),
                    Node("square", "core.square", "a", "b", "copy1"),
                    Node("copy2", "core.copy", "b", "c", "square"),
                    Node("exp", "core.exp", "c", "output", "copy2")],
            [new("input")], [new("output")]);
        var optimizer = Assembly.LoadFrom(Path.Combine(AppContext.BaseDirectory, "SharpInference.Vm.Optimization.dll"))
            .GetType("SharpInference.Vm.Optimization.VmGraphOptimizer", true)!;
        var program = (VmProgram)Invoke(optimizer.GetMethod("Optimize")!, null, graph, VmTarget.Direct3D12, null);
        Assert.Equal(3, program.Definitions.Count(d => d.Kind == VmDefinitionKind.Kernel));
        Assert.True(program.Slots.Count < resources.Length, "Parent allocator must actually reuse physical scratch.");
        var artifact = Compile(program);
        var (device, _) = Device();
        using (device)
        using (var executor = (IVmExecutable)Executor(device, artifact))
        {
            var context = program.Slots.Select(s => new byte[checked((int)s.Tensor.ByteLength)]).ToArray();
            var inputIndex = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "input").i;
            var outputIndex = program.Slots.Select((s, i) => (s, i)).Single(p => p.s.Id == "output").i;
            Bytes([0, 0.5f, 1, 1.5f]).CopyTo(context[inputIndex], 0);
            executor.Execute("forward", context);
            var actual = Floats(context[outputIndex]);
            float[] expected = [1, MathF.Exp(0.25f), MathF.Exp(1), MathF.Exp(2.25f)];
            for (var i = 0; i < expected.Length; i++)
                Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 2e-5f);
            output.WriteLine($"Parent mapper: {resources.Length} logical resources -> {program.Slots.Count} physical slots; " +
                "4 explicit dispatches, 3 deduplicated kernels, exact original ports, numerical hardware result verified.");
        }

        static SharpInference.Graphs.GraphResource Resource(string name, SharpInference.Graphs.GraphResourceKind kind) =>
            new(new(name), name, kind, kind is SharpInference.Graphs.GraphResourceKind.Input or SharpInference.Graphs.GraphResourceKind.Output
                ? SharpInference.Graphs.GraphResourceLifetime.External : SharpInference.Graphs.GraphResourceLifetime.Token,
                new(SharpInference.Graphs.GraphElementType.Float32, [4]));
        static SharpInference.Graphs.LogicalNode Node(string id, string op, string input, string result, string? dependency = null) =>
            new(new(id), new(op), new("root"),
                [new("input", new(input), SharpInference.Graphs.GraphResourceAccess.Read),
                        new("output", new(result), SharpInference.Graphs.GraphResourceAccess.Write)],
                dependency is null ? [] : [new(dependency)], new Dictionary<string, string>(),
                new(GraphElementType.Float32, GraphElementType.Float32));
    }
}
