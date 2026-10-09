using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.InteropServices;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Runtime;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class VmQueuedTaskTests
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(15);

    [Fact]
    public async Task GenerationScopesDoNotReserveWorkersAndResultsHaveIndependentStorage()
    {
        var program = CpuProgram();
        var created = 0;
        await using var engine = await VmInferenceEngine.CreateAsync(program, "token", "logits", Initialize,
            () => { Interlocked.Increment(ref created); return new RecordingExecutable(program); }, new(1, 1, 1, 1));
        var sessions = Enumerable.Range(0, 6).Select(_ => engine.CreateSession()).ToArray();
        var scopes = new List<VmGenerationLease>();
        try
        {
            foreach (var session in sessions)
                scopes.Add(await session.BeginGenerationAsync().AsTask().WaitAsync(Timeout));
            Assert.DoesNotContain(typeof(VmGenerationLease).GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
                field => field.FieldType.Name == "Worker");
            var first = await scopes[0].ForwardTokenAsync(2);
            foreach (var scope in scopes) Assert.All(await scope.ForwardTokenAsync(1), value =>
                Assert.Equal(ReferenceEquals(scope, scopes[0]) ? 3f : 1f, value));
            Assert.All(first, value => Assert.Equal(2f, value));
            Assert.Equal(2, created);
        }
        finally
        {
            foreach (var scope in scopes) await scope.DisposeAsync();
            foreach (var session in sessions) await session.DisposeAsync();
        }
    }

    [Fact]
    public async Task CancelledPendingInferenceDoesNotRunOrModifyItsState()
    {
        var program = CpuProgram();
        var started = Signal();
        using var release = new ManualResetEventSlim();
        var calls = 0;
        await using var engine = await VmInferenceEngine.CreateAsync(program, "token", "logits", Initialize,
            () => new RecordingExecutable(program, () =>
            {
                if (Interlocked.Increment(ref calls) == 1)
                {
                    started.TrySetResult();
                    Assert.True(release.Wait(Timeout));
                }
            }), new(2, 2, 1, 1));
        await using var first = engine.CreateSession();
        await using var second = engine.CreateSession();
        var active = first.ForwardAsync(1).AsTask();
        await started.Task.WaitAsync(Timeout);
        using var cancellation = new CancellationTokenSource();
        var pending = second.ForwardAsync(2, cancellation.Token).AsTask();
        cancellation.Cancel();
        try { await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending.WaitAsync(Timeout)); }
        finally { release.Set(); }
        await active.WaitAsync(Timeout);
        Assert.All(await second.ForwardAsync(3), value => Assert.Equal(3f, value));
        Assert.Equal(2, calls);
    }

    [Fact]
    public async Task RunningCancellationWaitsForCompletionInvalidatesStateAndRecoversWorker()
    {
        var program = CpuProgram();
        var started = Signal();
        using var release = new ManualResetEventSlim();
        var calls = 0;
        await using var engine = await VmInferenceEngine.CreateAsync(program, "token", "logits", Initialize,
            () => new RecordingExecutable(program, () =>
            {
                if (Interlocked.Increment(ref calls) == 1)
                {
                    started.TrySetResult();
                    Assert.True(release.Wait(Timeout));
                }
            }), new(2, 2, 1, 1));
        await using var session = engine.CreateSession();
        using var cancellation = new CancellationTokenSource();
        var task = session.ForwardAsync(1, cancellation.Token).AsTask();
        await started.Task.WaitAsync(Timeout);
        cancellation.Cancel();
        try { Assert.False(task.IsCompleted); }
        finally { release.Set(); }
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task.WaitAsync(Timeout));
        Assert.True(task.IsCanceled);
        await Assert.ThrowsAsync<InvalidOperationException>(() => session.ReadStateAsync().AsTask());
        await session.ResetAsync();
        Assert.All(await session.ForwardAsync(2), value => Assert.Equal(2f, value));
    }

    [Fact]
    public async Task PrefillCancellationStopsBeforeTheNextBatch()
    {
        var program = CpuProgram();
        using var cancellation = new CancellationTokenSource();
        var entries = new ConcurrentQueue<string>();
        await using var engine = await VmInferenceEngine.CreateAsync(program, "token", "logits", Initialize,
            () => new RecordingExecutable(program, after: entry =>
            {
                entries.Enqueue(entry);
                cancellation.Cancel();
            }), new(2, 2, 1, 1));
        await using var session = engine.CreateSession();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            session.PrefillAsync(Enumerable.Repeat(1, 9).ToArray(), cancellation.Token).AsTask());
        Assert.Equal(["prefill.8"], entries.ToArray());
        await Assert.ThrowsAsync<InvalidOperationException>(() => session.ReadStateAsync().AsTask());
    }

    [Fact]
    public void GpuTaskSwitchesSessionsWithoutRerecordingAndRejectsInvalidInputsBeforeExecution()
    {
        var program = D3D12VmTests.AccumulatingProgram();
        var artifact = new D3D12VmCompiler(SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create()).Compile(program);
        var (device, _) = D3D12VmDeviceFactory.Create();
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        using (var manager = new VmResourceManager(InitializeHalf, pool.Allocate))
        using (var first = manager.CreateSession(program))
        using (var second = manager.CreateSession(program))
        using (var bindings = manager.CreateBindings(program))
        using (var executor = artifact.CreateExecutor(pool))
        {
            Assert.Equal(1f, Run(first, 0));
            var address = executor.GetGpuVirtualAddress("state");
            var records = executor.ComputeRecordCount;
            var localUploads = pool.TaskStatistics.InitialLocalUploadBytes;
            Assert.Equal(2f, Run(second, 1));
            Assert.Equal(3f, Run(first, 1));
            Assert.Equal(address, executor.GetGpuVirtualAddress("state"));
            Assert.Equal(records, executor.ComputeRecordCount);
            Assert.Equal(1UL, records);
            Assert.Equal(localUploads, pool.TaskStatistics.InitialLocalUploadBytes);
            Assert.Equal(1UL, pool.GlobalUploadCount);
            using (var state = first.CreateStateBindings())
            using (var access = state.BeginStateAccess())
                access.Write("state", 0, BitConverter.GetBytes(10f));
            Assert.Equal(11f, Run(first, 0));
            var before = pool.TaskStatistics.CompletedTasks;
            Assert.Throws<ArgumentOutOfRangeException>(() => Run(second, 10));
            using var cancelled = new CancellationTokenSource();
            cancelled.Cancel();
            Assert.ThrowsAny<OperationCanceledException>(() => Run(second, 0, cancelled.Token));
            Assert.Equal(before, pool.TaskStatistics.CompletedTasks);
            Assert.Equal(4f, Run(second, 1));
            Assert.Equal(records, executor.ComputeRecordCount);
            Assert.Equal(1UL, executor.BindingRecordCount);

            float Run(VmSessionResources session, int token, CancellationToken cancellation = default)
            {
                manager.BindSession(bindings, session);
                try
                {
                    using var access = bindings.BeginExecution();
                    access.GetStorage("token").Write(0, BitConverter.GetBytes(token));
                    executor.ExecuteTask("forward", access, cancellation);
                    var bytes = new byte[4];
                    access.Read("output", 0, bytes);
                    return BitConverter.ToSingle(bytes);
                }
                finally { bindings.UnbindSession(); }
            }
        }
    }

    [Fact]
    public async Task GpuQueuedWorkersPreserveStateAcrossSnapshotsResetAndMultipleSessions()
    {
        var program = VmGraphOptimizer.Optimize(VmInferenceTests.Graph(), VmTarget.Direct3D12, new(PrefillCapacity: 8));
        var artifact = new D3D12VmCompiler(SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create()).Compile(program);
        var (device, _) = D3D12VmDeviceFactory.Create();
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        {
            var created = new ConcurrentBag<D3D12VmExecutor>();
            await using var engine = await VmInferenceEngine.CreateAsync(program, "token", "logits",
                (_, storage) =>
                {
                    var weights = Enumerable.Range(0, 8).Select(value => (float)value).ToArray();
                    storage.Write(0, MemoryMarshal.AsBytes(weights.AsSpan()));
                }, () =>
                {
                    var executor = artifact.CreateExecutor(pool);
                    created.Add(executor);
                    return executor;
                }, new(8, 8), pool.Allocate);
            await using var first = engine.CreateSession();
            await using var second = engine.CreateSession();
            var replies = await Task.WhenAll(first.PrefillAsync(new[] { 1, 2, 3 }).AsTask(),
                second.PrefillAsync(new[] { 2, 2 }).AsTask());
            Assert.All(replies[0], value => Assert.Equal(6f, value));
            Assert.All(replies[1], value => Assert.Equal(4f, value));
            Assert.All(await first.ForwardAsync(1), value => Assert.Equal(7f, value));
            using var snapshot = new MemoryStream();
            await first.ExportStateAsync(snapshot, "queued-model");
            snapshot.Position = 0;
            await second.ImportStateAsync(snapshot, "queued-model", 4);
            Assert.All(await second.ForwardAsync(2), value => Assert.Equal(9f, value));
            await first.ResetAsync();
            Assert.All(await first.ForwardAsync(3), value => Assert.Equal(3f, value));
            Assert.Equal(4, created.Count);
            Assert.Equal(1UL, pool.GlobalUploadCount);
            Assert.All(created, executor => Assert.InRange(executor.BindingRecordCount, 0UL, 1UL));
            Assert.True(pool.TaskStatistics.StateLoadBytes > 0);
            Assert.True(pool.TaskStatistics.StateCommitBytes > 0);
        }
    }

    private static VmProgram CpuProgram() =>
        VmGraphOptimizer.Optimize(VmInferenceTests.Graph(), VmTarget.Cpu, new(PrefillCapacity: 8));
    private static TaskCompletionSource Signal() => new(TaskCreationOptions.RunContinuationsAsynchronously);
    private static void Initialize(VmSlot _, IVmStorage storage) => storage.Write(0, new byte[(int)storage.ByteLength]);
    private static void InitializeHalf(VmSlot _, IVmStorage storage)
    {
        var values = Enumerable.Range(1, 4).Select(value => (Half)value).ToArray();
        storage.Write(0, MemoryMarshal.AsBytes(values.AsSpan()));
    }

    private sealed class RecordingExecutable(VmProgram program, Action? before = null,
        Action<string>? after = null) : IVmTaskExecutable
    {
        public VmProgram Program => program;
        public void ExecuteTask(string entryName, VmExecutionLease lease, CancellationToken cancellation)
        {
            cancellation.ThrowIfCancellationRequested();
            before?.Invoke();
            Execute(entryName, lease.GetBuffers());
            after?.Invoke(entryName);
            cancellation.ThrowIfCancellationRequested();
        }
        public void Execute(string entryName, byte[][] slots)
        {
            var input = Array.FindIndex(program.Slots.ToArray(), slot => slot.Id == "token");
            var state = Array.FindIndex(program.Slots.ToArray(), slot => slot.Id == program.State.Entries.Single().Slot);
            var output = Array.FindIndex(program.Slots.ToArray(), slot => slot.Id == "logits");
            var count = entryName == "forward" ? 1 : int.Parse(entryName.Split('.')[1]);
            var tokens = MemoryMarshal.Cast<byte, int>(slots[input].AsSpan());
            var values = MemoryMarshal.Cast<byte, float>(slots[state].AsSpan());
            for (var index = 0; index < count; index++) values[0] += tokens[index];
            MemoryMarshal.Cast<byte, float>(slots[output].AsSpan()).Fill(values[0]);
        }
        public void Dispose() { }
    }
}
