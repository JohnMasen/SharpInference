using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class VmEngineTests
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);

    [Fact]
    public async Task DefaultsCreateTwoIndependentInstancesForEachQueue()
    {
        var created = new List<Worker>();
        await using var engine = await VmEngine<Worker>.CreateAsync(new(4, 4),
            index => Create("prefill", index), index => Create("inference", index));
        Assert.Equal(2, created.Count(worker => worker.Kind == "prefill"));
        Assert.Equal(2, created.Count(worker => worker.Kind == "inference"));
        var started = Enumerable.Range(0, 4).Select(_ => Signal()).ToArray();
        var release = Signal();
        var tasks = new Task<string>[4];
        for (var i = 0; i < 2; i++)
        {
            var request = i;
            tasks[i] = engine.PrefillAsync(async (worker, token) =>
            {
                started[request].SetResult();
                await release.Task.WaitAsync(token);
                return worker.Kind;
            }).AsTask();
            tasks[i + 2] = engine.InferenceAsync(async (worker, token) =>
            {
                started[request + 2].SetResult();
                await release.Task.WaitAsync(token);
                return worker.Kind;
            }).AsTask();
        }
        try
        {
            await Task.WhenAll(started.Select(signal => signal.Task)).WaitAsync(Timeout);
        }
        finally { release.TrySetResult(); }
        var results = await Task.WhenAll(tasks).WaitAsync(Timeout);
        Assert.Equal(["prefill", "prefill", "inference", "inference"], results);
        Assert.Equal(4, created.Count);

        Worker Create(string kind, int index)
        {
            var worker = new Worker(kind, index);
            created.Add(worker);
            return worker;
        }
    }

    [Fact]
    public async Task FullQueueRejectsImmediatelyAndSingleWorkerPreservesFifo()
    {
        await using var engine = await CreateEngine(capacity: 2);
        var started = Signal();
        var release = Signal();
        var first = engine.PrefillAsync(async (_, token) =>
        {
            started.SetResult();
            await release.Task.WaitAsync(token);
            return 0;
        }).AsTask();
        await started.Task.WaitAsync(Timeout);
        var order = new List<int>();
        var second = engine.PrefillAsync((_, _) => { order.Add(1); return ValueTask.FromResult(1); }).AsTask();
        var third = engine.PrefillAsync((_, _) => { order.Add(2); return ValueTask.FromResult(2); }).AsTask();
        try
        {
            var error = Assert.Throws<VmQueueFullException>(() =>
                engine.PrefillAsync((_, _) => ValueTask.FromResult(3)));
            Assert.Equal("prefill", error.Queue);
        }
        finally { release.TrySetResult(); }
        var results = await Task.WhenAll(first, second, third).WaitAsync(Timeout);
        Assert.Equal([0, 1, 2], results);
        Assert.Equal([1, 2], order);
    }

    [Fact]
    public async Task EntireGenerationHoldsInferenceInstanceAcrossTokenSteps()
    {
        await using var engine = await CreateEngine();
        var token0 = Signal();
        var token1 = Signal();
        var continue0 = Signal();
        var continue1 = Signal();
        var generation = engine.InferenceAsync(async (worker, cancellation) =>
        {
            token0.SetResult();
            await continue0.Task.WaitAsync(cancellation);
            token1.SetResult();
            await continue1.Task.WaitAsync(cancellation);
            return worker;
        }).AsTask();
        await token0.Task.WaitAsync(Timeout);
        var next = engine.InferenceAsync((worker, _) => ValueTask.FromResult(worker)).AsTask();
        try
        {
            Assert.False(next.IsCompleted);
            continue0.SetResult();
            await token1.Task.WaitAsync(Timeout);
            Assert.False(next.IsCompleted);
        }
        finally
        {
            continue0.TrySetResult();
            continue1.TrySetResult();
        }
        Assert.Same(await generation.WaitAsync(Timeout), await next.WaitAsync(Timeout));
    }

    [Fact]
    public async Task CancelledPendingRequestDoesNotInvokeOperation()
    {
        await using var engine = await CreateEngine();
        var started = Signal();
        var release = Signal();
        var running = engine.PrefillAsync(async (_, token) =>
        {
            started.SetResult();
            await release.Task.WaitAsync(token);
            return 0;
        }).AsTask();
        await started.Task.WaitAsync(Timeout);
        using var cancelled = new CancellationTokenSource();
        var invoked = false;
        var pending = engine.PrefillAsync((_, _) =>
        {
            invoked = true;
            return ValueTask.FromResult(1);
        }, cancelled.Token).AsTask();
        cancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending.WaitAsync(Timeout));
        release.SetResult();
        await running.WaitAsync(Timeout);
        await engine.PrefillAsync((_, _) => ValueTask.FromResult(2)).AsTask().WaitAsync(Timeout);
        Assert.False(invoked);
    }

    [Fact]
    public async Task CancellationDoesNotReturnExecutingInstanceBeforeWorkCompletes()
    {
        await using var engine = await CreateEngine();
        using var cancellation = new CancellationTokenSource();
        var started = Signal();
        var sawCancellation = Signal();
        var completedGpuWork = Signal();
        var running = engine.PrefillAsync(async (_, token) =>
        {
            using var registration = token.Register(() => sawCancellation.TrySetResult());
            started.SetResult();
            await completedGpuWork.Task;
            return 0;
        }, cancellation.Token).AsTask();
        await started.Task.WaitAsync(Timeout);
        var pending = engine.PrefillAsync((_, _) => ValueTask.FromResult(1)).AsTask();
        cancellation.Cancel();
        try
        {
            await sawCancellation.Task.WaitAsync(Timeout);
            Assert.False(running.IsCompleted);
            Assert.False(pending.IsCompleted);
        }
        finally { completedGpuWork.TrySetResult(); }
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => running.WaitAsync(Timeout));
        Assert.Equal(1, await pending.WaitAsync(Timeout));
    }

    [Fact]
    public async Task FailedInstanceIsDisposedAndReplacedBeforeNextOperation()
    {
        var created = 0;
        var disposed = 0;
        await using var engine = await VmEngine<Worker>.CreateAsync(new(4, 4, 1, 1),
            _ => new Worker("prefill", Interlocked.Increment(ref created), () => Interlocked.Increment(ref disposed)),
            index => new Worker("inference", index));
        var failed = engine.PrefillAsync<int>((_, _) => throw new IOException("Execution failed")).AsTask();
        await Assert.ThrowsAsync<IOException>(() => failed.WaitAsync(Timeout));
        var next = await engine.PrefillAsync((worker, _) => ValueTask.FromResult(worker.Index)).AsTask().WaitAsync(Timeout);
        Assert.Equal(2, next);
        Assert.Equal(2, created);
        Assert.Equal(1, disposed);
    }

    [Fact]
    public async Task ShutdownCancelsPendingAndWaitsForExecutingWork()
    {
        var engine = await CreateEngine();
        var entered = Signal();
        var cancelled = Signal();
        var finish = Signal();
        var running = engine.InferenceAsync(async (_, token) =>
        {
            using var registration = token.Register(() => cancelled.TrySetResult());
            entered.SetResult();
            await finish.Task;
            return 0;
        }).AsTask();
        await entered.Task.WaitAsync(Timeout);
        var pending = engine.InferenceAsync((_, _) => ValueTask.FromResult(1)).AsTask();
        var shutdown = engine.DisposeAsync().AsTask();
        try
        {
            await cancelled.Task.WaitAsync(Timeout);
            Assert.False(shutdown.IsCompleted);
            Assert.Throws<ObjectDisposedException>(() => engine.InferenceAsync((_, _) => ValueTask.FromResult(2)));
        }
        finally { finish.TrySetResult(); }
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => running.WaitAsync(Timeout));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending.WaitAsync(Timeout));
        await shutdown.WaitAsync(Timeout);
        await engine.DisposeAsync();
    }

    [Fact]
    public async Task FactoryCannotShareOneInstanceBetweenBothQueues()
    {
        var worker = new Worker("shared", 0);
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            await VmEngine<Worker>.CreateAsync(new(4, 4, 1, 1), _ => worker, _ => worker));
        Assert.True(worker.Disposed);
    }

    [Theory]
    [InlineData(0, 1, 2, 2)]
    [InlineData(1, 0, 2, 2)]
    [InlineData(1, 1, 0, 2)]
    [InlineData(1, 1, 2, 0)]
    public void InvalidQueueOrInstanceCountsAreRejected(int prefillCapacity, int inferenceCapacity,
        int prefillCount, int inferenceCount) =>
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new VmEngineOptions(prefillCapacity, inferenceCapacity, prefillCount, inferenceCount).Validate());

    private static TaskCompletionSource Signal() => new(TaskCreationOptions.RunContinuationsAsynchronously);

    private static ValueTask<VmEngine<Worker>> CreateEngine(int capacity = 4) =>
        VmEngine<Worker>.CreateAsync(new(capacity, capacity, 1, 1),
            index => new Worker("prefill", index), index => new Worker("inference", index));

    private sealed class Worker(string kind, int index, Action? disposed = null) : IDisposable
    {
        public string Kind { get; } = kind;
        public int Index { get; } = index;
        public bool Disposed { get; private set; }
        public void Dispose()
        {
            Assert.False(Disposed);
            Disposed = true;
            disposed?.Invoke();
        }
    }
}
