using SharpInference.WebApi;

namespace SharpInference.Tests;

public sealed class GpuBatchSchedulerTests
{
    [Fact]
    public async Task CompiledQueuesDoNotAddAnotherGenerationWaitingRoom()
    {
        var cpu = new GpuBatchScheduler(new GpuBatchServiceOptions(), Host(1), useVmQueues: true);
        await using var first = await cpu.AcquireAsync(CancellationToken.None);
        await using var second = await cpu.AcquireAsync(CancellationToken.None);
        Assert.Throws<InvalidOperationException>(() => new GpuBatchScheduler(
            new GpuBatchServiceOptions { MaxInFlightGenerationBatches = 1 }, Host(1), useVmQueues: true));
    }

    [Fact]
    public async Task CompiledGpuResidentCapacityRejectsImmediatelyAndReleasesAfterUse()
    {
        var scheduler = new GpuBatchScheduler(
            new GpuBatchServiceOptions { MaxResidentGpuSessions = 1 }, Device(4), useVmQueues: true);
        var first = await scheduler.AcquireAsync(CancellationToken.None);
        await Assert.ThrowsAsync<ResidentSessionLimitException>(() => scheduler.AcquireAsync(CancellationToken.None).AsTask());
        await first.DisposeAsync();
        await using var next = await scheduler.AcquireAsync(CancellationToken.None);
    }

    [Fact]
    public async Task AutoLimitUsesAdvertisedDeviceConcurrency()
    {
        var scheduler = new GpuBatchScheduler(new GpuBatchServiceOptions(), Device(4));
        var leases = new List<IAsyncDisposable>();
        for (var index = 0; index < 4; index++)
            leases.Add(await scheduler.AcquireAsync(CancellationToken.None));
        var fifth = scheduler.AcquireAsync(CancellationToken.None).AsTask();
        Assert.False(fifth.IsCompleted);
        await leases[0].DisposeAsync();
        await using var admitted = await fifth;
        foreach (var lease in leases.Skip(1))
            await lease.DisposeAsync();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(8)]
    public async Task AutoHostLimitUsesAdvertisedConcurrency(int expected)
    {
        var scheduler = new GpuBatchScheduler(new GpuBatchServiceOptions(), Host(expected));
        var leases = new List<IAsyncDisposable>();
        for (var index = 0; index < expected; index++)
            leases.Add(await scheduler.AcquireAsync(CancellationToken.None));
        var next = scheduler.AcquireAsync(CancellationToken.None).AsTask();
        Assert.False(next.IsCompleted);
        await leases[0].DisposeAsync();
        await using var admitted = await next;
        foreach (var lease in leases.Skip(1))
            await lease.DisposeAsync();
    }

    [Fact]
    public async Task ExplicitLimitCanReduceAdvertisedConcurrency()
    {
        var options = new GpuBatchServiceOptions { MaxInFlightGenerationBatches = 2 };
        foreach (var execution in new[] { Host(12), Device(12) })
        {
            var scheduler = new GpuBatchScheduler(options, execution);
            await using var first = await scheduler.AcquireAsync(CancellationToken.None);
            await using var second = await scheduler.AcquireAsync(CancellationToken.None);
            var third = scheduler.AcquireAsync(CancellationToken.None).AsTask();
            Assert.False(third.IsCompleted);
            await second.DisposeAsync();
            await using var admitted = await third;
        }
    }

    [Fact]
    public async Task CpuLimitIsNotClampedByGpuResidentSessions()
    {
        var scheduler = new GpuBatchScheduler(
            new GpuBatchServiceOptions { MaxResidentGpuSessions = 1 }, Host(4));
        await using var first = await scheduler.AcquireAsync(CancellationToken.None);
        await using var second = await scheduler.AcquireAsync(CancellationToken.None);
    }

    [Fact]
    public void RejectsNegativeLimitAndMissingCapabilities()
    {
        Assert.Throws<InvalidOperationException>(() => new GpuBatchScheduler(
            new GpuBatchServiceOptions { MaxInFlightGenerationBatches = -1 }, Host(1)));
        Assert.Throws<ArgumentNullException>(() => new GpuBatchScheduler(new GpuBatchServiceOptions(), null!));
    }

    [Fact]
    public async Task ExplicitLimitCannotExceedSingleSessionCapability()
    {
        var scheduler = new GpuBatchScheduler(
            new GpuBatchServiceOptions { MaxInFlightGenerationBatches = 8 }, Host(1));
        await using var first = await scheduler.AcquireAsync(CancellationToken.None);
        using var cancellation = new CancellationTokenSource();
        var second = scheduler.AcquireAsync(cancellation.Token).AsTask();
        Assert.False(second.IsCompleted);
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => second);
        await first.DisposeAsync();
        await using var next = await scheduler.AcquireAsync(CancellationToken.None);
    }

    [Fact]
    public async Task CompiledDeviceAdmissionCannotExceedSingleSessionCapability()
    {
        var scheduler = new GpuBatchScheduler(new GpuBatchServiceOptions(), Device(1), useVmQueues: true);
        await using var first = await scheduler.AcquireAsync(CancellationToken.None);
        await Assert.ThrowsAsync<ResidentSessionLimitException>(() => scheduler.AcquireAsync(CancellationToken.None).AsTask());
        await first.DisposeAsync();
        await using var next = await scheduler.AcquireAsync(CancellationToken.None);
    }

    [Fact]
    public async Task FairGateAdmitsWaitersInArrivalOrder()
    {
        var gate = new FairAsyncGate(1);
        using var initialLease = await gate.AcquireAsync();
        var firstWaiter = gate.AcquireAsync().AsTask();
        var secondWaiter = gate.AcquireAsync().AsTask();

        Assert.False(firstWaiter.IsCompleted);
        Assert.False(secondWaiter.IsCompleted);

        initialLease.Dispose();
        using var firstLease = await firstWaiter;
        Assert.False(secondWaiter.IsCompleted);

        firstLease.Dispose();
        using var secondLease = await secondWaiter;
    }

    [Fact]
    public async Task SchedulerBoundsResidentSessionsBeforeGenerationBatches()
    {
        var scheduler = new GpuBatchScheduler(new GpuBatchServiceOptions
        {
            MaxResidentGpuSessions = 1,
            MaxInFlightGenerationBatches = 2,
        }, Device(4));

        await using var firstLease = await scheduler.AcquireAsync(CancellationToken.None);
        var secondWaiter = scheduler.AcquireAsync(CancellationToken.None).AsTask();

        Assert.False(secondWaiter.IsCompleted);

        await firstLease.DisposeAsync();
        await using var secondLease = await secondWaiter;
    }

    private static ProcessorExecutionCapabilities Host(int concurrency) => new(concurrency, 1, new HashSet<string> { "opaque.storage" });
    private static ProcessorExecutionCapabilities Device(int concurrency) => new(concurrency, 1, new HashSet<string> { "opaque.storage" },
        requiresResidentSessionAdmission: true);
}
