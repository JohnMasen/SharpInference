using SharpInference.WebApi;

namespace SharpInference.Tests;

public sealed class GpuBatchSchedulerTests
{
    [Fact]
    public async Task AutoGpuLimitAdmitsFourConcurrentGenerations()
    {
        var scheduler = new GpuBatchScheduler(new GpuBatchServiceOptions(), "vortice");
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
    [InlineData(1, 1)]
    [InlineData(3, 1)]
    [InlineData(8, 4)]
    public async Task AutoCpuLimitUsesHalfAvailableCores(int cores, int expected)
    {
        var scheduler = new GpuBatchScheduler(new GpuBatchServiceOptions(), "cpu", cores);
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
    public async Task ExplicitLimitOverridesBackendDefaults()
    {
        var options = new GpuBatchServiceOptions { MaxInFlightGenerationBatches = 2 };
        foreach (var kind in new[] { "cpu", "vortice" })
        {
            var scheduler = new GpuBatchScheduler(options, kind, 12);
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
            new GpuBatchServiceOptions { MaxResidentGpuSessions = 1 }, "cpu", 4);
        await using var first = await scheduler.AcquireAsync(CancellationToken.None);
        await using var second = await scheduler.AcquireAsync(CancellationToken.None);
    }

    [Fact]
    public void RejectsNegativeLimitAndUnknownBackend()
    {
        Assert.Throws<InvalidOperationException>(() => new GpuBatchScheduler(
            new GpuBatchServiceOptions { MaxInFlightGenerationBatches = -1 }, "cpu"));
        Assert.Throws<InvalidOperationException>(() =>
            new GpuBatchScheduler(new GpuBatchServiceOptions(), "unknown"));
        Assert.Throws<InvalidOperationException>(() =>
            new GpuBatchScheduler(new GpuBatchServiceOptions(), "portable-vortice"));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GpuBatchScheduler(new GpuBatchServiceOptions(), "cpu", 0));
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
        }, "vortice");

        await using var firstLease = await scheduler.AcquireAsync(CancellationToken.None);
        var secondWaiter = scheduler.AcquireAsync(CancellationToken.None).AsTask();

        Assert.False(secondWaiter.IsCompleted);

        await firstLease.DisposeAsync();
        await using var secondLease = await secondWaiter;
    }
}
