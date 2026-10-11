namespace SharpInference.WebApi;

/// <summary>Signals that no resident GPU session can be admitted at the configured capacity.</summary>
public sealed class ResidentSessionLimitException : InvalidOperationException
{
    /// <summary>Creates an exception reporting that resident GPU session capacity has been reached.</summary>
    public ResidentSessionLimitException() : base("The maximum resident session capacity was reached.") { }
}

/// <summary>Configures resident GPU session admission and non-VM generation concurrency.</summary>
public sealed class GpuBatchServiceOptions
{
    /// <summary>Gets the positive maximum number of admitted resident GPU sessions.</summary>
    public int MaxResidentGpuSessions { get; init; } = 64;
    /// <summary>Gets the non-VM generation concurrency limit; zero uses backend capacity and is required for VM-managed queues.</summary>
    public int MaxInFlightGenerationBatches { get; init; }

    internal void Validate()
    {
        if (MaxResidentGpuSessions <= 0)
        {
            throw new InvalidOperationException("Rwkv:GpuBatchService:MaxResidentGpuSessions must be positive.");
        }

        if (MaxInFlightGenerationBatches < 0)
        {
            throw new InvalidOperationException("Rwkv:GpuBatchService:MaxInFlightGenerationBatches must be non-negative.");
        }
    }
}

/// <summary>Bounds generation work across backends and GPU residency in first-come, first-served order.</summary>
public sealed class GpuBatchScheduler
{
    private readonly FairAsyncGate? residentSessions;
    private readonly FairAsyncGate? generationBatches;
    private readonly bool useVmQueues;

    /// <summary>Creates generation admission gates from configured limits and backend execution capabilities.</summary>
    /// <param name="options">The resident GPU session and generation admission limits.</param>
    /// <param name="execution">The backend&apos;s concurrency and resident-session admission capabilities.</param>
    /// <param name="useVmQueues">Whether VM engine queues own generation concurrency instead of this scheduler.</param>
    public GpuBatchScheduler(GpuBatchServiceOptions options, ProcessorExecutionCapabilities execution,
        bool useVmQueues = false)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(execution);
        options.Validate();
        this.useVmQueues = useVmQueues;
        if (useVmQueues && options.MaxInFlightGenerationBatches != 0)
            throw new InvalidOperationException(
                "Compiled VMs use Rwkv:Runtime:Vm:InferenceInstances; remove GpuBatchService:MaxInFlightGenerationBatches.");

        var generationLimit = options.MaxInFlightGenerationBatches == 0
            ? execution.MaximumConcurrentSessions
            : Math.Min(options.MaxInFlightGenerationBatches, execution.MaximumConcurrentSessions);
        generationBatches = useVmQueues ? null : new FairAsyncGate(generationLimit);
        if (execution.RequiresResidentSessionAdmission)
            residentSessions = new FairAsyncGate(Math.Min(options.MaxResidentGpuSessions,
                execution.MaximumConcurrentSessions));
    }

    /// <summary>Acquires resident-session and generation permits and returns a lease that releases them when disposed.</summary>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    public async ValueTask<IAsyncDisposable> AcquireAsync(CancellationToken cancellationToken)
    {
        var residentLease = residentSessions is null
            ? null
            : useVmQueues
                ? residentSessions.TryAcquire() ?? throw new ResidentSessionLimitException()
                : await residentSessions.AcquireAsync(cancellationToken);
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            var batchLease = generationBatches is null ? null : await generationBatches.AcquireAsync(cancellationToken);
            return new GenerationLease(residentLease, batchLease);
        }
        catch
        {
            residentLease?.Dispose();
            throw;
        }
    }

    private sealed class GenerationLease(IDisposable? residentLease, IDisposable? batchLease) : IAsyncDisposable
    {
        public ValueTask DisposeAsync()
        {
            batchLease?.Dispose();
            residentLease?.Dispose();
            return ValueTask.CompletedTask;
        }
    }
}

/// <summary>Provides disposable concurrency permits with first-come, first-served asynchronous admission.</summary>
public sealed class FairAsyncGate
{
    private readonly object sync = new();
    private readonly LinkedList<Waiter> waiters = [];
    private int availablePermits;

    /// <summary>Creates an admission gate with a positive number of permits.</summary>
    /// <param name="capacity">The positive number of concurrently available admission permits.</param>
    public FairAsyncGate(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        availablePermits = capacity;
    }

    /// <summary>Immediately acquires a permit only when one is available and no earlier waiter is queued.</summary>
    public IDisposable? TryAcquire()
    {
        lock (sync)
        {
            if (availablePermits <= 0 || waiters.Count != 0) return null;
            availablePermits--;
            return new Lease(this);
        }
    }

    /// <summary>Acquires a disposable permit immediately or waits in arrival order until admission or cancellation.</summary>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    public async ValueTask<IDisposable> AcquireAsync(CancellationToken cancellationToken = default)
    {
        Waiter? waiter;
        lock (sync)
        {
            if (availablePermits > 0 && waiters.Count == 0)
            {
                availablePermits--;
                return new Lease(this);
            }

            waiter = new Waiter();
            waiter.Node = waiters.AddLast(waiter);
        }

        try
        {
            await waiter.Completion.Task.WaitAsync(cancellationToken);
            return new Lease(this);
        }
        catch
        {
            lock (sync)
            {
                if (waiter.Node is not null)
                {
                    waiters.Remove(waiter.Node);
                    waiter.Node = null;
                }
            }

            throw;
        }
    }

    private void Release()
    {
        Waiter? waiter = null;
        lock (sync)
        {
            if (waiters.First is { } first)
            {
                waiter = first.Value;
                waiters.Remove(first);
                waiter.Node = null;
            }
            else
            {
                availablePermits++;
            }
        }

        waiter?.Completion.TrySetResult();
    }

    private sealed class Lease(FairAsyncGate owner) : IDisposable
    {
        private FairAsyncGate? owner = owner;

        public void Dispose() => Interlocked.Exchange(ref owner, null)?.Release();
    }

    private sealed class Waiter
    {
        public TaskCompletionSource Completion { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        public LinkedListNode<Waiter>? Node { get; set; }
    }
}
