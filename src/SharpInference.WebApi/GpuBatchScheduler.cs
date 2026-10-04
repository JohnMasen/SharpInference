namespace SharpInference.WebApi;

public sealed class ResidentSessionLimitException : InvalidOperationException
{
    public ResidentSessionLimitException() : base("The maximum resident session capacity was reached.") { }
}

public sealed class GpuBatchServiceOptions
{
    public int MaxResidentGpuSessions { get; init; } = 64;
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

    public GpuBatchScheduler(GpuBatchServiceOptions options, string? backendKind, int? processorCount = null,
        bool useVmQueues = false)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        if (processorCount is <= 0)
            throw new ArgumentOutOfRangeException(nameof(processorCount));
        this.useVmQueues = useVmQueues;
        if (useVmQueues && options.MaxInFlightGenerationBatches != 0)
            throw new InvalidOperationException(
                "Compiled VMs use Rwkv:Runtime:Vm:InferenceInstances; remove GpuBatchService:MaxInFlightGenerationBatches.");

        var cores = processorCount ?? Environment.ProcessorCount;
        var automaticLimit = backendKind?.ToLowerInvariant() switch
        {
            null or "" or "cpu" => Math.Max(1, cores / 2),
            "vortice" or "d3d12" => 4,
            _ => throw new InvalidOperationException($"Unsupported RWKV runtime kind '{backendKind}'."),
        };
        // Temporary backend-based heuristic; replace with measured admission control later.
        generationBatches = useVmQueues ? null : new FairAsyncGate(options.MaxInFlightGenerationBatches == 0
            ? automaticLimit
            : options.MaxInFlightGenerationBatches);
        if (string.Equals(backendKind, "vortice", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(backendKind, "d3d12", StringComparison.OrdinalIgnoreCase))
            residentSessions = new FairAsyncGate(options.MaxResidentGpuSessions);
    }

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

public sealed class FairAsyncGate
{
    private readonly object sync = new();
    private readonly LinkedList<Waiter> waiters = [];
    private int availablePermits;

    public FairAsyncGate(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        availablePermits = capacity;
    }

    public IDisposable? TryAcquire()
    {
        lock (sync)
        {
            if (availablePermits <= 0 || waiters.Count != 0) return null;
            availablePermits--;
            return new Lease(this);
        }
    }

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
