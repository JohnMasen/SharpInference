using System.Threading.Channels;
using System.Runtime.CompilerServices;

namespace SharpInference.Vm;

/// <summary>Configures bounded prefill and inference queues and their worker counts.</summary>
/// <param name="PrefillQueueCapacity">The maximum number of pending prefill requests.</param>
/// <param name="InferenceQueueCapacity">The maximum number of pending inference requests.</param>
/// <param name="PrefillInstances">The number of independently owned prefill workers.</param>
/// <param name="InferenceInstances">The number of independently owned inference workers.</param>
/// <param name="MaximumPrefillTokens">The maximum token count accepted by a single prefill request.</param>
public sealed record VmEngineOptions(int PrefillQueueCapacity, int InferenceQueueCapacity,
    int PrefillInstances = 2, int InferenceInstances = 2, int MaximumPrefillTokens = 65536)
{
    /// <summary>Rejects nonpositive queue capacities, worker counts, and prefill token limits.</summary>
    public void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(PrefillQueueCapacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(InferenceQueueCapacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(PrefillInstances);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(InferenceInstances);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaximumPrefillTokens);
    }
}

/// <summary>Indicates that a bounded VM work queue cannot accept another request.</summary>
/// <param name="queue">The name of the queue that rejected the request.</param>
public sealed class VmQueueFullException(string queue) :
    InvalidOperationException($"The '{queue}' VM queue is full.")
{
    /// <summary>Gets the name of the queue that rejected the request.</summary>
    public string Queue { get; } = queue;
}

/// <summary>Owns separate bounded worker pools for prefill and inference operations.</summary>
/// <typeparam name="TVm">The disposable, independently owned worker type.</typeparam>
public sealed class VmEngine<TVm> : IAsyncDisposable where TVm : class, IDisposable
{
    private readonly VmWorkQueue<TVm> prefill;
    private readonly VmWorkQueue<TVm> inference;

    private VmEngine(VmWorkQueue<TVm> prefill, VmWorkQueue<TVm> inference)
    {
        this.prefill = prefill;
        this.inference = inference;
    }

    /// <summary>Creates owned resources and worker pools, releasing partially created resources on failure.</summary>
    /// <param name="options">The positive capacities, worker counts, and prefill token limit.</param>
    /// <param name="prefillFactory">The factory creating a distinct owned prefill worker for each worker index.</param>
    /// <param name="inferenceFactory">The factory creating a distinct owned inference worker for each worker index.</param>
    public static async ValueTask<VmEngine<TVm>> CreateAsync(VmEngineOptions options,
        Func<int, TVm> prefillFactory, Func<int, TVm> inferenceFactory)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(prefillFactory);
        ArgumentNullException.ThrowIfNull(inferenceFactory);
        options.Validate();
        var owned = new ConditionalWeakTable<TVm, object>();
        var ownershipGate = new object();
        TVm Create(Func<int, TVm> factory, int index)
        {
            var instance = factory(index) ?? throw new InvalidOperationException("A VM factory returned null.");
            lock (ownershipGate)
            {
                if (owned.TryGetValue(instance, out _))
                {
                    throw new InvalidOperationException("VM factories must return distinct, newly owned instances.");
                }

                owned.Add(instance, new object());
            }
            return instance;
        }
        var prefill = new VmWorkQueue<TVm>("prefill", options.PrefillInstances,
            options.PrefillQueueCapacity, index => Create(prefillFactory, index));
        try
        {
            var inference = new VmWorkQueue<TVm>("inference", options.InferenceInstances,
                options.InferenceQueueCapacity, index => Create(inferenceFactory, index));
            return new VmEngine<TVm>(prefill, inference);
        }
        catch (Exception creationError)
        {
            try
            {
                await prefill.DisposeAsync().ConfigureAwait(false);
            }
            catch (Exception cleanupError)
            {
                throw new AggregateException("Engine construction and cleanup failed.", creationError, cleanupError);
            }
            throw;
        }
    }

    /// <summary>Processes prefill work through the dedicated worker pool.</summary>
    /// <param name="operation">The asynchronous callback owning the reserved worker until completion.</param>
    /// <param name="cancellationToken">The token used to cancel queued or active work.</param>
    /// <typeparam name="TResult">The result returned by the queued callback.</typeparam>
    /// <returns>The callback result after execution on an exclusively reserved prefill worker.</returns>
    /// <exception cref="VmQueueFullException">The bounded prefill queue cannot accept the request.</exception>
    public ValueTask<TResult> PrefillAsync<TResult>(Func<TVm, CancellationToken, ValueTask<TResult>> operation,
        CancellationToken cancellationToken = default) => prefill.Submit(operation, cancellationToken);

    // Each queued callback owns its instance until execution and cleanup complete.
    /// <summary>Submits generation work to the inference queue while reserving its worker until the callback completes.</summary>
    /// <param name="generation">The asynchronous generation callback owning a worker until it completes.</param>
    /// <param name="cancellationToken">The token used to cancel queued or active work.</param>
    /// <typeparam name="TResult">The result returned by the queued callback.</typeparam>
    /// <returns>The callback result after execution on an exclusively reserved inference worker.</returns>
    /// <exception cref="VmQueueFullException">The bounded inference queue cannot accept the request.</exception>
    public ValueTask<TResult> InferenceAsync<TResult>(Func<TVm, CancellationToken, ValueTask<TResult>> generation,
        CancellationToken cancellationToken = default) => inference.Submit(generation, cancellationToken);

    /// <summary>Stops accepting work, waits for active operations, and releases owned resources.</summary>
    public async ValueTask DisposeAsync() =>
        await Task.WhenAll(prefill.DisposeAsync().AsTask(), inference.DisposeAsync().AsTask()).ConfigureAwait(false);
}

internal sealed class VmWorkQueue<TVm> : IAsyncDisposable where TVm : class, IDisposable
{
    private readonly string name;
    private readonly object gate = new();
    private readonly Channel<Request> requests;
    private readonly CancellationTokenSource stopping = new();
    private readonly Func<int, TVm> factory;
    private readonly HashSet<TVm> instances = new(ReferenceEqualityComparer.Instance);
    private readonly Task[] workers;
    private Exception? fatalError;
    private Task? shutdown;

    public VmWorkQueue(string name, int count, int capacity, Func<int, TVm> factory)
    {
        this.name = name;
        this.factory = factory;
        requests = Channel.CreateBounded<Request>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = count == 1,
            AllowSynchronousContinuations = false,
        });
        var created = new List<TVm>();
        try
        {
            for (var index = 0; index < count; index++)
            {
                var instance = factory(index) ?? throw new InvalidOperationException("A VM factory returned null.");
                if (!instances.Add(instance))
                {
                    throw new InvalidOperationException("A VM factory returned an already owned instance.");
                }

                created.Add(instance);
            }
        }
        catch (Exception creationError)
        {
            var errors = new List<Exception> { creationError };
            foreach (var instance in created)
            {
                try
                {
                    instance.Dispose();
                }
                catch (Exception error)
                {
                    errors.Add(error);
                }
            }
            stopping.Dispose();
            if (errors.Count > 1)
            {
                throw new AggregateException("VM queue construction and cleanup failed.", errors);
            }

            throw;
        }
        workers = created.Select((instance, index) => Task.Run(() => RunWorker(index, instance))).ToArray();
    }

    public ValueTask<TResult> Submit<TResult>(
        Func<TVm, CancellationToken, ValueTask<TResult>> operation, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(operation);
        cancellationToken.ThrowIfCancellationRequested();
        lock (gate)
        {
            if (fatalError is not null)
            {
                throw new InvalidOperationException($"The '{name}' VM queue failed.", fatalError);
            }

            ObjectDisposedException.ThrowIf(shutdown is not null, this);
            var request = new Request<TResult>(operation, cancellationToken);
            if (!requests.Writer.TryWrite(request))
            {
                request.Abandon();
                throw new VmQueueFullException(name);
            }
            return new ValueTask<TResult>(request.Completion);
        }
    }

    private async Task RunWorker(int index, TVm initial)
    {
        TVm? instance = initial;
        try
        {
            await foreach (var request in requests.Reader.ReadAllAsync().ConfigureAwait(false))
            {
                if (stopping.IsCancellationRequested)
                {
                    request.Reject(fatalError ?? new OperationCanceledException(stopping.Token));
                    continue;
                }
                var healthy = await request.RunAsync(instance!, stopping.Token).ConfigureAwait(false);
                if (!healthy)
                {
                    var failed = instance!;
                    instance = null;
                    lock (gate)
                    {
                        instances.Remove(failed);
                    }

                    failed.Dispose();
                    if (!stopping.IsCancellationRequested)
                    {
                        var replacement = factory(index) ??
                            throw new InvalidOperationException("A VM factory returned null during recovery.");
                        lock (gate)
                        {
                            if (!instances.Add(replacement))
                            {
                                throw new InvalidOperationException("A VM factory reused an active instance during recovery.");
                            }

                            instance = replacement;
                        }
                    }
                }
            }
        }
        catch (Exception error)
        {
            Exception failure = error;
            lock (gate)
            {
                fatalError ??= error;
                requests.Writer.TryComplete();
                try
                {
                    stopping.Cancel();
                }
                catch (Exception cancellationError)
                {
                    failure = new AggregateException("VM worker and cancellation callbacks failed.",
                        error, cancellationError);
                    fatalError = failure;
                }
            }
            while (requests.Reader.TryRead(out var request))
            {
                request.Reject(failure);
            }

            if (!ReferenceEquals(failure, error))
            {
                throw failure;
            }

            throw;
        }
        finally
        {
            if (instance is not null)
            {
                lock (gate)
                {
                    instances.Remove(instance);
                }

                instance.Dispose();
            }
        }
    }

    public ValueTask DisposeAsync()
    {
        lock (gate)
        {
            shutdown ??= ShutdownAsync();
            return new ValueTask(shutdown);
        }
    }

    private async Task ShutdownAsync()
    {
        requests.Writer.TryComplete();
        List<Exception>? errors = null;
        try
        {
            stopping.Cancel();
        }
        catch (Exception error)
        {
            (errors ??= []).Add(error);
        }
        var completion = Task.WhenAll(workers);
        try
        {
            await completion.ConfigureAwait(false);
        }
        catch (Exception error)
        {
            (errors ??= []).Add(completion.Exception ?? error);
        }
        finally
        {
            stopping.Dispose();
        }
        if (errors is not null)
        {
            throw new AggregateException("VM queue shutdown failed.", errors);
        }
    }

    private abstract class Request
    {
        public abstract ValueTask<bool> RunAsync(TVm instance, CancellationToken stopping);
        public abstract void Reject(Exception error);
        public abstract void Abandon();
    }

    private sealed class Request<TResult> : Request
    {
        private readonly Func<TVm, CancellationToken, ValueTask<TResult>> operation;
        private readonly CancellationToken caller;
        private readonly TaskCompletionSource<TResult> completion =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly CancellationTokenRegistration registration;
        private int state;

        public Request(Func<TVm, CancellationToken, ValueTask<TResult>> operation, CancellationToken caller)
        {
            this.operation = operation;
            this.caller = caller;
            registration = caller.Register(() =>
            {
                if (Interlocked.CompareExchange(ref state, 2, 0) == 0)
                {
                    completion.TrySetCanceled(caller);
                }
            });
        }

        public Task<TResult> Completion => completion.Task;

        public override async ValueTask<bool> RunAsync(TVm instance, CancellationToken stopping)
        {
            if (Interlocked.CompareExchange(ref state, 1, 0) != 0)
            {
                registration.Dispose();
                return true;
            }
            var invoked = false;
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(caller, stopping);
            try
            {
                linked.Token.ThrowIfCancellationRequested();
                invoked = true;
                var result = await operation(instance, linked.Token).ConfigureAwait(false);
                if (linked.IsCancellationRequested)
                {
                    completion.TrySetCanceled(linked.Token);
                }
                else
                {
                    completion.TrySetResult(result);
                }

                return true;
            }
            catch (OperationCanceledException) when (linked.IsCancellationRequested)
            {
                completion.TrySetCanceled(linked.Token);
                return !invoked;
            }
            catch (Exception error)
            {
                completion.TrySetException(error);
                return false;
            }
            finally
            {
                Volatile.Write(ref state, 2);
                registration.Dispose();
            }
        }

        public override void Reject(Exception error)
        {
            if (Interlocked.CompareExchange(ref state, 2, 0) == 0)
            {
                if (error is OperationCanceledException cancelled)
                {
                    completion.TrySetCanceled(cancelled.CancellationToken);
                }
                else
                {
                    completion.TrySetException(error);
                }
            }
            registration.Dispose();
        }

        public override void Abandon()
        {
            Volatile.Write(ref state, 2);
            registration.Dispose();
        }
    }
}
