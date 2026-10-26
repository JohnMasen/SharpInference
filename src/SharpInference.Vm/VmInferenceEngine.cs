using System.Runtime.InteropServices;

namespace SharpInference.Vm;

/// <summary>Executes named VM entry points against program-ordered managed slot buffers.</summary>
public interface IVmExecutable : IDisposable
{
    /// <summary>Gets the program whose entry points and slot order this executable implements.</summary>
    VmProgram Program { get; }
    /// <summary>Executes a named entry point against managed buffers in program-slot order.</summary>
    /// <param name="entryName">The name of the program entry point to execute.</param>
    /// <param name="slots">The resource buffers in program-slot order.</param>
    void Execute(string entryName, byte[][] slots);
}

/// <summary>Combines an inference VM program, its token and logits slots, and an executable factory.</summary>
/// <param name="Program">The program executed by the factory-produced executable.</param>
/// <param name="InputSlot">The Int32 slot receiving input token identifiers.</param>
/// <param name="OutputSlot">The Float32 slot containing output logits.</param>
/// <param name="CreateExecutable">The factory returning a newly owned executable for the program.</param>
public sealed record VmInferenceProgram(VmProgram Program, string InputSlot, string OutputSlot,
    Func<IVmExecutable> CreateExecutable);

/// <summary>Coordinates shared model resources, worker pools, and stateful inference sessions.</summary>
public sealed class VmInferenceEngine : IAsyncDisposable
{
    private readonly VmResourceManager manager;
    private readonly VmEngine<Worker> queue;
    private readonly HashSet<VmInferenceSession> sessions = [];
    private readonly object gate = new();
    private Task? shutdown;

    private VmInferenceEngine(VmInferenceProgram prefill, VmInferenceProgram inference,
        VmResourceManager manager, VmEngine<Worker> queue, VmEngineOptions options)
    {
        Program = inference.Program;
        PrefillProgram = prefill.Program;
        this.manager = manager;
        this.queue = queue;
        MaximumPrefillTokens = options.MaximumPrefillTokens;
        VocabularySize = checked((int)inference.Program.Slots.Single(slot => slot.Id == inference.OutputSlot).Tensor.ElementCount);
    }

    /// <summary>Gets the program used for single-token inference.</summary>
    public VmProgram Program { get; }
    /// <summary>Gets the program used to process prefill token batches.</summary>
    public VmProgram PrefillProgram { get; }
    /// <summary>Gets the number of logits produced per inference step.</summary>
    public int VocabularySize { get; }
    /// <summary>Gets the maximum token count accepted by a single prefill request.</summary>
    public int MaximumPrefillTokens { get; }

    /// <summary>Creates owned resources and worker pools, releasing partially created resources on failure.</summary>
    /// <param name="program">The program used for both prefill and single-token inference.</param>
    /// <param name="inputSlot">The Int32 slot receiving input token identifiers.</param>
    /// <param name="outputSlot">The Float32 slot containing output logits.</param>
    /// <param name="initializeGlobal">The callback initializing newly allocated global slot storage.</param>
    /// <param name="factory">The factory returning a distinct newly owned executable for each worker.</param>
    /// <param name="options">The prefill and inference worker counts, queue capacities, and token limit.</param>
    /// <param name="allocate">The optional slot storage allocator; null uses managed byte arrays.</param>
    public static ValueTask<VmInferenceEngine> CreateAsync(VmProgram program, string inputSlot,
        string outputSlot, Action<VmSlot, IVmStorage> initializeGlobal, Func<IVmExecutable> factory,
        VmEngineOptions options, Func<VmSlot, IVmStorage>? allocate = null) => CreateAsync(new(program, inputSlot, outputSlot, factory),
            new(program, inputSlot, outputSlot, factory), initializeGlobal, options, allocate);

    /// <summary>Creates owned resources and worker pools, releasing partially created resources on failure.</summary>
    /// <param name="prefill">The prefill program, token/logits slot mapping, and executable factory.</param>
    /// <param name="inference">The inference program, token/logits slot mapping, and executable factory.</param>
    /// <param name="initializeGlobal">The callback initializing newly allocated global slot storage.</param>
    /// <param name="options">The prefill and inference worker counts, queue capacities, and token limit.</param>
    /// <param name="allocate">The optional slot storage allocator; null uses managed byte arrays.</param>
    public static async ValueTask<VmInferenceEngine> CreateAsync(VmInferenceProgram prefill,
        VmInferenceProgram inference, Action<VmSlot, IVmStorage> initializeGlobal, VmEngineOptions options,
        Func<VmSlot, IVmStorage>? allocate = null)
    {
        ArgumentNullException.ThrowIfNull(prefill);
        ArgumentNullException.ThrowIfNull(inference);
        options.Validate();
        Validate(prefill, isPrefill: true);
        Validate(inference, isPrefill: false);
        if (prefill.Program.Abi != inference.Program.Abi ||
            prefill.Program.Slots.Single(slot => slot.Id == prefill.OutputSlot).Tensor.ElementCount !=
            inference.Program.Slots.Single(slot => slot.Id == inference.OutputSlot).Tensor.ElementCount)
        {
            throw new InvalidDataException("Prefill and inference programs require matching model/output ABIs.");
        }

        var manager = new VmResourceManager(initializeGlobal, allocate);
        VmEngine<Worker>? queue = null;
        try
        {
            Worker Create(VmInferenceProgram component)
            {
                var bindings = manager.CreateBindings(component.Program);
                IVmExecutable? executable = null;
                try
                {
                    executable = component.CreateExecutable() ??
                        throw new InvalidOperationException("A VM executable factory returned null.");
                    if (VmProgramXml.Serialize(executable.Program) != VmProgramXml.Serialize(component.Program))
                    {
                        throw new InvalidDataException("The executable and resource program do not match.");
                    }

                    if (component.Program.Target == VmTarget.Direct3D12 && executable is not IVmTaskExecutable)
                    {
                        throw new NotSupportedException("GPU inference requires queued-task execution support.");
                    }

                    return new Worker(bindings, executable, component.InputSlot, component.OutputSlot);
                }
                catch (Exception error)
                {
                    var errors = new List<Exception> { error };
                    try { bindings.Dispose(); }
                    catch (Exception cleanup) { errors.Add(cleanup); }
                    try { executable?.Dispose(); }
                    catch (Exception cleanup) { errors.Add(cleanup); }
                    if (errors.Count > 1)
                    {
                        throw new AggregateException("VM worker construction failed.", errors);
                    }

                    throw;
                }
            }
            VmSessionResources.ValidateContracts([prefill.Program, inference.Program]);
            queue = await VmEngine<Worker>.CreateAsync(options, _ => Create(prefill), _ => Create(inference)).ConfigureAwait(false);
            return new VmInferenceEngine(prefill, inference, manager, queue, options);
        }
        catch (Exception error)
        {
            var errors = new List<Exception> { error };
            if (queue is not null)
            {
                try { await queue.DisposeAsync().ConfigureAwait(false); }
                catch (Exception cleanup) { errors.Add(cleanup); }
            }
            try { manager.Dispose(); }
            catch (Exception cleanup) { errors.Add(cleanup); }
            if (errors.Count > 1)
            {
                throw new AggregateException("Inference engine construction failed.", errors);
            }

            throw;
        }

        static void Validate(VmInferenceProgram component, bool isPrefill)
        {
            ArgumentNullException.ThrowIfNull(component.CreateExecutable);
            var input = component.Program.Slots.SingleOrDefault(slot => slot.Id == component.InputSlot);
            var output = component.Program.Slots.SingleOrDefault(slot => slot.Id == component.OutputSlot);
            if (input?.Tensor.ElementType != VmElementType.Int32 ||
                output?.Tensor.ElementType != VmElementType.Float32)
            {
                throw new ArgumentException("Inference requires an Int32 token slot and Float32 logits slot.");
            }

            var entry = isPrefill ? "prefill.1" : "forward";
            if (!component.Program.Entries.Any(candidate => candidate.Name == entry))
            {
                throw new InvalidDataException($"The VM program has no required '{entry}' entry.");
            }
        }
    }

    /// <summary>Allocates an independent session with state storage compatible with the engine's programs.</summary>
    public VmInferenceSession CreateSession()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(shutdown is not null, this);
            var session = new VmInferenceSession(this, manager.CreateSession(PrefillProgram, Program));
            sessions.Add(session);
            return session;
        }
    }

    internal void Remove(VmInferenceSession session)
    {
        lock (gate)
        {
            sessions.Remove(session);
        }
    }

    internal async ValueTask<float[]> PrefillAsync(VmSessionResources state, int[] tokens, CancellationTokenSource cancellation)
    {
        var entries = PrefillProgram.Entries.Where(entry => entry.Name.StartsWith("prefill.", StringComparison.Ordinal))
            .Select(entry => (entry.Name, Tokens: int.Parse(entry.Name.AsSpan("prefill.".Length),
                System.Globalization.CultureInfo.InvariantCulture))).OrderByDescending(entry => entry.Tokens).ToArray();
        var result = Array.Empty<float>();
        var advanced = false;
        try
        {
            for (var offset = 0; offset < tokens.Length;)
            {
                cancellation.Token.ThrowIfCancellationRequested();
                var entry = entries.FirstOrDefault(entry => entry.Tokens <= tokens.Length - offset);
                if (entry.Name is null)
                {
                    throw new InvalidDataException("The prefill program cannot execute this token count.");
                }

                var request = new InferenceRequest(state, entry.Name, tokens.AsSpan(offset, entry.Tokens).ToArray(), cancellation);
                result = await queue.PrefillAsync((worker, token) => ExecuteRequest(worker, request, token),
                    cancellation.Token).ConfigureAwait(false);
                advanced = true;
                offset += entry.Tokens;
            }
            return result;
        }
        catch
        {
            if (advanced)
            {
                using var bindings = state.CreateStateBindings();
                using var access = bindings.BeginStateAccess();
                access.InvalidateState();
            }
            throw;
        }
    }

    internal ValueTask<float[]> ForwardAsync(VmSessionResources state, int token, CancellationTokenSource cancellation)
    {
        ValidateToken(token);
        cancellation.Token.ThrowIfCancellationRequested();
        var request = new InferenceRequest(state, "forward", [token], cancellation);
        return queue.InferenceAsync((worker, stopped) => ExecuteRequest(worker, request, stopped), cancellation.Token);
    }

    private sealed record InferenceRequest(VmSessionResources State, string Entry, int[] Tokens,
        CancellationTokenSource Cancellation);

    private ValueTask<float[]> ExecuteRequest(Worker worker, InferenceRequest request, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        request.Cancellation.Token.ThrowIfCancellationRequested();
        lock (worker.Gate)
        {
            manager.BindSession(worker.Bindings, request.State);
            try { return ValueTask.FromResult(Run(worker, request.Entry, request.Tokens, cancellation)); }
            finally { worker.Bindings.UnbindSession(); }
        }
    }

    internal void ValidateToken(int token)
    {
        if (token < 0 || token >= VocabularySize)
        {
            throw new ArgumentOutOfRangeException(nameof(token));
        }
    }

    private float[] Run(Worker worker, string entry, ReadOnlySpan<int> tokens, CancellationToken cancellation)
    {
        using var access = worker.Bindings.BeginExecution();
        try
        {
            cancellation.ThrowIfCancellationRequested();
            var input = worker.Bindings.Program.Slots[worker.InputIndex].Id;
            var output = worker.Bindings.Program.Slots[worker.OutputIndex];
            access.GetStorage(input).Write(0, MemoryMarshal.AsBytes(tokens));
            if (worker.Executable is IVmTaskExecutable tasks)
            {
                tasks.ExecuteTask(entry, access, cancellation);
            }
            else
            {
                worker.Executable.Execute(entry, access.GetBuffers());
            }

            cancellation.ThrowIfCancellationRequested();
            var result = new float[checked((int)output.Tensor.ElementCount)];
            access.Read(output.Id, 0, MemoryMarshal.AsBytes(result.AsSpan()));
            return result;
        }
        catch
        {
            access.InvalidateState();
            throw;
        }
    }

    /// <summary>Stops accepting work, waits for active operations, and releases owned resources.</summary>
    public ValueTask DisposeAsync()
    {
        lock (gate)
        {
            shutdown ??= StopAsync();
            return new ValueTask(shutdown);
        }
    }

    private async Task StopAsync()
    {
        await Task.Yield();
        VmInferenceSession[] snapshot;
        lock (gate)
        {
            snapshot = sessions.ToArray();
        }

        foreach (var session in snapshot)
        {
            session.Stop();
        }

        List<Exception>? errors = null;
        try { await queue.DisposeAsync().ConfigureAwait(false); }
        catch (Exception error) { (errors ??= []).Add(error); }
        foreach (var session in snapshot)
        {
            try { await session.DisposeAsync().ConfigureAwait(false); }
            catch (Exception error) { (errors ??= []).Add(error); }
        }
        try { manager.Dispose(); }
        catch (Exception error) { (errors ??= []).Add(error); }
        if (errors is not null)
        {
            throw new AggregateException("Inference engine shutdown failed.", errors);
        }
    }

    internal sealed class Worker : IDisposable
    {
        public Worker(VmBindings bindings, IVmExecutable executable, string inputSlot, string outputSlot)
        {
            Bindings = bindings;
            Executable = executable;
            InputIndex = bindings.Program.Slots.Select((slot, index) => (slot, index)).Single(pair => pair.slot.Id == inputSlot).index;
            OutputIndex = bindings.Program.Slots.Select((slot, index) => (slot, index)).Single(pair => pair.slot.Id == outputSlot).index;
            var entries = new List<(string Name, int Tokens)>();
            foreach (var entry in bindings.Program.Entries.Where(entry => entry.Name.StartsWith("prefill.", StringComparison.Ordinal)))
            {
                if (!int.TryParse(entry.Name.AsSpan("prefill.".Length), System.Globalization.NumberStyles.None,
                    System.Globalization.CultureInfo.InvariantCulture, out var tokens) ||
                    tokens <= 0 || (ulong)tokens > bindings.Program.Slots[InputIndex].Tensor.ElementCount)
                {
                    throw new InvalidDataException($"Invalid prefill entry '{entry.Name}'.");
                }

                entries.Add((entry.Name, tokens));
            }
            PrefillEntries = entries.OrderByDescending(entry => entry.Tokens).ToArray();
        }
        public object Gate { get; } = new();
        public VmBindings Bindings { get; }
        public IVmExecutable Executable { get; }
        public int InputIndex { get; }
        public int OutputIndex { get; }
        public (string Name, int Tokens)[] PrefillEntries { get; }
        public void Dispose()
        {
            try { Bindings.Dispose(); }
            finally { Executable.Dispose(); }
        }
    }
}

/// <summary>Serializes token processing and state transfers for a single inference context.</summary>
public sealed class VmInferenceSession : IAsyncDisposable
{
    private readonly VmInferenceEngine engine;
    private readonly VmSessionResources state;
    private readonly SemaphoreSlim operations = new(1, 1);
    private readonly CancellationTokenSource lifetime = new();
    private readonly object gate = new();
    private Task? shutdown;
    private bool lifetimeDisposed;
    private VmGenerationLease? generation;

    internal VmInferenceSession(VmInferenceEngine engine, VmSessionResources state)
    {
        this.engine = engine;
        this.state = state;
    }

    /// <summary>Processes a nonempty bounded token batch and returns the final output logits.</summary>
    /// <param name="tokens">The nonempty vocabulary token sequence to process.</param>
    /// <param name="cancellation">The token used to cancel the operation.</param>
    public async ValueTask<float[]> PrefillAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellation = default)
    {
        if (tokens.IsEmpty)
        {
            throw new ArgumentException("At least one token is required.", nameof(tokens));
        }

        if (tokens.Length > engine.MaximumPrefillTokens)
        {
            throw new ArgumentOutOfRangeException(nameof(tokens), "Prefill exceeds the configured token budget.");
        }

        var snapshot = tokens.ToArray();
        foreach (var token in snapshot)
        {
            if (token < 0 || token >= engine.VocabularySize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokens));
            }
        }

        using var linked = Link(cancellation);
        await operations.WaitAsync(linked.Token).ConfigureAwait(false);
        try { return await engine.PrefillAsync(state, snapshot, linked).ConfigureAwait(false); }
        finally { operations.Release(); }
    }

    /// <summary>Reserves exclusive session access for a sequence of token-generation operations.</summary>
    /// <param name="cancellation">The token used to cancel the operation.</param>
    public async ValueTask<VmGenerationLease> BeginGenerationAsync(CancellationToken cancellation = default)
    {
        var linked = Link(cancellation);
        var acquired = false;
        try
        {
            await operations.WaitAsync(linked.Token).ConfigureAwait(false);
            acquired = true;
            lock (gate)
            {
                linked.Token.ThrowIfCancellationRequested();
                return generation = new VmGenerationLease(this, linked);
            }
        }
        catch
        {
            if (acquired)
            {
                operations.Release();
            }

            linked.Dispose();
            throw;
        }
    }

    internal ValueTask<float[]> ForwardScopedAsync(int token, CancellationTokenSource cancellation) =>
        engine.ForwardAsync(state, token, cancellation);

    internal void ValidateToken(int token)
    {
        engine.ValidateToken(token);
    }

    internal void EndGeneration(VmGenerationLease scope)
    {
        lock (gate)
        {
            if (!ReferenceEquals(generation, scope))
            {
                throw new InvalidOperationException("The generation scope does not own this session.");
            }

            generation = null;
            operations.Release();
        }
    }

    /// <summary>Processes one vocabulary token and returns the resulting logits.</summary>
    /// <param name="token">The vocabulary token identifier to process.</param>
    /// <param name="cancellation">The token used to cancel the operation.</param>
    public async ValueTask<float[]> ForwardAsync(int token, CancellationToken cancellation = default)
    {
        engine.ValidateToken(token);
        await using var generation = await BeginGenerationAsync(cancellation).ConfigureAwait(false);
        return await generation.ForwardTokenAsync(token).ConfigureAwait(false);
    }

    /// <summary>Exports valid session state with a context identifier and payload integrity hashes.</summary>
    /// <param name="destination">The writable state stream; the caller retains ownership of the stream.</param>
    /// <param name="contextId">The nonempty context identifier embedded in or required by the state stream.</param>
    /// <param name="cancellation">The token used to cancel the operation.</param>
    public async ValueTask ExportStateAsync(Stream destination, string contextId, CancellationToken cancellation = default)
    {
        using var linked = Link(cancellation);
        await operations.WaitAsync(linked.Token).ConfigureAwait(false);
        try
        {
            using var bindings = state.CreateStateBindings();
            using var access = bindings.BeginStateAccess();
            VmStateStream.Export(destination, state.StateProgram, access, contextId);
        }
        finally { operations.Release(); }
    }

    /// <summary>Validates and stages a context-compatible state stream before restoring session state.</summary>
    /// <param name="source">The readable state stream; the caller retains ownership of the stream.</param>
    /// <param name="contextId">The nonempty context identifier embedded in or required by the state stream.</param>
    /// <param name="budget">The maximum aggregate state payload size to stage in bytes.</param>
    /// <param name="cancellation">The token used to cancel the operation.</param>
    public async ValueTask ImportStateAsync(Stream source, string contextId, ulong budget,
        CancellationToken cancellation = default)
    {
        using var linked = Link(cancellation);
        await operations.WaitAsync(linked.Token).ConfigureAwait(false);
        try
        {
            using var bindings = state.CreateStateBindings();
            using var access = bindings.BeginStateAccess();
            VmStateStream.Import(source, state.StateProgram, access, contextId, budget);
        }
        finally { operations.Release(); }
    }

    /// <summary>Zeroes all registered state slots and marks the session state valid.</summary>
    /// <param name="cancellation">The token used to cancel the operation.</param>
    public async ValueTask ResetAsync(CancellationToken cancellation = default)
    {
        using var linked = Link(cancellation);
        await operations.WaitAsync(linked.Token).ConfigureAwait(false);
        try
        {
            using var bindings = state.CreateStateBindings();
            using var access = bindings.BeginStateAccess();
            foreach (var slot in state.StateProgram.Slots)
            {
                access.WriteState(slot.Id, new byte[checked((int)slot.Tensor.ByteLength)]);
            }

            access.ValidateState();
        }
        finally { operations.Release(); }
    }

    /// <summary>Gets the state-only program exposing canonical named session-state slots.</summary>
    public VmProgram StateProgram => state.StateProgram;

    /// <summary>Copies valid session-state payloads into arrays keyed by canonical state-entry name.</summary>
    /// <param name="cancellation">The token used to cancel the operation.</param>
    public async ValueTask<IReadOnlyDictionary<string, byte[]>> ReadStateAsync(CancellationToken cancellation = default)
    {
        using var linked = Link(cancellation);
        await operations.WaitAsync(linked.Token).ConfigureAwait(false);
        try
        {
            using var bindings = state.CreateStateBindings();
            using var access = bindings.BeginStateAccess();
            if (!access.StateValid)
            {
                throw new InvalidOperationException("Cannot read an invalid session state.");
            }

            var values = new Dictionary<string, byte[]>(StringComparer.Ordinal);
            foreach (var slot in state.StateProgram.Slots)
            {
                var bytes = new byte[checked((int)slot.Tensor.ByteLength)];
                access.Read(slot.Id, 0, bytes);
                values.Add(slot.Id, bytes);
            }
            return values;
        }
        finally { operations.Release(); }
    }

    /// <summary>Validates a complete state snapshot and restores it while holding exclusive session access.</summary>
    /// <param name="values">The complete state payload map keyed by registered state slot name.</param>
    /// <param name="cancellation">The token used to cancel the operation.</param>
    public async ValueTask WriteStateAsync(IReadOnlyDictionary<string, byte[]> values,
        CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Count != state.StateProgram.Slots.Count ||
            state.StateProgram.Slots.Any(slot => !values.TryGetValue(slot.Id, out var bytes) ||
                bytes is null || (ulong)bytes.Length != slot.Tensor.ByteLength))
        {
            throw new InvalidDataException("State entries do not match the session schema.");
        }

        var snapshot = values.ToDictionary(pair => pair.Key, pair => pair.Value.ToArray(), StringComparer.Ordinal);
        using var linked = Link(cancellation);
        await operations.WaitAsync(linked.Token).ConfigureAwait(false);
        try
        {
            using var bindings = state.CreateStateBindings();
            using var access = bindings.BeginStateAccess();
            try
            {
                foreach (var slot in state.StateProgram.Slots)
                {
                    access.WriteState(slot.Id, snapshot[slot.Id]);
                }

                access.ValidateState();
            }
            catch
            {
                access.InvalidateState();
                throw;
            }
        }
        finally { operations.Release(); }
    }

    private CancellationTokenSource Link(CancellationToken cancellation)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(shutdown is not null || lifetime.IsCancellationRequested, this);
            return CancellationTokenSource.CreateLinkedTokenSource(cancellation, lifetime.Token);
        }
    }

    internal void Stop()
    {
        lock (gate)
        {
            if (!lifetimeDisposed)
            {
                lifetime.Cancel();
            }
        }
    }

    /// <summary>Stops accepting work, waits for active operations, and releases owned resources.</summary>
    public ValueTask DisposeAsync()
    {
        lock (gate)
        {
            shutdown ??= StopAsync();
            return new ValueTask(shutdown);
        }
    }

    private async Task StopAsync()
    {
        await Task.Yield();
        Stop();
        VmGenerationLease? active;
        lock (gate)
        {
            active = generation;
        }

        if (active is not null)
        {
            try { await active.DisposeAsync().ConfigureAwait(false); }
            catch (OperationCanceledException) when (lifetime.IsCancellationRequested) { }
        }
        await operations.WaitAsync().ConfigureAwait(false);
        try { state.Dispose(); }
        finally
        {
            engine.Remove(this);
            operations.Release();
            lock (gate)
            {
                lifetimeDisposed = true;
                lifetime.Dispose();
            }
        }
    }
}

/// <summary>Reserves a session for a sequence of token-generation operations until released.</summary>
public sealed class VmGenerationLease : IAsyncDisposable
{
    private readonly VmInferenceSession session;
    private readonly CancellationTokenSource cancellation;
    private readonly SemaphoreSlim steps = new(1, 1);
    private readonly object gate = new();
    private Task? shutdown;
    private Exception? failure;

    internal VmGenerationLease(VmInferenceSession session, CancellationTokenSource cancellation)
    {
        this.session = session;
        this.cancellation = cancellation;
    }

    /// <summary>Synchronously processes one token within the reserved generation scope and returns logits.</summary>
    /// <param name="token">The vocabulary token identifier to process.</param>
    public float[] ForwardToken(int token) => ForwardTokenAsync(token).AsTask().GetAwaiter().GetResult();

    /// <summary>Processes one token within the reserved generation scope and returns logits.</summary>
    /// <param name="token">The vocabulary token identifier to process.</param>
    /// <param name="cancellationToken">The token used to cancel queued or active work.</param>
    public ValueTask<float[]> ForwardTokenAsync(int token, CancellationToken cancellationToken = default)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(shutdown is not null, this);
            session.ValidateToken(token);
            return ForwardCoreAsync(token, cancellationToken);
        }
    }

    private async ValueTask<float[]> ForwardCoreAsync(int token, CancellationToken caller)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellation.Token, caller);
        await steps.WaitAsync(linked.Token).ConfigureAwait(false);
        try { return await session.ForwardScopedAsync(token, linked).ConfigureAwait(false); }
        catch (Exception error)
        {
            lock (gate)
            {
                failure ??= error;
            }

            throw;
        }
        finally { steps.Release(); }
    }

    /// <summary>Cancels active token work, waits for completion, and releases the session's generation reservation.</summary>
    public ValueTask DisposeAsync()
    {
        lock (gate)
        {
            return new ValueTask(shutdown ??= DisposeCoreAsync());
        }
    }

    private async Task DisposeCoreAsync()
    {
        await Task.Yield();
        if (steps.CurrentCount == 0)
        {
            cancellation.Cancel();
        }

        await steps.WaitAsync().ConfigureAwait(false);
        var cancelled = cancellation.IsCancellationRequested;
        try
        {
            session.EndGeneration(this);
            if (failure is { } error)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error).Throw();
            }

            if (cancelled)
            {
                throw new OperationCanceledException(cancellation.Token);
            }
        }
        finally
        {
            cancellation.Dispose();
            steps.Release();
        }
    }
}
