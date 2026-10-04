using System.Runtime.InteropServices;

namespace SharpInference.Vm;

public interface IVmExecutable : IDisposable
{
    VmProgram Program { get; }
    void Execute(string entryName, byte[][] slots);
}

public sealed record VmInferenceProgram(VmProgram Program, string InputSlot, string OutputSlot,
    Func<IVmExecutable> CreateExecutable);

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

    public VmProgram Program { get; }
    public VmProgram PrefillProgram { get; }
    public int VocabularySize { get; }
    public int MaximumPrefillTokens { get; }

    public static ValueTask<VmInferenceEngine> CreateAsync(VmProgram program, string inputSlot,
        string outputSlot, Action<VmSlot, IVmStorage> initializeGlobal, Func<IVmExecutable> factory,
        VmEngineOptions options, Func<VmSlot, IVmStorage>? allocate = null) => CreateAsync(new(program, inputSlot, outputSlot, factory),
            new(program, inputSlot, outputSlot, factory), initializeGlobal, options, allocate);

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
            throw new InvalidDataException("Prefill and inference programs require matching model/output ABIs.");
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
                        throw new InvalidDataException("The executable and resource program do not match.");
                    return new Worker(bindings, executable, component.InputSlot, component.OutputSlot);
                }
                catch (Exception error)
                {
                    var errors = new List<Exception> { error };
                    try { bindings.Dispose(); }
                    catch (Exception cleanup) { errors.Add(cleanup); }
                    try { executable?.Dispose(); }
                    catch (Exception cleanup) { errors.Add(cleanup); }
                    if (errors.Count > 1) throw new AggregateException("VM worker construction failed.", errors);
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
            if (errors.Count > 1) throw new AggregateException("Inference engine construction failed.", errors);
            throw;
        }

        static void Validate(VmInferenceProgram component, bool isPrefill)
        {
            ArgumentNullException.ThrowIfNull(component.CreateExecutable);
            var input = component.Program.Slots.SingleOrDefault(slot => slot.Id == component.InputSlot);
            var output = component.Program.Slots.SingleOrDefault(slot => slot.Id == component.OutputSlot);
            if (input?.Tensor.ElementType != VmElementType.Int32 ||
                output?.Tensor.ElementType != VmElementType.Float32)
                throw new ArgumentException("Inference requires an Int32 token slot and Float32 logits slot.");
            var entry = isPrefill ? "prefill.1" : "forward";
            if (!component.Program.Entries.Any(candidate => candidate.Name == entry))
                throw new InvalidDataException($"The VM program has no required '{entry}' entry.");
        }
    }

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

    internal void Remove(VmInferenceSession session) { lock (gate) sessions.Remove(session); }

    internal ValueTask<float[]> PrefillAsync(VmSessionResources state, int[] tokens, CancellationToken cancellation)
    {
        return queue.PrefillAsync((worker, token) =>
        {
            lock (worker.Gate)
            {
                manager.BindSession(worker.Bindings, state);
                try
                {
                    var result = Array.Empty<float>();
                    for (var offset = 0; offset < tokens.Length;)
                    {
                        token.ThrowIfCancellationRequested();
                        var entry = worker.PrefillEntries.FirstOrDefault(entry => entry.Tokens <= tokens.Length - offset);
                        if (entry.Name is null)
                            throw new InvalidDataException("The prefill program cannot execute this token count.");
                        result = Run(worker, entry.Name, tokens.AsSpan(offset, entry.Tokens));
                        offset += entry.Tokens;
                    }
                    return ValueTask.FromResult(result);
                }
                catch
                {
                    using var access = worker.Bindings.BeginStateAccess();
                    access.InvalidateState();
                    throw;
                }
                finally { worker.Bindings.UnbindSession(); }
            }
        }, cancellation);
    }

    internal Task StartGeneration(VmSessionResources state, TaskCompletionSource<VmGenerationLease> ready,
        CancellationToken cancellation)
    {
        return queue.InferenceAsync(async (worker, token) =>
        {
            VmGenerationLease? lease = null;
            lock (worker.Gate)
            {
                manager.BindSession(worker.Bindings, state);
                lease = new VmGenerationLease(this, worker, token);
                ready.TrySetResult(lease);
            }
            try
            {
                await lease.End.Task.WaitAsync(token).ConfigureAwait(false);
                return 0;
            }
            catch
            {
                lock (worker.Gate)
                {
                    using var access = worker.Bindings.BeginStateAccess();
                    access.InvalidateState();
                }
                throw;
            }
            finally
            {
                lock (worker.Gate)
                {
                    lease.Close();
                    worker.Bindings.UnbindSession();
                }
            }
        }, cancellation).AsTask();
    }

    internal float[] Forward(Worker worker, int token)
    {
        return Run(worker, "forward", [token]);
    }

    internal void ValidateToken(int token)
    {
        if (token < 0 || token >= VocabularySize) throw new ArgumentOutOfRangeException(nameof(token));
    }

    private float[] Run(Worker worker, string entry, ReadOnlySpan<int> tokens)
    {
        using var access = worker.Bindings.BeginExecution();
        try
        {
            var buffers = access.GetBuffers();
            tokens.CopyTo(MemoryMarshal.Cast<byte, int>(buffers[worker.InputIndex].AsSpan()));
            worker.Executable.Execute(entry, buffers);
            return MemoryMarshal.Cast<byte, float>(buffers[worker.OutputIndex]).ToArray();
        }
        catch
        {
            access.InvalidateState();
            throw;
        }
    }

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
        lock (gate) snapshot = sessions.ToArray();
        foreach (var session in snapshot) session.Stop();
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
        if (errors is not null) throw new AggregateException("Inference engine shutdown failed.", errors);
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
                    throw new InvalidDataException($"Invalid prefill entry '{entry.Name}'.");
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

public sealed class VmInferenceSession : IAsyncDisposable
{
    private readonly VmInferenceEngine engine;
    private readonly VmSessionResources state;
    private readonly SemaphoreSlim operations = new(1, 1);
    private readonly CancellationTokenSource lifetime = new();
    private readonly object gate = new();
    private Task? shutdown;
    private bool lifetimeDisposed;

    internal VmInferenceSession(VmInferenceEngine engine, VmSessionResources state)
    {
        this.engine = engine;
        this.state = state;
    }

    public async ValueTask<float[]> PrefillAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellation = default)
    {
        if (tokens.IsEmpty) throw new ArgumentException("At least one token is required.", nameof(tokens));
        if (tokens.Length > engine.MaximumPrefillTokens)
            throw new ArgumentOutOfRangeException(nameof(tokens), "Prefill exceeds the configured token budget.");
        var snapshot = tokens.ToArray();
        foreach (var token in snapshot)
            if (token < 0 || token >= engine.VocabularySize) throw new ArgumentOutOfRangeException(nameof(tokens));
        using var linked = Link(cancellation);
        await operations.WaitAsync(linked.Token).ConfigureAwait(false);
        try { return await engine.PrefillAsync(state, snapshot, linked.Token).ConfigureAwait(false); }
        finally { operations.Release(); }
    }

    public async ValueTask<VmGenerationLease> BeginGenerationAsync(CancellationToken cancellation = default)
    {
        var linked = Link(cancellation);
        var acquired = false;
        var transferred = false;
        try
        {
            await operations.WaitAsync(linked.Token).ConfigureAwait(false);
            acquired = true;
            var ready = new TaskCompletionSource<VmGenerationLease>(TaskCreationOptions.RunContinuationsAsynchronously);
            var request = engine.StartGeneration(state, ready, linked.Token);
            var completion = FinishOperationAsync(request, linked);
            transferred = true;
            if (await Task.WhenAny(ready.Task, completion).ConfigureAwait(false) == completion)
                await completion.ConfigureAwait(false);
            var lease = await ready.Task.ConfigureAwait(false);
            lease.Completion = completion;
            return lease;
        }
        finally
        {
            if (!transferred)
            {
                if (acquired) operations.Release();
                linked.Dispose();
            }
        }
    }

    private async Task FinishOperationAsync(Task request, CancellationTokenSource linked)
    {
        try { await request.ConfigureAwait(false); }
        finally
        {
            operations.Release();
            linked.Dispose();
        }
    }

    public async ValueTask<float[]> ForwardAsync(int token, CancellationToken cancellation = default)
    {
        engine.ValidateToken(token);
        await using var generation = await BeginGenerationAsync(cancellation).ConfigureAwait(false);
        return generation.ForwardToken(token);
    }

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

    public async ValueTask ResetAsync(CancellationToken cancellation = default)
    {
        using var linked = Link(cancellation);
        await operations.WaitAsync(linked.Token).ConfigureAwait(false);
        try
        {
            using var bindings = state.CreateStateBindings();
            using var access = bindings.BeginStateAccess();
            foreach (var slot in state.StateProgram.Slots)
                access.WriteState(slot.Id, new byte[checked((int)slot.Tensor.ByteLength)]);
            access.ValidateState();
        }
        finally { operations.Release(); }
    }

    public VmProgram StateProgram => state.StateProgram;

    public async ValueTask<IReadOnlyDictionary<string, byte[]>> ReadStateAsync(CancellationToken cancellation = default)
    {
        using var linked = Link(cancellation);
        await operations.WaitAsync(linked.Token).ConfigureAwait(false);
        try
        {
            using var bindings = state.CreateStateBindings();
            using var access = bindings.BeginStateAccess();
            if (!access.StateValid) throw new InvalidOperationException("Cannot read an invalid session state.");
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

    public async ValueTask WriteStateAsync(IReadOnlyDictionary<string, byte[]> values,
        CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Count != state.StateProgram.Slots.Count ||
            state.StateProgram.Slots.Any(slot => !values.TryGetValue(slot.Id, out var bytes) ||
                bytes is null || (ulong)bytes.Length != slot.Tensor.ByteLength))
            throw new InvalidDataException("State entries do not match the session schema.");
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
                    access.WriteState(slot.Id, snapshot[slot.Id]);
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
            if (!lifetimeDisposed) lifetime.Cancel();
        }
    }

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

public sealed class VmGenerationLease : IAsyncDisposable
{
    private readonly VmInferenceEngine engine;
    private readonly VmInferenceEngine.Worker worker;
    private readonly CancellationToken cancellation;
    private bool closed;
    internal TaskCompletionSource End { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
    internal Task Completion { get; set; } = Task.CompletedTask;

    internal VmGenerationLease(VmInferenceEngine engine, VmInferenceEngine.Worker worker, CancellationToken cancellation)
    {
        this.engine = engine;
        this.worker = worker;
        this.cancellation = cancellation;
    }

    public float[] ForwardToken(int token)
    {
        lock (worker.Gate)
        {
            ObjectDisposedException.ThrowIf(closed, this);
            cancellation.ThrowIfCancellationRequested();
            engine.ValidateToken(token);
            try { return engine.Forward(worker, token); }
            catch (Exception error)
            {
                End.TrySetException(error);
                throw;
            }
        }
    }

    internal void Close() => closed = true;

    public async ValueTask DisposeAsync()
    {
        lock (worker.Gate)
        {
            closed = true;
            End.TrySetResult();
        }
        await Completion.ConfigureAwait(false);
    }
}
