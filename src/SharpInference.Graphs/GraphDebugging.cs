namespace SharpInference.Graphs;

/// <summary>
/// A separate prepared plan whose execution dispatches per-node debug notifications.
/// </summary>
public interface IDebugBackendExecutablePlan : IBackendExecutablePlan
{
}

/// <summary>
/// Opt-in capability for preparing an instrumented plan without changing ordinary preparation.
/// </summary>
public interface IDebuggableBackend
{
    BackendPreparationResult PrepareDebug(ExecutionGraph graph);
}

public enum GraphDebugSessionState
{
    Idle,
    Running,
    Paused,
    Completed,
    Stopped,
}

public sealed record GraphDebugResourceDescriptor(
    GraphResource Resource,
    string Port,
    GraphResourceAccess Access)
{
    public ResourceId ResourceId => Resource.Id;
}

public abstract class GraphDebugNodeEventArgs : EventArgs
{
    private protected GraphDebugNodeEventArgs(
        ExecutionNodeId executionNodeId,
        IReadOnlyList<GraphDebugResourceDescriptor> inputs)
    {
        ExecutionNodeId = executionNodeId;
        Inputs = inputs;
    }

    public ExecutionNodeId ExecutionNodeId { get; }
    public IReadOnlyList<GraphDebugResourceDescriptor> Inputs { get; }
    public bool PauseRequested { get; set; }
}

public sealed class GraphDebugBeforeNodeEventArgs : GraphDebugNodeEventArgs
{
    internal GraphDebugBeforeNodeEventArgs(
        ExecutionNodeId executionNodeId,
        IReadOnlyList<GraphDebugResourceDescriptor> inputs) : base(executionNodeId, inputs)
    {
    }
}

public sealed class GraphDebugAfterNodeEventArgs : GraphDebugNodeEventArgs
{
    internal GraphDebugAfterNodeEventArgs(
        ExecutionNodeId executionNodeId,
        IReadOnlyList<GraphDebugResourceDescriptor> inputs,
        IReadOnlyList<GraphDebugResourceDescriptor> outputs) : base(executionNodeId, inputs)
    {
        Outputs = outputs;
    }

    public IReadOnlyList<GraphDebugResourceDescriptor> Outputs { get; }
}

/// <summary>
/// Owns debugging and cancellation for one session. Node callbacks run synchronously on the
/// inference thread; Resume and Stop may be called from another thread.
/// </summary>
public sealed class GraphDebugSession : IDisposable
{
    private readonly object _gate = new();
    private readonly CancellationTokenSource _stopSource = new();
    private GraphDebugSessionState _state;
    private int? _inferenceThread;
    private ExecutionNodeId? _pendingNode;
    private bool _invocationActive;
    private bool _pauseRequested;
    private bool _cancellationInProgress;
    private bool _disposed;

    public event EventHandler<GraphDebugBeforeNodeEventArgs>? BeforeNode;
    public event EventHandler<GraphDebugAfterNodeEventArgs>? AfterNode;

    public GraphDebugSessionState State
    {
        get { lock (_gate) { ThrowIfDisposed(); return _state; } }
    }

    public bool PauseRequested
    {
        get { lock (_gate) { ThrowIfDisposed(); return _pauseRequested; } }
    }

    public CancellationToken StopToken
    {
        get { lock (_gate) { ThrowIfDisposed(); return _stopSource.Token; } }
    }

    public static IDebugBackendExecutablePlan PrepareDebugPlan(
        IExecutionGraphBackend backend,
        ExecutionGraph graph)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(graph);
        if (backend is not IDebuggableBackend debugBackend)
        {
            throw new NotSupportedException(
                $"Backend '{backend.GetType().Name}' cannot prepare a debug execution plan.");
        }

        var result = debugBackend.PrepareDebug(graph)
            ?? throw new InvalidOperationException("The backend returned no debug preparation result.");
        var plan = result.GetPlanOrThrow(backend.GetType().Name);
        return plan as IDebugBackendExecutablePlan
            ?? throw new InvalidOperationException(
                $"Backend '{backend.GetType().Name}' returned an ordinary plan instead of a debug execution plan.");
    }

    public void BeginInvocation()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            if (_state != GraphDebugSessionState.Idle && _state != GraphDebugSessionState.Completed)
            {
                throw new InvalidOperationException($"Cannot begin an invocation while debugging is {_state}.");
            }

            _inferenceThread = Environment.CurrentManagedThreadId;
            _invocationActive = true;
            _pendingNode = null;
            _pauseRequested = false;
            _state = GraphDebugSessionState.Running;
        }
    }

    public void RequestPause()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            if (_state != GraphDebugSessionState.Running || _pauseRequested)
            {
                throw new InvalidOperationException("Pause can only be requested once during a running invocation.");
            }

            _pauseRequested = true;
        }
    }

    public void Resume()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            if (_state != GraphDebugSessionState.Paused)
            {
                throw new InvalidOperationException("Only a paused invocation can be resumed.");
            }

            _pauseRequested = false;
            _state = GraphDebugSessionState.Running;
            Monitor.PulseAll(_gate);
        }
    }

    public void Stop()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            if (_state != GraphDebugSessionState.Running && _state != GraphDebugSessionState.Paused)
            {
                throw new InvalidOperationException("Only an active invocation can be stopped.");
            }

            _pauseRequested = false;
            _state = GraphDebugSessionState.Stopped;
            _cancellationInProgress = true;
            Monitor.PulseAll(_gate);
        }

        CancelStopSource();
    }

    public void Dispose()
    {
        bool cancel;
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            _pauseRequested = false;
            _state = GraphDebugSessionState.Stopped;
            BeforeNode = null;
            AfterNode = null;
            cancel = !_cancellationInProgress;
            if (cancel) _cancellationInProgress = true;
            Monitor.PulseAll(_gate);
        }

        if (cancel) CancelStopSource();
    }

    private void CancelStopSource()
    {
        try
        {
            _stopSource.Cancel();
        }
        finally
        {
            lock (_gate)
            {
                _cancellationInProgress = false;
                if (_disposed) _stopSource.Dispose();
            }
        }
    }

    public void NotifyBeforeNode(
        ExecutionNodeId executionNodeId,
        IEnumerable<GraphDebugResourceDescriptor> inputs)
    {
        lock (_gate) ThrowIfDisposed();
        var snapshot = Snapshot(inputs);
        if (snapshot.Any(resource => resource.Access == GraphResourceAccess.Write))
        {
            throw new ArgumentException("BeforeNode resources must be readable inputs.", nameof(inputs));
        }

        lock (_gate)
        {
            EnsureInferenceThread();
            EnsureRunning();
            if (_pendingNode is not null)
            {
                throw new InvalidOperationException("The previous node has not completed.");
            }

            _pendingNode = executionNodeId;
        }

        try
        {
            var args = new GraphDebugBeforeNodeEventArgs(executionNodeId, snapshot);
            BeforeNode?.Invoke(this, args);
            ApplyEventPauseRequest(args);
            WaitIfPaused();
        }
        catch
        {
            lock (_gate) _pendingNode = null;
            throw;
        }
    }

    public void NotifyAfterNode(
        ExecutionNodeId executionNodeId,
        IEnumerable<GraphDebugResourceDescriptor> inputs,
        IEnumerable<GraphDebugResourceDescriptor> outputs)
    {
        lock (_gate) ThrowIfDisposed();
        var inputSnapshot = Snapshot(inputs);
        var outputSnapshot = Snapshot(outputs);
        if (inputSnapshot.Any(resource => resource.Access == GraphResourceAccess.Write))
        {
            throw new ArgumentException("AfterNode inputs must be readable.", nameof(inputs));
        }

        if (outputSnapshot.Any(resource => resource.Access == GraphResourceAccess.Read))
        {
            throw new ArgumentException("AfterNode resources must be writable outputs.", nameof(outputs));
        }

        lock (_gate)
        {
            ThrowIfDisposed();
            EnsureInferenceThread();
            EnsureRunning();
            if (_pendingNode != executionNodeId)
            {
                throw new InvalidOperationException("AfterNode must match the pending BeforeNode.");
            }

            _pendingNode = null;
        }

        var args = new GraphDebugAfterNodeEventArgs(executionNodeId, inputSnapshot, outputSnapshot);
        AfterNode?.Invoke(this, args);
        ApplyEventPauseRequest(args);
        WaitIfPaused();
    }

    private void ApplyEventPauseRequest(GraphDebugNodeEventArgs args)
    {
        if (!args.PauseRequested)
        {
            return;
        }

        lock (_gate)
        {
            ThrowIfDisposed();
            EnsureInferenceThread();
            EnsureRunning();
            _pauseRequested = true;
        }
    }

    public void EndInvocation()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            EnsureInferenceThread();
            if (!_invocationActive)
            {
                throw new InvalidOperationException("There is no active invocation to end.");
            }

            _invocationActive = false;
            _inferenceThread = null;
            _pendingNode = null;
            _pauseRequested = false;
            if (_state != GraphDebugSessionState.Stopped)
            {
                _state = GraphDebugSessionState.Completed;
            }
        }
    }

    private void WaitIfPaused()
    {
        lock (_gate)
        {
            EnsureInferenceThread();
            EnsureRunning();
            if (_pauseRequested)
            {
                _state = GraphDebugSessionState.Paused;
                while (_state == GraphDebugSessionState.Paused)
                {
                    Monitor.Wait(_gate);
                }
            }

            EnsureRunning();
        }
    }

    private void EnsureInferenceThread()
    {
        ThrowIfDisposed();
        if (!_invocationActive || _inferenceThread != Environment.CurrentManagedThreadId)
        {
            throw new InvalidOperationException("Node notifications and invocation completion belong to the inference thread.");
        }
    }

    private void EnsureRunning()
    {
        ThrowIfDisposed();
        if (_state == GraphDebugSessionState.Stopped)
        {
            throw new OperationCanceledException("The debug session was stopped.", _stopSource.Token);
        }

        if (_state != GraphDebugSessionState.Running)
        {
            throw new InvalidOperationException($"The debug session is {_state}, not running.");
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    private static IReadOnlyList<GraphDebugResourceDescriptor> Snapshot(
        IEnumerable<GraphDebugResourceDescriptor> descriptors)
    {
        ArgumentNullException.ThrowIfNull(descriptors);
        var values = descriptors.ToArray();
        if (values.Any(value => value is null || value.Resource is null))
        {
            throw new ArgumentException("Resources and resource descriptors cannot be null.", nameof(descriptors));
        }

        return Array.AsReadOnly(values);
    }
}
