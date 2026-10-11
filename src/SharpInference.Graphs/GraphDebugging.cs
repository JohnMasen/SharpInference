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
    /// <summary>Prepares a separately instrumented execution graph plan or returns failure diagnostics.</summary>
    /// <param name="graph">The graph supplying operations and resource binding metadata.</param>
    BackendPreparationResult PrepareDebug(ExecutionGraph graph);
}

/// <summary>Describes the lifecycle state of a graph debugging invocation.</summary>
public enum GraphDebugSessionState
{
    /// <summary>No debug invocation is active.</summary>
    Idle,
    /// <summary>A debug invocation is executing.</summary>
    Running,
    /// <summary>Execution is paused at a debug node boundary.</summary>
    Paused,
    /// <summary>The debug invocation has completed.</summary>
    Completed,
    /// <summary>The debug session has been cancelled.</summary>
    Stopped,
}

/// <summary>Describes a node port&apos;s resource and access for debug notifications.</summary>
/// <param name="Resource">The graph resource descriptor.</param>
/// <param name="Port">The node port name.</param>
/// <param name="Access">The permitted read and write access.</param>
public sealed record GraphDebugResourceDescriptor(
    GraphResource Resource,
    string Port,
    GraphResourceAccess Access)
{
    /// <summary>Gets the identifier of the resource described by this debug port.</summary>
    public ResourceId ResourceId => Resource.Id;
}

/// <summary>Provides the current execution node, input descriptors, and a callback pause request.</summary>
public abstract class GraphDebugNodeEventArgs : EventArgs
{
    private protected GraphDebugNodeEventArgs(
        ExecutionNodeId executionNodeId,
        IReadOnlyList<GraphDebugResourceDescriptor> inputs)
    {
        ExecutionNodeId = executionNodeId;
        Inputs = inputs;
    }

    /// <summary>Gets the execution node associated with this notification.</summary>
    public ExecutionNodeId ExecutionNodeId { get; }
    /// <summary>Gets the node&apos;s inputs resource descriptors.</summary>
    public IReadOnlyList<GraphDebugResourceDescriptor> Inputs { get; }
    /// <summary>Gets or sets whether the callback requests a pause at this node boundary.</summary>
    public bool PauseRequested { get; set; }
}

/// <summary>Provides information for a notification raised before executing a graph node.</summary>
public sealed class GraphDebugBeforeNodeEventArgs : GraphDebugNodeEventArgs
{
    internal GraphDebugBeforeNodeEventArgs(
        ExecutionNodeId executionNodeId,
        IReadOnlyList<GraphDebugResourceDescriptor> inputs) : base(executionNodeId, inputs)
    {
    }
}

/// <summary>Provides input and output information after executing a graph node.</summary>
public sealed class GraphDebugAfterNodeEventArgs : GraphDebugNodeEventArgs
{
    internal GraphDebugAfterNodeEventArgs(
        ExecutionNodeId executionNodeId,
        IReadOnlyList<GraphDebugResourceDescriptor> inputs,
        IReadOnlyList<GraphDebugResourceDescriptor> outputs) : base(executionNodeId, inputs)
    {
        Outputs = outputs;
    }

    /// <summary>Gets the node&apos;s outputs resource descriptors.</summary>
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

    /// <summary>Occurs synchronously on the inference thread before a node executes.</summary>
    public event EventHandler<GraphDebugBeforeNodeEventArgs>? BeforeNode;
    /// <summary>Occurs synchronously on the inference thread after a node executes.</summary>
    public event EventHandler<GraphDebugAfterNodeEventArgs>? AfterNode;

    /// <summary>Gets the current debug invocation lifecycle state.</summary>
    public GraphDebugSessionState State
    {
        get { lock (_gate) { ThrowIfDisposed(); return _state; } }
    }

    /// <summary>Gets whether a pause has been requested at a debug node boundary.</summary>
    public bool PauseRequested
    {
        get { lock (_gate) { ThrowIfDisposed(); return _pauseRequested; } }
    }

    /// <summary>Gets the cancellation token signalled when the debug session stops.</summary>
    public CancellationToken StopToken
    {
        get { lock (_gate) { ThrowIfDisposed(); return _stopSource.Token; } }
    }

    /// <summary>Prepares an instrumented plan and rejects unsupported backends, preparation failures, or ordinary plans.</summary>
    /// <param name="backend">The backend that prepares the execution graph.</param>
    /// <param name="graph">The graph supplying operations and resource binding metadata.</param>
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

    /// <summary>Begins a debug invocation on the current inference thread.</summary>
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

    /// <summary>Requests a pause at the next debug node boundary during a running invocation.</summary>
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

    /// <summary>Resumes a paused debug invocation.</summary>
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

    /// <summary>Cancels the debug session and releases a paused inference thread.</summary>
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

    /// <inheritdoc/>
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

    /// <summary>Raises the synchronous before-node notification and honors debug pause and stop requests.</summary>
    /// <param name="executionNodeId">The execution node identifier.</param>
    /// <param name="inputs">The node input resource descriptors.</param>
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

    /// <summary>Raises the synchronous after-node notification and honors debug pause and stop requests.</summary>
    /// <param name="executionNodeId">The execution node identifier.</param>
    /// <param name="inputs">The node input resource descriptors.</param>
    /// <param name="outputs">The node output resource descriptors.</param>
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

    /// <summary>Ends the current debug invocation and updates its completion state.</summary>
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
