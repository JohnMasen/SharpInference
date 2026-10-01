using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class GraphDebuggingTests
{
    private static readonly ExecutionNodeId Node = new("layer.0");

    [Fact]
    public void DebugPreparation_RequiresDistinctInstrumentedPlanAndDoesNotFallBack()
    {
        var graph = CreateGraph();
        var ordinaryBackend = new TestBackend();
        var ordinary = ordinaryBackend.Prepare(graph).GetPlanOrThrow("test");
        Assert.IsType<TestPlan>(ordinary);
        Assert.Throws<NotSupportedException>(() => GraphDebugSession.PrepareDebugPlan(ordinaryBackend, graph));
        Assert.Equal(1, ordinaryBackend.NormalPreparations);

        var debugBackend = new DebuggableTestBackend(graph);
        var debug = GraphDebugSession.PrepareDebugPlan(debugBackend, graph);
        Assert.IsType<TestDebugPlan>(debug);
        Assert.Same(graph, debug.Graph);
        Assert.Equal(0, debugBackend.NormalPreparations);
        Assert.Equal(1, debugBackend.DebugPreparations);

        debugBackend.DebugResult = new BackendPreparationResult.Success(new TestPlan(graph));
        Assert.Throws<InvalidOperationException>(() => GraphDebugSession.PrepareDebugPlan(debugBackend, graph));
        Assert.Equal(0, debugBackend.NormalPreparations);

        debugBackend.DebugResult = new BackendPreparationResult.Failure(
            [new BackendPreparationDiagnostic(null, null, [],
                BackendPreparationFailureReason.UnsupportedOperation, "No debug implementation.", null, [])]);
        var failure = Assert.Throws<BackendPreparationException>(
            () => GraphDebugSession.PrepareDebugPlan(debugBackend, graph));
        Assert.Contains("No debug implementation.", failure.Message);
        Assert.Equal(0, debugBackend.NormalPreparations);
    }

    [Fact]
    public void Events_AreOrderedAndContainOnlyRelevantResourceSnapshots()
    {
        var session = new GraphDebugSession();
        var order = new List<string>();
        var input = new GraphDebugResourceDescriptor(Resource("input"), "x", GraphResourceAccess.Read);
        var output = new GraphDebugResourceDescriptor(Resource("output"), "y", GraphResourceAccess.Write);
        var inputs = new List<GraphDebugResourceDescriptor> { input };
        var outputs = new List<GraphDebugResourceDescriptor> { output };
        GraphDebugBeforeNodeEventArgs? before = null;
        GraphDebugAfterNodeEventArgs? after = null;
        session.BeforeNode += (_, e) => { order.Add("before"); before = e; };
        session.AfterNode += (_, e) => { order.Add("after"); after = e; };

        session.BeginInvocation();
        session.NotifyBeforeNode(Node, inputs);
        order.Add("execute");
        session.NotifyAfterNode(Node, inputs, outputs);
        session.EndInvocation();
        inputs.Clear();
        outputs.Clear();

        Assert.Equal(["before", "execute", "after"], order);
        Assert.Equal(Node, before!.ExecutionNodeId);
        Assert.Equal([input], before.Inputs);
        Assert.DoesNotContain(typeof(GraphDebugBeforeNodeEventArgs).GetProperties(), property => property.Name == "Outputs");
        Assert.Equal(Node, after!.ExecutionNodeId);
        Assert.Equal([input], after.Inputs);
        Assert.Equal([output], after.Outputs);
        Assert.Equal(new ResourceId("input"), after.Inputs[0].ResourceId);
        Assert.Equal(GraphElementType.Float32, after.Inputs[0].Resource.Tensor.ElementType);
        Assert.Equal(GraphDebugSessionState.Completed, session.State);
        Assert.False(session.StopToken.IsCancellationRequested);
    }

    [Fact]
    public async Task Resume_FromAnotherThread_UnblocksPausedInference()
    {
        var session = new GraphDebugSession();
        using var reached = new ManualResetEventSlim();
        var eventThread = 0;
        session.BeforeNode += (_, args) =>
        {
            eventThread = Environment.CurrentManagedThreadId;
            args.PauseRequested = true;
            reached.Set();
        };
        var invocation = Task.Run(() =>
        {
            session.BeginInvocation();
            try
            {
                session.NotifyBeforeNode(Node, []);
                session.NotifyAfterNode(Node, [], []);
            }
            finally
            {
                session.EndInvocation();
            }
            return Environment.CurrentManagedThreadId;
        });

        Assert.True(reached.Wait(TimeSpan.FromSeconds(5)));
        Assert.True(SpinWait.SpinUntil(() => session.State == GraphDebugSessionState.Paused, TimeSpan.FromSeconds(5)));
        Assert.True(session.PauseRequested);
        Assert.False(invocation.IsCompleted);
        session.Resume();
        var inferenceThread = await invocation.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(inferenceThread, eventThread);
        Assert.False(session.PauseRequested);
        Assert.Equal(GraphDebugSessionState.Completed, session.State);
        Assert.Throws<InvalidOperationException>(session.Resume);
    }

    [Fact]
    public async Task AfterNode_EventArgsCanPauseWithBothInputAndOutputDescriptors()
    {
        var session = new GraphDebugSession();
        using var afterReached = new ManualResetEventSlim();
        var input = new GraphDebugResourceDescriptor(Resource("input"), "x", GraphResourceAccess.Read);
        var output = new GraphDebugResourceDescriptor(Resource("output"), "y", GraphResourceAccess.Write);
        session.AfterNode += (_, args) =>
        {
            Assert.Equal([input], args.Inputs);
            Assert.Equal([output], args.Outputs);
            args.PauseRequested = true;
            afterReached.Set();
        };
        var invocation = Task.Run(() =>
        {
            session.BeginInvocation();
            try
            {
                session.NotifyBeforeNode(Node, [input]);
                session.NotifyAfterNode(Node, [input], [output]);
            }
            finally
            {
                session.EndInvocation();
            }
        });

        Assert.True(afterReached.Wait(TimeSpan.FromSeconds(5)));
        Assert.True(SpinWait.SpinUntil(() => session.State == GraphDebugSessionState.Paused, TimeSpan.FromSeconds(5)));
        Assert.False(invocation.IsCompleted);
        session.Resume();
        await invocation.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(GraphDebugSessionState.Completed, session.State);
    }

    [Fact]
    public async Task Stop_CancelsOnlyItsOwnSessionAndInvalidatesInvocation()
    {
        var stopped = new GraphDebugSession();
        var unaffected = new GraphDebugSession();
        using var reached = new ManualResetEventSlim();
        var otherEvents = 0;
        stopped.BeforeNode += (_, _) => { stopped.RequestPause(); reached.Set(); };
        unaffected.BeforeNode += (_, _) => otherEvents++;
        var invocation = Task.Run(() =>
        {
            stopped.BeginInvocation();
            try
            {
                stopped.NotifyBeforeNode(Node, []);
                stopped.NotifyAfterNode(Node, [], []);
            }
            finally
            {
                stopped.EndInvocation();
            }
        });

        Assert.True(reached.Wait(TimeSpan.FromSeconds(5)));
        Assert.True(SpinWait.SpinUntil(() => stopped.State == GraphDebugSessionState.Paused, TimeSpan.FromSeconds(5)));
        stopped.Stop();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => invocation.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal(GraphDebugSessionState.Stopped, stopped.State);
        Assert.True(stopped.StopToken.IsCancellationRequested);
        Assert.Throws<InvalidOperationException>(stopped.BeginInvocation);
        Assert.Throws<InvalidOperationException>(stopped.Resume);
        Assert.Throws<InvalidOperationException>(stopped.Stop);

        unaffected.BeginInvocation();
        unaffected.NotifyBeforeNode(Node, []);
        unaffected.NotifyAfterNode(Node, [], []);
        unaffected.EndInvocation();
        Assert.Equal(1, otherEvents);
        Assert.Equal(GraphDebugSessionState.Completed, unaffected.State);
        Assert.False(unaffected.StopToken.IsCancellationRequested);
    }

    [Fact]
    public void InvalidTransitionsAndUnmatchedNodesAreRejected()
    {
        var session = new GraphDebugSession();
        Assert.Throws<InvalidOperationException>(session.RequestPause);
        Assert.Throws<InvalidOperationException>(session.Resume);
        Assert.Throws<InvalidOperationException>(session.Stop);
        Assert.Throws<InvalidOperationException>(() => session.NotifyBeforeNode(Node, []));
        session.BeginInvocation();
        Assert.Throws<InvalidOperationException>(session.BeginInvocation);
        Assert.Throws<InvalidOperationException>(() => session.NotifyAfterNode(Node, [], []));
        Assert.Throws<ArgumentException>(() => session.NotifyBeforeNode(Node,
            [new GraphDebugResourceDescriptor(Resource("output"), "y", GraphResourceAccess.Write)]));
        Assert.Throws<ArgumentException>(() => session.NotifyAfterNode(Node, [],
            [new GraphDebugResourceDescriptor(Resource("input"), "x", GraphResourceAccess.Read)]));
        Assert.Throws<ArgumentException>(() => session.NotifyAfterNode(Node,
            [new GraphDebugResourceDescriptor(Resource("output"), "y", GraphResourceAccess.Write)], []));
        session.NotifyBeforeNode(Node, []);
        Assert.Throws<InvalidOperationException>(() => session.NotifyBeforeNode(Node, []));
        Assert.Throws<InvalidOperationException>(() => session.NotifyAfterNode(new("wrong"), [], []));
        session.NotifyAfterNode(Node, [], []);
        session.EndInvocation();
        Assert.Throws<InvalidOperationException>(session.EndInvocation);
    }

    [Fact]
    public void InferenceCallbacks_AreOwnedByInvocationThread_WhileControlsAreExternal()
    {
        var session = new GraphDebugSession();
        session.BeginInvocation();
        Exception? workerFailure = null;
        var worker = new Thread(() =>
        {
            try
            {
                Assert.Throws<InvalidOperationException>(() => session.NotifyBeforeNode(Node, []));
                Assert.Throws<InvalidOperationException>(session.EndInvocation);
                session.RequestPause();
            }
            catch (Exception exception)
            {
                workerFailure = exception;
            }
        });
        worker.Start();
        Assert.True(worker.Join(TimeSpan.FromSeconds(5)));
        Assert.Null(workerFailure);
        Assert.True(session.PauseRequested);
        session.Stop();
        Assert.Throws<OperationCanceledException>(() => session.NotifyBeforeNode(Node, []));
        session.EndInvocation();
    }

    [Fact]
    public void Dispose_CancelsOwnedTokenAndRejectsFurtherOperations()
    {
        var session = new GraphDebugSession();
        var token = session.StopToken;
        session.Dispose();
        session.Dispose();

        Assert.True(token.IsCancellationRequested);
        Assert.Throws<ObjectDisposedException>(() => _ = session.State);
        Assert.Throws<ObjectDisposedException>(() => _ = session.PauseRequested);
        Assert.Throws<ObjectDisposedException>(() => _ = session.StopToken);
        Assert.Throws<ObjectDisposedException>(session.BeginInvocation);
        Assert.Throws<ObjectDisposedException>(session.RequestPause);
        Assert.Throws<ObjectDisposedException>(session.Resume);
        Assert.Throws<ObjectDisposedException>(session.Stop);
        Assert.Throws<ObjectDisposedException>(() => session.NotifyBeforeNode(Node, []));
        Assert.Throws<ObjectDisposedException>(() => session.NotifyAfterNode(Node, [], []));
        Assert.Throws<ObjectDisposedException>(session.EndInvocation);
    }

    [Fact]
    public async Task Dispose_FromAnotherThread_UnblocksPausedInference()
    {
        var session = new GraphDebugSession();
        using var reached = new ManualResetEventSlim();
        var token = session.StopToken;
        session.BeforeNode += (_, args) =>
        {
            args.PauseRequested = true;
            reached.Set();
        };
        var invocation = Task.Run(() =>
        {
            session.BeginInvocation();
            try
            {
                session.NotifyBeforeNode(Node, []);
            }
            finally
            {
                Assert.Throws<ObjectDisposedException>(session.EndInvocation);
            }
        });

        Assert.True(reached.Wait(TimeSpan.FromSeconds(5)));
        Assert.True(SpinWait.SpinUntil(() => session.State == GraphDebugSessionState.Paused, TimeSpan.FromSeconds(5)));
        session.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => invocation.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.True(token.IsCancellationRequested);
    }

    [Fact]
    public void Dispose_DuringStopCancellation_DoesNotRaceTheTokenSource()
    {
        var session = new GraphDebugSession();
        session.BeginInvocation();
        var token = session.StopToken;
        using var registration = token.Register(session.Dispose);
        session.Stop();

        Assert.True(token.IsCancellationRequested);
        Assert.Throws<ObjectDisposedException>(session.Stop);
        session.Dispose();
    }

    private sealed record TestPlan(ExecutionGraph Graph) : IBackendExecutablePlan;
    private sealed record TestDebugPlan(ExecutionGraph Graph) : IDebugBackendExecutablePlan;

    private static GraphResource Resource(string id) =>
        new(new ResourceId(id), id, GraphResourceKind.TokenTransient, GraphResourceLifetime.Token,
            new TensorDescriptor(GraphElementType.Float32, [1]));

    private static ExecutionGraph CreateGraph() =>
        new(new GraphIdentity("test", 1, "debug-test"),
            new GraphModelSignature(1, 1, 1, 1, 1, "test"),
            [],
            [new GraphRegion(new RegionId("root"), null, GraphRegionTypes.Graph, null, "root",
                new Dictionary<string, string>())],
            [], [], []);

    private class TestBackend : IExecutionGraphBackend
    {
        public int NormalPreparations { get; private set; }
        public IPrimitiveOperatorBackend PrimitiveOperators => null!;
        public IExecutionKernelCatalog KernelCatalog => null!;
        public IReadOnlyList<OperatorImplementationDescription> GetOperatorImplementations(ExecutionGraph graph, ExecutionNode node) => [];
        public BackendPreparationResult Prepare(ExecutionGraph graph)
        {
            NormalPreparations++;
            return new BackendPreparationResult.Success(new TestPlan(graph));
        }

        public IProcessorSessionExecutor CreateSessionExecutor(
            IRwkvModel model, IRwkvState state, IBackendExecutablePlan plan) =>
            throw new NotSupportedException("This backend only tests graph preparation.");
    }

    private sealed class DebuggableTestBackend(ExecutionGraph graph) : TestBackend, IDebuggableBackend
    {
        public int DebugPreparations { get; private set; }
        public BackendPreparationResult DebugResult { get; set; } =
            new BackendPreparationResult.Success(new TestDebugPlan(graph));

        public BackendPreparationResult PrepareDebug(ExecutionGraph executionGraph)
        {
            Assert.Same(graph, executionGraph);
            DebugPreparations++;
            return DebugResult;
        }
    }
}
