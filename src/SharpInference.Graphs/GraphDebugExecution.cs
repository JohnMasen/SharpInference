namespace SharpInference.Graphs;

/// <summary>Executes token forward passes for a processor session.</summary>
public interface IProcessorSessionExecutor : IDisposable
{
    void ForwardToken(int token, Span<float> logits);

    void ForwardTokens(ReadOnlySpan<int> tokens, Span<float> logits)
    {
        foreach (var token in tokens)
        {
            ForwardToken(token, logits);
        }
    }
}

/// <summary>Provides an optional callback for per-layer trace values.</summary>
public interface IProcessorLayerTraceExecutor
{
    Action<int, float[]>? LayerTrace { get; set; }
}

/// <summary>Creates processor session executors from prepared backend plans.</summary>
public interface IProcessorSessionBackend
{
    IProcessorSessionExecutor CreateSessionExecutor(
        IRwkvModel model, IRwkvState state, IBackendExecutablePlan plan);
}

/// <summary>Combines execution-graph and debug backend capabilities for processor sessions.</summary>
public interface IProcessorDebugBackend : IExecutionGraphBackend, IDebuggableBackend
{
    IProcessorSessionExecutor CreateDebugSession(
        IRwkvModel model, IRwkvState state, IDebugBackendExecutablePlan plan,
        GraphDebugSession debugSession);
}
