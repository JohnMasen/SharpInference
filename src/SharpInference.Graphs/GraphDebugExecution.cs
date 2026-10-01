namespace SharpInference.Graphs;

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

public interface IProcessorLayerTraceExecutor
{
    Action<int, float[]>? LayerTrace { get; set; }
}

public interface IProcessorSessionBackend
{
    IProcessorSessionExecutor CreateSessionExecutor(
        IRwkvModel model, IRwkvState state, IBackendExecutablePlan plan);
}

public interface IProcessorDebugBackend : IExecutionGraphBackend, IDebuggableBackend
{
    IProcessorSessionExecutor CreateDebugSession(
        IRwkvModel model, IRwkvState state, IDebugBackendExecutablePlan plan,
        GraphDebugSession debugSession);
}
