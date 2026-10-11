namespace SharpInference.Graphs;

/// <summary>Executes token forward passes for a processor session.</summary>
public interface IProcessorSessionExecutor : IDisposable
{
    /// <summary>Processes one token and writes its prediction logits using the current session state.</summary>
    /// <param name="token">The input token identifier.</param>
    /// <param name="logits">The output buffer receiving token prediction logits.</param>
    void ForwardToken(int token, Span<float> logits);

    /// <summary>Processes tokens in order and writes prediction logits to the supplied output buffer.</summary>
    /// <param name="tokens">The input token identifiers in processing order.</param>
    /// <param name="logits">The output buffer receiving token prediction logits.</param>
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
    /// <summary>Gets or sets the optional callback receiving a layer index and its trace values.</summary>
    Action<int, float[]>? LayerTrace { get; set; }
}

/// <summary>Creates processor session executors from prepared backend plans.</summary>
public interface IProcessorSessionBackend
{
    /// <summary>Creates a processor executor using the supplied model, state, and prepared plans.</summary>
    /// <param name="model">The model executed by the processor session.</param>
    /// <param name="state">The model session state.</param>
    /// <param name="plan">The prepared backend executable plan.</param>
    IProcessorSessionExecutor CreateSessionExecutor(
        IModel model, IModelState state, IBackendExecutablePlan plan);
}

/// <summary>Combines execution-graph and debug backend capabilities for processor sessions.</summary>
public interface IProcessorDebugBackend : IExecutionGraphBackend, IDebuggableBackend
{
    /// <summary>Creates a processor executor using an instrumented plan and debug session.</summary>
    /// <param name="model">The model executed by the processor session.</param>
    /// <param name="state">The model session state.</param>
    /// <param name="plan">The prepared backend executable plan.</param>
    /// <param name="debugSession">The session controlling debug callbacks, pausing, and cancellation.</param>
    IProcessorSessionExecutor CreateDebugSession(
        IModel model, IModelState state, IDebugBackendExecutablePlan plan,
        GraphDebugSession debugSession);
}
