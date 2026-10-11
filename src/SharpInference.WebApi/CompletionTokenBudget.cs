namespace SharpInference.WebApi;

/// <summary>Limits completion length to the remaining logical context window.</summary>
public static class CompletionTokenBudget
{
    /// <summary>Returns the smaller of the requested completion limit and remaining context tokens, rejecting invalid or exhausted windows.</summary>
    /// <param name="contextWindowTokens">The positive logical context-window size in tokens.</param>
    /// <param name="promptTokens">The number of tokens in the prompt.</param>
    /// <param name="requestedMaxTokens">The positive completion token limit requested by the client.</param>
    public static int Calculate(int contextWindowTokens, int promptTokens, int requestedMaxTokens)
    {
        if (contextWindowTokens <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(contextWindowTokens));
        }

        if (promptTokens < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(promptTokens));
        }

        if (requestedMaxTokens <= 0)
        {
            throw new ArgumentException("max_tokens must be positive.", nameof(requestedMaxTokens));
        }

        if (promptTokens >= contextWindowTokens)
        {
            throw new ContextWindowExceededException(contextWindowTokens, promptTokens);
        }

        return Math.Min(requestedMaxTokens, contextWindowTokens - promptTokens);
    }
}

/// <summary>Signals that the prompt already meets or exceeds the configured token context window.</summary>
/// <param name="contextWindowTokens">The positive logical context-window size in tokens.</param>
/// <param name="promptTokens">The number of tokens in the prompt.</param>
public sealed class ContextWindowExceededException(int contextWindowTokens, int promptTokens)
    : Exception(
        $"The prompt contains {promptTokens} tokens, which meets or exceeds the {contextWindowTokens}-token context window.")
{
    /// <summary>Gets the configured logical context-window size in tokens.</summary>
    public int ContextWindowTokens { get; } = contextWindowTokens;
    /// <summary>Gets the number of prompt tokens that exhausted the context window.</summary>
    public int PromptTokens { get; } = promptTokens;
}
