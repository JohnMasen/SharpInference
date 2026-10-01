namespace SharpInference.WebApi;

public static class CompletionTokenBudget
{
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

public sealed class ContextWindowExceededException(int contextWindowTokens, int promptTokens)
    : Exception(
        $"The prompt contains {promptTokens} tokens, which meets or exceeds the {contextWindowTokens}-token context window.")
{
    public int ContextWindowTokens { get; } = contextWindowTokens;
    public int PromptTokens { get; } = promptTokens;
}
