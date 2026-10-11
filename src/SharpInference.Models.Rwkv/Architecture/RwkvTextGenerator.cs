namespace SharpInference;

/// <summary>Provides the RWKV World tokenizer adapter for model-neutral text generation.</summary>
public static class RwkvTextGenerator
{
    /// <summary>Generates RWKV output one UTF-16 character at a time from a text prompt.</summary>
    /// <param name="session">The model generation session.</param>
    /// <param name="tokenizer">The RWKV World tokenizer.</param>
    /// <param name="prompt">The initial text prompt.</param>
    /// <param name="options">The optional sampling and stopping settings.</param>
    /// <param name="cancellationToken">A token used to cancel generation.</param>
    /// <returns>The generated characters.</returns>
    public static IAsyncEnumerable<char> GenerateCharactersAsync(ITokenGenerationSession session,
        RwkvWorldTokenizer tokenizer, string prompt, RwkvGenerationOptions? options = null,
        CancellationToken cancellationToken = default) =>
        TextGenerator.GenerateCharactersAsync(session, tokenizer, prompt, options, cancellationToken);

    /// <summary>Encodes a text prompt and generates RWKV output chunks.</summary>
    /// <param name="session">The model generation session.</param>
    /// <param name="tokenizer">The RWKV World tokenizer.</param>
    /// <param name="prompt">The initial text prompt.</param>
    /// <param name="options">The optional sampling and stopping settings.</param>
    /// <param name="cancellationToken">A token used to cancel generation.</param>
    /// <returns>The generated text chunks.</returns>
    public static IAsyncEnumerable<string> GenerateAsync(ITokenGenerationSession session,
        RwkvWorldTokenizer tokenizer, string prompt, RwkvGenerationOptions? options = null,
        CancellationToken cancellationToken = default) =>
        TextGenerator.GenerateAsync(session, tokenizer, prompt, options, cancellationToken);

    /// <summary>Prefills tokenized prompt input and generates RWKV output chunks.</summary>
    /// <param name="session">The model generation session.</param>
    /// <param name="tokenizer">The RWKV World tokenizer.</param>
    /// <param name="promptTokens">The ordered prompt token identifiers.</param>
    /// <param name="options">The optional sampling and stopping settings.</param>
    /// <param name="cancellationToken">A token used to cancel generation.</param>
    /// <returns>The generated text chunks.</returns>
    public static IAsyncEnumerable<string> GenerateAsync(ITokenGenerationSession session,
        RwkvWorldTokenizer tokenizer, IReadOnlyList<int> promptTokens, RwkvGenerationOptions? options = null,
        CancellationToken cancellationToken = default) =>
        TextGenerator.GenerateAsync(session, tokenizer, promptTokens, options, cancellationToken);

    /// <summary>Generates RWKV output using logits from an already-prefilled session.</summary>
    /// <param name="session">The prefilled model generation session.</param>
    /// <param name="tokenizer">The RWKV World tokenizer.</param>
    /// <param name="initialLogits">The next-token logits produced by prefill.</param>
    /// <param name="options">The optional sampling and stopping settings.</param>
    /// <param name="cancellationToken">A token used to cancel generation.</param>
    /// <returns>The generated text chunks.</returns>
    public static IAsyncEnumerable<string> GenerateFromPrefilledAsync(ITokenGenerationSession session,
        RwkvWorldTokenizer tokenizer, ReadOnlyMemory<float> initialLogits, RwkvGenerationOptions? options = null,
        CancellationToken cancellationToken = default) =>
        TextGenerator.GenerateFromPrefilledAsync(session, tokenizer, initialLogits, options, cancellationToken);
}
