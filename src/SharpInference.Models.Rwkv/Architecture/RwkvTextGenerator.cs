namespace SharpInference;

/// <summary>Provides the RWKV World tokenizer adapter for model-neutral text generation.</summary>
public static class RwkvTextGenerator
{
    public static IAsyncEnumerable<char> GenerateCharactersAsync(ITokenGenerationSession session,
        RwkvWorldTokenizer tokenizer, string prompt, RwkvGenerationOptions? options = null,
        CancellationToken cancellationToken = default) =>
        TextGenerator.GenerateCharactersAsync(session, tokenizer, prompt, options, cancellationToken);

    public static IAsyncEnumerable<string> GenerateAsync(ITokenGenerationSession session,
        RwkvWorldTokenizer tokenizer, string prompt, RwkvGenerationOptions? options = null,
        CancellationToken cancellationToken = default) =>
        TextGenerator.GenerateAsync(session, tokenizer, prompt, options, cancellationToken);

    public static IAsyncEnumerable<string> GenerateAsync(ITokenGenerationSession session,
        RwkvWorldTokenizer tokenizer, IReadOnlyList<int> promptTokens, RwkvGenerationOptions? options = null,
        CancellationToken cancellationToken = default) =>
        TextGenerator.GenerateAsync(session, tokenizer, promptTokens, options, cancellationToken);

    public static IAsyncEnumerable<string> GenerateFromPrefilledAsync(ITokenGenerationSession session,
        RwkvWorldTokenizer tokenizer, ReadOnlyMemory<float> initialLogits, RwkvGenerationOptions? options = null,
        CancellationToken cancellationToken = default) =>
        TextGenerator.GenerateFromPrefilledAsync(session, tokenizer, initialLogits, options, cancellationToken);
}
