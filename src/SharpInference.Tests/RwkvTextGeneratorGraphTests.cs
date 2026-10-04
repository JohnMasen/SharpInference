using System.Text;

namespace SharpInference.Tests;

public sealed class RwkvTextGeneratorGraphTests
{
    [Fact]
    public async Task GeneratesEachTokenThroughSessionForward()
    {
        var tokenizer = RwkvWorldTokenizer.LoadBundled();
        var token = Assert.Single(tokenizer.Encode("a"));
        var logits = new float[tokenizer.TokenIds[^1] + 1];
        Array.Fill(logits, -100f);
        logits[token] = 100f;
        var session = new CountingSession(logits);
        var output = new StringBuilder();

        await foreach (var text in RwkvTextGenerator.GenerateAsync(
            session, tokenizer, "x", new RwkvGenerationOptions { MaxTokens = 2, TopK = 1 }))
            output.Append(text);

        Assert.Equal("aa", output.ToString());
        Assert.Equal(1, session.PrefillCount);
        Assert.Equal([token, token], session.ForwardedTokens);
    }

    [Fact]
    public async Task GeneratorUsesAsyncPrefillAndOneScopeForEntireReply()
    {
        var tokenizer = RwkvWorldTokenizer.LoadBundled();
        var token = Assert.Single(tokenizer.Encode("a"));
        var logits = new float[tokenizer.TokenIds[^1] + 1];
        Array.Fill(logits, -100f);
        logits[token] = 100f;
        var session = new ScopedSession(logits);
        var output = new StringBuilder();
        await foreach (var text in RwkvTextGenerator.GenerateAsync(session, tokenizer, "x",
            new RwkvGenerationOptions { MaxTokens = 3, TopK = 1 }))
            output.Append(text);
        Assert.Equal("aaa", output.ToString());
        Assert.Equal(1, session.Prefills);
        Assert.Equal(1, session.Scopes);
        Assert.Equal(3, session.Steps);
        Assert.Equal(1, session.Releases);
    }

    [Fact]
    public async Task AbandonedGenerationReleasesItsScope()
    {
        var tokenizer = RwkvWorldTokenizer.LoadBundled();
        var token = Assert.Single(tokenizer.Encode("a"));
        var logits = new float[tokenizer.TokenIds[^1] + 1];
        Array.Fill(logits, -100f);
        logits[token] = 100f;
        var session = new ScopedSession(logits);
        await foreach (var text in RwkvTextGenerator.GenerateFromPrefilledAsync(session, tokenizer, logits,
            new RwkvGenerationOptions { MaxTokens = 10, TopK = 1 }))
            break;
        Assert.Equal(1, session.Scopes);
        Assert.Equal(1, session.Releases);
    }

    private sealed class ScopedSession(float[] logits) : IRwkvScopedGenerationSession, IRwkvAsyncPrefillSession
    {
        public int Prefills, Scopes, Steps, Releases;
        public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens) => throw new InvalidOperationException("Use asynchronous prefill.");
        public ReadOnlyMemory<float> ForwardToken(int token) => throw new InvalidOperationException("Use the generation lease.");
        public ValueTask<ReadOnlyMemory<float>> PrefillAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Prefills++;
            return ValueTask.FromResult<ReadOnlyMemory<float>>(logits);
        }
        public ValueTask<IRwkvGenerationScope> BeginGenerationAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Scopes++;
            return ValueTask.FromResult<IRwkvGenerationScope>(new Scope(this, logits));
        }
        private sealed class Scope(ScopedSession owner, float[] logits) : IRwkvGenerationScope, IRwkvGenerationSession
        {
            public IRwkvGenerationSession Session => this;
            public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens) => throw new InvalidOperationException();
            public ReadOnlyMemory<float> ForwardToken(int token) { owner.Steps++; return logits; }
            public ValueTask DisposeAsync() { owner.Releases++; return ValueTask.CompletedTask; }
        }
    }

    private sealed class CountingSession(float[] logits) : IRwkvGenerationSession
    {
        public int PrefillCount { get; private set; }
        public List<int> ForwardedTokens { get; } = [];

        public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens)
        {
            PrefillCount++;
            return logits;
        }

        public ReadOnlyMemory<float> ForwardToken(int token)
        {
            ForwardedTokens.Add(token);
            return logits;
        }
    }
}
