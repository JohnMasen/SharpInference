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
