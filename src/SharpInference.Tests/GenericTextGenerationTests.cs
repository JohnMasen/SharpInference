using System.Text;

namespace SharpInference.Tests;

public sealed class GenericTextGenerationTests
{
    [Fact]
    public async Task IndependentTokenizerWithUnsortedIdsGeneratesWithoutModelTypes()
    {
        var session = new Session();
        var result = new StringBuilder();
        await foreach (var chunk in TextGenerator.GenerateAsync(session, new Tokenizer(), "prompt",
            new TextGenerationOptions { Temperature = 0, MaxTokens = 4, StopTokenIds = new HashSet<int> { 3 } }))
            result.Append(chunk);
        Assert.Equal("ba", result.ToString());
        Assert.Equal([4, 1], session.Forwarded);
    }

    private sealed class Tokenizer : IByteTextTokenizer
    {
        public IReadOnlyList<int> TokenIds => [4, 1, 3];
        public IReadOnlyList<int> Encode(string text) => [1];
        public ReadOnlyMemory<byte> DecodeBytes(int token) => Encoding.UTF8.GetBytes(token switch
        {
            4 => "b", 1 => "a", 3 => "", _ => throw new ArgumentOutOfRangeException(nameof(token)),
        });
    }

    private sealed class Session : ITokenGenerationSession
    {
        public List<int> Forwarded { get; } = [];
        public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens) => new float[] { 0, 0, 0, 0, 5 };
        public ReadOnlyMemory<float> ForwardToken(int token)
        {
            Forwarded.Add(token);
            return token == 4 ? new float[] { 0, 5, 0, 0, 0 } : new float[] { 0, 0, 0, 5, 0 };
        }
    }
}
