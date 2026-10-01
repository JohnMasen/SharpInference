namespace SharpInference.Tests;

public sealed class RwkvWorldTokenizerTests
{
    [Fact]
    public void LoadBundled_LoadsOfficialWorldVocabulary()
    {
        var tokenizer = RwkvWorldTokenizer.LoadBundled();

        Assert.Equal(65529, tokenizer.TokenIds.Count);
        Assert.Equal([98], tokenizer.Encode("a"));
        Assert.Equal("a", tokenizer.Decode([98]));
    }

    [Fact]
    public void Encode_UsesLongestUtf8TokenAndDecodeRoundTrips()
    {
        var path = Path.Combine(Path.GetTempPath(), $"sharpinference-vocab-{Guid.NewGuid():N}.txt");
        File.WriteAllText(path, "1 'a' 1\n2 'ab' 2\n3 b'\\xe4\\xb8\\xad' 3\n");
        try
        {
            var tokenizer = RwkvWorldTokenizer.Load(path);
            var tokens = tokenizer.Encode("ab中");

            Assert.Equal([2, 3], tokens);
            Assert.Equal("ab中", tokenizer.Decode(tokens));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
