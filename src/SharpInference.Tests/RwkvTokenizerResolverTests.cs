namespace SharpInference.Tests;

public sealed class RwkvTokenizerResolverTests
{
    [Theory]
    [InlineData("rwkv-6")]
    [InlineData("rwkv-7")]
    public void LoadForModel_Rwkv6OrRwkv7_UsesBundledWorldVocabulary(string architectureId)
    {
        var tokenizer = RwkvTokenizerResolver.LoadForModel(CreateMetadata(architectureId));

        Assert.Equal([98], tokenizer.Encode("a"));
    }

    [Fact]
    public void LoadForModel_UnsupportedArchitecture_ThrowsExplicitError()
    {
        var exception = Assert.Throws<NotSupportedException>(
            () => RwkvTokenizerResolver.LoadForModel(CreateMetadata("rwkv-5")));

        Assert.Contains("rwkv-5", exception.Message, StringComparison.Ordinal);
    }

    private static RwkvModelMetadata CreateMetadata(string architectureId) =>
        new(65536, 128, 1, 2, 64, architectureId);
}
