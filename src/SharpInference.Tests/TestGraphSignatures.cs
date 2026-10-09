using SharpInference.Graphs;

namespace SharpInference.Tests;

internal static class TestGraphSignatures
{
    public static GraphModelSignature Create(int vocabularySize, int embeddingSize, int layerCount,
        int headCount, int headSize, string stateAbiId) => new("test", stateAbiId,
        new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["vocabulary"] = vocabularySize,
            ["embedding"] = embeddingSize,
            ["layers"] = layerCount,
            ["attentionHeads"] = headCount,
            ["attentionHeadSize"] = headSize,
        }, new Dictionary<string, string>(StringComparer.Ordinal) { ["execution.token-prefill"] = "true" });
}
