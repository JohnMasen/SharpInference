using SharpInference.Graphs;

namespace SharpInference;

public static class RwkvGraphSignatures
{
    public static GraphModelSignature Create(int vocabularySize, int embeddingSize, int layerCount,
        int headCount, int headSize, string stateAbiId) => new("rwkv", stateAbiId,
        new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["vocabulary"] = vocabularySize,
            ["embedding"] = embeddingSize,
            ["layers"] = layerCount,
            ["attentionHeads"] = headCount,
            ["attentionHeadSize"] = headSize,
        }, new Dictionary<string, string>(StringComparer.Ordinal) { ["execution.token-prefill"] = "true" });

    public static RwkvModelMetadata Read(GraphModelSignature signature, string architectureId) =>
        RwkvModelMetadata.FromModelMetadata(new ModelMetadata(architectureId,
            signature.Dimensions.ToDictionary(value => value.Key, value => (long)value.Value,
                StringComparer.Ordinal), signature.Attributes));

    public static bool IsCompatible(GraphModelSignature signature)
    {
        ArgumentNullException.ThrowIfNull(signature);
        int Dimension(string key) => signature.Dimensions.GetValueOrDefault(key);
        return Dimension("vocabulary") > 0 && Dimension("embedding") > 0 && Dimension("layers") > 0 &&
            Dimension("attentionHeads") > 0 && Dimension("attentionHeadSize") > 0 &&
            Dimension("attentionHeads") * (long)Dimension("attentionHeadSize") == Dimension("embedding");
    }
}
