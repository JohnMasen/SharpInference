using SharpInference.Graphs;

namespace SharpInference;

/// <summary>Creates and interprets architecture-neutral graph signatures for RWKV models.</summary>
public static class RwkvGraphSignatures
{
    /// <summary>Creates an RWKV graph signature with dimensions, state ABI, and token-prefill support.</summary>
    /// <param name="vocabularySize">The vocabulary token count.</param>
    /// <param name="embeddingSize">The embedding width.</param>
    /// <param name="layerCount">The recurrent layer count.</param>
    /// <param name="headCount">The attention head count.</param>
    /// <param name="headSize">The attention head width.</param>
    /// <param name="stateAbiId">The recurrent-state ABI identifier.</param>
    /// <returns>The graph model signature.</returns>
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

    /// <summary>Reads RWKV dimensions from a graph signature for the specified architecture.</summary>
    /// <param name="signature">The signature containing required RWKV dimensions.</param>
    /// <param name="architectureId">The model architecture identifier.</param>
    /// <returns>The RWKV model metadata.</returns>
    public static RwkvModelMetadata Read(GraphModelSignature signature, string architectureId) =>
        RwkvModelMetadata.FromModelMetadata(new ModelMetadata(architectureId,
            signature.Dimensions.ToDictionary(value => value.Key, value => (long)value.Value,
                StringComparer.Ordinal), signature.Attributes));

    /// <summary>Checks that required RWKV dimensions are positive and the head layout matches the embedding.</summary>
    /// <param name="signature">The graph signature to inspect.</param>
    /// <returns>Whether the dimensions describe a compatible RWKV layout.</returns>
    public static bool IsCompatible(GraphModelSignature signature)
    {
        ArgumentNullException.ThrowIfNull(signature);
        int Dimension(string key) => signature.Dimensions.GetValueOrDefault(key);
        return Dimension("vocabulary") > 0 && Dimension("embedding") > 0 && Dimension("layers") > 0 &&
            Dimension("attentionHeads") > 0 && Dimension("attentionHeadSize") > 0 &&
            Dimension("attentionHeads") * (long)Dimension("attentionHeadSize") == Dimension("embedding");
    }
}
