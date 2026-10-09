namespace SharpInference;

/// <summary>Resolves the bundled World tokenizer for supported RWKV model architectures.</summary>
public static class RwkvTokenizerResolver
{
    /// <summary>Loads the tokenizer associated with the architecture in the supplied model metadata.</summary>
    /// <param name="metadata">The model metadata containing the architecture identifier.</param>
    /// <returns>The bundled tokenizer for the model architecture.</returns>
    public static RwkvWorldTokenizer LoadForModel(RwkvModelMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        return LoadForArchitecture(metadata.ArchitectureId);
    }

    /// <summary>Loads the bundled tokenizer for a supported architecture identifier.</summary>
    /// <param name="architectureId">The architecture identifier to resolve.</param>
    /// <returns>The bundled World tokenizer.</returns>
    /// <exception cref="NotSupportedException">The architecture does not have a bundled tokenizer.</exception>
    public static RwkvWorldTokenizer LoadForArchitecture(string architectureId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(architectureId);
        return architectureId switch
        {
            "rwkv-6" or "rwkv-7" => RwkvWorldTokenizer.LoadBundled(),
            _ => throw new NotSupportedException($"No bundled tokenizer is available for architecture '{architectureId}'. Only RWKV-6 and RWKV-7 World models are supported."),
        };
    }
}
