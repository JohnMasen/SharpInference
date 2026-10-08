namespace SharpInference;

/// <summary>Identifies the supported RWKV architecture represented by a model file's tensor names.</summary>
public static class RwkvModelArchitectureDetector
{
    /// <summary>Opens a model file and identifies its supported RWKV architecture.</summary>
    /// <param name="modelPath">The path to the GGML model file.</param>
    /// <returns><c>rwkv-7</c> or <c>rwkv-6</c>, depending on the tensor contract found in the file.</returns>
    public static string Detect(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        using var catalog = GgmlModelFile.Open(modelPath);
        return Detect(catalog);
    }

    /// <summary>Identifies the supported RWKV architecture represented by a tensor catalog.</summary>
    /// <param name="catalog">The tensor catalog to inspect.</param>
    /// <returns><c>rwkv-7</c> when RWKV-7 marker tensors are present, or <c>rwkv-6</c> for RWKV-6 markers.</returns>
    /// <exception cref="NotSupportedException">The catalog does not match either supported tensor contract.</exception>
    public static string Detect(IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        if (catalog.TryGet("blocks.0.att.r_k", out _) &&
            catalog.TryGet("blocks.0.att.x_rwkvag", out _))
        {
            return "rwkv-7";
        }

        if (catalog.TryGet("blocks.0.att.time_maa_x", out _) &&
            catalog.TryGet("blocks.0.att.time_faaaa", out _))
        {
            return "rwkv-6";
        }

        throw new NotSupportedException("The GGML model does not match a supported RWKV-6 or RWKV-7 tensor contract.");
    }
}
