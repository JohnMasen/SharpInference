namespace SharpInference;

public static class RwkvModelArchitectureDetector
{
    public static string Detect(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        using var catalog = GgmlModelFile.Open(modelPath);
        return Detect(catalog);
    }

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
