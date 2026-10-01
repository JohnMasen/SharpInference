namespace SharpInference;

public static class RwkvTokenizerResolver
{
    public static RwkvWorldTokenizer LoadForModel(RwkvModelMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        return LoadForArchitecture(metadata.ArchitectureId);
    }

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
