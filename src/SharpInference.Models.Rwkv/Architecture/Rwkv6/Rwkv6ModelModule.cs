using SharpInference.Graphs;

namespace SharpInference.Architectures.Rwkv6;

public sealed class Rwkv6ModelModule : IModelGraphModule
{
    private readonly PortableRwkv6GraphProvider provider = new();
    public string ArchitectureId => provider.ArchitectureId;
    public IGraphModelSignatureReader LegacySignatureReader { get; } = new RwkvLegacyGraphSignatureReader();

    public bool CanLoad(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        return tensors.TryGet("blocks.0.att.time_maa_x", out _) &&
            tensors.TryGet("blocks.0.att.time_faaaa", out _) &&
            !tensors.TryGet("blocks.0.att.k_k", out _) &&
            !tensors.TryGet("blocks.0.att.r_k", out _);
    }

    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        if (!CanLoad(tensors)) throw new InvalidDataException("The tensor catalog does not match RWKV-6.");
        var dimensions = RwkvTensorCatalogDimensions.Read(tensors);
        var heads = tensors.GetRequired("blocks.0.att.time_faaaa").Dimensions;
        if (heads.Count != 3 || heads[0] != 1 || heads[1] <= 0 || heads[2] <= 0 ||
            heads[1] * (long)heads[2] != dimensions.EmbeddingSize)
            throw new InvalidDataException("RWKV-6 head dimensions do not match the embedding tensor.");
        return new RwkvModelMetadata(dimensions.VocabularySize, dimensions.EmbeddingSize,
            dimensions.LayerCount, heads[2], heads[1], ArchitectureId).ToModelMetadata();
    }

    public LogicalGraph Build(IModelTensorCatalog tensors) => provider.Build(tensors);
}
