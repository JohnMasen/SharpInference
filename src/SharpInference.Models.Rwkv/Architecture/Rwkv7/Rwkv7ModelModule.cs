using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

namespace SharpInference.Architectures.Rwkv7;

/// <summary>Recognizes RWKV-7 tensor catalogs and supplies metadata and portable logical graphs.</summary>
public sealed class Rwkv7ModelModule
{
    private readonly PortableRwkv7GraphProvider provider = new();
    /// <inheritdoc/>
    public string ArchitectureId => provider.ArchitectureId;
    /// <inheritdoc/>
    public bool CanLoad(IModelTensorCatalog tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        return tensors.TryGet("blocks.0.att.r_k", out _) &&
            tensors.TryGet("blocks.0.att.x_rwkvag", out _);
    }

    /// <inheritdoc/>
    public ModelMetadata ReadMetadata(IModelTensorCatalog tensors)
    {
        if (!CanLoad(tensors))
        {
            throw new InvalidDataException("The tensor catalog does not match RWKV-7.");
        }

        var dimensions = RwkvTensorCatalogDimensions.Read(tensors);
        var heads = tensors.GetRequired("blocks.0.att.r_k").Dimensions;
        if (heads.Count != 2 || heads[0] <= 0 || heads[1] <= 0 ||
            heads[0] * (long)heads[1] != dimensions.EmbeddingSize)
        {
            throw new InvalidDataException("RWKV-7 head dimensions do not match the embedding tensor.");
        }

        return new RwkvModelMetadata(dimensions.VocabularySize, dimensions.EmbeddingSize,
            dimensions.LayerCount, heads[1], heads[0], ArchitectureId).ToModelMetadata();
    }

    /// <inheritdoc/>
    public GraphPackage<LogicalGraph> Build(IModelTensorCatalog tensors) => provider.Build(tensors);
}
