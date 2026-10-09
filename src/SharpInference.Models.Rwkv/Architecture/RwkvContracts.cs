namespace SharpInference;

/// <summary>Describes the dimensions and architecture identifier of an RWKV model.</summary>
public sealed record RwkvModelMetadata(
    int VocabularySize,
    int EmbeddingSize,
    int LayerCount,
    int HeadCount,
    int HeadSize,
    string ArchitectureId)
{
    public static implicit operator ModelMetadata(RwkvModelMetadata metadata) => metadata.ToModelMetadata();

    public static RwkvModelMetadata FromModelMetadata(ModelMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        int Dimension(string name) => metadata.Dimensions.TryGetValue(name, out var value)
            ? checked((int)value)
            : throw new InvalidDataException($"RWKV metadata requires dimension '{name}'.");
        return new(Dimension("vocabulary"), Dimension("embedding"), Dimension("layers"),
            Dimension("attentionHeads"), Dimension("attentionHeadSize"), metadata.ArchitectureId);
    }

    /// <summary>Converts the RWKV-specific dimensions to the general model metadata contract.</summary>
    public ModelMetadata ToModelMetadata() => new(
        ArchitectureId,
        new Dictionary<string, long>(StringComparer.Ordinal)
        {
            ["vocabulary"] = VocabularySize,
            ["embedding"] = EmbeddingSize,
            ["layers"] = LayerCount,
            ["attentionHeads"] = HeadCount,
            ["attentionHeadSize"] = HeadSize,
        });
}

/// <summary>Exposes RWKV-specific metadata for a bound model.</summary>
public interface IRwkvModel : IModel
{
    RwkvModelMetadata Metadata { get; }
    ModelMetadata IModel.ModelMetadata => Metadata.ToModelMetadata();
}

/// <summary>Represents mutable recurrent state used by an RWKV model.</summary>
public interface IRwkvState : IModelState
{
    int ElementCount { get; }
    new IRwkvState Clone();
    IModelState IModelState.Clone() => Clone();
    void CopyTo(Span<float> destination);
    void Restore(ReadOnlySpan<float> source);
}

/// <summary>A named, mutable float32 state buffer for an RWKV model.</summary>
public sealed record RwkvStateView(string Name, IReadOnlyList<int> Dimensions, float[] Values);

/// <summary>Optional named access to the buffers underlying an RWKV state snapshot.</summary>
public interface INamedRwkvState : IRwkvState
{
    IReadOnlyList<RwkvStateView> Views { get; }
    void CommitViews();
}

/// <summary>Defines model binding, state creation, and token-forward operations for an RWKV architecture.</summary>
public interface IRwkvArchitecture
{
    string Id { get; }
    bool CanLoad(IModelTensorCatalog tensors);
    IRwkvModel Bind(IModelTensorCatalog tensors);
    IRwkvState CreateState(IRwkvModel model);
    void ForwardToken(IRwkvModel model, int token, IRwkvState state, Span<float> logits);

    void ForwardTokens(IRwkvModel model, ReadOnlySpan<int> tokens, IRwkvState state, Span<float> logits)
    {
        foreach (var token in tokens)
            ForwardToken(model, token, state, logits);
    }
}

/// <summary>Reads RWKV model dimensions from its architecture-specific tensor names.</summary>
public sealed record RwkvTensorCatalogDimensions(int VocabularySize, int EmbeddingSize, int LayerCount)
{
    public static RwkvTensorCatalogDimensions Read(IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        var embedding = catalog.GetRequired("emb.weight").Dimensions;
        if (embedding.Count != 2 || embedding.Any(size => size <= 0))
            throw new InvalidDataException("The RWKV embedding tensor must have two positive dimensions.");
        var layers = new HashSet<int>();
        foreach (var name in catalog.Names)
        {
            if (!name.StartsWith("blocks.", StringComparison.Ordinal)) continue;
            var end = name.IndexOf('.', "blocks.".Length);
            if (end < 0 || !int.TryParse(name.AsSpan("blocks.".Length, end - "blocks.".Length),
                    System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture,
                    out var layer) || layer < 0)
                throw new InvalidDataException($"Invalid RWKV block tensor name '{name}'.");
            layers.Add(layer);
        }
        if (layers.Count == 0 || layers.Max() != layers.Count - 1)
            throw new InvalidDataException("RWKV block indices must be contiguous and start at zero.");
        return new(embedding[1], embedding[0], layers.Count);
    }
}
