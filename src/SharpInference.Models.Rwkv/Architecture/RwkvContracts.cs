namespace SharpInference;

/// <summary>Describes the dimensions and architecture identifier of an RWKV model.</summary>
/// <param name="VocabularySize">The number of vocabulary tokens.</param>
/// <param name="EmbeddingSize">The embedding width.</param>
/// <param name="LayerCount">The recurrent layer count.</param>
/// <param name="HeadCount">The attention head count.</param>
/// <param name="HeadSize">The number of elements per attention head.</param>
/// <param name="ArchitectureId">The RWKV architecture identifier.</param>
public sealed record RwkvModelMetadata(
    int VocabularySize,
    int EmbeddingSize,
    int LayerCount,
    int HeadCount,
    int HeadSize,
    string ArchitectureId)
{
    /// <summary>Converts RWKV dimensions to architecture-neutral model metadata.</summary>
    /// <param name="metadata">The RWKV metadata to convert.</param>
    /// <returns>The general model metadata.</returns>
    public static implicit operator ModelMetadata(RwkvModelMetadata metadata) => metadata.ToModelMetadata();

    /// <summary>Reads the required RWKV dimensions from general model metadata.</summary>
    /// <param name="metadata">Metadata containing vocabulary, embedding, layer, and head dimensions.</param>
    /// <returns>The RWKV-specific metadata.</returns>
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
    /// <summary>Gets the model's RWKV-specific dimensions and architecture identifier.</summary>
    RwkvModelMetadata Metadata { get; }
    ModelMetadata IModel.ModelMetadata => Metadata.ToModelMetadata();
}

/// <summary>Represents mutable recurrent state used by an RWKV model.</summary>
public interface IRwkvState : IModelState
{
    /// <summary>Gets the number of float32 values in the flattened recurrent state.</summary>
    int ElementCount { get; }
    /// <summary>Creates an independent copy of the current recurrent state.</summary>
    /// <returns>The copied RWKV state.</returns>
    new IRwkvState Clone();
    IModelState IModelState.Clone() => Clone();
    /// <summary>Copies the flattened recurrent state into caller-owned storage.</summary>
    /// <param name="destination">Storage for all state elements.</param>
    void CopyTo(Span<float> destination);
    /// <summary>Restores recurrent state from its flattened representation.</summary>
    /// <param name="source">The state elements in the implementation's snapshot order.</param>
    void Restore(ReadOnlySpan<float> source);
}

/// <summary>A named, mutable float32 state buffer for an RWKV model.</summary>
/// <param name="Name">The state buffer name.</param>
/// <param name="Dimensions">The buffer's tensor dimensions.</param>
/// <param name="Values">The mutable float32 buffer values.</param>
public sealed record RwkvStateView(string Name, IReadOnlyList<int> Dimensions, float[] Values);

/// <summary>Optional named access to the buffers underlying an RWKV state snapshot.</summary>
public interface INamedRwkvState : IRwkvState
{
    /// <summary>Gets named host-accessible buffers representing the recurrent state.</summary>
    IReadOnlyList<RwkvStateView> Views { get; }
    /// <summary>Synchronizes edits to named host views back to the state implementation.</summary>
    void CommitViews();
}

/// <summary>Defines model binding, state creation, and token-forward operations for an RWKV architecture.</summary>
public interface IRwkvArchitecture
{
    /// <summary>Gets the supported architecture identifier.</summary>
    string Id { get; }
    /// <summary>Determines whether the tensor catalog matches this architecture.</summary>
    /// <param name="tensors">The model tensor catalog.</param>
    /// <returns>Whether this architecture can load the catalog.</returns>
    bool CanLoad(IModelTensorCatalog tensors);
    /// <summary>Binds model tensors to the architecture's inference representation.</summary>
    /// <param name="tensors">The compatible model tensor catalog.</param>
    /// <returns>The bound RWKV model.</returns>
    IRwkvModel Bind(IModelTensorCatalog tensors);
    /// <summary>Creates initial recurrent state for a bound model.</summary>
    /// <param name="model">The bound model.</param>
    /// <returns>The initial model state.</returns>
    IRwkvState CreateState(IRwkvModel model);
    /// <summary>Processes one token, updating recurrent state and writing next-token logits.</summary>
    /// <param name="model">The bound model.</param>
    /// <param name="token">The input token identifier.</param>
    /// <param name="state">The recurrent state to update.</param>
    /// <param name="logits">The destination for vocabulary logits.</param>
    void ForwardToken(IRwkvModel model, int token, IRwkvState state, Span<float> logits);

    /// <summary>Processes tokens in sequence, leaving the final token's logits in the destination.</summary>
    /// <param name="model">The bound model.</param>
    /// <param name="tokens">The ordered input tokens.</param>
    /// <param name="state">The recurrent state to update.</param>
    /// <param name="logits">The destination for the final next-token logits.</param>
    void ForwardTokens(IRwkvModel model, ReadOnlySpan<int> tokens, IRwkvState state, Span<float> logits)
    {
        foreach (var token in tokens)
        {
            ForwardToken(model, token, state, logits);
        }
    }
}

/// <summary>Reads RWKV model dimensions from its architecture-specific tensor names.</summary>
/// <param name="VocabularySize">The number of vocabulary tokens.</param>
/// <param name="EmbeddingSize">The embedding width.</param>
/// <param name="LayerCount">The number of contiguous model blocks.</param>
public sealed record RwkvTensorCatalogDimensions(int VocabularySize, int EmbeddingSize, int LayerCount)
{
    /// <summary>Validates the embedding tensor and contiguous block indices to derive model dimensions.</summary>
    /// <param name="catalog">The model tensor catalog.</param>
    /// <returns>The vocabulary, embedding, and layer dimensions.</returns>
    public static RwkvTensorCatalogDimensions Read(IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        var embedding = catalog.GetRequired("emb.weight").Dimensions;
        if (embedding.Count != 2 || embedding.Any(size => size <= 0))
        {
            throw new InvalidDataException("The RWKV embedding tensor must have two positive dimensions.");
        }

        var layers = new HashSet<int>();
        foreach (var name in catalog.Names)
        {
            if (!name.StartsWith("blocks.", StringComparison.Ordinal))
            {
                continue;
            }

            var end = name.IndexOf('.', "blocks.".Length);
            if (end < 0 || !int.TryParse(name.AsSpan("blocks.".Length, end - "blocks.".Length),
                    System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture,
                    out var layer) || layer < 0)
            {
                throw new InvalidDataException($"Invalid RWKV block tensor name '{name}'.");
            }

            layers.Add(layer);
        }
        if (layers.Count == 0 || layers.Max() != layers.Count - 1)
        {
            throw new InvalidDataException("RWKV block indices must be contiguous and start at zero.");
        }

        return new(embedding[1], embedding[0], layers.Count);
    }
}
