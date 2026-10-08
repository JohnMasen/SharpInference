namespace SharpInference;

/// <summary>Identifies the floating-point storage format used by a model tensor.</summary>
public enum RwkvTensorDataType : uint
{
    Float32 = 0,
    Float16 = 1,
}

/// <summary>Describes the dimensions and architecture identifier of an RWKV model.</summary>
public sealed record RwkvModelMetadata(
    int VocabularySize,
    int EmbeddingSize,
    int LayerCount,
    int HeadCount,
    int HeadSize,
    string ArchitectureId)
{
    /// <summary>Converts the RWKV-specific dimensions to the general model metadata contract.</summary>
    /// <returns>A model metadata instance containing the architecture and positive dimensions.</returns>
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

/// <summary>Stores an architecture identifier and named model dimensions and attributes.</summary>
public sealed class ModelMetadata
{
    /// <summary>Creates model metadata and validates all supplied keys and values.</summary>
    /// <param name="architectureId">The non-empty architecture identifier.</param>
    /// <param name="dimensions">Optional positive model dimensions keyed by name.</param>
    /// <param name="attributes">Optional non-empty model attributes keyed by name.</param>
    public ModelMetadata(
        string architectureId,
        IReadOnlyDictionary<string, long>? dimensions = null,
        IReadOnlyDictionary<string, string>? attributes = null)
    {
        ArchitectureId = string.IsNullOrWhiteSpace(architectureId)
            ? throw new ArgumentException("A model architecture identifier is required.", nameof(architectureId))
            : architectureId;
        Dimensions = new Dictionary<string, long>(
            dimensions ?? new Dictionary<string, long>(), StringComparer.Ordinal);
        Attributes = new Dictionary<string, string>(
            attributes ?? new Dictionary<string, string>(), StringComparer.Ordinal);
        if (Dimensions.Any(value => string.IsNullOrWhiteSpace(value.Key) || value.Value <= 0))
            throw new ArgumentException("Model dimensions require names and positive values.", nameof(dimensions));
        if (Attributes.Any(value => string.IsNullOrWhiteSpace(value.Key) || value.Value is null))
            throw new ArgumentException("Model attributes require names and values.", nameof(attributes));
    }

    public string ArchitectureId { get; }
    public IReadOnlyDictionary<string, long> Dimensions { get; }
    public IReadOnlyDictionary<string, string> Attributes { get; }
}

/// <summary>Exposes the metadata and typed values of one model tensor.</summary>
public interface IModelTensor
{
    string Name { get; }
    RwkvTensorDataType DataType { get; }
    IReadOnlyList<int> Dimensions { get; }
    ReadOnlySpan<float> FloatValues { get; }
    ReadOnlySpan<Half> HalfValues { get; }
}

/// <summary>Provides model-wide metadata and named tensor lookup.</summary>
public interface IModelTensorCatalog
{
    int VocabularySize { get; }
    int EmbeddingSize { get; }
    int LayerCount { get; }
    IReadOnlyCollection<string> Names { get; }
    bool TryGet(string name, out IModelTensor tensor);
    IModelTensor GetRequired(string name);
}

/// <summary>Represents an opened model file and its tensor catalog.</summary>
public interface IModelFile : IModelTensorCatalog, IDisposable
{
    string Path { get; }
}

/// <summary>Opens model files from paths.</summary>
public interface IModelReader
{
    IModelFile Open(string path);
}

/// <summary>Reports whether model weights must have CPU-owned copies.</summary>
public interface IModelWeightOwnershipPolicy
{
    bool RequiresCpuWeightCopy { get; }
}

/// <summary>Prepares a weight plan from a model tensor catalog.</summary>
public interface IModelWeightPreflight
{
    void PrepareWeightPlan(IModelTensorCatalog tensors);
}

/// <summary>Exposes general metadata for a bound model.</summary>
public interface IModel
{
    ModelMetadata ModelMetadata { get; }
}

/// <summary>Exposes RWKV-specific metadata for a bound model.</summary>
public interface IRwkvModel : IModel
{
    RwkvModelMetadata Metadata { get; }
    ModelMetadata IModel.ModelMetadata => Metadata.ToModelMetadata();
}

/// <summary>Represents mutable recurrent state used by an RWKV model.</summary>
public interface IRwkvState
{
    string ArchitectureId { get; }
    int ElementCount { get; }
    IRwkvState Clone();
    void Reset();
    void CopyTo(Span<float> destination);
    void Restore(ReadOnlySpan<float> source);
}

/// <summary>A named, mutable float32 state buffer; dimensions use the model's head and embedding sizes.</summary>
public sealed record RwkvStateView(string Name, IReadOnlyList<int> Dimensions, float[] Values);

/// <summary>Optional named access to the buffers underlying a state snapshot.</summary>
public interface INamedRwkvState : IRwkvState
{
    IReadOnlyList<RwkvStateView> Views { get; }

    /// <summary>Publish edits to view arrays to an attached backend before the next forward pass.</summary>
    void CommitViews();
}

/// <summary>Defines model binding, state creation, and token-forward operations for an RWKV architecture.</summary>
public interface IRwkvArchitecture
{
    /// <summary>Gets the architecture identifier.</summary>
    string Id { get; }

    /// <summary>Determines whether the tensor catalog matches this architecture.</summary>
    /// <param name="tensors">The tensor catalog to inspect.</param>
    /// <returns><see langword="true"/> when the architecture can bind the catalog.</returns>
    bool CanLoad(IModelTensorCatalog tensors);

    /// <summary>Binds model tensors to an architecture-specific model.</summary>
    /// <param name="tensors">The tensor catalog to bind.</param>
    /// <returns>The bound architecture-specific model.</returns>
    IRwkvModel Bind(IModelTensorCatalog tensors);

    /// <summary>Creates recurrent state for a bound model.</summary>
    /// <param name="model">The model whose dimensions define the state.</param>
    /// <returns>New mutable recurrent state.</returns>
    IRwkvState CreateState(IRwkvModel model);

    /// <summary>Processes one token and writes the resulting logits.</summary>
    /// <param name="model">The bound model to execute.</param>
    /// <param name="token">The input token identifier.</param>
    /// <param name="state">The recurrent state to update.</param>
    /// <param name="logits">The destination for output logits.</param>
    void ForwardToken(IRwkvModel model, int token, IRwkvState state, Span<float> logits);

    void ForwardTokens(IRwkvModel model, ReadOnlySpan<int> tokens, IRwkvState state, Span<float> logits)
    {
        foreach (var token in tokens)
        {
            ForwardToken(model, token, state, logits);
        }
    }
}
