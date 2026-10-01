namespace SharpInference;

public enum RwkvTensorDataType : uint
{
    Float32 = 0,
    Float16 = 1,
}

public sealed record RwkvModelMetadata(
    int VocabularySize,
    int EmbeddingSize,
    int LayerCount,
    int HeadCount,
    int HeadSize,
    string ArchitectureId);

public interface IModelTensor
{
    string Name { get; }
    RwkvTensorDataType DataType { get; }
    IReadOnlyList<int> Dimensions { get; }
    ReadOnlySpan<float> FloatValues { get; }
    ReadOnlySpan<Half> HalfValues { get; }
}

public interface IModelTensorCatalog
{
    int VocabularySize { get; }
    int EmbeddingSize { get; }
    int LayerCount { get; }
    IReadOnlyCollection<string> Names { get; }
    bool TryGet(string name, out IModelTensor tensor);
    IModelTensor GetRequired(string name);
}

public interface IModelFile : IModelTensorCatalog, IDisposable
{
    string Path { get; }
}

public interface IModelReader
{
    IModelFile Open(string path);
}

public interface IModelWeightOwnershipPolicy
{
    bool RequiresCpuWeightCopy { get; }
}

public interface IModelWeightPreflight
{
    void PrepareWeightPlan(IModelTensorCatalog tensors);
}

public interface IRwkvModel
{
    RwkvModelMetadata Metadata { get; }
}

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
        {
            ForwardToken(model, token, state, logits);
        }
    }
}
