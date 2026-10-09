namespace SharpInference;

/// <summary>Identifies the floating-point storage format used by a model tensor.</summary>
public enum TensorDataType : uint
{
    Float32 = 0,
    Float16 = 1,
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
    TensorDataType DataType { get; }
    IReadOnlyList<int> Dimensions { get; }
    ReadOnlySpan<float> FloatValues { get; }
    ReadOnlySpan<Half> HalfValues { get; }
}

/// <summary>Provides model-wide metadata and named tensor lookup.</summary>
public interface IModelTensorCatalog
{
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

/// <summary>Defines model binding and architecture-owned state creation.</summary>
public interface IModelArchitecture
{
    string Id { get; }
    bool CanLoad(IModelTensorCatalog tensors);
    IModel Bind(IModelTensorCatalog tensors);
    IModelState CreateState(IModel model);
}

/// <summary>Represents mutable model state without assuming a tensor layout or element type.</summary>
public interface IModelState
{
    string ArchitectureId { get; }
    IModelState Clone();
    void Reset();
}
