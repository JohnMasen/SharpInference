namespace SharpInference;

/// <summary>Identifies the floating-point storage format used by a model tensor.</summary>
public enum TensorDataType : uint
{
    /// <summary>IEEE 754 single-precision storage.</summary>
    Float32 = 0,
    /// <summary>IEEE 754 half-precision storage.</summary>
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
        {
            throw new ArgumentException("Model dimensions require names and positive values.", nameof(dimensions));
        }

        if (Attributes.Any(value => string.IsNullOrWhiteSpace(value.Key) || value.Value is null))
        {
            throw new ArgumentException("Model attributes require names and values.", nameof(attributes));
        }
    }

    /// <summary>Gets the architecture identifier used to select model implementations.</summary>
    public string ArchitectureId { get; }
    /// <summary>Gets positive model dimensions keyed by their architecture-defined names.</summary>
    public IReadOnlyDictionary<string, long> Dimensions { get; }
    /// <summary>Gets additional architecture-defined string attributes.</summary>
    public IReadOnlyDictionary<string, string> Attributes { get; }
}

/// <summary>Exposes the metadata and typed values of one model tensor.</summary>
public interface IModelTensor
{
    /// <summary>Gets the tensor's catalog binding name.</summary>
    string Name { get; }
    /// <summary>Gets the element storage format.</summary>
    TensorDataType DataType { get; }
    /// <summary>Gets the ordered tensor dimensions.</summary>
    IReadOnlyList<int> Dimensions { get; }
    /// <summary>Gets the single-precision values when the tensor uses single-precision storage.</summary>
    ReadOnlySpan<float> FloatValues { get; }
    /// <summary>Gets the half-precision values when the tensor uses half-precision storage.</summary>
    ReadOnlySpan<Half> HalfValues { get; }
}

/// <summary>Provides model-wide metadata and named tensor lookup.</summary>
public interface IModelTensorCatalog
{
    /// <summary>Gets the available tensor binding names.</summary>
    IReadOnlyCollection<string> Names { get; }
    /// <summary>Looks up a named tensor without requiring that it exists.</summary>
    /// <param name="name">The catalog binding name.</param>
    /// <param name="tensor">The tensor when found.</param>
    /// <returns>Whether the catalog contains the tensor.</returns>
    bool TryGet(string name, out IModelTensor tensor);
    /// <summary>Returns a required tensor or reports that its binding is missing.</summary>
    /// <param name="name">The catalog binding name.</param>
    /// <returns>The matching tensor.</returns>
    IModelTensor GetRequired(string name);
}

/// <summary>Represents an opened model file and its tensor catalog.</summary>
public interface IModelFile : IModelTensorCatalog, IDisposable
{
    /// <summary>Gets the path of the opened model file.</summary>
    string Path { get; }
}

/// <summary>Opens model files from paths.</summary>
public interface IModelReader
{
    /// <summary>Opens a model file whose lifetime is owned by the caller.</summary>
    /// <param name="path">The model file path.</param>
    /// <returns>The opened tensor catalog.</returns>
    IModelFile Open(string path);
}

/// <summary>Reports whether model weights must have CPU-owned copies.</summary>
public interface IModelWeightOwnershipPolicy
{
    /// <summary>Gets whether binding requires CPU-owned copies instead of borrowing catalog storage.</summary>
    bool RequiresCpuWeightCopy { get; }
}

/// <summary>Prepares a weight plan from a model tensor catalog.</summary>
public interface IModelWeightPreflight
{
    /// <summary>Validates and plans weight bindings before execution resources are created.</summary>
    /// <param name="tensors">The model tensor catalog.</param>
    void PrepareWeightPlan(IModelTensorCatalog tensors);
}

/// <summary>Exposes general metadata for a bound model.</summary>
public interface IModel
{
    /// <summary>Gets the bound model's architecture, dimensions and attributes.</summary>
    ModelMetadata ModelMetadata { get; }
}

/// <summary>Defines model binding and architecture-owned state creation.</summary>
public interface IModelArchitecture
{
    /// <summary>Gets the architecture identifier accepted by this implementation.</summary>
    string Id { get; }
    /// <summary>Checks whether the supplied tensor catalog can be bound.</summary>
    /// <param name="tensors">The candidate model tensors.</param>
    /// <returns>Whether the catalog satisfies the architecture's model contract.</returns>
    bool CanLoad(IModelTensorCatalog tensors);
    /// <summary>Binds compatible tensors to an architecture-owned model.</summary>
    /// <param name="tensors">The model tensors to bind.</param>
    /// <returns>The bound model.</returns>
    IModel Bind(IModelTensorCatalog tensors);
    /// <summary>Creates mutable state matching the bound model.</summary>
    /// <param name="model">The model whose state is required.</param>
    /// <returns>A new independent state instance.</returns>
    IModelState CreateState(IModel model);
}

/// <summary>Represents mutable model state without assuming a tensor layout or element type.</summary>
public interface IModelState
{
    /// <summary>Gets the architecture identifier governing the state layout.</summary>
    string ArchitectureId { get; }
    /// <summary>Creates an independent copy of the current state.</summary>
    /// <returns>The copied state.</returns>
    IModelState Clone();
    /// <summary>Restores the architecture-defined initial state.</summary>
    void Reset();
}
