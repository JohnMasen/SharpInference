using System.Collections.ObjectModel;

namespace SharpInference.Graphs.Execution;

/// <summary>Stores model and backend identities with ordinal, read-only snapshots of compilation metadata.</summary>
public sealed class ExecutionGraphCompilationContext
{
    /// <summary>Creates a compilation context, copying each supplied dictionary.</summary>
    /// <param name="modelType">The nonblank model type identifier.</param>
    /// <param name="backend">The nonblank backend identifier.</param>
    /// <param name="modelAttributes">Optional model metadata; null produces an empty dictionary.</param>
    /// <param name="backendAttributes">Optional backend metadata; null produces an empty dictionary.</param>
    /// <param name="options">Optional compilation options; null produces an empty dictionary.</param>
    /// <exception cref="ArgumentException">The model type or backend is null, empty, or whitespace.</exception>
    public ExecutionGraphCompilationContext(
        string modelType,
        string backend,
        IReadOnlyDictionary<string, string>? modelAttributes = null,
        IReadOnlyDictionary<string, string>? backendAttributes = null,
        IReadOnlyDictionary<string, string>? options = null)
    {
        ModelType = Require(modelType, nameof(modelType));
        Backend = Require(backend, nameof(backend));
        ModelAttributes = Copy(modelAttributes);
        BackendAttributes = Copy(backendAttributes);
        Options = Copy(options);
    }

    /// <summary>Gets the model type identifier.</summary>
    public string ModelType { get; }
    /// <summary>Gets the backend identifier.</summary>
    public string Backend { get; }
    /// <summary>Gets the read-only model metadata snapshot.</summary>
    public IReadOnlyDictionary<string, string> ModelAttributes { get; }
    /// <summary>Gets the read-only backend metadata snapshot.</summary>
    public IReadOnlyDictionary<string, string> BackendAttributes { get; }
    /// <summary>Gets the read-only compilation options snapshot.</summary>
    public IReadOnlyDictionary<string, string> Options { get; }

    /// <summary>Checks that a context identifier contains non-whitespace text without trimming it.</summary>
    /// <param name="value">The identifier to validate.</param>
    /// <param name="parameter">The parameter name used in validation diagnostics.</param>
    /// <returns>The original identifier.</returns>
    /// <exception cref="ArgumentException">The value is null, empty, or whitespace.</exception>
    private static string Require(string value, string parameter) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException("A nonempty value is required.", parameter)
            : value;

    /// <summary>Copies optional metadata into an ordinal dictionary wrapped as read-only.</summary>
    /// <param name="values">The source metadata, or null for an empty snapshot.</param>
    /// <returns>A read-only dictionary independent of the source dictionary.</returns>
    private static IReadOnlyDictionary<string, string> Copy(
        IReadOnlyDictionary<string, string>? values) =>
        new ReadOnlyDictionary<string, string>(
            new Dictionary<string, string>(
                values ?? new Dictionary<string, string>(),
                StringComparer.Ordinal));
}
