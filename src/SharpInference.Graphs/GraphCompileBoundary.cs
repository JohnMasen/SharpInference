
namespace SharpInference.Graphs;

/// <summary>Expresses the logical author's compilation-boundary intent.</summary>
/// <remarks>Optimizers determine concrete transformations independently of the recorded intent.</remarks>
public enum GraphCompileBoundary
{
    /// <summary>Delegates the boundary decision to the backend optimizer.</summary>
    Auto,
    /// <summary>Expresses an intent to retain the graph as one compilation unit.</summary>
    Atomic,
    /// <summary>Expresses an intent to expose the graph body to cross-boundary optimization.</summary>
    Transparent,
}
