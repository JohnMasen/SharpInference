
namespace SharpInference.Graphs;

/// <summary>Records compilation-boundary intent on graph definitions, execution graphs, and execution calls.</summary>
/// <remarks>
/// These values are preserved as metadata. The default converter retains definitions and calls for all
/// three values, and explicit execution-graph inlining does not enforce them.
/// </remarks>
public enum GraphCompileBoundary
{
    /// <summary>Leaves the boundary decision unspecified rather than requesting an explicit boundary policy.</summary>
    Auto,
    /// <summary>Records an intent to retain the graph as one compilation unit; this is not an enforced inlining restriction.</summary>
    Atomic,
    /// <summary>Records an intent to expose the graph across its boundary; this does not automatically inline calls.</summary>
    Transparent,
}
