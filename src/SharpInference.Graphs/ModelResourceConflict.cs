
namespace SharpInference.Graphs;

/// <summary>One pair of operations with conflicting access to a resource.</summary>
/// <typeparam name="TNode">The model operation node type returned by the queried view.</typeparam>
/// <param name="Resource">The shared resource whose accesses conflict.</param>
/// <param name="First">The earlier operation in the queried node declaration order.</param>
/// <param name="Second">The later operation in the queried node declaration order.</param>
/// <param name="Kind">The combined conflict roles relative to First and Second, not execution order.</param>
/// <param name="Ordered">Whether a visible dependency path connects the operations in either direction.</param>
public sealed record ModelResourceConflict<TNode>(
    IModelResourceNode Resource,
    TNode First,
    TNode Second,
    ModelResourceConflictKind Kind,
    bool Ordered)
    where TNode : IModelGraphNode;
