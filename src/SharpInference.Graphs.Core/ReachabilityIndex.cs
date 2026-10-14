namespace SharpInference.Graphs.Core;

/// <summary>A fixed, endpoint-scoped index of zero-or-more-step directed reachability.</summary>
/// <typeparam name="TNode">The node type accepted by node-instance queries.</typeparam>
public sealed class ReachabilityIndex<TNode> where TNode : IGraphNode
{
    private readonly Dictionary<GraphNodeId, ulong[]> rows;
    private readonly Dictionary<GraphNodeId, int> columns;

    /// <summary>Creates an index from precomputed source bit rows and target bit positions.</summary>
    /// <param name="rows">The reachability bit row for each indexed source identity.</param>
    /// <param name="columns">The bit position assigned to each indexed target identity.</param>
    internal ReachabilityIndex(
        Dictionary<GraphNodeId, ulong[]> rows,
        Dictionary<GraphNodeId, int> columns)
    {
        this.rows = rows;
        this.columns = columns;
    }

    /// <summary>Gets the number of distinct indexed source identities.</summary>
    public int SourceCount => rows.Count;
    /// <summary>Gets the number of distinct indexed target identities.</summary>
    public int TargetCount => columns.Count;

    /// <summary>Tests a precomputed reachability relation using structural identities.</summary>
    /// <param name="source">An identity in the index's source scope.</param>
    /// <param name="target">An identity in the index's target scope.</param>
    /// <returns>Whether a zero-or-more-edge path existed when the index was built.</returns>
    /// <exception cref="KeyNotFoundException">The source or target is outside its indexed scope.</exception>
    public bool IsReachable(GraphNodeId source, GraphNodeId target)
    {
        if (!rows.TryGetValue(source, out var row))
            throw new KeyNotFoundException($"Source '{source}' is outside the reachability index.");
        if (!columns.TryGetValue(target, out var column))
            throw new KeyNotFoundException($"Target '{target}' is outside the reachability index.");
        return (row[column / 64] & (1UL << (column % 64))) != 0;
    }

    /// <summary>Tests reachability by extracting the structural identities of two nodes.</summary>
    /// <param name="source">The node whose identity must belong to the source scope.</param>
    /// <param name="target">The node whose identity must belong to the target scope.</param>
    /// <returns>Whether a zero-or-more-edge path existed when the index was built.</returns>
    /// <exception cref="ArgumentNullException">Either node is null.</exception>
    /// <exception cref="KeyNotFoundException">Either identity is outside its indexed scope.</exception>
    public bool IsReachable(TNode source, TNode target)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(target);
        return IsReachable(source.Id, target.Id);
    }
}
