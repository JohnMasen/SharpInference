namespace SharpInference.Graphs.Core;

/// <summary>A node constraint, optionally anchored to an exact identity.</summary>
public sealed class NodePattern
{
    /// <summary>Creates a node constraint from its optional identity anchor and predicate.</summary>
    /// <param name="id">The exact identity anchor, or null to enumerate visible nodes.</param>
    /// <param name="predicate">The combined condition evaluated for candidate nodes.</param>
    /// <param name="unrestricted">Whether every visible node is accepted without evaluating a condition.</param>
    internal NodePattern(GraphNodeId? id, Func<IGraphNode, bool> predicate, bool unrestricted = false)
    {
        Id = id;
        Predicate = predicate;
        Unrestricted = unrestricted;
    }

    /// <summary>Gets the optional identity anchor used to avoid a full node scan.</summary>
    internal GraphNodeId? Id { get; }
    /// <summary>Gets the combined condition for a candidate node.</summary>
    internal Func<IGraphNode, bool> Predicate { get; }
    /// <summary>Gets whether this constraint accepts every visible node without an identity anchor.</summary>
    internal bool Unrestricted { get; }

    /// <summary>Adds a type restriction and predicate without modifying the original constraint.</summary>
    /// <typeparam name="TRequested">The node type required by the additional condition.</typeparam>
    /// <param name="predicate">The condition invoked only for nodes of the requested type.</param>
    /// <returns>A new constraint combining the existing and additional conditions with logical AND.</returns>
    /// <exception cref="ArgumentNullException">The predicate is null.</exception>
    public NodePattern Where<TRequested>(Func<TRequested, bool> predicate)
        where TRequested : IGraphNode
    {
        ArgumentNullException.ThrowIfNull(predicate);
        return new(Id, node => Predicate(node) &&
            node is TRequested requested && predicate(requested));
    }
}
