namespace SharpInference.Graphs.Core;

/// <summary>A constraint on one direct, directed edge.</summary>
public sealed class EdgePattern
{
    /// <summary>Creates a direct-edge constraint from a predicate.</summary>
    /// <param name="predicate">The combined condition evaluated for candidate edges.</param>
    internal EdgePattern(Func<IGraphEdge, bool> predicate) => Predicate = predicate;
    /// <summary>Gets the combined condition for a candidate edge.</summary>
    internal Func<IGraphEdge, bool> Predicate
    {
        get;
    }

    /// <summary>Adds a type restriction and predicate without modifying the original constraint.</summary>
    /// <typeparam name="TRequested">The edge type required by the additional condition.</typeparam>
    /// <param name="predicate">The condition invoked only for edges of the requested type.</param>
    /// <returns>A new constraint combining the existing and additional conditions with logical AND.</returns>
    /// <exception cref="ArgumentNullException">The predicate is null.</exception>
    public EdgePattern Where<TRequested>(Func<TRequested, bool> predicate)
        where TRequested : IGraphEdge
    {
        ArgumentNullException.ThrowIfNull(predicate);
        return new(edge => Predicate(edge) &&
            edge is TRequested requested && predicate(requested));
    }
}
