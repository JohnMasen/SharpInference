namespace SharpInference.Graphs.Core;

/// <summary>Factories for node and edge constraints.</summary>
public static class PatternElements
{
    /// <summary>Creates a node constraint with no identity, name, or type restriction.</summary>
    /// <returns>A constraint accepting any visible node.</returns>
    public static NodePattern Node() => new(null, _ => true);

    /// <summary>Creates a node constraint anchored to an exact structural identity.</summary>
    /// <param name="id">The node identity, not its display name or a capture alias.</param>
    /// <returns>A constraint that permits direct identity lookup during matching.</returns>
    /// <exception cref="ArgumentException">The identifier is null, empty, or whitespace.</exception>
    public static NodePattern Node(GraphNodeId id)
    {
        if (string.IsNullOrWhiteSpace(id.Value))
            throw new ArgumentException("A node identifier is required.", nameof(id));
        return new(id, node => node.Id == id);
    }

    /// <summary>Creates a constraint matching the actual name of a named node.</summary>
    /// <param name="name">The node name to compare using ordinal, case-sensitive equality.</param>
    /// <returns>A constraint accepting only named nodes with the specified name.</returns>
    /// <remarks>
    /// The string overload matches names even though strings can convert to node identifiers.
    /// Supply a GraphNodeId to match by identity; this argument is not a capture alias.
    /// </remarks>
    /// <exception cref="ArgumentException">The name is null, empty, or whitespace.</exception>
    public static NodePattern Node(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("A node name is required.", nameof(name));
        return new(null, node => node is INamedGraphNode named &&
            string.Equals(named.Name, name, StringComparison.Ordinal));
    }

    /// <summary>Creates a constraint accepting any single direct, directed edge.</summary>
    /// <returns>An unrestricted edge constraint, not a variable-length path constraint.</returns>
    public static EdgePattern Edge() => new(_ => true);
}
