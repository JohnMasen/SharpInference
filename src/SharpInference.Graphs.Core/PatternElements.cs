namespace SharpInference.Graphs.Core;

/// <summary>Factories for node and edge constraints.</summary>
public static class PatternElements
{
    /// <summary>Creates a node constraint with no identity or type restriction.</summary>
    /// <returns>A constraint accepting any visible node.</returns>
    public static NodePattern Node() => new(null, _ => true, unrestricted: true);

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

    /// <summary>Creates a constraint accepting any single direct, directed edge.</summary>
    /// <returns>An unrestricted edge constraint, not a variable-length path constraint.</returns>
    public static EdgePattern Edge() => new(_ => true);
}
