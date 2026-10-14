namespace SharpInference.Graphs.Core;

/// <summary>The public entry point for creating a linear pattern.</summary>
public static class Pattern
{
    /// <summary>Creates an immutable linear pattern beginning with one node constraint.</summary>
    /// <param name="first">The initial node constraint.</param>
    /// <returns>A complete single-node pattern that can be extended with an edge constraint.</returns>
    /// <exception cref="ArgumentNullException">The initial constraint is null.</exception>
    public static MatchContext Create(NodePattern first)
    {
        ArgumentNullException.ThrowIfNull(first);
        return MatchContext.Start(first);
    }
}
