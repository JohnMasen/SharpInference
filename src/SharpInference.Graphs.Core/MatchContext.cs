namespace SharpInference.Graphs.Core;

/// <summary>An immutable sequence of alternating node and edge constraints.</summary>
public sealed class MatchContext
{
    private readonly NodePattern[] nodes;
    private readonly EdgePattern[] edges;

    /// <summary>Creates a pattern from alternating node and edge constraints.</summary>
    /// <param name="nodes">The node constraints in path order.</param>
    /// <param name="edges">The intervening edge constraints in path order.</param>
    private MatchContext(NodePattern[] nodes, EdgePattern[] edges)
    {
        this.nodes = nodes;
        this.edges = edges;
    }

    /// <summary>Creates a complete single-node pattern.</summary>
    /// <param name="first">The initial node constraint.</param>
    /// <returns>A pattern containing the supplied node constraint and no edges.</returns>
    internal static MatchContext Start(NodePattern first) => new([first], []);
    /// <summary>Gets node constraints in path order.</summary>
    internal IReadOnlyList<NodePattern> Nodes => nodes;
    /// <summary>Gets edge constraints in path order.</summary>
    internal IReadOnlyList<EdgePattern> Edges => edges;
    /// <summary>Gets whether the sequence ends with a node and is ready for matching.</summary>
    internal bool Complete => nodes.Length == edges.Length + 1;

    /// <summary>Appends an edge constraint to a pattern that ends with a node.</summary>
    /// <param name="context">The original complete pattern, which remains unchanged.</param>
    /// <param name="edge">The direct-edge constraint to append.</param>
    /// <returns>A new incomplete pattern requiring a trailing node constraint.</returns>
    /// <exception cref="ArgumentNullException">The pattern or edge constraint is null.</exception>
    /// <exception cref="ArgumentException">The original pattern already ends with an edge.</exception>
    public static MatchContext operator -(MatchContext context, EdgePattern edge)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(edge);
        if (!context.Complete)
        {
            throw new ArgumentException("An edge must follow a node.", nameof(context));
        }

        return new(context.nodes, [.. context.edges, edge]);
    }

    /// <summary>Appends a node constraint to a pattern that ends with an edge.</summary>
    /// <param name="context">The original incomplete pattern, which remains unchanged.</param>
    /// <param name="node">The node constraint to append.</param>
    /// <returns>A new complete pattern ready for matching or further extension.</returns>
    /// <exception cref="ArgumentNullException">The pattern or node constraint is null.</exception>
    /// <exception cref="ArgumentException">The original pattern already ends with a node.</exception>
    public static MatchContext operator -(MatchContext context, NodePattern node)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(node);
        if (context.Complete)
        {
            throw new ArgumentException("A node must follow an edge.", nameof(context));
        }

        return new([.. context.nodes, node], context.edges);
    }
}
