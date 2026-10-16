using SharpInference.Graphs.Logical;

namespace SharpInference.Graphs;

/// <summary>Identifies either the containing graph or one direct graph-call instance as a binding scope.</summary>
/// <param name="Call">The direct call identity, or null for the containing graph.</param>
public readonly record struct GraphBindingOwner(GraphDefinitionNodeId? Call)
{
    /// <summary>Gets the containing-graph scope, represented by a null call identity.</summary>
    public static GraphBindingOwner Parent => new(null);
    /// <summary>Creates a binding scope for one direct graph-call instance.</summary>
    /// <param name="call">The call identity to store without performing graph membership validation.</param>
    /// <returns>A scope whose call identity is the supplied value.</returns>
    public static GraphBindingOwner Instance(GraphDefinitionNodeId call) => new(call);
    /// <summary>Gets whether no call identity is set and the binding belongs to the containing graph.</summary>
    public bool IsParent => Call is null;
    /// <summary>Formats the call identity, or the literal parent marker for containing-graph bindings.</summary>
    /// <returns>The stored call's value, or <c>parent</c>.</returns>
    public override string ToString() => Call?.Value ?? "parent";
}
