using SharpInference.Graphs.Core;

namespace SharpInference.Graphs;

/// <summary>Identifies one resource node in a containing graph or direct call instance.</summary>
/// <param name="Owner">The containing-graph or direct call-instance binding scope.</param>
/// <param name="Resource">The resource identity within that scope.</param>
public readonly record struct ModelResourceNodeId(
    GraphBindingOwner Owner,
    ResourceId Resource)
{
    /// <summary>Gets a structural identity prefixed with the parent scope or the owning call identity.</summary>
    /// <remarks>Parent and instance resources use distinct prefixes even when their resource names match.</remarks>
    public GraphNodeId GraphId => Owner.IsParent
        ? new($"resource:parent:{Resource.Value}")
        : new($"resource:instance:{Owner.Call!.Value.Value}:{Resource.Value}");

    /// <inheritdoc/>
    public override string ToString() => Owner.IsParent
        ? Resource.Value
        : $"{Owner}/{Resource.Value}";
}
