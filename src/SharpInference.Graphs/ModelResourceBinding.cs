using SharpInference.Instructions;

namespace SharpInference.Graphs;

/// <summary>One resource access reconstructed from graph edges.</summary>
/// <param name="Port">The operation port shared by read and write edges with this binding identity.</param>
/// <param name="Resource">The scoped resource accessed by the operation.</param>
/// <param name="Access">Read, write, or read/write when both edge directions form a paired binding.</param>
/// <param name="BindingId">The identity used to pair accesses and order reconstructed bindings.</param>
public sealed record ModelResourceBinding(
    string Port,
    ModelResourceNodeId Resource,
    ResourceAccess Access,
    string BindingId);
