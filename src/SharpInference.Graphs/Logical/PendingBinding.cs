
namespace SharpInference.Graphs.Logical;

/// <summary>A configured resource access awaiting conversion to logical graph edges.</summary>
/// <param name="Port">The operation or call port name.</param>
/// <param name="Resource">The referenced declaration.</param>
/// <param name="Access">Whether to emit a read edge, a write edge, or both.</param>
/// <param name="Owner">The containing parent's scope or the referenced call-instance scope.</param>
internal sealed record PendingBinding(
    string Port,
    GraphDefinitionResource Resource,
    GraphResourceAccess Access,
    GraphBindingOwner Owner);
