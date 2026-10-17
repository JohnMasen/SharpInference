using SharpInference.Instructions;

namespace SharpInference.Graphs.Logical;

/// <summary>Describes a tensor resource declared by a logical graph.</summary>
/// <param name="Id">The resource identity within its declaring graph.</param>
/// <param name="Name">The resource display name.</param>
/// <param name="Kind">The resource's public-port, temporary, or read-only role.</param>
/// <param name="Tensor">The element type, layout, and dimensions.</param>
/// <param name="Owner">The declaring graph identity, not a containing call-instance scope.</param>
/// <param name="BindingKey">The optional external binding key.</param>
public sealed record GraphDefinitionResource(
    ResourceId Id,
    string Name,
    GraphDefinitionResourceKind Kind,
    TensorDescriptor Tensor,
    GraphDefinitionId Owner,
    string? BindingKey = null);
