
namespace SharpInference.Graphs.Execution;

/// <summary>Describes a model-level resource independently of its owning scope in an execution graph.</summary>
/// <param name="Id">The resource identity within its scope.</param>
/// <param name="Name">The resource's descriptive name.</param>
/// <param name="Kind">The resource's interface or temporary classification.</param>
/// <param name="Tensor">The tensor type, dimensions, and layout.</param>
/// <param name="BindingKey">An optional external binding key.</param>
public sealed record ExecutionGraphResource(
    ResourceId Id,
    string Name,
    GraphDefinitionResourceKind Kind,
    TensorDescriptor Tensor,
    string? BindingKey = null);
