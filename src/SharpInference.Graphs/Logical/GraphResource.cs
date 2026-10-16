
namespace SharpInference.Graphs.Logical;

/// <summary>A typed handle over a logical resource declaration.</summary>
/// <typeparam name="T">A resource type marker that is not checked against the tensor element type.</typeparam>
public class GraphResource<T>
{
    /// <summary>Wraps a resource declaration without copying or validating it.</summary>
    /// <param name="resource">The declaration to expose.</param>
    internal GraphResource(GraphDefinitionResource resource) => Resource = resource;
    /// <summary>Gets the underlying declaration, including its declaring graph owner.</summary>
    public GraphDefinitionResource Resource { get; }
    /// <summary>Gets the declaration's resource identity.</summary>
    public ResourceId Id => Resource.Id;
    /// <summary>Gets the declaration's tensor contract.</summary>
    public TensorDescriptor Tensor => Resource.Tensor;
}
