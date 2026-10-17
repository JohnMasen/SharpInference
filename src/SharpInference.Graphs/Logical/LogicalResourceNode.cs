using SharpInference.Instructions;
using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Logical;

/// <summary>A resource declaration viewed in a parent or call-instance scope.</summary>
/// <param name="ResourceNodeId">The resource identity qualified by its binding owner.</param>
/// <param name="Resource">The underlying declaration, whose graph owner is independent of that scope.</param>
public sealed record LogicalResourceNode(
    ModelResourceNodeId ResourceNodeId,
    GraphDefinitionResource Resource) : LogicalGraphNode, IModelResourceNode
{
    /// <summary>Gets the graph identity of the scoped resource.</summary>
    public override GraphNodeId GraphId => ResourceNodeId.GraphId;
    /// <summary>Gets the tensor descriptor of the underlying declaration.</summary>
    public TensorDescriptor Tensor => Resource.Tensor;
    /// <summary>Gets the role of the underlying declaration.</summary>
    public GraphDefinitionResourceKind Kind => Resource.Kind;
}
