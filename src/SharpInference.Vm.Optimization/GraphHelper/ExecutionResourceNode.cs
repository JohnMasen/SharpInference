using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Graphs.Core;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Places a resource description in a parent or call-instance scope.</summary>
/// <param name="ResourceNodeId">The owner-qualified resource identity used by bindings.</param>
/// <param name="Resource">The resource's tensor and interface metadata.</param>
public sealed record ExecutionResourceNode(
    ModelResourceNodeId ResourceNodeId,
    ExecutionGraphResource Resource)
    : ExecutionGraphStructureNode, IModelResourceNode
{
    /// <summary>Gets the structural identity derived from the owner-qualified resource identity.</summary>
    public override GraphNodeId GraphId => ResourceNodeId.GraphId;
    /// <summary>Gets the tensor descriptor from the resource description.</summary>
    public TensorDescriptor Tensor => Resource.Tensor;
    /// <summary>Gets the interface or temporary classification from the resource description.</summary>
    public GraphDefinitionResourceKind Kind => Resource.Kind;
}
