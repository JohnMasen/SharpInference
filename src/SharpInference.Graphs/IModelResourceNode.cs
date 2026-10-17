using SharpInference.Instructions;

namespace SharpInference.Graphs;

/// <summary>Common behavior for resource nodes.</summary>
public interface IModelResourceNode : IModelGraphNode
{
    /// <summary>Gets the scoped resource identity.</summary>
    ModelResourceNodeId ResourceNodeId { get; }

    /// <summary>Gets the tensor contract.</summary>
    TensorDescriptor Tensor { get; }

    /// <summary>Gets the resource role.</summary>
    GraphDefinitionResourceKind Kind { get; }
}
