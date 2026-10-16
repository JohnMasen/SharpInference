
namespace SharpInference.Graphs;

/// <summary>Common behavior for resource-write edges.</summary>
public interface IResourceWriteEdge : IModelGraphEdge
{
    /// <summary>Gets the bound operation port.</summary>
    string Port { get; }

    /// <summary>Gets the binding shared with a paired read edge.</summary>
    string BindingId { get; }
}
