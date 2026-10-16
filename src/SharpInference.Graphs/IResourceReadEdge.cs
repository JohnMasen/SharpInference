
namespace SharpInference.Graphs;

/// <summary>Common behavior for resource-read edges.</summary>
public interface IResourceReadEdge : IModelGraphEdge
{
    /// <summary>Gets the bound operation port.</summary>
    string Port { get; }

    /// <summary>Gets the binding shared with a paired write edge.</summary>
    string BindingId { get; }
}
