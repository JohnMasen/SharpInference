using SharpInference.Graphs.Core;

namespace SharpInference.Graphs;

/// <summary>Identifies a directed relationship and its meaning in an inference graph.</summary>
public interface IInferenceGraphEdge : IGraphEdge
{
    /// <summary>Gets the category interpreted and validated by the containing graph.</summary>
    string EdgeKind { get; }
}
