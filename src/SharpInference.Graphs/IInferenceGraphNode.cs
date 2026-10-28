using SharpInference.Graphs.Core;

namespace SharpInference.Graphs;

/// <summary>Identifies a node and its meaning in an inference graph.</summary>
public interface IInferenceGraphNode : IGraphNode
{
    /// <summary>Gets the category interpreted and validated by the containing graph.</summary>
    string NodeKind
    {
        get;
    }
}
