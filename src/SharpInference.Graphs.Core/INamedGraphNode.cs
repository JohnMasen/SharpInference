namespace SharpInference.Graphs.Core;

/// <summary>Optional name contract used by named node patterns.</summary>
public interface INamedGraphNode : IGraphNode
{
    /// <summary>Gets the name used by name-based patterns, independently of the node identity.</summary>
    string Name { get; }
}
