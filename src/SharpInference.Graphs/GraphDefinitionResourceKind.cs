
namespace SharpInference.Graphs;

/// <summary>Classifies a resource's graph role and the access and storage scope assigned during VM emission.</summary>
public enum GraphDefinitionResourceKind
{
    /// <summary>A graph input emitted as read-only session storage.</summary>
    Input,
    /// <summary>A graph output emitted as writable session storage.</summary>
    Output,
    /// <summary>A graph input and output emitted as writable session storage.</summary>
    InOut,
    /// <summary>A writable graph-local intermediate eligible for temporary-resource reuse.</summary>
    Temporary,
    /// <summary>A read-only model weight emitted as shared global storage.</summary>
    Weight,
    /// <summary>Mutable session storage also included in the emitted model state contract.</summary>
    State,
    /// <summary>A read-only constant emitted as shared global storage.</summary>
    Constant,
}
