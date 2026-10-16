
namespace SharpInference.Graphs.Logical;

/// <summary>Pairs a validated logical definition with its typed definition-port contract.</summary>
/// <typeparam name="TContract">The self-creating public port contract.</typeparam>
public sealed class GraphDefinition<TContract> where TContract : IGraphPortContract<TContract>
{
    /// <summary>Stores a definition and its already-created contract.</summary>
    /// <param name="definition">The logical definition.</param>
    /// <param name="ports">The definition-level port handles.</param>
    internal GraphDefinition(LogicalGraphDefinition definition, TContract ports)
    {
        Definition = definition;
        Ports = ports;
    }

    /// <summary>Gets the validated logical definition.</summary>
    public LogicalGraphDefinition Definition { get; }
    /// <summary>Gets the public ports declared on this definition, rather than a call instance.</summary>
    public TContract Ports { get; }
}
