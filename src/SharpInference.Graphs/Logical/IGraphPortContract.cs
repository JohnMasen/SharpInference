
namespace SharpInference.Graphs.Logical;

/// <summary>Creates the same typed public-port shape for a definition and each of its calls.</summary>
/// <typeparam name="TSelf">The implementing contract type returned by its factory method.</typeparam>
public interface IGraphPortContract<TSelf> where TSelf : IGraphPortContract<TSelf>
{
    /// <summary>Creates a contract by requesting its ports from the supplied factory.</summary>
    /// <param name="ports">The factory that declares definition ports or resolves call-instance ports.</param>
    /// <returns>The typed contract exposing the requested port handles.</returns>
    static abstract TSelf Create(IGraphPortFactory ports);
}
