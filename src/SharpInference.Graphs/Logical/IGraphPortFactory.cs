
namespace SharpInference.Graphs.Logical;

/// <summary>Creates typed public-port handles for a definition or an instantiated call.</summary>
public interface IGraphPortFactory
{
    /// <summary>Requests a public read-only input port.</summary>
    /// <typeparam name="T">The resource type marker, independent of the tensor element type.</typeparam>
    /// <param name="id">The public port identifier.</param>
    /// <param name="tensor">The required element type, layout, and dimensions.</param>
    /// <returns>The input port handle in the factory's scope.</returns>
    GraphInput<T> AddInput<T>(string id, TensorDescriptor tensor);
    /// <summary>Requests a public write-only output port.</summary>
    /// <typeparam name="T">The resource type marker, independent of the tensor element type.</typeparam>
    /// <param name="id">The public port identifier.</param>
    /// <param name="tensor">The required element type, layout, and dimensions.</param>
    /// <returns>The output port handle in the factory's scope.</returns>
    GraphOutput<T> AddOutput<T>(string id, TensorDescriptor tensor);
    /// <summary>Requests a public input/output port with read/write access.</summary>
    /// <typeparam name="T">The resource type marker, independent of the tensor element type.</typeparam>
    /// <param name="id">The public port identifier.</param>
    /// <param name="tensor">The required element type, layout, and dimensions.</param>
    /// <returns>The input/output port handle in the factory's scope.</returns>
    GraphInOut<T> AddInOut<T>(string id, TensorDescriptor tensor);
}
