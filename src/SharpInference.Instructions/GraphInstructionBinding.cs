using SharpInference.Graphs.Execution;
using SharpInference.Graphs;

namespace SharpInference.Instructions;

/// <summary>Maps a versioned graph operation to a target instruction and optional dispatch policy.</summary>
/// <param name="Operation">The versioned graph operation mapped to the target instruction.</param>
/// <param name="CollectionId">The identity of the collection providing this implementation.</param>
/// <param name="InstructionName">The registered instruction name.</param>
/// <param name="Target">The execution architecture required by this contract.</param>
/// <param name="Dispatch">The optional policy computing thread-group and dispatch dimensions.</param>
/// <param name="ReadWritePorts">The ports requiring both reads and writes, or null when there are none.</param>
public sealed record GraphInstructionBinding(
    OperationId Operation, Guid CollectionId, string InstructionName, ExecutionTarget Target,
    Func<GraphInstructionDispatchContext, ExecutionDispatch>? Dispatch = null,
    IReadOnlyList<string>? ReadWritePorts = null);

/// <summary>Supplies tensor descriptors and attributes to a graph operation's dispatch policy.</summary>
/// <param name="Tensors">The tensor descriptor associated with each graph port.</param>
/// <param name="Attributes">The instruction attributes supplied to code generation.</param>
public sealed record GraphInstructionDispatchContext(
    IReadOnlyDictionary<string, TensorDescriptor> Tensors, IReadOnlyDictionary<string, string> Attributes);

/// <summary>Maps model-owned graph operations to explicitly registered target instructions.</summary>
public interface IGraphInstructionProvider
{
    /// <summary>Returns registered mappings from graph operations to target instructions.</summary>
    IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings();
}
