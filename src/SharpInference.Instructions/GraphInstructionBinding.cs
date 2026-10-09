using SharpInference.Graphs;

namespace SharpInference.Instructions;

public sealed record GraphInstructionBinding(
    GraphOperationId Operation, Guid CollectionId, string InstructionName, InstructionTarget Target,
    Func<GraphInstructionDispatchContext, GraphInstructionDispatch>? Dispatch = null,
    IReadOnlyList<string>? ReadWritePorts = null);

public sealed record GraphInstructionDispatchContext(
    IReadOnlyDictionary<string, TensorDescriptor> Tensors, IReadOnlyDictionary<string, string> Attributes);

public sealed record GraphInstructionDispatch(
    uint ThreadsX, uint ThreadsY, uint ThreadsZ, uint GroupsX, uint GroupsY, uint GroupsZ);

/// <summary>Maps model-owned graph operations to explicitly registered target instructions.</summary>
public interface IGraphInstructionProvider
{
    IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings();
}
