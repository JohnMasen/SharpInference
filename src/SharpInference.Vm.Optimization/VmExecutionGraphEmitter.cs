using SharpInference.Graphs.Execution;
using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

public sealed record VmExecutionGraphEmitterOptions(
    string? Name = null,
    string Abi = "vm:execution-graph:v1",
    string Entry = "forward",
    string StateSchema = "execution-graph",
    int StateVersion = 1);

/// <summary>Emits VM definitions from selected execution implementations and storage.</summary>
public sealed partial class VmExecutionGraphEmitter
{
    private readonly VmTarget target;
    private readonly ExecutionTarget instructionTarget;
    private readonly InstructionRegistry registry;
    private readonly ExecutionImplementationSelector implementationValidator;

    public VmExecutionGraphEmitter(
        VmTarget target, IEnumerable<IInstructionCollectionProvider> instructionCollections)
    {
        this.target = target;
        instructionTarget = target switch
        {
            VmTarget.Cpu => ExecutionTarget.Cpu,
            VmTarget.Direct3D12 => ExecutionTarget.Direct3D12,
            _ => throw new NotSupportedException($"No VM emitter is installed for target '{target}'."),
        };
        registry = new InstructionRegistry(instructionCollections);
        implementationValidator = new([registry]);
    }

    public IInstructionCollectionProvider InstructionCollections => registry;

    private static VmAccess Access(ResourceAccess access) =>
        access == ResourceAccess.Read ? VmAccess.ReadOnly : VmAccess.ReadWrite;

    private static VmTensor Tensor(TensorDescriptor tensor)
    {
        if (tensor.Layout != "dense")
        {
            throw new NotSupportedException($"VM emission does not support tensor layout '{tensor.Layout}'.");
        }

        return new(tensor.ElementType switch
        {
            ElementType.Byte => VmElementType.Byte,
            ElementType.Int32 => VmElementType.Int32,
            ElementType.UInt32 => VmElementType.UInt32,
            ElementType.Float16 => VmElementType.Float16,
            ElementType.Float32 => VmElementType.Float32,
            _ => throw new NotSupportedException($"VM emission does not support element type '{tensor.ElementType}'."),
        }, tensor.Dimensions);
    }
}
