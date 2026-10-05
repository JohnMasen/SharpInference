using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

public static class GpuMatVecReferencePrograms
{
    public static VmProgram Create(int rows, int columns, bool cooperative,
        VmElementType matrixType = VmElementType.Float32, VmElementType inputType = VmElementType.Float32,
        VmElementType outputType = VmElementType.Float32, int paddingElements = 0,
        bool weightPort = false, uint serialThreads = 64)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(paddingElements);
        var matrixPort = weightPort ? "weight" : "matrix";
        VmParameter[] parameters =
        [
            new(matrixPort, VmAccess.ReadOnly, new(matrixType, [rows, columns])),
            new("input", VmAccess.ReadOnly, new(inputType, [columns])),
            new("output", VmAccess.ReadWrite, new(outputType, [rows])),
        ];
        var configuration = cooperative ? GpuMatVecExecution.Cooperative : GpuMatVecExecution.Serial;
        var operation = new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32),
            outputType == VmElementType.Float16 ? InstructionCollectionIds.TierZeroFloat16 : InstructionCollectionIds.TierZeroFloat32,
            "core.mat-vec", parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)),
            executionConfiguration: configuration);
        var slots = parameters.Select(parameter => new VmSlot(parameter.Name,
            parameter.Name == matrixPort ? VmSlotScope.Global : VmSlotScope.Local, parameter.Access,
            paddingElements == 0 ? parameter.Tensor : new(parameter.Tensor.ElementType,
                [checked((int)parameter.Tensor.ElementCount + paddingElements * 2)]))).ToArray();
        var arguments = parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name,
            checked((ulong)paddingElements * (ulong)parameter.Tensor.ElementSize))).ToArray();
        var grid = cooperative ? GpuMatVecExecution.Groups((ulong)rows) :
            (checked((uint)(((ulong)rows + serialThreads - 1) / serialThreads)), 1u);
        var kernel = new VmDefinition("matvec", VmDefinitionKind.Kernel, parameters,
            [new("body", operation)], new(cooperative ? GpuMatVecExecution.Threads : serialThreads));
        var rootParameters = slots.Select(slot => new VmParameter(slot.Id, slot.Access, slot.Tensor)).ToArray();
        var forward = new VmDefinition("forward", VmDefinitionKind.Orchestration, rootParameters,
            [new("compute", new VmDispatch(kernel.Id, arguments, new(grid.Item1, grid.Item2))),
             new("barrier", new VmBarrier(["output"]), ["compute"])]);
        return new("matvec-reference", "matvec.fp32", VmTarget.Direct3D12, slots, [kernel, forward],
            [new("forward", "forward", rootParameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)))],
            new("none", 1, []));
    }
}
