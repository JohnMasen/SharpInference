using SharpInference.Instructions;

namespace SharpInference.Vm;

internal static class VmGpuLaunchContracts
{
    public static void Validate(VmProgram program, IReadOnlyDictionary<string, VmDefinition> definitions)
    {
        var grids = new Dictionary<string, VmThreadGroup>(StringComparer.Ordinal);
        IEnumerable<(VmOperator Operation, VmDefinition Definition)> Operations(VmDefinition definition)
        {
            foreach (var node in definition.Nodes)
                if (node.Instruction is VmOperator operation)
                    yield return (operation, definition);
                else if (node.Instruction is VmCall call)
                    foreach (var item in Operations(definitions[call.Definition]))
                        yield return item;
        }
        foreach (var kernel in program.Definitions.Where(definition => definition.Kind == VmDefinitionKind.Kernel))
        {
            var operations = Operations(kernel).ToArray();
            var cooperativeMatVec = operations.Where(item =>
                item.Operation.ExecutionConfiguration == GpuMatVecExecution.Cooperative).ToArray();
            var cooperativeMatrix = operations.Where(item =>
                item.Operation.ExecutionConfiguration ==
                    GpuMatrixMultiplyExecution.Cooperative).ToArray();
            var tiledMatrix = operations.Where(item =>
                item.Operation.ExecutionConfiguration ==
                    GpuMatrixMultiplyExecution.Tiled).ToArray();
            if (cooperativeMatVec.Length > 0)
            {
                if (operations.Length != 1 ||
                    cooperativeMatVec[0].Operation.Operation != "core.mat-vec" ||
                    cooperativeMatVec[0].Operation.InstructionCollectionId !=
                        InstructionCollectionIds.TierZeroFloat32 &&
                    cooperativeMatVec[0].Operation.InstructionCollectionId !=
                        InstructionCollectionIds.TierZeroFloat16 ||
                    kernel.Threads != new VmThreadGroup(GpuMatVecExecution.Threads))
                    throw new InvalidDataException(
                        $"Kernel '{kernel.Id}' requires one isolated cooperative MatVec and a 64x1x1 thread group.");
                var (operation, definition) = cooperativeMatVec[0];
                var output = operation.Arguments.SingleOrDefault(
                        argument => argument.Parameter == "output")
                    ?? throw new InvalidDataException(
                        $"Kernel '{kernel.Id}' has no MatVec output.");
                var tensor = definition.Parameters.Single(
                    parameter => parameter.Name == output.Source).Tensor;
                var grid = GpuMatVecExecution.Groups(tensor.ElementCount);
                grids.Add(kernel.Id, new(grid.X, grid.Y));
            }
            else if (cooperativeMatrix.Length > 0)
            {
                if (operations.Length != 1 ||
                    cooperativeMatrix[0].Operation.Operation is not (
                        "core.matrix-multiply" or "core.affine") ||
                    cooperativeMatrix[0].Operation.InstructionCollectionId !=
                        InstructionCollectionIds.TierZeroFloat32 ||
                    kernel.Threads != new VmThreadGroup(GpuMatrixMultiplyExecution.Threads))
                    throw new InvalidDataException(
                        $"Kernel '{kernel.Id}' requires one isolated cooperative matrix operation and a 64x1x1 thread group.");
                var (operation, definition) = cooperativeMatrix[0];
                var output = operation.Arguments.SingleOrDefault(
                        argument => argument.Parameter == "output")
                    ?? throw new InvalidDataException(
                        $"Kernel '{kernel.Id}' has no matrix output.");
                var tensor = definition.Parameters.Single(
                    parameter => parameter.Name == output.Source).Tensor;
                var grid = GpuMatrixMultiplyExecution.Groups(tensor.ElementCount);
                grids.Add(kernel.Id, new(grid.X, grid.Y));
            }
            else if (tiledMatrix.Length > 0)
            {
                if (operations.Length != 1 ||
                    tiledMatrix[0].Operation.Operation is not (
                        "core.matrix-multiply" or "core.affine") ||
                    tiledMatrix[0].Operation.InstructionCollectionId !=
                        InstructionCollectionIds.TierZeroFloat32 ||
                    kernel.Threads != new VmThreadGroup(
                        GpuMatrixMultiplyExecution.TileSize,
                        GpuMatrixMultiplyExecution.TileSize))
                    throw new InvalidDataException(
                        $"Kernel '{kernel.Id}' requires one isolated tiled matrix operation and an 8x8x1 thread group.");
                var (operation, definition) = tiledMatrix[0];
                var output = operation.Arguments.SingleOrDefault(
                        argument => argument.Parameter == "output")
                    ?? throw new InvalidDataException(
                        $"Kernel '{kernel.Id}' has no matrix output.");
                var tensor = definition.Parameters.Single(
                    parameter => parameter.Name == output.Source).Tensor;
                if (tensor.Dimensions.Count != 2)
                    throw new InvalidDataException(
                        $"Kernel '{kernel.Id}' requires a rank-two matrix output.");
                var grid = GpuMatrixMultiplyExecution.TileGroups(
                    tensor.Dimensions[0], tensor.Dimensions[1]);
                grids.Add(kernel.Id, new(grid.X, grid.Y));
            }
        }
        foreach (var dispatch in program.Definitions.SelectMany(definition => definition.Nodes)
                     .Select(node => node.Instruction).OfType<VmDispatch>())
            if (grids.TryGetValue(dispatch.Definition, out var grid) && dispatch.Groups != grid)
                throw new InvalidDataException(
                    $"Dispatch of '{dispatch.Definition}' does not match its cooperative grid.");
    }
}
