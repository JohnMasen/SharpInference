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
            var cooperative = operations.Where(item =>
                item.Operation.ExecutionConfiguration == GpuMatVecExecution.Cooperative).ToArray();
            if (cooperative.Length == 0) continue;
            if (operations.Length != 1 || cooperative[0].Operation.Operation != "core.mat-vec" ||
                cooperative[0].Operation.InstructionCollectionId != InstructionCollectionIds.TierZeroFloat32 &&
                cooperative[0].Operation.InstructionCollectionId != InstructionCollectionIds.TierZeroFloat16 ||
                kernel.Threads != new VmThreadGroup(GpuMatVecExecution.Threads))
                throw new InvalidDataException($"Kernel '{kernel.Id}' requires one isolated cooperative MatVec and a 64x1x1 thread group.");
            var (operation, definition) = cooperative[0];
            var output = operation.Arguments.SingleOrDefault(argument => argument.Parameter == "output")
                ?? throw new InvalidDataException($"Kernel '{kernel.Id}' has no MatVec output.");
            var tensor = definition.Parameters.Single(parameter => parameter.Name == output.Source).Tensor;
            var grid = GpuMatVecExecution.Groups(tensor.ElementCount);
            grids.Add(kernel.Id, new(grid.X, grid.Y));
        }
        foreach (var dispatch in program.Definitions.SelectMany(definition => definition.Nodes)
                     .Select(node => node.Instruction).OfType<VmDispatch>())
            if (grids.TryGetValue(dispatch.Definition, out var grid) && dispatch.Groups != grid)
                throw new InvalidDataException($"Dispatch of '{dispatch.Definition}' does not match its cooperative MatVec grid.");
    }
}
