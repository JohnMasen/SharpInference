using SharpInference.Graphs;
using SharpInference.Vm;

namespace SharpInference.Vm.Optimization;

public static class VmGraphBindingValidator
{
    public static void Validate(VmProgram program, ExecutionGraph graph, bool allowTokenPrefill = false)
    {
        ArgumentNullException.ThrowIfNull(program);
        ArgumentNullException.ThrowIfNull(graph);
        VmProgramValidator.Validate(program);
        if (program.Abi != $"vm:{graph.Model.StateAbiId}" || program.State.Schema != graph.GraphState.Schema.Name ||
            program.State.Version != 1 || program.State.Entries.Count != graph.GraphState.Count)
            throw new InvalidDataException("The VM program and graph have incompatible state ABIs.");
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var slots = program.Slots.ToDictionary(slot => slot.Id, StringComparer.Ordinal);
        foreach (var entry in graph.GraphState)
        {
            var binding = program.State.Entries.SingleOrDefault(candidate => candidate.Name == entry.Name);
            if (binding is null || !slots.TryGetValue(binding.Slot, out var slot) || slot.Scope != VmSlotScope.Session)
                throw new InvalidDataException($"The VM state omitted session resource '{entry.Name}'.");
            ValidateTensor(resources[entry.Resource].Tensor, slot.Tensor, false, entry.Name);
        }
        foreach (var id in graph.Inputs.Concat(graph.Outputs))
        {
            if (!slots.TryGetValue(id.Value, out var slot))
                throw new InvalidDataException($"The VM program omitted graph IO resource '{id}'.");
            var tokenCapacity = allowTokenPrefill && graph.Inputs.Count == 1 && graph.Inputs[0] == id &&
                resources[id].Tensor.ElementType == GraphElementType.Int32 &&
                resources[id].Tensor.Dimensions.SequenceEqual([1]) &&
                program.Entries.Any(entry => entry.Name == "prefill.1");
            ValidateTensor(resources[id].Tensor, slot.Tensor, tokenCapacity, id.Value);
        }
    }

    private static void ValidateTensor(TensorDescriptor expected, VmTensor actual, bool tokenCapacity, string name)
    {
        var elementType = expected.ElementType switch
        {
            GraphElementType.Byte => VmElementType.Byte,
            GraphElementType.Int32 => VmElementType.Int32,
            GraphElementType.UInt32 => VmElementType.UInt32,
            GraphElementType.Float16 => VmElementType.Float16,
            GraphElementType.Float32 => VmElementType.Float32,
            _ => throw new NotSupportedException($"Unsupported graph tensor type '{expected.ElementType}'."),
        };
        if (expected.Layout != "dense" || actual.ElementType != elementType ||
            !(tokenCapacity ? actual.Dimensions.Count == 1 && actual.Dimensions[0] >= 1 :
                actual.Dimensions.SequenceEqual(expected.Dimensions)))
            throw new InvalidDataException($"VM resource '{name}' has an incompatible tensor descriptor.");
    }
}
