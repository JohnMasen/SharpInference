namespace SharpInference.Instructions;

public sealed class TargetInstructionCollection(IInstructionCollectionProvider provider, InstructionTarget target)
    : IInstructionCollectionProvider, IInstructionOptimizationProvider, IGraphInstructionProvider
{
    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        Array.AsReadOnly(provider.QueryInstructionCollection()
            .Where(collection => collection.Architecture == target).ToArray());
    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        Array.AsReadOnly(provider.QueryInstruction(collectionId, instructionName)
            .Where(instruction => instruction.Target == target).ToArray());
    public IReadOnlyList<InstructionOptimizationCapability> QueryOptimizationCapabilities() =>
        provider is IInstructionOptimizationProvider optimization
            ? Array.AsReadOnly(optimization.QueryOptimizationCapabilities()
                .Where(capability => capability.Target == target).ToArray())
            : [];
    public IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings() =>
        provider is IGraphInstructionProvider graph
            ? Array.AsReadOnly(graph.QueryGraphInstructionBindings()
                .Where(binding => binding.Target == target).ToArray())
            : [];
}
