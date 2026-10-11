namespace SharpInference.Instructions;

/// <summary>Filters a provider's collections, implementations, and graph mappings to one architecture.</summary>
/// <param name="provider">The instruction collection provider to filter.</param>
/// <param name="target">The CPU or Direct3D12 VM execution backend.</param>
public sealed class TargetInstructionCollection(IInstructionCollectionProvider provider, InstructionTarget target)
    : IInstructionCollectionProvider, IInstructionOptimizationProvider, IGraphInstructionProvider
{
    /// <summary>Returns descriptions of instruction collections exposed by the provider.</summary>
    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        Array.AsReadOnly(provider.QueryInstructionCollection()
            .Where(collection => collection.Architecture == target).ToArray());
    /// <summary>Returns available target implementations of the specified collection and instruction name.</summary>
    /// <param name="collectionId">The identity of the collection providing this implementation.</param>
    /// <param name="instructionName">The registered instruction name.</param>
    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        Array.AsReadOnly(provider.QueryInstruction(collectionId, instructionName)
            .Where(instruction => instruction.Target == target).ToArray());
    /// <summary>Returns instruction implementations available for graph fusion selection.</summary>
    public IReadOnlyList<InstructionOptimizationCapability> QueryOptimizationCapabilities() =>
        provider is IInstructionOptimizationProvider optimization
            ? Array.AsReadOnly(optimization.QueryOptimizationCapabilities()
                .Where(capability => capability.Target == target).ToArray())
            : [];
    /// <summary>Returns registered mappings from graph operations to target instructions.</summary>
    public IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings() =>
        provider is IGraphInstructionProvider graph
            ? Array.AsReadOnly(graph.QueryGraphInstructionBindings()
                .Where(binding => binding.Target == target).ToArray())
            : [];
}
