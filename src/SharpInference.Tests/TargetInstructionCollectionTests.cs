using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;

namespace SharpInference.Tests;

public sealed class TargetInstructionCollectionTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphBindingsAreForwardedOnlyForSelectedTarget(bool gpu)
    {
        var target = gpu ? InstructionTarget.Direct3D12 : InstructionTarget.Cpu;
        var provider = new GraphProvider();
        var filtered = new TargetInstructionCollection(provider, target);
        var registry = new InstructionRegistry([filtered]);
        var binding = Assert.Single(registry.QueryGraphInstructionBindings());
        var original = provider.Bindings.Single(value => value.Target == target);
        Assert.Same(original, binding);
        Assert.Same(original.Dispatch, binding.Dispatch);
        Assert.Equal(new GraphInstructionDispatch(8, 1, 1, 2, 1, 1),
            binding.Dispatch!(new(new Dictionary<string, TensorDescriptor>(), new Dictionary<string, string>())));
    }

    [Fact]
    public void ProviderWithoutGraphMappingsReturnsEmptyBindings()
    {
        var filtered = new TargetInstructionCollection(new CpuFloat32InstructionCollection(), InstructionTarget.Cpu);
        Assert.Empty(filtered.QueryGraphInstructionBindings());
    }

    private sealed class GraphProvider : IInstructionCollectionProvider, IGraphInstructionProvider
    {
        public IReadOnlyList<GraphInstructionBinding> Bindings { get; } =
            new[] { InstructionTarget.Cpu, InstructionTarget.Direct3D12 }.Select(target =>
                new GraphInstructionBinding(new("test.target-mapping", 1), Guid.NewGuid(), "test.kernel", target,
                    _ => new(8, 1, 1, 2, 1, 1))).ToArray();

        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
            Bindings.Select(binding => new InstructionCollectionDescription(binding.CollectionId,
                "Target mapping test", 2, binding.Target)).ToArray();
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
            Bindings.Where(binding => binding.CollectionId == collectionId && binding.InstructionName == instructionName)
                .Select(binding => (Instruction)new TestInstruction(binding)).ToArray();
        public IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings() => Bindings;
    }

    private sealed class TestInstruction(GraphInstructionBinding binding) : Instruction
    {
        public override Guid CollectionId => binding.CollectionId;
        public override string Name => binding.InstructionName;
        public override InstructionTarget Target => binding.Target;
        public override IReadOnlyList<InstructionSignature> Signatures =>
            [new([], [], new(GraphElementType.Float32, GraphElementType.Float32))];
        protected override InstructionRecording Generate(InstructionParameter[] parameters) => new("");
    }
}
