using SharpInference.Graphs;
using SharpInference.Instructions.D3D12;

namespace SharpInference.Instructions.Phi4.D3D12;

public sealed class Phi4D3D12TextInstructionCollection : IInstructionCollectionProvider, IGraphInstructionProvider
{
    private readonly GpuTransformerInstructionCollection transformer = new();
    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [new(Phi4InstructionCollectionIds.TextFloat32, "Phi4 FP32 decoder operations", 1, InstructionTarget.Direct3D12)];

    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == Phi4InstructionCollectionIds.TextFloat32
            ? transformer.QueryInstruction(InstructionCollectionIds.TransformerFloat32, instructionName)
                .Select(instruction => (Instruction)new TextInstruction(instruction)).ToArray()
            : [];

    public IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings() =>
        Phi4TextGraphOperations.Bindings(InstructionTarget.Direct3D12)
            .Where(binding => binding.Operation != Phi4TextGraphOperations.AdvancePosition)
            .Select(binding => binding with { Dispatch = context => Dispatch(context, binding.Operation) }).ToArray();

    private static GraphInstructionDispatch Dispatch(GraphInstructionDispatchContext c, GraphOperationId operation)
    {
        ulong work;
        if (operation == Phi4TextGraphOperations.RmsNorm) work = 1;
        else if (operation == Phi4TextGraphOperations.CausalSoftmax) work = (ulong)c.Tensors["scores"].Dimensions[0];
        else
        {
            var tensor = c.Tensors[operation == Phi4TextGraphOperations.RopeKeyValueWrite ? "qkv" :
                operation == Phi4TextGraphOperations.GroupedQueryScores ? "scores" : "output"];
            var count = tensor.Dimensions.Aggregate(1UL, (total, size) => checked(total * (ulong)size));
            work = (count + 63) / 64;
        }
        var groups = GpuMatVecExecution.Groups(work);
        return new(64, 1, 1, groups.X, groups.Y, 1);
    }

    private sealed class TextInstruction(Instruction inner) : Instruction
    {
        public override Guid CollectionId => Phi4InstructionCollectionIds.TextFloat32;
        public override string Name => inner.Name;
        public override InstructionTarget Target => inner.Target;
        public override IReadOnlyList<InstructionSignature> Signatures => inner.Signatures.Select(signature =>
            new InstructionSignature(signature.Ports.Select(port =>
                Name == "transformer.rope-kv-write" && port.Name is "key_cache" or "value_cache"
                    ? port with { Access = GraphResourceAccess.ReadWrite } : port).ToArray(), signature.Attributes, signature.Precision)).ToArray();
        public override IReadOnlyList<InstructionIndexBound> IndexBounds => Name switch
        {
            "transformer.rope-kv-write" or "transformer.gqa-scores" => [new("position", "key_cache", 0)],
            "transformer.gqa-values" => [new("position", "value_cache", 0)],
            "transformer.causal-softmax" => [new("position", "scores", 1)],
            _ => [],
        };

        protected override InstructionRecording Generate(InstructionParameter[] parameters)
        {
            var recorder = new Recorder();
            inner.Invoke(recorder, parameters, new(GraphElementType.Float32, GraphElementType.Float32));
            return recorder.Recording ?? throw new InvalidOperationException("The transformer instruction recorded no shader.");
        }
    }

    private sealed class Recorder : IInstructionRecorder
    {
        public InstructionTarget Target => InstructionTarget.Direct3D12;
        public InstructionRecording? Recording { get; private set; }
        public void Record(InstructionRecording recording)
        {
            if (Recording is not null) throw new InvalidOperationException("Expected a single transformer recording.");
            Recording = recording;
        }
    }
}
