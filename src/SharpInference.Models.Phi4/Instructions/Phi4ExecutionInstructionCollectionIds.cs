using SharpInference.Graphs.Execution;

namespace SharpInference.Instructions;

/// <summary>
/// Defines the GPU-only Phi-4 execution protocol, not a general transformer instruction set.
/// Cache tensors use [context, keyValueHeads, headSize]. Scalar position selects the current cache token.
/// Batched control is Int32 [startPosition, validCount], affecting causal positions and cache writes.
/// No CPU implementation of this protocol is provided.
/// </summary>
public static class Phi4ExecutionInstructionCollectionIds
{
    public static readonly Guid Float32 = new("6e233c7f-1312-4c0f-bf54-8f493d7cd405");

    /// <summary>No CPU implementation currently implements this position/cache/control protocol.</summary>
    public static IReadOnlyList<ExecutionTarget> SupportedTargets { get; } =
        Array.AsReadOnly(new[] { ExecutionTarget.Direct3D12 });

    public static IReadOnlyList<string> RequiredInstructions { get; } = Array.AsReadOnly(new[]
    {
        "transformer.rope-kv-write",
        "transformer.gqa-scores",
        "transformer.causal-softmax",
        "transformer.gqa-values",
        "transformer.grouped-query-attention",
        "transformer.batch-rope-kv-write",
        "transformer.batch-gqa-scores",
        "transformer.batch-causal-softmax",
        "transformer.batch-gqa-values",
        "transformer.batch-select-last",
        "transformer.batch-gather-rows",
    });

    public static InstructionCollectionDescription Describe() =>
        new(Float32, "Phi-4 FP32 position/cache/control protocol (GPU only)", 1,
            ExecutionTarget.Direct3D12, "phi4", InstructionCategory.Specialized, RequiredInstructions);
}
