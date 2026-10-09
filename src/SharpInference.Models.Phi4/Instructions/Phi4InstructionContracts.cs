using SharpInference.Graphs;

namespace SharpInference.Instructions.Phi4;

public static class Phi4AudioGraphOperations
{
    public static readonly GraphOperationId NormalizeFeatures = new(Phi4AudioInstructionNames.NormalizeFeatures, 1);
    public static readonly GraphOperationId Conv2D = new(Phi4AudioInstructionNames.Conv2D, 1);
    public static readonly GraphOperationId FlattenSubsampling = new(Phi4AudioInstructionNames.FlattenSubsampling, 1);
    public static readonly GraphOperationId BiasActivation = new(Phi4AudioInstructionNames.BiasActivation, 1);
    public static readonly GraphOperationId Conv1D = new(Phi4AudioInstructionNames.Conv1D, 1);
    public static readonly GraphOperationId LayerNorm = new(Phi4AudioInstructionNames.LayerNorm, 1);
    public static readonly GraphOperationId SwiGlu = new(Phi4AudioInstructionNames.SwiGlu, 1);
    public static readonly GraphOperationId Residual = new(Phi4AudioInstructionNames.Residual, 1);
    public static readonly GraphOperationId RelativeAttention = new(Phi4AudioInstructionNames.RelativeAttention, 1);

    public static IReadOnlyList<GraphInstructionBinding> Bindings(InstructionTarget target) =>
        [new(NormalizeFeatures, Phi4InstructionCollectionIds.AudioFloat32, Phi4AudioInstructionNames.NormalizeFeatures, target),
         new(Conv2D, Phi4InstructionCollectionIds.AudioFloat32, Phi4AudioInstructionNames.Conv2D, target),
         new(FlattenSubsampling, Phi4InstructionCollectionIds.AudioFloat32, Phi4AudioInstructionNames.FlattenSubsampling, target),
         new(BiasActivation, Phi4InstructionCollectionIds.AudioFloat32, Phi4AudioInstructionNames.BiasActivation, target),
         new(Conv1D, Phi4InstructionCollectionIds.AudioFloat32, Phi4AudioInstructionNames.Conv1D, target),
         new(LayerNorm, Phi4InstructionCollectionIds.AudioFloat32, Phi4AudioInstructionNames.LayerNorm, target),
         new(SwiGlu, Phi4InstructionCollectionIds.AudioFloat32, Phi4AudioInstructionNames.SwiGlu, target),
         new(Residual, Phi4InstructionCollectionIds.AudioFloat32, Phi4AudioInstructionNames.Residual, target),
         new(RelativeAttention, Phi4InstructionCollectionIds.AudioFloat32, Phi4AudioInstructionNames.RelativeAttention, target)];
}

public static class Phi4VisionGraphOperations
{
    public static readonly GraphOperationId PatchEmbedding = new(Phi4VisionInstructionNames.PatchEmbedding, 1);
    public static readonly GraphOperationId LayerNorm = new(Phi4VisionInstructionNames.LayerNorm, 1);
    public static readonly GraphOperationId Attention = new(Phi4VisionInstructionNames.Attention, 1);
    public static readonly GraphOperationId Gelu = new(Phi4VisionInstructionNames.Gelu, 1);
    public static readonly GraphOperationId Pool2x2 = new(Phi4VisionInstructionNames.Pool2x2, 1);
    public static readonly GraphOperationId HdGather = new(Phi4VisionInstructionNames.HdGather, 1);

    public static IReadOnlyList<GraphInstructionBinding> Bindings(InstructionTarget target) =>
        [new(PatchEmbedding, Phi4InstructionCollectionIds.VisionFloat32, Phi4VisionInstructionNames.PatchEmbedding, target),
         new(LayerNorm, Phi4InstructionCollectionIds.VisionFloat32, Phi4VisionInstructionNames.LayerNorm, target),
         new(Attention, Phi4InstructionCollectionIds.VisionFloat32, Phi4VisionInstructionNames.Attention, target),
         new(Gelu, Phi4InstructionCollectionIds.VisionFloat32, Phi4VisionInstructionNames.Gelu, target),
         new(Pool2x2, Phi4InstructionCollectionIds.VisionFloat32, Phi4VisionInstructionNames.Pool2x2, target),
         new(HdGather, Phi4InstructionCollectionIds.VisionFloat32, Phi4VisionInstructionNames.HdGather, target)];
}

public static class Phi4TextGraphOperations
{
    public static readonly GraphOperationId RmsNorm = new("phi4.text.rms-norm", 1);
    public static readonly GraphOperationId SwiGlu = new("phi4.text.swiglu", 1);
    public static readonly GraphOperationId RopeKeyValueWrite = new("phi4.text.rope-kv-write", 1);
    public static readonly GraphOperationId GroupedQueryScores = new("phi4.text.gqa-scores", 1);
    public static readonly GraphOperationId CausalSoftmax = new("phi4.text.causal-softmax", 1);
    public static readonly GraphOperationId GroupedQueryValues = new("phi4.text.gqa-values", 1);
    public static readonly GraphOperationId ScaledAdd = new("phi4.text.scaled-add", 1);
    public static readonly GraphOperationId AdvancePosition = new("phi4.text.advance-position", 1);

    public static IReadOnlyList<GraphInstructionBinding> Bindings(InstructionTarget target) =>
        [new(RmsNorm, Phi4InstructionCollectionIds.TextFloat32, "transformer.rms-norm", target),
         new(SwiGlu, Phi4InstructionCollectionIds.TextFloat32, "transformer.swiglu", target),
         new(RopeKeyValueWrite, Phi4InstructionCollectionIds.TextFloat32, "transformer.rope-kv-write", target,
             ReadWritePorts: ["key_cache", "value_cache"]),
         new(GroupedQueryScores, Phi4InstructionCollectionIds.TextFloat32, "transformer.gqa-scores", target),
         new(CausalSoftmax, Phi4InstructionCollectionIds.TextFloat32, "transformer.causal-softmax", target),
         new(GroupedQueryValues, Phi4InstructionCollectionIds.TextFloat32, "transformer.gqa-values", target),
         new(ScaledAdd, Phi4InstructionCollectionIds.TextFloat32, "transformer.scaled-add", target),
         new(AdvancePosition, Phi4InstructionCollectionIds.TextFloat32, "phi4.text.advance-position", target)];
}

public static class Phi4InstructionCollectionIds
{
    public static readonly Guid AudioFloat32 =
        new("6c4a8c67-0fd8-4a7d-9e99-f5369584e7df");
    public static readonly Guid VisionFloat32 =
        new("9757d3a4-a4d6-42d8-89e2-1333a38dca4b");
    public static readonly Guid TextFloat32 =
        new("2eb00e68-a9a1-40c1-9c94-65f811be33e2");
}

public static class Phi4AudioInstructionNames
{
    public const string NormalizeFeatures = "phi4.audio.normalize-features";
    public const string Conv1D = "phi4.audio.conv1d";
    public const string Conv2D = "phi4.audio.conv2d";
    public const string FlattenSubsampling = "phi4.audio.flatten-subsampling";
    public const string LayerNorm = "phi4.audio.layer-norm";
    public const string BiasActivation = "phi4.audio.bias-activation";
    public const string SwiGlu = "phi4.audio.swi-glu";
    public const string Residual = "phi4.audio.residual";
    public const string RelativeAttention = "phi4.audio.relative-attention";
}

public static class Phi4VisionInstructionNames
{
    public const string PatchEmbedding = "phi4.vision.patch-embedding";
    public const string LayerNorm = "phi4.vision.layer-norm";
    public const string Attention = "phi4.vision.self-attention";
    public const string Gelu = "phi4.vision.gelu-tanh";
    public const string Pool2x2 = "phi4.vision.pool-2x2";
    public const string HdGather = "phi4.vision.hd-gather";
}
