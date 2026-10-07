namespace SharpInference.Instructions.Phi4;

public static class Phi4InstructionCollectionIds
{
    public static readonly Guid AudioFloat32 =
        new("6c4a8c67-0fd8-4a7d-9e99-f5369584e7df");
    public static readonly Guid VisionFloat32 =
        new("9757d3a4-a4d6-42d8-89e2-1333a38dca4b");
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
