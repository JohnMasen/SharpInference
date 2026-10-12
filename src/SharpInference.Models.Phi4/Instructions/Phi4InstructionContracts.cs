using SharpInference.Graphs;

namespace SharpInference.Instructions.Phi4;

/// <summary>Defines versioned graph operation identifiers and instruction bindings for Phi-4 audio processing.</summary>
public static class Phi4AudioGraphOperations
{
    /// <summary>The versioned graph operation identifier for log-mel feature normalization.</summary>
    public static readonly GraphOperationId NormalizeFeatures = new(Phi4AudioInstructionNames.NormalizeFeatures, 1);
    /// <summary>The versioned graph operation identifier for two-dimensional convolution.</summary>
    public static readonly GraphOperationId Conv2D = new(Phi4AudioInstructionNames.Conv2D, 1);
    /// <summary>The versioned graph operation identifier for subsampled feature flattening.</summary>
    public static readonly GraphOperationId FlattenSubsampling = new(Phi4AudioInstructionNames.FlattenSubsampling, 1);
    /// <summary>The versioned graph operation identifier for bias addition and activation.</summary>
    public static readonly GraphOperationId BiasActivation = new(Phi4AudioInstructionNames.BiasActivation, 1);
    /// <summary>The versioned graph operation identifier for one-dimensional convolution.</summary>
    public static readonly GraphOperationId Conv1D = new(Phi4AudioInstructionNames.Conv1D, 1);
    /// <summary>The versioned graph operation identifier for layer normalization.</summary>
    public static readonly GraphOperationId LayerNorm = new(Phi4AudioInstructionNames.LayerNorm, 1);
    /// <summary>The versioned graph operation identifier for swish-gated feed-forward activation.</summary>
    public static readonly GraphOperationId SwiGlu = new(Phi4AudioInstructionNames.SwiGlu, 1);
    /// <summary>The versioned graph operation identifier for scaled residual addition.</summary>
    public static readonly GraphOperationId Residual = new(Phi4AudioInstructionNames.Residual, 1);
    /// <summary>The versioned graph operation identifier for relative-position audio attention.</summary>
    public static readonly GraphOperationId RelativeAttention = new(Phi4AudioInstructionNames.RelativeAttention, 1);

}

/// <summary>Defines versioned graph operation identifiers and instruction bindings for Phi-4 vision processing.</summary>
public static class Phi4VisionGraphOperations
{
    /// <summary>The versioned graph operation identifier for image patch embedding.</summary>
    public static readonly GraphOperationId PatchEmbedding = new(Phi4VisionInstructionNames.PatchEmbedding, 1);
    /// <summary>The versioned graph operation identifier for layer normalization.</summary>
    public static readonly GraphOperationId LayerNorm = new(Phi4VisionInstructionNames.LayerNorm, 1);
    /// <summary>The versioned graph operation identifier for masked vision self-attention.</summary>
    public static readonly GraphOperationId Attention = new(Phi4VisionInstructionNames.Attention, 1);
    /// <summary>The versioned graph operation identifier for tanh-approximated GELU activation.</summary>
    public static readonly GraphOperationId Gelu = new(Phi4VisionInstructionNames.Gelu, 1);
    /// <summary>The versioned graph operation identifier for two-by-two image patch pooling.</summary>
    public static readonly GraphOperationId Pool2x2 = new(Phi4VisionInstructionNames.Pool2x2, 1);
    /// <summary>The versioned graph operation identifier for high-definition image embedding gathering.</summary>
    public static readonly GraphOperationId HdGather = new(Phi4VisionInstructionNames.HdGather, 1);

}

/// <summary>Defines versioned graph operation identifiers and instruction bindings for Phi-4 text decoding.</summary>
public static class Phi4TextGraphOperations
{
    /// <summary>The versioned graph operation identifier for root-mean-square normalization.</summary>
    public static readonly GraphOperationId RmsNorm = new("phi4.text.rms-norm", 1);
    /// <summary>The versioned graph operation identifier for swish-gated feed-forward activation.</summary>
    public static readonly GraphOperationId SwiGlu = new("phi4.text.swiglu", 1);
    /// <summary>The versioned graph operation identifier for rotary encoding and key/value cache writes.</summary>
    public static readonly GraphOperationId RopeKeyValueWrite = new("phi4.text.rope-kv-write", 1);
    /// <summary>The versioned graph operation identifier for grouped-query attention score computation.</summary>
    public static readonly GraphOperationId GroupedQueryScores = new("phi4.text.gqa-scores", 1);
    /// <summary>The versioned graph operation identifier for causal attention softmax.</summary>
    public static readonly GraphOperationId CausalSoftmax = new("phi4.text.causal-softmax", 1);
    /// <summary>The versioned graph operation identifier for grouped-query value accumulation.</summary>
    public static readonly GraphOperationId GroupedQueryValues = new("phi4.text.gqa-values", 1);
    /// <summary>The versioned graph operation identifier for scaled vector addition.</summary>
    public static readonly GraphOperationId ScaledAdd = new("phi4.text.scaled-add", 1);
    /// <summary>The versioned graph operation identifier for decoder position advancement.</summary>
    public static readonly GraphOperationId AdvancePosition = new("phi4.text.advance-position", 1);

}

/// <summary>Defines stable identifiers for the Phi-4 FP32 audio, vision, and text instruction collections.</summary>
public static class Phi4InstructionCollectionIds
{
    /// <summary>The stable collection identifier for Phi-4 FP32 audio instructions.</summary>
    public static readonly Guid AudioFloat32 =
        new("6c4a8c67-0fd8-4a7d-9e99-f5369584e7df");
    /// <summary>The stable collection identifier for Phi-4 FP32 vision instructions.</summary>
    public static readonly Guid VisionFloat32 =
        new("9757d3a4-a4d6-42d8-89e2-1333a38dca4b");
    /// <summary>The stable collection identifier for Phi-4 FP32 text instructions.</summary>
    public static readonly Guid TextFloat32 =
        new("2eb00e68-a9a1-40c1-9c94-65f811be33e2");
}

/// <summary>Defines instruction names for the Phi-4 audio encoder kernels.</summary>
public static class Phi4AudioInstructionNames
{
    /// <summary>The instruction name for log-mel feature normalization.</summary>
    public const string NormalizeFeatures = "phi4.audio.normalize-features";
    /// <summary>The instruction name for one-dimensional convolution.</summary>
    public const string Conv1D = "phi4.audio.conv1d";
    /// <summary>The instruction name for two-dimensional convolution.</summary>
    public const string Conv2D = "phi4.audio.conv2d";
    /// <summary>The instruction name for subsampled feature flattening.</summary>
    public const string FlattenSubsampling = "phi4.audio.flatten-subsampling";
    /// <summary>The instruction name for layer normalization.</summary>
    public const string LayerNorm = "phi4.audio.layer-norm";
    /// <summary>The instruction name for bias addition and activation.</summary>
    public const string BiasActivation = "phi4.audio.bias-activation";
    /// <summary>The instruction name for swish-gated feed-forward activation.</summary>
    public const string SwiGlu = "phi4.audio.swi-glu";
    /// <summary>The instruction name for scaled residual addition.</summary>
    public const string Residual = "phi4.audio.residual";
    /// <summary>The instruction name for relative-position audio attention.</summary>
    public const string RelativeAttention = "phi4.audio.relative-attention";
}

/// <summary>Defines instruction names for the Phi-4 vision encoder kernels.</summary>
public static class Phi4VisionInstructionNames
{
    /// <summary>The instruction name for image patch embedding.</summary>
    public const string PatchEmbedding = "phi4.vision.patch-embedding";
    /// <summary>The instruction name for layer normalization.</summary>
    public const string LayerNorm = "phi4.vision.layer-norm";
    /// <summary>The instruction name for masked vision self-attention.</summary>
    public const string Attention = "phi4.vision.self-attention";
    /// <summary>The instruction name for tanh-approximated GELU activation.</summary>
    public const string Gelu = "phi4.vision.gelu-tanh";
    /// <summary>The instruction name for two-by-two image patch pooling.</summary>
    public const string Pool2x2 = "phi4.vision.pool-2x2";
    /// <summary>The instruction name for high-definition image embedding gathering.</summary>
    public const string HdGather = "phi4.vision.hd-gather";
}
