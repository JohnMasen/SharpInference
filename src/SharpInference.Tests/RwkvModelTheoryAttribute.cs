namespace SharpInference.Tests;

/// <summary>Skips external RWKV theories when neither explicit configuration nor local models are available.</summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class RwkvModelTheoryAttribute : TheoryAttribute
{
    /// <summary>Sets the shared model availability reason during test discovery.</summary>
    public RwkvModelTheoryAttribute()
    {
        Skip = RwkvTestModels.SkipReason;
    }
}
