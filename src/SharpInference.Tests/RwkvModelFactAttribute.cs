namespace SharpInference.Tests;

/// <summary>Skips external RWKV facts when neither explicit configuration nor local models are available.</summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class RwkvModelFactAttribute : FactAttribute
{
    /// <summary>Sets the shared model availability reason during test discovery.</summary>
    public RwkvModelFactAttribute()
    {
        Skip = RwkvTestModels.SkipReason;
    }
}
