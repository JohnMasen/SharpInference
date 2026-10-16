namespace SharpInference.Tests;

/// <summary>Verifies local model defaults, explicit overrides, and discovery-time skipping.</summary>
public sealed class RwkvTestModelsTests
{
    /// <summary>Checks that an available default enables facts and theories without a build property.</summary>
    [Fact]
    public void AvailableDefaultRunsWithoutExplicitConfiguration()
    {
        var directory = Path.GetTempPath();
        Assert.Equal(directory, RwkvTestModels.GetDirectory(null, directory));
        Assert.Equal(directory, RwkvTestModels.GetDirectory(" ", directory));
        Assert.Null(RwkvTestModels.GetSkipReason(null, directory));
    }

    /// <summary>Checks that missing defaults produce an actionable skip reason rather than a passing no-op.</summary>
    [Fact]
    public void MissingDefaultSkipsWhenNotExplicitlyConfigured()
    {
        var missing = Path.Combine(Path.GetTempPath(), $"rwkv-missing-{Guid.NewGuid():N}");
        var reason = RwkvTestModels.GetSkipReason(null, missing);
        Assert.NotNull(reason);
        Assert.Contains(missing, reason);
        Assert.Contains("RwkvTestModelDirectory", reason);
        Assert.Throws<InvalidOperationException>(() => RwkvTestModels.GetDirectory(null, missing));
    }

    /// <summary>Checks that explicit configuration takes priority over the default.</summary>
    [Fact]
    public void ExplicitDirectoryOverridesDefault()
    {
        var directory = Path.GetTempPath();
        var missing = Path.Combine(directory, $"rwkv-missing-{Guid.NewGuid():N}");
        Assert.Equal(directory, RwkvTestModels.GetDirectory(directory, missing));
        Assert.Null(RwkvTestModels.GetSkipReason(directory, missing));
    }

    /// <summary>Checks that invalid overrides fail execution rather than falling back or skipping.</summary>
    [Fact]
    public void InvalidExplicitDirectoryFailsInsteadOfSkipping()
    {
        var directory = Path.GetTempPath();
        var missing = Path.Combine(directory, $"rwkv-missing-{Guid.NewGuid():N}");
        foreach (var configured in new[] { missing, "relative-model-directory" })
        {
            Assert.Null(RwkvTestModels.GetSkipReason(configured, directory));
            var exception = Assert.Throws<InvalidOperationException>(
                () => RwkvTestModels.GetDirectory(configured, directory));
            Assert.Contains(configured, exception.Message);
        }
    }

    /// <summary>Checks that both xUnit attributes use the shared discovery configuration.</summary>
    [Fact]
    public void ModelAttributesShareTheDirectoryResolver()
    {
        Assert.Equal(RwkvTestModels.SkipReason, new RwkvModelFactAttribute().Skip);
        Assert.Equal(RwkvTestModels.SkipReason, new RwkvModelTheoryAttribute().Skip);
    }
}
