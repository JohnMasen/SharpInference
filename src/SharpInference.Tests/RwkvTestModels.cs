using System.Reflection;

namespace SharpInference.Tests;

/// <summary>Resolves the shared external model directory for RWKV tests and discovery.</summary>
internal static class RwkvTestModels
{
    /// <summary>Gets the local fallback directory used when no build property is configured.</summary>
    internal const string DefaultModelDirectory = @"D:\GithubRoot\rwkvcpp\tests";

    /// <summary>Gets the explicit directory embedded by the test project's build property.</summary>
    internal static string? ConfiguredDirectory
    {
        get;
    } = typeof(RwkvTestModels).Assembly
        .GetCustomAttributes<AssemblyMetadataAttribute>()
        .SingleOrDefault(attribute => attribute.Key == "RwkvTestModelDirectory")?.Value;

    /// <summary>Gets the discovery-time skip reason, or null when tests should execute.</summary>
    internal static string? SkipReason => GetSkipReason(ConfiguredDirectory, DefaultModelDirectory);

    /// <summary>Gets the validated shared directory before executing a model test.</summary>
    /// <returns>An existing absolute model directory.</returns>
    /// <exception cref="InvalidOperationException">No usable model directory is available.</exception>
    internal static string GetDirectory() => GetDirectory(ConfiguredDirectory, DefaultModelDirectory);

    /// <summary>Determines whether absent local model configuration should skip discovery.</summary>
    /// <param name="configuredDirectory">The explicit build property, if supplied.</param>
    /// <param name="defaultDirectory">The fallback directory to use without an explicit property.</param>
    /// <returns>A skip reason only when no explicit path is supplied and the fallback is absent.</returns>
    internal static string? GetSkipReason(string? configuredDirectory, string defaultDirectory) =>
        ResolveDirectory(configuredDirectory, defaultDirectory) is null
            ? $"RWKV model tests skipped: default directory '{defaultDirectory}' does not exist. " +
                "Set the RwkvTestModelDirectory build property to an existing absolute directory."
            : null;

    /// <summary>Resolves and validates a model directory without hiding explicit configuration errors.</summary>
    /// <param name="configuredDirectory">The explicit build property, if supplied.</param>
    /// <param name="defaultDirectory">The fallback directory to use without an explicit property.</param>
    /// <returns>The existing absolute directory selected by the configuration.</returns>
    /// <exception cref="InvalidOperationException">The selected directory is absent or not absolute.</exception>
    internal static string GetDirectory(string? configuredDirectory, string defaultDirectory)
    {
        var directory = ResolveDirectory(configuredDirectory, defaultDirectory);
        if (directory is null || !Path.IsPathFullyQualified(directory) || !Directory.Exists(directory))
        {
            throw new InvalidOperationException(
                $"RwkvTestModelDirectory must identify an existing absolute directory. " +
                $"Selected directory: '{directory ?? defaultDirectory}'.");
        }

        return directory;
    }

    /// <summary>Selects the explicit path or an available fallback without validating explicit input.</summary>
    /// <param name="configuredDirectory">The explicit build property, if supplied.</param>
    /// <param name="defaultDirectory">The fallback directory used only without an explicit property.</param>
    /// <returns>The selected path, or null when the unconfigured fallback is absent.</returns>
    private static string? ResolveDirectory(string? configuredDirectory, string defaultDirectory) =>
        !string.IsNullOrWhiteSpace(configuredDirectory)
            ? configuredDirectory
            : Directory.Exists(defaultDirectory) ? defaultDirectory : null;
}
