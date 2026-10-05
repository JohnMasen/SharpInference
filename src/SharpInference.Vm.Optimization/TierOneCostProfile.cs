using System.Numerics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

public sealed class TierOneCostMeasurement
{
    [JsonConstructor]
    public TierOneCostMeasurement(string operation, string implementationFingerprint, string shape,
        string inputAliases, uint threadsPerGroup, bool reuseLocalStorage,
        IReadOnlyList<double> referenceMicroseconds, IReadOnlyList<double> optimizedMicroseconds)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(operation);
        ArgumentException.ThrowIfNullOrWhiteSpace(implementationFingerprint);
        ArgumentException.ThrowIfNullOrWhiteSpace(shape);
        ArgumentException.ThrowIfNullOrWhiteSpace(inputAliases);
        ArgumentNullException.ThrowIfNull(referenceMicroseconds);
        ArgumentNullException.ThrowIfNull(optimizedMicroseconds);
        if (threadsPerGroup is 0 or > 1024 || referenceMicroseconds.Count < 7 || optimizedMicroseconds.Count < 7 ||
            referenceMicroseconds.Concat(optimizedMicroseconds).Any(value => !double.IsFinite(value) || value <= 0))
            throw new ArgumentException("Costs require at least seven positive finite samples per implementation and a valid thread group.");
        Operation = operation;
        ImplementationFingerprint = implementationFingerprint;
        Shape = shape;
        InputAliases = inputAliases;
        ThreadsPerGroup = threadsPerGroup;
        ReuseLocalStorage = reuseLocalStorage;
        ReferenceMicroseconds = Array.AsReadOnly(referenceMicroseconds.ToArray());
        OptimizedMicroseconds = Array.AsReadOnly(optimizedMicroseconds.ToArray());
    }

    public string Operation { get; }
    public string ImplementationFingerprint { get; }
    public string Shape { get; }
    public string InputAliases { get; }
    public uint ThreadsPerGroup { get; }
    public bool ReuseLocalStorage { get; }
    public IReadOnlyList<double> ReferenceMicroseconds { get; }
    public IReadOnlyList<double> OptimizedMicroseconds { get; }
    [JsonIgnore]
    public double ConservativeSavingMicroseconds =>
        Quantile(ReferenceMicroseconds, 0.25) * 0.98 - Quantile(OptimizedMicroseconds, 0.75) * 1.02;
    [JsonIgnore]
    public string Key => $"{Operation}|{ImplementationFingerprint}|{Shape}|{InputAliases}|{ThreadsPerGroup}|{ReuseLocalStorage}";

    private static double Quantile(IReadOnlyList<double> samples, double fraction)
    {
        var sorted = samples.Order().ToArray();
        return sorted[(int)((sorted.Length - 1) * fraction)];
    }
}

public sealed class TierOneCostProfile
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    };

    [JsonConstructor]
    public TierOneCostProfile(string environmentFingerprint, DateTimeOffset measuredAt,
        IReadOnlyList<TierOneCostMeasurement> measurements)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(environmentFingerprint);
        ArgumentNullException.ThrowIfNull(measurements);
        if (measuredAt == default || measurements.Any(measurement => measurement is null) ||
            measurements.Select(measurement => measurement.Key).Distinct(StringComparer.Ordinal).Count() != measurements.Count)
            throw new ArgumentException("Cost profiles require a timestamp and unique nonnull measurement keys.");
        EnvironmentFingerprint = environmentFingerprint;
        MeasuredAt = measuredAt;
        Measurements = Array.AsReadOnly(measurements.ToArray());
    }

    public string EnvironmentFingerprint { get; }
    public DateTimeOffset MeasuredAt { get; }
    public IReadOnlyList<TierOneCostMeasurement> Measurements { get; }
    public string Serialize() => JsonSerializer.Serialize(this, JsonOptions);
    public static TierOneCostProfile Deserialize(string json) =>
        JsonSerializer.Deserialize<TierOneCostProfile>(json, JsonOptions) ??
        throw new InvalidDataException("Missing offline cost profile.");
}

public static class TierOneEnvironmentFingerprint
{
    public static string Create(string hardwareIdentity)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(hardwareIdentity);
        var identity = string.Join("|", hardwareIdentity, Environment.MachineName,
            RuntimeInformation.OSDescription, RuntimeInformation.ProcessArchitecture,
            RuntimeInformation.FrameworkDescription, Environment.ProcessorCount, Vector<float>.Count,
            System.Runtime.GCSettings.IsServerGC, typeof(VmGraphOptimizer).Assembly.ManifestModule.ModuleVersionId,
            typeof(VmProgram).Assembly.ManifestModule.ModuleVersionId,
            Environment.GetEnvironmentVariable("DOTNET_TieredCompilation"),
            Environment.GetEnvironmentVariable("DOTNET_TieredPGO"),
            Environment.GetEnvironmentVariable("DOTNET_EnableHWIntrinsic"),
            Environment.GetEnvironmentVariable("DOTNET_EnableAVX2"),
            Environment.GetEnvironmentVariable("DOTNET_EnableAVX512F"));
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(identity)));
    }
}

public sealed class TierOneOptimizationSettings
{
    public TierOneOptimizationSettings(TierOneCostProfile profile, string environmentFingerprint,
        int maximumSearchStates = 100_000, TimeSpan? maximumProfileAge = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentException.ThrowIfNullOrWhiteSpace(environmentFingerprint);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumSearchStates);
        Profile = profile;
        EnvironmentFingerprint = environmentFingerprint;
        MaximumSearchStates = maximumSearchStates;
        MaximumProfileAge = maximumProfileAge ?? TimeSpan.FromDays(30);
        if (MaximumProfileAge <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(maximumProfileAge));
    }

    public TierOneCostProfile Profile { get; }
    public string EnvironmentFingerprint { get; }
    public int MaximumSearchStates { get; }
    public TimeSpan MaximumProfileAge { get; }
}

public enum TierOneSelectionStatus { BaselineRetained, ModelOptimal, BestFoundWithinBudget }

public sealed record TierOneOptimizationReport(
    TierOneSelectionStatus Status, string CostModel, string Diagnostic, int MatchedCandidates,
    int TrustedCandidates, int SelectedCandidates, int SearchStates, double ConservativeSavingMicroseconds,
    int OriginalNodeCount, int OptimizedNodeCount);

public sealed record VmOptimizationResult(VmProgram Program, TierOneOptimizationReport Report,
    GpuMatVecOptimizationReport? MatVecReport = null);
