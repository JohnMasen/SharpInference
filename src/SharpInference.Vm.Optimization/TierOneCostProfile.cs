using System.Numerics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

/// <summary>Stores baseline and optimized timing samples for a specific implementation and launch domain.</summary>
public sealed class TierOneCostMeasurement
{
    /// <summary>Snapshots at least seven positive finite timing samples per implementation and validates the launch domain.</summary>
    /// <param name="operation">The instruction name whose baseline and optimized costs were measured.</param>
    /// <param name="implementationFingerprint">The implementation identity used to match trusted measurements.</param>
    /// <param name="shape">The serialized tensor-shape and element-type key identifying this measurement domain.</param>
    /// <param name="inputAliases">The canonical input alias pattern used to match measurement domains.</param>
    /// <param name="threadsPerGroup">The number of GPU threads in each dispatch group.</param>
    /// <param name="reuseLocalStorage">Whether compatible non-overlapping temporary lifetimes share storage.</param>
    /// <param name="referenceMicroseconds">The positive baseline execution-time samples in microseconds.</param>
    /// <param name="optimizedMicroseconds">The positive optimized execution-time samples in microseconds.</param>
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

    /// <summary>Gets the instruction name whose baseline and optimized costs were measured.</summary>
    public string Operation { get; }
    /// <summary>Gets the implementation identity used to match trusted measurements.</summary>
    public string ImplementationFingerprint { get; }
    /// <summary>Gets the tensor-shape and element-type key for this measurement domain.</summary>
    public string Shape { get; }
    /// <summary>Gets the canonical input alias pattern used to match measurement domains.</summary>
    public string InputAliases { get; }
    /// <summary>Gets the number of GPU threads in each dispatch group.</summary>
    public uint ThreadsPerGroup { get; }
    /// <summary>Gets whether compatible non-overlapping temporary lifetimes share storage.</summary>
    public bool ReuseLocalStorage { get; }
    /// <summary>Gets the positive baseline execution-time samples in microseconds.</summary>
    public IReadOnlyList<double> ReferenceMicroseconds { get; }
    /// <summary>Gets the positive optimized execution-time samples in microseconds.</summary>
    public IReadOnlyList<double> OptimizedMicroseconds { get; }
    /// <summary>Gets the estimated savings in microseconds using pessimistic timing quantiles and margins.</summary>
    [JsonIgnore]
    public double ConservativeSavingMicroseconds =>
        Quantile(ReferenceMicroseconds, 0.25) * 0.98 - Quantile(OptimizedMicroseconds, 0.75) * 1.02;
    /// <summary>Gets the composite implementation, shape, alias, launch, and storage measurement key.</summary>
    [JsonIgnore]
    public string Key => $"{Operation}|{ImplementationFingerprint}|{Shape}|{InputAliases}|{ThreadsPerGroup}|{ReuseLocalStorage}";

    private static double Quantile(IReadOnlyList<double> samples, double fraction)
    {
        var sorted = samples.Order().ToArray();
        return sorted[(int)((sorted.Length - 1) * fraction)];
    }
}

/// <summary>Stores timestamped offline optimization measurements for a particular environment.</summary>
public sealed class TierOneCostProfile
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    };

    /// <summary>Creates a timestamped profile with unique nonnull measurement keys.</summary>
    /// <param name="environmentFingerprint">The current measurement environment identity required to trust offline costs.</param>
    /// <param name="measuredAt">The timestamp at which the offline profile was measured.</param>
    /// <param name="measurements">The measurements with unique implementation and launch-domain keys.</param>
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

    /// <summary>Gets the current measurement environment identity required to trust offline costs.</summary>
    public string EnvironmentFingerprint { get; }
    /// <summary>Gets the timestamp at which the offline profile was measured.</summary>
    public DateTimeOffset MeasuredAt { get; }
    /// <summary>Gets the measurements with unique implementation and launch-domain keys.</summary>
    public IReadOnlyList<TierOneCostMeasurement> Measurements { get; }
    /// <summary>Serializes the offline cost profile as indented JSON.</summary>
    public string Serialize() => JsonSerializer.Serialize(this, JsonOptions);
    /// <summary>Parses an offline cost profile from JSON, rejecting unknown fields and invalid measurements.</summary>
    /// <param name="json">The JSON containing an offline cost profile.</param>
    public static TierOneCostProfile Deserialize(string json) =>
        JsonSerializer.Deserialize<TierOneCostProfile>(json, JsonOptions) ??
        throw new InvalidDataException("Missing offline cost profile.");
}

/// <summary>Hashes hardware, runtime, build, and JIT settings that affect optimization measurements.</summary>
public static class TierOneEnvironmentFingerprint
{
    /// <summary>Hashes hardware identity and current machine, runtime, module, SIMD, GC, and JIT settings.</summary>
    /// <param name="hardwareIdentity">The nonempty hardware identity supplied by the caller.</param>
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

/// <summary>Configures trusted offline costs, profile freshness, and a bounded fusion search.</summary>
public sealed class TierOneOptimizationSettings
{
    /// <summary>Configures profile trust and a positive search budget, defaulting profile freshness to thirty days.</summary>
    /// <param name="profile">The offline timing profile used for cost-driven selection.</param>
    /// <param name="environmentFingerprint">The current measurement environment identity required to trust offline costs.</param>
    /// <param name="maximumSearchStates">The maximum number of fusion selection states to explore.</param>
    /// <param name="maximumProfileAge">The positive maximum trusted profile age; null selects thirty days.</param>
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

    /// <summary>Gets the offline timing profile used for cost-driven selection.</summary>
    public TierOneCostProfile Profile { get; }
    /// <summary>Gets the current measurement environment identity required to trust offline costs.</summary>
    public string EnvironmentFingerprint { get; }
    /// <summary>Gets the maximum number of fusion selection states to explore.</summary>
    public int MaximumSearchStates { get; }
    /// <summary>Gets the maximum age of a trusted offline cost profile.</summary>
    public TimeSpan MaximumProfileAge { get; }
}

/// <summary>Indicates whether the baseline or a cost-model-selected fusion plan was retained.</summary>
public enum TierOneSelectionStatus
{
    /// <summary>Retains the unfused graph because no trusted beneficial selection was found.</summary>
    BaselineRetained,
    /// <summary>Completes the search and selects the optimum under the supplied cost model.</summary>
    ModelOptimal,
    /// <summary>Retains the best selection found before exhausting the search budget.</summary>
    BestFoundWithinBudget
}

/// <summary>Reports fusion matching, cost-model trust, search effort, and estimated conservative savings.</summary>
/// <param name="Status">The outcome of the bounded cost-model selection search.</param>
/// <param name="CostModel">The identifier of the cost model used for selection.</param>
/// <param name="Diagnostic">The selection or failure diagnostic.</param>
/// <param name="MatchedCandidates">The number of structurally matching fusion candidates.</param>
/// <param name="TrustedCandidates">The number of candidates with compatible trusted offline costs.</param>
/// <param name="SelectedCandidates">The number of non-overlapping fusion candidates selected.</param>
/// <param name="SearchStates">The number of fusion-selection states explored.</param>
/// <param name="ConservativeSavingMicroseconds">The estimated savings in microseconds using pessimistic timing quantiles and margins.</param>
/// <param name="OriginalNodeCount">The node count before fusion selection.</param>
/// <param name="OptimizedNodeCount">The node count after fusion selection.</param>
public sealed record TierOneOptimizationReport(
    TierOneSelectionStatus Status, string CostModel, string Diagnostic, int MatchedCandidates,
    int TrustedCandidates, int SelectedCandidates, int SearchStates, double ConservativeSavingMicroseconds,
    int OriginalNodeCount, int OptimizedNodeCount);

/// <summary>Contains an optimized VM artifact and its fusion and matrix-vector selection reports.</summary>
/// <param name="Program">The optimized and validated VM program.</param>
/// <param name="Report">The tier-one fusion selection report.</param>
/// <param name="MatVecReport">The GPU matrix-vector selection report.</param>
public sealed record VmOptimizationResult(VmProgram Program, TierOneOptimizationReport Report,
    GpuMatVecOptimizationReport? MatVecReport = null);
