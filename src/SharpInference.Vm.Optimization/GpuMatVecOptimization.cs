using SharpInference.Instructions;
using SharpInference.Vm;

namespace SharpInference.Vm.Optimization;

public enum GpuMatVecMode { Serial, Cooperative, Profile, Default }

public sealed class GpuMatVecOptimizationSettings
{
    public GpuMatVecOptimizationSettings(GpuMatVecMode mode, TierOneCostProfile? profile = null,
        string? environmentFingerprint = null)
    {
        if (!Enum.IsDefined(mode)) throw new ArgumentOutOfRangeException(nameof(mode));
        if (mode == GpuMatVecMode.Profile &&
            (profile is null || string.IsNullOrWhiteSpace(environmentFingerprint)))
            throw new ArgumentException("Profile selection requires explicit costs and a current environment fingerprint.");
        if (mode != GpuMatVecMode.Profile && (profile is not null || environmentFingerprint is not null))
            throw new ArgumentException("Costs are only accepted in profile mode.");
        Mode = mode;
        Profile = profile;
        EnvironmentFingerprint = environmentFingerprint;
    }

    public GpuMatVecMode Mode { get; }
    public TierOneCostProfile? Profile { get; }
    public string? EnvironmentFingerprint { get; }

    public static string Shape(IReadOnlyList<VmParameter> parameters)
    {
        var matrix = parameters.Single(parameter => parameter.Name is "matrix" or "weight");
        var input = parameters.Single(parameter => parameter.Name == "input");
        var output = parameters.Single(parameter => parameter.Name == "output");
        return $"{string.Join("x", matrix.Tensor.Dimensions)}|{matrix.Tensor.ElementType},{input.Tensor.ElementType},{output.Tensor.ElementType}";
    }

    internal (bool Cooperative, string Diagnostic) Select(IReadOnlyList<VmParameter> parameters,
        VmOptimizationOptions options)
    {
        if (Mode == GpuMatVecMode.Serial) return (false, "Explicit serial MatVec.");
        if (Mode == GpuMatVecMode.Default) return (true, "Default GPU MatVec implementation: cooperative.");
        if (Mode == GpuMatVecMode.Cooperative) return (true, "Explicit cooperative MatVec; not a measured selection.");
        var profile = Profile!;
        var now = DateTimeOffset.UtcNow;
        if (profile.EnvironmentFingerprint != EnvironmentFingerprint)
            return (false, "MatVec profile environment fingerprint mismatch.");
        if (profile.MeasuredAt > now.AddMinutes(5) || now - profile.MeasuredAt > TimeSpan.FromDays(30))
            return (false, "MatVec profile timestamp is stale or in the future.");
        var measurement = profile.Measurements.SingleOrDefault(measurement =>
            measurement.Operation == "core.mat-vec" &&
            measurement.ImplementationFingerprint == GpuMatVecExecution.ImplementationFingerprint &&
            measurement.Shape == Shape(parameters) && measurement.InputAliases == "0,1" &&
            measurement.ThreadsPerGroup == options.ThreadsPerGroup &&
            measurement.ThreadsPerGroup == GpuMatVecExecution.Threads &&
            measurement.ReuseLocalStorage == options.ReuseLocalStorage);
        if (measurement is null) return (false, "No trusted MatVec costs for this shape/storage/launch domain.");
        return measurement.ConservativeSavingMicroseconds > 0
            ? (true, "Cooperative MatVec selected from positive conservative offline savings.")
            : (false, "MatVec costs do not establish a conservative benefit; serial retained.");
    }
}

public sealed record GpuMatVecOptimizationReport(int Calls, int CooperativeCalls, IReadOnlyList<string> Diagnostics);
