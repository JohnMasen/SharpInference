using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime;

/// <summary>Configures VM queueing, compilation, storage, and optimization behavior.</summary>
public sealed class VmRuntimeConfig
{
    public int PrefillInstances { get; init; } = 2;
    public int InferenceInstances { get; init; } = 2;
    public int PrefillQueueCapacity { get; init; } = 16;
    public int InferenceQueueCapacity { get; init; } = 16;
    public int MaximumPrefillTokens { get; init; } = 65536;
    public int PrefillCapacity { get; init; } = 64;
    public uint ThreadsPerGroup { get; init; } = 64;
    public bool ReuseLocalStorage { get; init; } = true;
    public bool NativeHalfWeights { get; init; } = true;
    public bool WeightViews { get; init; } = true;
    public GpuMatVecMode GpuMatVecMode { get; init; } = GpuMatVecMode.Default;
    public TierOneCostProfile? GpuMatVecCostProfile { get; init; }
    public TierOneCostProfile? TierOneCostProfile { get; init; }
    public int TierOneMaximumSearchStates { get; init; } = 100_000;
    public string? ProgramPath { get; init; }
    public string? ArtifactDirectory { get; init; }
    public string? PrefillProgramPath { get; init; }
    public string? PrefillArtifactDirectory { get; init; }

    /// <summary>Builds and validates VM engine queue and capacity options.</summary>
    /// <returns>Validated engine options.</returns>
    public VmEngineOptions EngineOptions()
    {
        if (!Enum.IsDefined(GpuMatVecMode))
            throw new InvalidOperationException("Unknown GPU MatVec mode.");
        if (GpuMatVecMode == GpuMatVecMode.Profile && GpuMatVecCostProfile is null ||
            GpuMatVecMode != GpuMatVecMode.Profile && GpuMatVecCostProfile is not null)
            throw new InvalidOperationException("GPU MatVec profile mode requires an explicit profile, and other modes cannot accept one.");
        var options = new VmEngineOptions(PrefillQueueCapacity, InferenceQueueCapacity,
            PrefillInstances, InferenceInstances, MaximumPrefillTokens);
        options.Validate();
        if (PrefillCapacity is < 1 or > 1024 || ThreadsPerGroup is 0 or > 1024)
            throw new InvalidOperationException("VM prefill capacity and thread group dimensions must be in 1..1024.");
        if (ProgramPath is not null && ArtifactDirectory is not null)
            throw new InvalidOperationException("Select a program to compile or an existing artifact, not both.");
        if (PrefillProgramPath is not null && PrefillArtifactDirectory is not null)
            throw new InvalidOperationException("Select a prefill program or prefill artifact, not both.");
        return options;
    }

    /// <summary>Builds optimization options for the selected target and environment fingerprint.</summary>
    /// <param name="tierOneEnvironmentFingerprint">The hardware/runtime fingerprint required for tier-one profiling.</param>
    /// <param name="target">The selected VM execution target.</param>
    /// <returns>VM optimization options derived from this configuration.</returns>
    public VmOptimizationOptions OptimizationOptions(string? tierOneEnvironmentFingerprint = null, VmTarget? target = null) =>
        new(ReuseLocalStorage, ThreadsPerGroup, PrefillCapacity, NativeHalfWeights, WeightViews,
            TierOneCostProfile is null ? null : new(TierOneCostProfile,
                tierOneEnvironmentFingerprint ?? throw new InvalidOperationException("A current hardware/runtime fingerprint is required to enable T1."),
                TierOneMaximumSearchStates),
            GpuMatVecMode == GpuMatVecMode.Default ||
                target == VmTarget.Cpu && GpuMatVecMode == GpuMatVecMode.Serial ? null :
                new(GpuMatVecMode, GpuMatVecCostProfile,
                    GpuMatVecMode == GpuMatVecMode.Profile ? tierOneEnvironmentFingerprint : null));
}
