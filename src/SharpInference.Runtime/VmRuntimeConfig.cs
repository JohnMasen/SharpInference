using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime;

/// <summary>Configures VM queueing, compilation, storage, and optimization behavior.</summary>
public sealed class VmRuntimeConfig
{
    /// <summary>Gets the number of reusable VM instances dedicated to prompt prefill.</summary>
    public int PrefillInstances { get; init; } = 2;
    /// <summary>Gets the number of VM instances available for whole-generation inference.</summary>
    public int InferenceInstances { get; init; } = 2;
    /// <summary>Gets the maximum number of queued prefill requests.</summary>
    public int PrefillQueueCapacity { get; init; } = 16;
    /// <summary>Gets the maximum number of queued inference requests.</summary>
    public int InferenceQueueCapacity { get; init; } = 16;
    /// <summary>Gets the maximum token count accepted by a prefill request.</summary>
    public int MaximumPrefillTokens { get; init; } = 65536;
    /// <summary>Gets the compiled prefill capacity used by VM optimization.</summary>
    public int PrefillCapacity { get; init; } = 64;
    /// <summary>Gets the thread-group size used by generated GPU dispatches.</summary>
    public uint ThreadsPerGroup { get; init; } = 64;
    /// <summary>Gets whether optimization may reuse local storage across non-overlapping resource lifetimes.</summary>
    public bool ReuseLocalStorage { get; init; } = true;
    /// <summary>Gets whether optimization may preserve weights in native half precision.</summary>
    public bool NativeHalfWeights { get; init; } = true;
    /// <summary>Gets whether optimization may use views of weight storage instead of materialized copies.</summary>
    public bool WeightViews { get; init; } = true;
    /// <summary>Gets whether optimization may represent eligible tensor transformations as storage views.</summary>
    public bool TensorViews { get; init; } = true;
    /// <summary>Gets the GPU matrix-vector implementation selection mode.</summary>
    public GpuMatVecMode GpuMatVecMode { get; init; } = GpuMatVecMode.Default;
    /// <summary>Gets the explicit cost profile required when GPU matrix-vector selection uses profile mode.</summary>
    public TierOneCostProfile? GpuMatVecCostProfile { get; init; }
    /// <summary>Gets the optional measured cost profile enabling tier-one plan optimization.</summary>
    public TierOneCostProfile? TierOneCostProfile { get; init; }
    /// <summary>Gets the maximum number of states explored during tier-one optimization.</summary>
    public int TierOneMaximumSearchStates { get; init; } = 100_000;
    /// <summary>Gets the optional XML VM program path, mutually exclusive with an execution plan or artifact directory.</summary>
    public string? ProgramPath { get; init; }
    /// <summary>Gets the optional offline target execution-plan XML path, mutually exclusive with a program or artifact.</summary>
    public string? ExecutionPlanPath { get; init; }
    /// <summary>Gets the optional directory containing an existing compiled inference artifact.</summary>
    public string? ArtifactDirectory { get; init; }
    /// <summary>Gets the optional XML prefill program path, mutually exclusive with a prefill artifact directory.</summary>
    public string? PrefillProgramPath { get; init; }
    /// <summary>Gets the optional directory containing a separately compiled prefill artifact.</summary>
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
        if (new[] { ProgramPath, ArtifactDirectory, ExecutionPlanPath }.Count(path => path is not null) > 1)
            throw new InvalidOperationException("Select an execution plan, a program, or an existing artifact, not multiple inputs.");
        if (ExecutionPlanPath is not null && (TierOneCostProfile is not null || GpuMatVecCostProfile is not null))
            throw new InvalidOperationException("An offline execution plan cannot be replanned using optimization profiles.");
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
                    GpuMatVecMode == GpuMatVecMode.Profile ? tierOneEnvironmentFingerprint : null), TensorViews);
}
