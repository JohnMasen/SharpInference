using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime;

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
    public string? ProgramPath { get; init; }
    public string? ArtifactDirectory { get; init; }
    public string? PrefillProgramPath { get; init; }
    public string? PrefillArtifactDirectory { get; init; }

    public VmEngineOptions EngineOptions()
    {
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

    public VmOptimizationOptions OptimizationOptions() => new(ReuseLocalStorage, ThreadsPerGroup, PrefillCapacity, NativeHalfWeights, WeightViews);
}
