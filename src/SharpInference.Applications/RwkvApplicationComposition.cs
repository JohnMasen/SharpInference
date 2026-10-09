using Microsoft.Extensions.Configuration;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;
using SharpInference.Runtime;
using SharpInference.Runtime.Cpu;
using SharpInference.Runtime.D3D12;

namespace SharpInference.Applications;

/// <summary>Application-owned model registration and configuration aliases, never a runtime default.</summary>
public sealed class RwkvApplicationComposition(ModelGraphModuleRegistry modules)
{
    public static ModelGraphModuleRegistry CreateModelModules()
    {
        var registry = new ModelGraphModuleRegistry();
        registry.Register(new Rwkv6ModelModule());
        registry.Register(new Rwkv7ModelModule());
        return registry;
    }

    public RwkvApplicationRuntime CreateRuntime(IConfigurationSection? runtimeSection, IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        var module = modules.Resolve(catalog);
        var kind = runtimeSection?["Kind"];
        var vmSection = runtimeSection?.GetSection("Vm");
        if (vmSection is not null)
        {
            string[] supported =
            [
                nameof(VmRuntimeConfig.PrefillInstances), nameof(VmRuntimeConfig.InferenceInstances),
                nameof(VmRuntimeConfig.PrefillQueueCapacity), nameof(VmRuntimeConfig.InferenceQueueCapacity),
                nameof(VmRuntimeConfig.MaximumPrefillTokens), nameof(VmRuntimeConfig.PrefillCapacity),
                nameof(VmRuntimeConfig.ThreadsPerGroup), nameof(VmRuntimeConfig.ReuseLocalStorage),
                nameof(VmRuntimeConfig.NativeHalfWeights), nameof(VmRuntimeConfig.ProgramPath),
                nameof(VmRuntimeConfig.WeightViews), nameof(VmRuntimeConfig.GpuMatVecMode),
                nameof(VmRuntimeConfig.ArtifactDirectory),
                nameof(VmRuntimeConfig.PrefillProgramPath), nameof(VmRuntimeConfig.PrefillArtifactDirectory),
            ];
            foreach (var option in vmSection.GetChildren())
                if (!supported.Contains(option.Key, StringComparer.OrdinalIgnoreCase))
                    throw new InvalidOperationException($"Unsupported RWKV VM option '{option.Key}'.");
        }
        var vmConfiguration = vmSection?.Get<VmRuntimeConfig>() ?? new();
        vmConfiguration.EngineOptions();
        var vorticeSection = runtimeSection?.GetSection("Vortice");
        if (vorticeSection is not null)
        {
            foreach (var option in vorticeSection.GetChildren())
                if (!string.Equals(option.Key, "AdapterIndex", StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException($"Unsupported RWKV Vortice runtime option '{option.Key}'.");
        }
        Func<VmGraphBackend> backendFactory = kind?.ToLowerInvariant() switch
        {
            null or "" or "cpu" => () => CpuVmBackendFactory.Create(vmConfiguration,
                instructionCollections: CpuInstructionCollections.Create()),
            "vortice" or "d3d12" => () => D3D12VmBackendFactory.Create(vmConfiguration,
                vorticeSection?.GetValue<int>("AdapterIndex") ?? 0,
                instructionCollections: D3D12InstructionCollections.Create()),
            _ => throw new InvalidOperationException($"The configured RWKV runtime '{kind}' is not registered."),
        };
        return new(module, RwkvTokenizerResolver.LoadForArchitecture(module.ArchitectureId), backendFactory);
    }
}

public sealed class RwkvApplicationRuntime(
    IModelGraphModule module, RwkvWorldTokenizer tokenizer, Func<VmGraphBackend> backendFactory)
{
    public string ArchitectureId => Module.ArchitectureId;
    public IModelGraphModule Module { get; } = module;
    public RwkvWorldTokenizer Tokenizer { get; } = tokenizer;
    public ILogicalGraphProvider Provider => Module;
    public VmGraphBackend CreateBackend() => backendFactory();
}
