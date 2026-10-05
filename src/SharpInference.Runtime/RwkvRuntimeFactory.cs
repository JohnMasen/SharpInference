using Microsoft.Extensions.Configuration;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Runtime;

public sealed class RwkvRuntimeFactory
{
    public static RwkvExecutionGraphGenerator CreateGraphGenerator(IModelTensorCatalog catalog, InstructionTarget architecture,
        IEnumerable<IInstructionCollectionProvider>? instructionCollections = null)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        if (architecture != InstructionTarget.Cpu && architecture != InstructionTarget.Direct3D12)
            throw new NotSupportedException($"No RWKV graph generator is installed for '{architecture}'.");
        var collections = instructionCollections ?? DefaultInstructionCollections.Create()
            .Select(provider => new TargetInstructionCollection(provider, architecture));
        return new(RwkvModelArchitectureDetector.Detect(catalog), architecture, collections);
    }
    public RwkvRuntimeSelection CreateRuntime(IConfigurationSection? runtimeSection, IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        var architectureId = RwkvModelArchitectureDetector.Detect(catalog);
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
                nameof(VmRuntimeConfig.WeightViews),
                nameof(VmRuntimeConfig.GpuMatVecMode),
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
            {
                if (!string.Equals(option.Key, "AdapterIndex", StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException(
                        $"Unsupported RWKV Vortice runtime option '{option.Key}'.");
            }
        }
        var target = kind?.ToLowerInvariant() switch
        {
            null or "" or "cpu" => InstructionTarget.Cpu,
            "vortice" or "d3d12" => InstructionTarget.Direct3D12,
            _ => throw new InvalidOperationException($"The configured RWKV runtime '{kind}' is not registered."),
        };
        var generator = CreateGraphGenerator(catalog, target);
        Func<VmGraphBackend> backendFactory = kind?.ToLowerInvariant() switch
        {
            null or "" or "cpu" => () => VmBackendFactory.CreateCpu(vmConfiguration,
                generatorCollections: [generator.InstructionCollections]),
            "vortice" or "d3d12" => () => VmBackendFactory.CreateD3D12(vmConfiguration,
                vorticeSection?.GetValue<int>("AdapterIndex") ?? 0, generatorCollections: [generator.InstructionCollections]),
            _ => throw new InvalidOperationException($"The configured RWKV runtime '{kind}' is not registered."),
        };
        return new RwkvRuntimeSelection(architectureId,
            RwkvTokenizerResolver.LoadForArchitecture(architectureId),
            CreateGraphProvider(architectureId), backendFactory);
    }

    public static ILogicalGraphProvider CreateGraphProvider(string architectureId) =>
        architectureId switch
        {
            "rwkv-6" => new PortableRwkv6GraphProvider(),
            "rwkv-7" => new PortableRwkv7GraphProvider(),
            _ => throw new NotSupportedException($"No graph provider is registered for '{architectureId}'."),
        };
}

public sealed class RwkvRuntimeSelection(
    string architectureId,
    RwkvWorldTokenizer tokenizer,
    ILogicalGraphProvider provider,
    Func<VmGraphBackend> backendFactory)
{
    public string ArchitectureId { get; } = architectureId;
    public RwkvWorldTokenizer Tokenizer { get; } = tokenizer;
    public ILogicalGraphProvider Provider { get; } = provider;

    public VmGraphBackend CreateBackend() => backendFactory();
}
