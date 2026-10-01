using Microsoft.Extensions.Configuration;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Backends.Cpu;
using SharpInference.Backends.Vortice;
using SharpInference.Graphs;

namespace SharpInference.Runtime;

public sealed class RwkvRuntimeFactory
{
    public RwkvRuntimeSelection CreateRuntime(IConfigurationSection? runtimeSection, IModelTensorCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        var architectureId = RwkvModelArchitectureDetector.Detect(catalog);
        var kind = runtimeSection?["Kind"];
        var vorticeSection = runtimeSection?.GetSection("Vortice");
        if (vorticeSection is not null)
        {
            foreach (var option in vorticeSection.GetChildren())
            {
                if (!string.Equals(option.Key, nameof(VorticeRuntimeConfig.AdapterIndex), StringComparison.OrdinalIgnoreCase) &&
                    !string.Equals(option.Key, nameof(VorticeRuntimeConfig.EnableCommandReplay), StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException(
                        $"Unsupported RWKV Vortice runtime option '{option.Key}'.");
            }
        }
        if (runtimeSection?.GetValue<bool>("Vortice:EnableCommandReplay") == true &&
            !string.Equals(kind, "vortice", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException(
                "Rwkv:Runtime:Vortice:EnableCommandReplay requires runtime kind 'vortice'.");

        Func<IExecutionGraphBackend> backendFactory = kind?.ToLowerInvariant() switch
        {
            null or "" or "cpu" => static () => CpuPrimitiveGraphBackend.Instance,
            "vortice" => () => VorticePrimitiveGraphBackend.FromConfig(
                vorticeSection!.Get<VorticeRuntimeConfig>() ?? new VorticeRuntimeConfig()),
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
    Func<IExecutionGraphBackend> backendFactory)
{
    public string ArchitectureId { get; } = architectureId;
    public RwkvWorldTokenizer Tokenizer { get; } = tokenizer;
    public ILogicalGraphProvider Provider { get; } = provider;

    public IExecutionGraphBackend CreateBackend() => backendFactory();
}
