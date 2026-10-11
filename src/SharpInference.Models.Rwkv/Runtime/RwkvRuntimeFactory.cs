using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Runtime;

/// <summary>Creates RWKV processors and graph generators for supported model and backend architectures.</summary>
public static class RwkvRuntimeFactory
{
    /// <summary>Loads an RWKV-6 or RWKV-7 model using the supplied VM backend.</summary>
    /// <param name="path">The model file path.</param>
    /// <param name="backend">The backend that prepares and executes the model.</param>
    /// <returns>The loaded processor, which the caller must dispose.</returns>
    public static Processor Load(string path, VmGraphBackend backend)
    {
        var modules = new ModelGraphModuleRegistry();
        modules.Register(new Rwkv6ModelModule());
        modules.Register(new Rwkv7ModelModule());
        return Processor.Load(path, new GgmlModelReader(), modules, backend);
    }

    /// <summary>Detects the RWKV model and creates a generator with target-specific instruction collections.</summary>
    /// <param name="catalog">The model tensor catalog used for architecture detection.</param>
    /// <param name="architecture">The CPU or Direct3D12 instruction target.</param>
    /// <param name="instructionCollections">The available instruction collection providers.</param>
    /// <returns>The configured RWKV graph generator.</returns>
    public static RwkvExecutionGraphGenerator CreateGraphGenerator(IModelTensorCatalog catalog, InstructionTarget architecture,
        IEnumerable<IInstructionCollectionProvider> instructionCollections)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentNullException.ThrowIfNull(instructionCollections);
        if (architecture != InstructionTarget.Cpu && architecture != InstructionTarget.Direct3D12)
            throw new NotSupportedException($"No RWKV graph generator is installed for '{architecture}'.");
        var collections = instructionCollections.Select(provider => new TargetInstructionCollection(provider, architecture));
        return new(RwkvModelArchitectureDetector.Detect(catalog), architecture, collections);
    }

    /// <summary>Creates the portable logical graph provider for a supported RWKV architecture.</summary>
    /// <param name="architectureId">The RWKV-6 or RWKV-7 architecture identifier.</param>
    /// <returns>The matching graph provider.</returns>
    public static ILogicalGraphProvider CreateGraphProvider(string architectureId) => architectureId switch
    {
        "rwkv-6" => new PortableRwkv6GraphProvider(),
        "rwkv-7" => new PortableRwkv7GraphProvider(),
        _ => throw new NotSupportedException($"No graph provider is registered for '{architectureId}'."),
    };
}
