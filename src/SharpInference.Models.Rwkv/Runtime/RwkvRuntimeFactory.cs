using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Runtime;

public static class RwkvRuntimeFactory
{
    public static Processor Load(string path, VmGraphBackend backend)
    {
        var modules = new ModelGraphModuleRegistry();
        modules.Register(new Rwkv6ModelModule());
        modules.Register(new Rwkv7ModelModule());
        return Processor.Load(path, new GgmlModelReader(), modules, backend);
    }

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

    public static ILogicalGraphProvider CreateGraphProvider(string architectureId) => architectureId switch
    {
        "rwkv-6" => new PortableRwkv6GraphProvider(),
        "rwkv-7" => new PortableRwkv7GraphProvider(),
        _ => throw new NotSupportedException($"No graph provider is registered for '{architectureId}'."),
    };
}
