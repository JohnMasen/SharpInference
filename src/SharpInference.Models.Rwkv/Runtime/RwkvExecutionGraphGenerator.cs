using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime;

/// <summary>Generates VM programs from RWKV logical graphs for a selected instruction target.</summary>
public sealed class RwkvExecutionGraphGenerator
{
    private readonly ILogicalGraphProvider provider;
    private readonly VmExecutionGraphGenerator generator;

    internal RwkvExecutionGraphGenerator(string modelArchitecture, InstructionTarget target,
        IEnumerable<IInstructionCollectionProvider> collections)
    {
        ModelArchitecture = modelArchitecture;
        provider = RwkvRuntimeFactory.CreateGraphProvider(modelArchitecture);
        generator = new(target, collections);
    }

    public string ModelArchitecture { get; }
    public InstructionTarget Architecture => generator.Architecture;
    public IInstructionCollectionProvider InstructionCollections => generator.InstructionCollections;

    /// <summary>Builds and generates a VM program for a model tensor catalog.</summary>
    /// <param name="catalog">The model tensor catalog used to build the logical graph.</param>
    /// <param name="options">Optional VM optimization settings.</param>
    /// <returns>The generated VM program.</returns>
    public VmProgram Generate(IModelTensorCatalog catalog, VmOptimizationOptions? options = null)
    {
        if (RwkvModelArchitectureDetector.Detect(catalog) != ModelArchitecture)
            throw new InvalidDataException("Model metadata does not match the selected execution graph generator.");
        return generator.Generate(provider.Build(catalog), options);
    }
}
