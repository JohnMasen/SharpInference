using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime;

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

    public VmProgram Generate(IModelTensorCatalog catalog, VmOptimizationOptions? options = null)
    {
        if (RwkvModelArchitectureDetector.Detect(catalog) != ModelArchitecture)
            throw new InvalidDataException("Model metadata does not match the selected execution graph generator.");
        return generator.Generate(provider.Build(catalog), options);
    }
}
