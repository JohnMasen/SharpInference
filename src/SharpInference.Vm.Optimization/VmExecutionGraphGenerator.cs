using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

public sealed class VmExecutionGraphGenerator
{
    private readonly InstructionRegistry instructions;
    public VmExecutionGraphGenerator(InstructionTarget architecture, IEnumerable<IInstructionCollectionProvider> collections)
    {
        Architecture = architecture;
        instructions = new(collections);
    }
    public InstructionTarget Architecture { get; }
    public IInstructionCollectionProvider InstructionCollections => instructions;

    public VmProgram Generate(LogicalGraph graph, VmOptimizationOptions? options = null)
    {
        var target = Architecture == InstructionTarget.Cpu ? VmTarget.Cpu :
            Architecture == InstructionTarget.Direct3D12 ? VmTarget.Direct3D12 :
            throw new NotSupportedException($"No VM lowering is installed for '{Architecture}'.");
        var program = VmGraphOptimizer.Optimize(graph, target, options);
        foreach (var definition in program.Definitions)
            foreach (var operation in definition.Nodes.Select(node => node.Instruction).OfType<VmOperator>())
            {
                var instruction = instructions.Resolve(operation.InstructionCollectionId, operation.InstructionName, Architecture);
                var parameters = VmInstructionParameters.Create(operation, definition.Parameters, parameter => (parameter.Name, "0"));
                instruction.GetSignature(parameters);
            }
        return VmInstructionContracts.Bind(program, instructions);
    }
}
