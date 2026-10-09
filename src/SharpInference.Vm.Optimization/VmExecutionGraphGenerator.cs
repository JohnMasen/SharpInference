using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

public sealed class VmExecutionGraphGenerator
{
    private readonly InstructionRegistry instructions;
    private readonly IGraphOperationLowerer? operationLowerer;
    public VmExecutionGraphGenerator(InstructionTarget architecture, IEnumerable<IInstructionCollectionProvider> collections,
        IGraphOperationLowerer? operationLowerer = null)
    {
        Architecture = architecture;
        instructions = new(collections);
        this.operationLowerer = operationLowerer;
    }
    public InstructionTarget Architecture { get; }
    public IInstructionCollectionProvider InstructionCollections => instructions;
    public TierOneOptimizationReport? LastOptimizationReport { get; private set; }
    public GpuMatVecOptimizationReport? LastMatVecOptimizationReport { get; private set; }

    public VmProgram Generate(LogicalGraph graph, VmOptimizationOptions? options = null)
    {
        graph = GraphOperationLowering.Apply(graph, operationLowerer);
        var target = Architecture == InstructionTarget.Cpu ? VmTarget.Cpu :
            Architecture == InstructionTarget.Direct3D12 ? VmTarget.Direct3D12 :
            throw new NotSupportedException($"No VM lowering is installed for '{Architecture}'.");
        var result = VmGraphOptimizer.OptimizeWithReport(graph, target, options, instructions.QueryOptimizationCapabilities(),
            instructions.QueryGraphInstructionBindings());
        var program = result.Program;
        foreach (var definition in program.Definitions)
            foreach (var operation in definition.Nodes.Select(node => node.Instruction).OfType<VmOperator>())
            {
                var instruction = instructions.Resolve(operation.InstructionCollectionId, operation.InstructionName, Architecture);
                var parameters = VmInstructionParameters.Create(operation, definition.Parameters, parameter => (parameter.Name, "0"));
                instruction.Adapt(parameters, operation.Precision, operation.ExecutionConfiguration);
            }
        var bound = VmInstructionContracts.Bind(program, instructions);
        LastOptimizationReport = result.Report;
        LastMatVecOptimizationReport = result.MatVecReport;
        return bound;
    }
}
