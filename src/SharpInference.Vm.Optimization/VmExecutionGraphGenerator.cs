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
    public TargetExecutionPlan? LastPlan { get; private set; }

    public TargetExecutionPlan BindPlan(TargetExecutionPlan plan) => ExecutionPlanAssembler.Bind(plan, instructions);

    public void ValidateOperations(IlExecutionPlan il)
    {
        var supported = TierZeroOperationContracts.Contracts.Select(contract => contract.Operation)
            .Concat(instructions.QueryGraphInstructionBindings().Where(binding => binding.Target == Architecture)
                .Select(binding => binding.Operation)).ToHashSet();
        var diagnostics = il.Nodes.Where(node => !supported.Contains(node.Operation)).Select(node =>
            new BackendPreparationDiagnostic(node.Id, node.Operation, node.Source.LogicalNodes,
                BackendPreparationFailureReason.UnsupportedOperation,
                $"No registered {Architecture} instruction or portable lowering for '{node.Operation}'.",
                node.Requirements, [])).ToArray();
        if (diagnostics.Length != 0)
            throw new BackendPreparationException($"vm.{Architecture}", diagnostics);
    }

    public VmProgram Generate(LogicalGraph graph, VmOptimizationOptions? options = null)
        => ExecutionPlanAssembler.Assemble(GeneratePlan(new IlExecutionPlan(graph), options));

    public TargetExecutionPlan GeneratePlan(IlExecutionPlan il, VmOptimizationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(il);
        var target = Architecture == InstructionTarget.Cpu ? VmTarget.Cpu :
            Architecture == InstructionTarget.Direct3D12 ? VmTarget.Direct3D12 :
            throw new NotSupportedException($"No VM lowering is installed for '{Architecture}'.");
        var lowered = operationLowerer is null ? il : new IlExecutionPlan(
            GraphOperationLowering.Apply(il.ToLogicalGraph(), operationLowerer));
        var result = VmGraphOptimizer.BuildPlan(lowered, target, options, instructions.QueryOptimizationCapabilities(),
            instructions.QueryGraphInstructionBindings());
        var bound = ExecutionPlanAssembler.Bind(result.Plan, instructions);
        LastOptimizationReport = result.Report;
        LastMatVecOptimizationReport = result.MatVecReport;
        LastPlan = bound;
        return bound;
    }
}
