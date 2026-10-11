using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

/// <summary>Lowers graphs and binds target execution plans against registered instruction providers.</summary>
public sealed class VmExecutionGraphGenerator
{
    private readonly InstructionRegistry instructions;
    private readonly IGraphOperationLowerer? operationLowerer;
    /// <summary>Creates a graph generator with target providers and optional portable operation lowering.</summary>
    /// <param name="architecture">The architecture identifier used to select implementations.</param>
    /// <param name="collections">The instruction collection providers available to this generator.</param>
    /// <param name="operationLowerer">The optional portable lowerer applied before target instruction selection.</param>
    public VmExecutionGraphGenerator(InstructionTarget architecture, IEnumerable<IInstructionCollectionProvider> collections,
        IGraphOperationLowerer? operationLowerer = null)
    {
        Architecture = architecture;
        instructions = new(collections);
        this.operationLowerer = operationLowerer;
    }
    /// <summary>Gets the architecture identifier used to select implementations.</summary>
    public InstructionTarget Architecture { get; }
    /// <summary>Gets the validated registry of instruction collection providers.</summary>
    public IInstructionCollectionProvider InstructionCollections => instructions;
    /// <summary>Gets the fusion selection report from the last successful plan generation.</summary>
    public TierOneOptimizationReport? LastOptimizationReport { get; private set; }
    /// <summary>Gets the matrix-vector selection report from the last successful plan generation.</summary>
    public GpuMatVecOptimizationReport? LastMatVecOptimizationReport { get; private set; }
    /// <summary>Gets the bound plan from the last successful generation.</summary>
    public TargetExecutionPlan? LastPlan { get; private set; }

    /// <summary>Resolves and validates a plan's instruction contracts against the installed registry.</summary>
    /// <param name="plan">The target execution plan to validate, bind, or assemble.</param>
    public TargetExecutionPlan BindPlan(TargetExecutionPlan plan) => ExecutionPlanAssembler.Bind(plan, instructions);

    /// <summary>Rejects graph operations without a standard contract or registered target instruction mapping.</summary>
    /// <param name="il">The intermediate execution graph to lower and optimize.</param>
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

    /// <summary>Creates a bound optimized VM program from the supplied logical graph.</summary>
    /// <param name="graph">The graph or binding metadata to inspect.</param>
    /// <param name="options">The lowering, resource reuse, launch, and optional offline-selection settings; null uses defaults.</param>
    public VmProgram Generate(LogicalGraph graph, VmOptimizationOptions? options = null)
        => ExecutionPlanAssembler.Assemble(GeneratePlan(new IlExecutionPlan(graph), options));

    /// <summary>Lowers and optimizes an IL graph, binds its instructions, and records the successful plan and reports.</summary>
    /// <param name="il">The intermediate execution graph to lower and optimize.</param>
    /// <param name="options">The lowering, resource reuse, launch, and optional offline-selection settings; null uses defaults.</param>
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
