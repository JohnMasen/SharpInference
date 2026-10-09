using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class TierOneOptimizerTests
{
    private static readonly InstructionRegistry Registry = new(SharpInference.Runtime.Cpu.CpuInstructionCollections.Create().Concat(SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create()));
    private static readonly IReadOnlyList<InstructionOptimizationCapability> Capabilities = Registry.QueryOptimizationCapabilities();

    [Theory]
    [InlineData(VmTarget.Cpu)]
    [InlineData(VmTarget.Direct3D12)]
    public void UnknownStaleNoisyOrUnprofitableCostsKeepBaseline(VmTarget target)
    {
        var graph = TierOneReferencePrograms.CreateGraph("multiply-add", [64]);
        var baseline = VmGraphOptimizer.Optimize(graph, target);
        foreach (var options in new VmOptimizationOptions[]
                 {
                     new(),
                     Options(target, [], environment: "stale"),
                     Options(target, []),
                     Options(target, [Measurement(target, "multiply-add", 10, 20)]),
                     Options(target, [Measurement(target, "multiply-add", 10, 9.8)]),
                     Options(target, [Measurement(target, "multiply-add", 20, 10, fingerprint: "stale")]),
                     Options(target, [Measurement(target, "multiply-add", 20, 10, shape: "17")]),
                 })
        {
            var result = VmGraphOptimizer.OptimizeWithReport(graph, target, options, Capabilities);
            Assert.Equal(TierOneSelectionStatus.BaselineRetained, result.Report.Status);
            Assert.Equal(0, result.Report.SelectedCandidates);
            Assert.Equal(VmProgramXml.Serialize(baseline), VmProgramXml.Serialize(result.Program));
            Assert.NotEmpty(result.Report.Diagnostic);
        }
    }

    [Theory]
    [InlineData(VmTarget.Cpu)]
    [InlineData(VmTarget.Direct3D12)]
    public void SelectedPlanReallocatesAndKeepsConfigurationThroughBindingAndXml(VmTarget target)
    {
        var graph = TierOneReferencePrograms.CreateGraph("multiply-add", [64]);
        var options = Options(target, [Measurement(target, "multiply-add", 20, 10)]);
        var generator = new VmExecutionGraphGenerator(target == VmTarget.Cpu ? InstructionTarget.Cpu : InstructionTarget.Direct3D12,
            SharpInference.Runtime.Cpu.CpuInstructionCollections.Create().Concat(SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create()));
        var program = generator.Generate(graph, options);
        Assert.Equal(TierOneSelectionStatus.ModelOptimal, generator.LastOptimizationReport!.Status);
        Assert.Equal(1, generator.LastOptimizationReport.SelectedCandidates);
        Assert.Equal(2, generator.LastOptimizationReport.OriginalNodeCount);
        Assert.Equal(1, generator.LastOptimizationReport.OptimizedNodeCount);
        var operation = Assert.Single(program.Definitions.SelectMany(definition => definition.Nodes)
            .Select(node => node.Instruction).OfType<VmOperator>());
        Assert.Equal(InstructionCollectionIds.TierOneFloat32, operation.InstructionCollectionId);
        Assert.NotNull(operation.ExecutionConfiguration);
        Assert.Equal(operation.ExecutionConfiguration, Assert.Single(VmProgramXml.Deserialize(VmProgramXml.Serialize(program))
            .Definitions.SelectMany(definition => definition.Nodes).Select(node => node.Instruction).OfType<VmOperator>()).ExecutionConfiguration);
        Assert.DoesNotContain(program.Slots, slot => slot.Id.StartsWith("local.", StringComparison.Ordinal));
        Assert.Equal(target == VmTarget.Cpu ? 0 : 1, program.Definitions.SelectMany(definition => definition.Nodes)
            .Count(node => node.Instruction is VmBarrier));
        Assert.All(program.Definitions.SelectMany(definition => definition.Parameters), parameter =>
            Assert.DoesNotContain("t0", parameter.Name));
    }

    [Fact]
    public void MatcherFollowsValuesAcrossUnrelatedNodesRatherThanAdjacency()
    {
        var graph = WithUnrelatedNode(controlCycle: false, externalConsumer: false);
        var result = VmGraphOptimizer.OptimizeWithReport(graph, VmTarget.Cpu,
            Options(VmTarget.Cpu, [Measurement(VmTarget.Cpu, "multiply-add", 20, 10)]), Capabilities);
        Assert.Equal(1, result.Report.SelectedCandidates);
        Assert.Equal(2, result.Report.OptimizedNodeCount);
        Assert.Contains(result.Program.Definitions.SelectMany(definition => definition.Nodes), node =>
            node.Instruction is VmOperator operation && operation.InstructionName == "core.square");
    }

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void CyclicContractionOrExternalIntermediateConsumerIsRejected(bool cycle, bool consumer)
    {
        var result = VmGraphOptimizer.OptimizeWithReport(WithUnrelatedNode(cycle, consumer), VmTarget.Cpu,
            Options(VmTarget.Cpu, [Measurement(VmTarget.Cpu, "multiply-add", 20, 10)]), Capabilities);
        Assert.Equal(TierOneSelectionStatus.BaselineRetained, result.Report.Status);
        Assert.Equal(0, result.Report.SelectedCandidates);
    }

    [Theory]
    [InlineData(100_000, TierOneSelectionStatus.ModelOptimal)]
    [InlineData(1, TierOneSelectionStatus.BestFoundWithinBudget)]
    public void JointContractionsMustRemainAcyclicAndBudgetStatusIsHonest(int budget, TierOneSelectionStatus status)
    {
        var graph = TwoCrossDependentPatterns();
        var result = VmGraphOptimizer.OptimizeWithReport(graph, VmTarget.Cpu,
            Options(VmTarget.Cpu, [Measurement(VmTarget.Cpu, "multiply-add", 20, 10)], budget), Capabilities);
        Assert.Equal(2, result.Report.TrustedCandidates);
        Assert.Equal(1, result.Report.SelectedCandidates);
        Assert.Equal(status, result.Report.Status);
        Assert.InRange(result.Report.SearchStates, 1, budget);
        VmProgramValidator.Validate(result.Program);
    }

    [Fact]
    public void IntervalDpChoosesTwoModerateGainsOverOneLargerOverlappingGain()
    {
        var original = TierOneReferencePrograms.CreateGraph("multiply-add-multiply", [64]);
        var tensor = original.Resources[0].Tensor;
        var nodes = original.Nodes.Select(node => node with
        {
            Resources = node.Resources.Select(binding => binding.Resource == new ResourceId("output")
                ? binding with { Resource = new("t2") } : binding).ToArray(),
        }).Append(new(new("stage3"), PrimitiveGraphOperations.Add, new("pointwise"),
            [new("left", new("t2"), GraphResourceAccess.Read), new("right", new("e"), GraphResourceAccess.Read),
                new("output", new("output"), GraphResourceAccess.Write)], [new("stage2")],
            new Dictionary<string, string>(), new(GraphElementType.Float32, GraphElementType.Float32))).ToArray();
        var graph = new LogicalGraph(original.Identity, original.Model,
            original.Resources.Append(new(new("t2"), "t2", GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, tensor))
                .Append(new(new("e"), "e", GraphResourceKind.Input, GraphResourceLifetime.External, tensor)),
            original.Regions, nodes, original.Inputs.Append(new("e")), original.Outputs, original.GraphState);
        var shortCost = Measurement(VmTarget.Cpu, "multiply-add", 15, 8);
        var longCost = Measurement(VmTarget.Cpu, "multiply-add-multiply", 20, 10);
        Assert.True(longCost.ConservativeSavingMicroseconds > shortCost.ConservativeSavingMicroseconds);
        var result = VmGraphOptimizer.OptimizeWithReport(graph, VmTarget.Cpu,
            Options(VmTarget.Cpu, [shortCost, longCost]), Capabilities);
        Assert.Equal(TierOneSelectionStatus.ModelOptimal, result.Report.Status);
        Assert.Equal(2, result.Report.SelectedCandidates);
        Assert.Equal(shortCost.ConservativeSavingMicroseconds * 2, result.Report.ConservativeSavingMicroseconds);
        Assert.All(result.Program.Definitions.SelectMany(definition => definition.Nodes).Select(node => node.Instruction)
            .OfType<VmOperator>(), operation => Assert.Equal("multiply-add", operation.InstructionName));
    }

    [Fact]
    public void OfflineProfilesRoundTripStrictlyAndRejectInvalidEvidence()
    {
        var profile = Options(VmTarget.Cpu, [Measurement(VmTarget.Cpu, "multiply-add", 20, 10)]).TierOne!.Profile;
        Assert.Equal(profile.Serialize(), TierOneCostProfile.Deserialize(profile.Serialize()).Serialize());
        Assert.Throws<System.Text.Json.JsonException>(() => TierOneCostProfile.Deserialize(profile.Serialize().Replace(
            "\"Measurements\":", "\"Unknown\": 1, \"Measurements\":", StringComparison.Ordinal)));
        Assert.Throws<ArgumentException>(() => new TierOneCostMeasurement("multiply-add", "fingerprint", "64", "0,1,2", 64, true,
            [1, 2], [1, 2]));
        Assert.Throws<ArgumentException>(() => new TierOneCostProfile("environment", DateTimeOffset.UtcNow,
            [profile.Measurements[0], profile.Measurements[0]]));
    }

    [Theory]
    [InlineData(-31)]
    [InlineData(1)]
    public void ExpiredAndFutureProfilesRetainBaseline(int days)
    {
        var profile = new TierOneCostProfile("environment", DateTimeOffset.UtcNow.AddDays(days),
            [Measurement(VmTarget.Cpu, "multiply-add", 20, 10)]);
        var result = VmGraphOptimizer.OptimizeWithReport(TierOneReferencePrograms.CreateGraph("multiply-add", [64]),
            VmTarget.Cpu, new(TierOne: new(profile, "environment")), Capabilities);
        Assert.Equal(TierOneSelectionStatus.BaselineRetained, result.Report.Status);
        Assert.Contains("timestamp", result.Report.Diagnostic);
    }

    private static TierOneCostMeasurement Measurement(VmTarget target, string operation, double reference, double optimized,
        string? fingerprint = null, string shape = "64")
    {
        var capability = Capabilities.Single(capability => capability.Name == operation && capability.Target ==
            (target == VmTarget.Cpu ? InstructionTarget.Cpu : InstructionTarget.Direct3D12));
        return new(operation, fingerprint ?? capability.ImplementationFingerprint, shape,
            string.Join(",", Enumerable.Range(0, capability.Definition.Inputs.Count)), 64, true,
            Enumerable.Repeat(reference, 7).ToArray(), Enumerable.Repeat(optimized, 7).ToArray());
    }

    private static VmOptimizationOptions Options(VmTarget target, IReadOnlyList<TierOneCostMeasurement> measurements,
        int budget = 100_000, string environment = "test-environment") =>
        new(TierOne: new(new("test-environment", DateTimeOffset.UtcNow, measurements), environment, budget));

    private static LogicalGraph WithUnrelatedNode(bool controlCycle, bool externalConsumer)
    {
        var original = TierOneReferencePrograms.CreateGraph("multiply-add", [64]);
        var tensor = original.Resources[0].Tensor;
        var extra = new LogicalNode(new("extra"), PrimitiveGraphOperations.Square, new("pointwise"),
            [new("input", new(externalConsumer ? "t0" : "c"), GraphResourceAccess.Read),
                new("output", new("extra-output"), GraphResourceAccess.Write)],
            controlCycle || externalConsumer ? [new("stage0")] : [], new Dictionary<string, string>(),
            new(GraphElementType.Float32, GraphElementType.Float32));
        return new(original.Identity, original.Model,
            original.Resources.Append(new(new("extra-output"), "extra-output", GraphResourceKind.Temporary,
                GraphResourceLifetime.Invocation, tensor)), original.Regions,
            [original.Nodes[0], extra, original.Nodes[1] with
            {
                Dependencies = controlCycle ? [new("stage0"), new("extra")] : original.Nodes[1].Dependencies,
            }], original.Inputs, original.Outputs, original.GraphState);
    }

    private static LogicalGraph TwoCrossDependentPatterns()
    {
        var original = TierOneReferencePrograms.CreateGraph("multiply-add", [64]);
        var tensor = original.Resources[0].Tensor;
        var resources = new[] { "a", "b", "c", "d", "e", "f" }.Select(port =>
            new GraphResource(new(port), port, GraphResourceKind.Input, GraphResourceLifetime.External, tensor)).ToList();
        resources.AddRange(new[] { "ta", "tb", "oa", "ob" }.Select(port => new GraphResource(new(port), port,
            port.StartsWith('o') ? GraphResourceKind.Output : GraphResourceKind.Temporary,
            port.StartsWith('o') ? GraphResourceLifetime.External : GraphResourceLifetime.Invocation, tensor)));
        LogicalNode Node(string id, GraphOperationId operation, string left, string right, string output, params string[] dependencies) =>
            new(new(id), operation, new("pointwise"),
                [new("left", new(left), GraphResourceAccess.Read), new("right", new(right), GraphResourceAccess.Read),
                    new("output", new(output), GraphResourceAccess.Write)],
                dependencies.Select(dependency => new LogicalNodeId(dependency)).ToArray(), new Dictionary<string, string>(),
                new(GraphElementType.Float32, GraphElementType.Float32));
        return new(original.Identity, original.Model, resources, original.Regions,
            [Node("a0", PrimitiveGraphOperations.Multiply, "a", "b", "ta"),
                Node("b0", PrimitiveGraphOperations.Multiply, "c", "d", "tb"),
                Node("a1", PrimitiveGraphOperations.Add, "ta", "e", "oa", "a0", "b0"),
                Node("b1", PrimitiveGraphOperations.Add, "tb", "f", "ob", "b0", "a0")],
            new[] { "a", "b", "c", "d", "e", "f" }.Select(port => new ResourceId(port)), [new("oa"), new("ob")], original.GraphState);
    }
}
