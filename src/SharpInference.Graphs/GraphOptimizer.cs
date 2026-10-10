namespace SharpInference.Graphs;

public enum OptimizationBoundary
{
    Off,
    WithinStage,
    WithinLayer,
    Unrestricted,
}

public enum GraphDefinitionPolicy
{
    Canonicalize,
    PreserveExpanded,
}

public sealed record GraphOptimizationOptions(
    OptimizationBoundary Boundary = OptimizationBoundary.WithinStage,
    int MaximumPasses = 16,
    GraphDefinitionPolicy DefinitionPolicy = GraphDefinitionPolicy.Canonicalize);

public interface IExecutionCapabilityProvider
{
    bool Supports(GraphOperationId operation);
}

public sealed class PortableExecutionCapabilities : IExecutionCapabilityProvider
{
    public static PortableExecutionCapabilities Instance { get; } = new();
    private PortableExecutionCapabilities() { }
    public bool Supports(GraphOperationId operation) => true;
}

public sealed record SequenceFusionRule(
    string Id,
    IReadOnlyList<GraphOperationId> Pattern,
    GraphOperationId Replacement,
    OptimizationBoundary Scope = OptimizationBoundary.Unrestricted,
    IReadOnlyList<FusionArgumentBinding>? Arguments = null);

public sealed record FusionArgumentBinding(string Name, int NodeIndex, string SourcePort);

public sealed record GraphNodeDefinition(
    string Id,
    GraphOperationId Operation,
    IReadOnlyList<GraphOperationId> Body,
    OptimizationBoundary Scope = OptimizationBoundary.WithinStage,
    IReadOnlyList<FusionArgumentBinding>? Arguments = null)
{
    internal SequenceFusionRule ToFusionRule() => new(Id, Body, Operation, Scope, Arguments);
}

public interface IExecutionKernelCatalog : IExecutionCapabilityProvider
{
    string BackendId { get; }
    IReadOnlySet<GraphOperationId> SupportedOperations { get; }
    IReadOnlyList<GraphNodeDefinition> NodeDefinitions { get; }
}

public sealed class ExecutionKernelCatalog : IExecutionKernelCatalog, IFusedOperatorProvider
{
    public ExecutionKernelCatalog(
        string backendId,
        IEnumerable<GraphOperationId> supportedOperations,
        IEnumerable<GraphNodeDefinition>? nodeDefinitions = null,
        IEnumerable<FusedOperatorDescription>? fusedOperators = null)
    {
        BackendId = string.IsNullOrWhiteSpace(backendId)
            ? throw new ArgumentException("A backend identifier is required.", nameof(backendId))
            : backendId;
        SupportedOperations = supportedOperations?.ToHashSet()
            ?? throw new ArgumentNullException(nameof(supportedOperations));
        NodeDefinitions = nodeDefinitions?.ToArray() ?? [];
        FusedOperators = fusedOperators?.ToArray() ?? [];
    }

    public string BackendId { get; }
    public IReadOnlySet<GraphOperationId> SupportedOperations { get; }
    public IReadOnlyList<GraphNodeDefinition> NodeDefinitions { get; }
    public IReadOnlyList<FusedOperatorDescription> FusedOperators { get; }
    public bool Supports(GraphOperationId operation) => SupportedOperations.Contains(operation);
}

public sealed class GraphOptimizer
{
    private static readonly IReadOnlySet<GraphOperationId> PureOperations = new HashSet<GraphOperationId>
    {
        PrimitiveGraphOperations.Copy,
        PrimitiveGraphOperations.Add,
        PrimitiveGraphOperations.Subtract,
        PrimitiveGraphOperations.Multiply,
        PrimitiveGraphOperations.Divide,
        PrimitiveGraphOperations.Maximum,
        PrimitiveGraphOperations.Exp,
        PrimitiveGraphOperations.Tanh,
        PrimitiveGraphOperations.Sigmoid,
        PrimitiveGraphOperations.ReciprocalSquareRoot,
        PrimitiveGraphOperations.Square,
        PrimitiveGraphOperations.Relu,
        PrimitiveGraphOperations.ReduceSum,
        PrimitiveGraphOperations.ReduceMean,
        PrimitiveGraphOperations.MatVec,
        PrimitiveGraphOperations.GatherRow,
        PortableTensorOperationContracts.Fill,
        PortableTensorOperationContracts.CastFp16ToFp32,
        PortableTensorOperationContracts.Reshape,
        PortableTensorOperationContracts.Slice,
        PortableTensorOperationContracts.Broadcast,
        PortableTensorOperationContracts.BatchedMatVec,
        PortableTensorOperationContracts.ReduceLastSum,
        PortableTensorOperationContracts.ReduceLastMean,
        PortableTensorOperationContracts.HeadOuter,
    };

    public ExecutionGraph Optimize(
        LogicalGraph graph,
        GraphOptimizationOptions options,
        IExecutionKernelCatalog kernelCatalog)
    {
        return Optimize(
            graph,
            options,
            kernelCatalog,
            kernelCatalog as IFusedOperatorProvider);
    }

    public ExecutionGraph Optimize(
        LogicalGraph graph,
        GraphOptimizationOptions options,
        IExecutionKernelCatalog kernelCatalog,
        IFusedOperatorProvider? fusedOperatorProvider)
    {
        ArgumentNullException.ThrowIfNull(kernelCatalog);
        var definitionRules = options.DefinitionPolicy == GraphDefinitionPolicy.Canonicalize
            ? kernelCatalog.NodeDefinitions.Select(definition => definition.ToFusionRule())
            : [];
        if (options.Boundary == OptimizationBoundary.Off)
        {
            if (options.DefinitionPolicy == GraphDefinitionPolicy.PreserveExpanded)
            {
                return OptimizeCore(graph, options, null, kernelCatalog, false, null);
            }

            return OptimizeCore(
                graph,
                options with { Boundary = OptimizationBoundary.Unrestricted },
                definitionRules,
                kernelCatalog,
                false, null);
        }

        var fusedOperators = (fusedOperatorProvider?.FusedOperators ?? [])
            .ToArray();
        var duplicateId = fusedOperators
            .GroupBy(fused => fused.Id, StringComparer.Ordinal)
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicateId is not null)
        {
            throw new InvalidDataException($"Fused operator identifier '{duplicateId.Key}' is registered more than once.");
        }

        return OptimizeCore(
            graph,
            options,
            definitionRules.Concat(fusedOperators.Select(fused => fused.Rule)),
            kernelCatalog,
            true,
            fusedOperatorProvider as IFusedElementwiseExpressionProvider ??
                kernelCatalog as IFusedElementwiseExpressionProvider);
    }

    public ExecutionGraph Optimize(
        LogicalGraph graph,
        GraphOptimizationOptions? options = null,
        IEnumerable<SequenceFusionRule>? rules = null,
        IExecutionCapabilityProvider? capabilities = null)
    {
        return OptimizeCore(graph, options, rules, capabilities,
            options?.Boundary != OptimizationBoundary.Off,
            capabilities as IFusedElementwiseExpressionProvider);
    }

    private static ExecutionGraph OptimizeCore(
        LogicalGraph graph,
        GraphOptimizationOptions? options,
        IEnumerable<SequenceFusionRule>? rules,
        IExecutionCapabilityProvider? capabilities,
        bool eliminateDeadNodes,
        IFusedElementwiseExpressionProvider? expressionProvider)
    {
        ArgumentNullException.ThrowIfNull(graph);
        options ??= new GraphOptimizationOptions();
        if (graph.Nodes.Any(node => node.Resources.Any(binding => binding.View is not null)))
        {
            if (options.Boundary != OptimizationBoundary.Off || options.DefinitionPolicy != GraphDefinitionPolicy.PreserveExpanded)
                throw new NotSupportedException("View-bound graphs currently require expanded definitions and disabled fusion.");
            eliminateDeadNodes = false;
        }
        if (options.MaximumPasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "MaximumPasses must be positive.");
        }

        capabilities ??= PortableExecutionCapabilities.Instance;
        var nodes = graph.Nodes.Select((node, index) => new MutableNode(
            new ExecutionNodeId($"e{index:D4}"),
            node.Operation,
            node.Region,
            node.Resources.ToList(),
            new List<NodeResourceBinding>(),
            node.Dependencies.Select(dependency => new ExecutionNodeId($"e{IndexOf(graph.Nodes, dependency):D4}")).ToHashSet(),
            new Dictionary<string, string>(node.Attributes, StringComparer.Ordinal),
            node.Requirements,
            [node.Id],
            null)).ToList();

        var enabledRules = options.Boundary == OptimizationBoundary.Off
            ? []
            : (rules ?? []).ToArray();
        foreach (var rule in enabledRules)
        {
            ValidateRule(rule);
        }
        if (capabilities is IExecutionKernelCatalog)
        {
            var reachable = new HashSet<GraphOperationId>(
                enabledRules.Where(rule => capabilities.Supports(rule.Replacement))
                    .Select(rule => rule.Replacement));
            bool changed;
            do
            {
                changed = false;
                foreach (var rule in enabledRules)
                {
                    if (reachable.Contains(rule.Replacement))
                    {
                        foreach (var operation in rule.Pattern)
                        {
                            changed |= reachable.Add(operation);
                        }
                    }
                }
            } while (changed);

            enabledRules = enabledRules
                .Where(rule => capabilities.Supports(rule.Replacement) || reachable.Contains(rule.Replacement))
                .ToArray();
        }
        var regionMap = graph.Regions.ToDictionary(region => region.Id);
        for (var pass = 0; pass < options.MaximumPasses; pass++)
        {
            var changed = false;
            foreach (var rule in enabledRules)
            {
                for (var index = 0; index <= nodes.Count - rule.Pattern.Count;)
                {
                    var candidates = nodes.GetRange(index, rule.Pattern.Count);
                    if (!candidates.Select(node => node.Operation).SequenceEqual(rule.Pattern) ||
                        !BoundaryAllows(candidates.Select(node => node.Region), options.Boundary, regionMap) ||
                        !BoundaryAllows(candidates.Select(node => node.Region), rule.Scope, regionMap))
                    {
                        index++;
                        continue;
                    }

                    Fuse(graph, nodes, index, candidates, rule, regionMap);
                    changed = true;
                    index++;
                }
            }

            if (!changed)
            {
                if (eliminateDeadNodes && capabilities is IExecutionKernelCatalog catalog &&
                    catalog.Supports(FusedElementwiseExpressionContract.Operation) &&
                    expressionProvider is not null)
                {
                    FuseElementwiseExpressions(graph, nodes, options.Boundary, expressionProvider);
                }

                if (eliminateDeadNodes)
                {
                    EliminateDeadPureNodes(graph, nodes);
                }

                var execution = CreateExecutionGraph(graph, nodes);
                var unsupported = execution.Nodes
                    .Where(node => !capabilities.Supports(node.Operation))
                    .Select(node => new BackendPreparationDiagnostic(
                        node.Id,
                        node.Operation,
                        node.Source.LogicalNodes,
                        BackendPreparationFailureReason.UnsupportedOperation,
                        $"Operation '{node.Operation}' is not supported.",
                        node.Requirements,
                        Array.Empty<OperatorImplementationDescription>()))
                    .ToArray();
                if (unsupported.Length != 0)
                {
                    throw new BackendPreparationException(
                        (capabilities as IExecutionKernelCatalog)?.BackendId ?? "execution",
                        unsupported);
                }

                return execution;
            }
        }

        throw new InvalidOperationException($"Graph optimization did not converge within {options.MaximumPasses} passes.");
    }

    private static void EliminateDeadPureNodes(LogicalGraph graph, List<MutableNode> nodes)
    {
        var resourceMap = graph.Resources.ToDictionary(resource => resource.Id);
        var observable = graph.Inputs.Concat(graph.Outputs)
            .Concat(graph.GraphState.Select(entry => entry.Resource))
            .ToHashSet();

        for (var index = nodes.Count - 1; index >= 0;)
        {
            var node = nodes[index];
            var writes = node.Resources
                .Where(binding => binding.Access != GraphResourceAccess.Read)
                .Select(binding => binding.Resource)
                .ToArray();
            var reads = node.Resources
                .Where(binding => binding.Access != GraphResourceAccess.Write)
                .Select(binding => binding.Resource)
                .ToHashSet();
            var removable = node.RuleId is null &&
                node.InternalResources.Count == 0 &&
                node.FirstWrites.Count == 0 &&
                PureOperations.Contains(node.Operation) &&
                writes.Length == 1 &&
                !reads.Contains(writes[0]) &&
                node.Resources.All(binding => binding.Access != GraphResourceAccess.ReadWrite) &&
                !observable.Contains(writes[0]) &&
                resourceMap.TryGetValue(writes[0], out var resource) &&
                resource.Kind is GraphResourceKind.Temporary or GraphResourceKind.TokenTransient &&
                resource.Scope == GraphResourceScope.Local &&
                resource.BindingKey is null &&
                !nodes.Any(other => other.Id != node.Id &&
                    other.Resources.Any(binding => binding.Resource == writes[0]));
            if (!removable)
            {
                index--;
                continue;
            }

            nodes.RemoveAt(index);
            foreach (var dependent in nodes.Where(other => other.Dependencies.Remove(node.Id)))
            {
                dependent.Dependencies.UnionWith(node.Dependencies);
            }

            index = nodes.Count - 1;
        }
    }

    private static void FuseElementwiseExpressions(
        LogicalGraph graph,
        List<MutableNode> nodes,
        OptimizationBoundary boundary,
        IFusedElementwiseExpressionProvider provider)
    {
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var regions = graph.Regions.ToDictionary(region => region.Id);
        var observable = graph.Inputs.Concat(graph.Outputs)
            .Concat(graph.GraphState.Select(entry => entry.Resource)).ToHashSet();
        for (var start = 0; start < nodes.Count - 1;)
        {
            if (!IsElementwise(nodes[start], resources))
            {
                start++;
                continue;
            }

            var end = start + 1;
            while (end < nodes.Count &&
                IsElementwise(nodes[end], resources) &&
                BoundaryAllows(nodes.GetRange(start, end - start + 1).Select(node => node.Region),
                    boundary, regions) &&
                CanChain(nodes, nodes[end - 1], nodes[end], resources, observable))
            {
                end++;
            }

            var fused = false;
            for (var count = end - start; count >= 2 && !fused; count--)
            {
                var candidates = nodes.GetRange(start, count);
                if (HasExternalDependencyCycle(nodes, candidates))
                    continue;
                var inputs = new Dictionary<ResourceId, int>();
                var steps = new List<ElementwiseInstruction>();
                for (var stepIndex = 0; stepIndex < count; stepIndex++)
                {
                    var node = candidates[stepIndex];
                    var previousOutput = stepIndex == 0 ? (ResourceId?)null : Output(candidates[stepIndex - 1]);
                    var arguments = OrderedInputs(node).Select(binding =>
                    {
                        if (binding.Resource == previousOutput)
                            return new ElementwiseOperand(StepIndex: stepIndex - 1);
                        if (!inputs.TryGetValue(binding.Resource, out var inputIndex))
                        {
                            inputIndex = inputs.Count;
                            inputs.Add(binding.Resource, inputIndex);
                        }
                        return new ElementwiseOperand(InputIndex: inputIndex);
                    }).ToArray();
                    steps.Add(new ElementwiseInstruction(node.Operation, arguments));
                }

                var output = Output(candidates[^1]);
                if (inputs.ContainsKey(output) ||
                    candidates.Take(count - 1).Any(node => inputs.ContainsKey(Output(node))) ||
                    nodes.Except(candidates).Any(node => node.Resources.Any(binding =>
                        binding.Resource == output && binding.Access != GraphResourceAccess.Read)))
                    continue;
                var signature = new OperatorSignature(
                    Enumerable.Repeat(GraphElementType.Float32, inputs.Count),
                    [GraphElementType.Float32]);
                var implementations = provider.GetExpressionImplementations(signature, resources[output].Tensor)
                    ?? throw new InvalidOperationException("The expression provider returned null implementations.");
                if (OperatorImplementationSelector.Select(FusedElementwiseExpressionContract.Operation,
                        signature,
                        new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32),
                        implementations) is null)
                    continue;

                var replaced = candidates.Select(node => node.Id).ToHashSet();
                var fusedId = candidates[0].Id;
                var fusedNode = new MutableNode(
                    fusedId,
                    FusedElementwiseExpressionContract.Operation,
                    LowestCommonAncestor(candidates.Select(node => node.Region), regions),
                    [.. inputs.OrderBy(pair => pair.Value)
                        .Select(pair => GraphBindings.Read($"input{pair.Value}", pair.Key.Value)),
                        GraphBindings.Write("output", output.Value)],
                    [],
                    candidates.SelectMany(node => node.Dependencies)
                        .Where(id => !replaced.Contains(id)).ToHashSet(),
                    FusedElementwiseExpressionContract.ToAttributes(
                        new FusedElementwiseExpression(inputs.Count, steps)),
                    new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32),
                    candidates.SelectMany(node => node.Sources).ToList(),
                    "core.fuse-elementwise-expression");
                nodes.RemoveRange(start, count);
                nodes.Insert(start, fusedNode);
                foreach (var dependent in nodes.Where(node => node.Id != fusedId &&
                    node.Dependencies.RemoveWhere(replaced.Contains) != 0))
                {
                    dependent.Dependencies.Add(fusedId);
                }
                fused = true;
            }
            start++;
        }
    }

    private static bool HasExternalDependencyCycle(
        IReadOnlyList<MutableNode> nodes, IReadOnlyList<MutableNode> candidates)
    {
        var selected = candidates.Select(node => node.Id).ToHashSet();
        var byId = nodes.ToDictionary(node => node.Id);
        var visited = new HashSet<ExecutionNodeId>();
        var pending = new Stack<ExecutionNodeId>(
            candidates.SelectMany(node => node.Dependencies).Where(id => !selected.Contains(id)));
        while (pending.TryPop(out var id))
        {
            if (selected.Contains(id))
                return true;
            if (visited.Add(id))
            {
                foreach (var dependency in byId[id].Dependencies)
                    pending.Push(dependency);
            }
        }
        return false;
    }

    private static bool IsElementwise(
        MutableNode node, IReadOnlyDictionary<ResourceId, GraphResource> resources)
    {
        var arity = FusedElementwiseExpressionContract.Arity(node.Operation);
        if (arity == 0 || node.RuleId is not null || node.InternalResources.Count != 0 ||
            node.FirstWrites.Count != 0 || node.Attributes.Count != 0 ||
            node.Resources.Count != arity + 1 ||
            node.Requirements.MinimumArithmeticType != GraphElementType.Float32 ||
            node.Requirements.MinimumAccumulatorType != GraphElementType.Float32 ||
            node.Resources.Count(binding => binding.Port == "output" &&
                binding.Access == GraphResourceAccess.Write) != 1)
            return false;

        var ports = arity == 1 ? new[] { "input" } : new[] { "left", "right" };
        if (ports.Any(port => node.Resources.Count(binding =>
                binding.Port == port && binding.Access == GraphResourceAccess.Read) != 1))
            return false;
        var output = Output(node);
        if (node.Resources.Any(binding => binding.Port != "output" && binding.Resource == output))
            return false;
        var target = resources[output];
        if (target.Kind is GraphResourceKind.Input or GraphResourceKind.Weight or
            GraphResourceKind.Constant or GraphResourceKind.SessionState ||
            target.BindingKey is not null || target.Scope != GraphResourceScope.Local)
            return false;
        return node.Resources.All(binding =>
        {
            var tensor = resources[binding.Resource].Tensor;
            return tensor.ElementType == GraphElementType.Float32 &&
                tensor.Layout == "dense" &&
                tensor.Dimensions.SequenceEqual(target.Tensor.Dimensions);
        });
    }

    private static bool CanChain(
        IReadOnlyList<MutableNode> nodes,
        MutableNode first,
        MutableNode second,
        IReadOnlyDictionary<ResourceId, GraphResource> resources,
        IReadOnlySet<ResourceId> observable)
    {
        var intermediate = Output(first);
        var resource = resources[intermediate];
        return second.Dependencies.Contains(first.Id) &&
            OrderedInputs(second).Count(binding => binding.Resource == intermediate) == 1 &&
            resource.Kind is GraphResourceKind.Temporary or GraphResourceKind.TokenTransient &&
            !observable.Contains(intermediate) &&
            !nodes.Any(node => node != first && node != second &&
                node.Resources.Any(binding => binding.Resource == intermediate));
    }

    private static ResourceId Output(MutableNode node) =>
        node.Resources.Single(binding => binding.Port == "output").Resource;

    private static IEnumerable<NodeResourceBinding> OrderedInputs(MutableNode node) =>
        FusedElementwiseExpressionContract.Arity(node.Operation) == 1
            ? [node.Resources.Single(binding => binding.Port == "input")]
            : [node.Resources.Single(binding => binding.Port == "left"),
                node.Resources.Single(binding => binding.Port == "right")];

    private static void Fuse(
        LogicalGraph graph,
        List<MutableNode> nodes,
        int index,
        IReadOnlyList<MutableNode> candidates,
        SequenceFusionRule rule,
        IReadOnlyDictionary<RegionId, GraphRegion> regionMap)
    {
        var replacedIds = candidates.Select(node => node.Id).ToHashSet();
        var fusedId = candidates[0].Id;
        var dependencies = candidates
            .SelectMany(node => node.Dependencies)
            .Where(dependency => !replacedIds.Contains(dependency))
            .ToHashSet();
        var allBindings = candidates
            .SelectMany((node, nodeIndex) => node.Resources.Select(binding => (nodeIndex, binding)))
            .ToArray();
        var privateResources = candidates.SelectMany(node => node.InternalResources.Select(binding => binding.Resource)).ToHashSet();
        var resources = new List<NodeResourceBinding>();
        var internalResources = candidates.SelectMany(node => node.InternalResources).ToList();
        var firstWrites = new HashSet<ResourceId>();
        var graphEndpoints = graph.Inputs.Concat(graph.Outputs).ToHashSet();
        var resourceMap = graph.Resources.ToDictionary(resource => resource.Id);
        foreach (var group in allBindings.GroupBy(item => item.binding.Resource))
        {
            var first = group.First();
            var firstNodeWrites = group
                .Where(item => item.nodeIndex == first.nodeIndex)
                .All(item => item.binding.Access == GraphResourceAccess.Write ||
                    candidates[item.nodeIndex].FirstWrites.Contains(group.Key));
            var resource = resourceMap[group.Key];
            var isPrivate = privateResources.Contains(group.Key) ||
                resource.Kind is GraphResourceKind.Temporary or GraphResourceKind.TokenTransient &&
                firstNodeWrites &&
                !graphEndpoints.Contains(group.Key) &&
                !nodes.Except(candidates).Any(node =>
                    node.Resources.Any(binding => binding.Resource == group.Key));
            var access = MergeAccess(group.Select(item => item.binding.Access));
            if (isPrivate)
            {
                var existing = internalResources.FindIndex(binding => binding.Resource == group.Key);
                if (existing >= 0)
                {
                    internalResources[existing] = internalResources[existing] with
                    {
                        Access = MergeAccess([internalResources[existing].Access, access]),
                    };
                }
                else
                {
                    internalResources.Add(new NodeResourceBinding(
                        $"scratch.{group.Key.Value}", group.Key, access,
                        InitializedBeforeRead: firstNodeWrites));
                }
                continue;
            }
            if (firstNodeWrites && access == GraphResourceAccess.ReadWrite)
                firstWrites.Add(group.Key);

            var explicitArguments = rule.Arguments?.Where(argument =>
                group.Any(item => item.nodeIndex == argument.NodeIndex &&
                    item.binding.Port == argument.SourcePort)).ToArray();
            if (explicitArguments is { Length: > 0 })
            {
                foreach (var argument in explicitArguments)
                {
                    var source = group.Single(item => item.nodeIndex == argument.NodeIndex &&
                        item.binding.Port == argument.SourcePort);
                    resources.Add(new NodeResourceBinding(argument.Name, group.Key, source.binding.Access));
                }
            }
            else
            {
                resources.Add(new NodeResourceBinding(
                    $"node{first.nodeIndex}.{first.binding.Port}", group.Key, access));
            }
        }
        if (rule.Arguments is not null && rule.Arguments.Any(argument =>
            !allBindings.Any(item => item.nodeIndex == argument.NodeIndex &&
                item.binding.Port == argument.SourcePort &&
                resources.Any(binding => binding.Resource == item.binding.Resource))))
        {
            throw new InvalidDataException($"Fusion rule '{rule.Id}' names an internal or missing source port.");
        }
        if (resources.Select(binding => binding.Port).Distinct(StringComparer.Ordinal).Count() != resources.Count)
        {
            throw new InvalidDataException($"Fusion rule '{rule.Id}' produces duplicate argument names.");
        }
        var fused = new MutableNode(
            fusedId,
            rule.Replacement,
            LowestCommonAncestor(candidates.Select(node => node.Region), regionMap),
            resources,
            internalResources,
            dependencies,
            new Dictionary<string, string>(StringComparer.Ordinal),
            PrecisionRequirement.Merge(candidates.Select(candidate => candidate.Requirements)),
            candidates.SelectMany(node => node.Sources).Distinct().ToList(),
            rule.Id)
        {
            FirstWrites = firstWrites,
        };

        nodes.RemoveRange(index, candidates.Count);
        nodes.Insert(index, fused);
        foreach (var node in nodes)
        {
            if (node.Id == fusedId)
            {
                continue;
            }

            if (node.Dependencies.RemoveWhere(replacedIds.Contains) != 0)
            {
                node.Dependencies.Add(fusedId);
            }
        }
    }

    private static ExecutionGraph CreateExecutionGraph(
        LogicalGraph graph,
        IReadOnlyList<MutableNode> nodes)
    {
        return new(
            graph.Identity with { Name = $"{graph.Identity.Name}.execution" },
            graph.Model,
            graph.Resources,
            graph.Regions,
            nodes.Select(node => new ExecutionNode(
                node.Id,
                node.Operation,
                node.Region,
                node.Resources,
                node.Dependencies.OrderBy(dependency => dependency.Value, StringComparer.Ordinal).ToArray(),
                node.Attributes,
                node.Requirements,
                new ExecutionSourceMap(node.Sources, node.RuleId))
            {
                InternalResources = node.InternalResources,
                FirstWriteResources = node.FirstWrites.OrderBy(id => id.Value, StringComparer.Ordinal).ToArray(),
            }),
            graph.Inputs,
            graph.Outputs,
            graph.GraphState);
    }

    private static bool BoundaryAllows(
        IEnumerable<RegionId> regions,
        OptimizationBoundary boundary,
        IReadOnlyDictionary<RegionId, GraphRegion> regionMap)
    {
        if (boundary == OptimizationBoundary.Unrestricted)
        {
            return true;
        }

        var lca = regionMap[LowestCommonAncestor(regions, regionMap)];
        while (lca.Type == GraphRegionTypes.Architecture && lca.ParentId is { } parent)
            lca = regionMap[parent];
        return boundary switch
        {
            OptimizationBoundary.WithinStage => string.Equals(lca.Type, GraphRegionTypes.Stage, StringComparison.Ordinal),
            OptimizationBoundary.WithinLayer => string.Equals(lca.Type, GraphRegionTypes.Stage, StringComparison.Ordinal) ||
                                                string.Equals(lca.Type, GraphRegionTypes.Layer, StringComparison.Ordinal),
            _ => false,
        };
    }

    private static RegionId LowestCommonAncestor(
        IEnumerable<RegionId> regions,
        IReadOnlyDictionary<RegionId, GraphRegion> regionMap)
    {
        var paths = regions.Select(region => Ancestors(region, regionMap).ToArray()).ToArray();
        if (paths.Length == 0)
        {
            throw new InvalidOperationException("At least one region is required.");
        }

        foreach (var candidate in paths[0])
        {
            if (paths.All(path => path.Contains(candidate)))
            {
                return candidate;
            }
        }

        throw new InvalidDataException("Regions do not share a graph root.");
    }

    private static IEnumerable<RegionId> Ancestors(
        RegionId region,
        IReadOnlyDictionary<RegionId, GraphRegion> regionMap)
    {
        var current = region;
        while (true)
        {
            yield return current;
            var parent = regionMap[current].ParentId;
            if (parent is not RegionId parentId)
            {
                yield break;
            }

            current = parentId;
        }
    }

    private static int IndexOf(IReadOnlyList<LogicalNode> nodes, LogicalNodeId id)
    {
        for (var index = 0; index < nodes.Count; index++)
        {
            if (nodes[index].Id == id) return index;
        }

        throw new InvalidDataException($"Unknown logical dependency '{id}'.");
    }

    private static void ValidateRule(SequenceFusionRule rule)
    {
        if (string.IsNullOrWhiteSpace(rule.Id) || rule.Pattern.Count < 2 ||
            rule.Pattern.Any(operation => string.IsNullOrWhiteSpace(operation.Name)) ||
            string.IsNullOrWhiteSpace(rule.Replacement.Name) ||
            rule.Scope == OptimizationBoundary.Off)
        {
            throw new InvalidDataException("Fusion rules require an identifier, at least two valid pattern operations, a replacement, and a non-Off scope.");
        }
        if (rule.Arguments is not null &&
            (rule.Arguments.Any(argument => string.IsNullOrWhiteSpace(argument.Name) ||
                string.IsNullOrWhiteSpace(argument.SourcePort) ||
                argument.NodeIndex < 0 || argument.NodeIndex >= rule.Pattern.Count) ||
             rule.Arguments.Select(argument => (argument.NodeIndex, argument.SourcePort)).Distinct().Count() != rule.Arguments.Count ||
             rule.Arguments.Select(argument => argument.Name).Distinct(StringComparer.Ordinal).Count() != rule.Arguments.Count))
        {
            throw new InvalidDataException($"Fusion rule '{rule.Id}' has invalid argument bindings.");
        }
    }

    private static GraphResourceAccess MergeAccess(IEnumerable<GraphResourceAccess> accesses)
    {
        var reads = false;
        var writes = false;
        foreach (var access in accesses)
        {
            reads |= access != GraphResourceAccess.Write;
            writes |= access != GraphResourceAccess.Read;
        }

        return (reads, writes) switch
        {
            (true, true) => GraphResourceAccess.ReadWrite,
            (true, false) => GraphResourceAccess.Read,
            (false, true) => GraphResourceAccess.Write,
            _ => throw new InvalidDataException("A fused resource has no access mode."),
        };
    }

    private sealed record MutableNode(
        ExecutionNodeId Id,
        GraphOperationId Operation,
        RegionId Region,
        List<NodeResourceBinding> Resources,
        List<NodeResourceBinding> InternalResources,
        HashSet<ExecutionNodeId> Dependencies,
        IReadOnlyDictionary<string, string> Attributes,
        PrecisionRequirement Requirements,
        List<LogicalNodeId> Sources,
        string? RuleId)
    {
        public HashSet<ResourceId> FirstWrites { get; init; } = [];
    }
}
