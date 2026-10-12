using System.Collections.ObjectModel;

namespace SharpInference.Graphs;

/// <summary>Validated structural rewrites used by execution graph postprocessors.</summary>
public static class ExecutionGraphRewriter
{
    /// <summary>Replaces one graph call with one concrete operation while preserving its boundary bindings and dependencies.</summary>
    public static ExecutionGraphProgram ReplaceGraphCallWithOperation(
        ExecutionGraphProgram program,
        ExecutionGraphId graphId,
        ExecutionNodeId callId,
        GraphOperationId operation,
        IReadOnlyDictionary<string, string>? attributes = null,
        PrecisionRequirement? requirements = null,
        string? transformation = null)
    {
        ArgumentNullException.ThrowIfNull(program);
        var graph = RequiredGraph(program, graphId);
        var call = graph.Nodes.SingleOrDefault(node => node.Id == callId) as ExecutionGraphCallNode
            ?? throw new ArgumentException($"Graph '{graphId}' has no call '{callId}'.", nameof(callId));
        var replacement = new ExecutionOperationNode(
            call.Id,
            operation,
            call.Bindings,
            call.Dependencies,
            new ReadOnlyDictionary<string, string>(
                new Dictionary<string, string>(
                    attributes ?? new Dictionary<string, string>(),
                    StringComparer.Ordinal)),
            requirements ?? new PrecisionRequirement(
                GraphElementType.Float32,
                GraphElementType.Float32),
            call.Source with
            {
                Transformation = transformation ?? "replace-graph-call-with-operation",
            },
            call.Architecture);
        return ReplaceGraph(
            program,
            CopyGraph(graph, nodes: graph.Nodes.Select(node =>
                node.Id == callId ? replacement : node)));
    }

    /// <summary>Fuses operation nodes into one supplied operation and reconnects incoming and outgoing dependencies.</summary>
    public static ExecutionGraphProgram FuseNodesToOperation(
        ExecutionGraphProgram program,
        ExecutionGraphId graphId,
        IEnumerable<ExecutionNodeId> nodeIds,
        ExecutionNodeId replacementId,
        GraphOperationId operation,
        IEnumerable<ExecutionGraphBinding> bindings,
        IReadOnlyDictionary<string, string>? attributes = null,
        PrecisionRequirement? requirements = null,
        GraphArchitecture? architecture = null,
        string? transformation = null)
    {
        ArgumentNullException.ThrowIfNull(program);
        var graph = RequiredGraph(program, graphId);
        var selected = nodeIds?.ToHashSet()
            ?? throw new ArgumentNullException(nameof(nodeIds));
        if (selected.Count < 2)
            throw new ArgumentException("Fusion requires at least two nodes.", nameof(nodeIds));
        var sources = graph.Nodes.Where(node => selected.Contains(node.Id)).ToArray();
        if (sources.Length != selected.Count || sources.Any(node => node is not ExecutionOperationNode))
            throw new ArgumentException("Every fused node must be an operation in the selected graph.", nameof(nodeIds));
        if (graph.Nodes.Any(node => node.Id == replacementId && !selected.Contains(node.Id)))
            throw new ArgumentException($"Replacement node '{replacementId}' already exists.", nameof(replacementId));

        var dependencies = sources.SelectMany(node => node.Dependencies)
            .Where(dependency => !selected.Contains(dependency))
            .Distinct().ToArray();
        var replacement = new ExecutionOperationNode(
            replacementId,
            operation,
            bindings?.ToArray() ?? throw new ArgumentNullException(nameof(bindings)),
            dependencies,
            new ReadOnlyDictionary<string, string>(
                new Dictionary<string, string>(
                    attributes ?? new Dictionary<string, string>(),
                    StringComparer.Ordinal)),
            requirements ?? MergeRequirements(sources.Cast<ExecutionOperationNode>()),
            new ExecutionSourceMap(
                sources.SelectMany(node => node.Source.LogicalNodes).Distinct().ToArray(),
                transformation ?? "fuse-nodes-to-operation"),
            architecture);
        var nodes = graph.Nodes.Where(node => !selected.Contains(node.Id))
            .Select(node => RewriteDependencies(node, selected, replacementId))
            .Append(replacement)
            .ToArray();
        return ReplaceGraph(program, CopyGraph(graph, nodes: nodes));
    }

    /// <summary>Extracts operation nodes into a new reusable graph and replaces them with one graph call.</summary>
    public static ExecutionGraphProgram ExtractNodesToGraph(
        ExecutionGraphProgram program,
        ExecutionGraphId sourceGraphId,
        IEnumerable<ExecutionNodeId> nodeIds,
        ExecutionGraphId extractedGraphId,
        ExecutionNodeId callId,
        GraphCompileBoundary boundary,
        GraphArchitecture? architecture = null)
    {
        ArgumentNullException.ThrowIfNull(program);
        if (program.Graphs.Any(graph => graph.Id == extractedGraphId))
            throw new ArgumentException($"Graph '{extractedGraphId}' already exists.", nameof(extractedGraphId));
        var source = RequiredGraph(program, sourceGraphId);
        var selected = nodeIds?.ToHashSet()
            ?? throw new ArgumentNullException(nameof(nodeIds));
        var selectedNodes = source.Nodes.Where(node => selected.Contains(node.Id)).ToArray();
        if (selectedNodes.Length != selected.Count ||
            selectedNodes.Any(node => node is not ExecutionOperationNode))
            throw new ArgumentException("Extraction currently requires operation nodes from one graph.", nameof(nodeIds));
        if (selected.Count == 0)
            throw new ArgumentException("Extraction requires at least one node.", nameof(nodeIds));
        if (source.Nodes.Any(node => node.Id == callId && !selected.Contains(node.Id)))
            throw new ArgumentException($"Call node '{callId}' already exists.", nameof(callId));

        var resourceMap = source.Resources.ToDictionary(resource => resource.Id);
        var outside = source.Nodes.Where(node => !selected.Contains(node.Id)).ToArray();
        var extractedResources = new List<ExecutionGraphResource>();
        var callBindings = new List<ExecutionGraphBinding>();
        foreach (var resourceId in selectedNodes.SelectMany(node => node.Bindings)
            .Select(binding => binding.Resource).Distinct())
        {
            var resource = resourceMap[resourceId];
            var bindings = selectedNodes.SelectMany(node => node.Bindings)
                .Where(binding => binding.Resource == resourceId).ToArray();
            var reads = bindings.Any(binding => binding.Access != GraphResourceAccess.Write);
            var writes = bindings.Any(binding => binding.Access != GraphResourceAccess.Read);
            var externallyRead = outside.SelectMany(node => node.Bindings).Any(binding =>
                binding.Resource == resourceId && binding.Access != GraphResourceAccess.Write);
            var externallyWritten = outside.SelectMany(node => node.Bindings).Any(binding =>
                binding.Resource == resourceId && binding.Access != GraphResourceAccess.Read);
            var portKind = reads && externallyWritten
                ? GraphDefinitionResourceKind.InOut
                : reads && !writes
                    ? GraphDefinitionResourceKind.Input
                    : writes && (externallyRead ||
                        resource.Kind is GraphDefinitionResourceKind.Output
                            or GraphDefinitionResourceKind.InOut)
                        ? GraphDefinitionResourceKind.Output
                        : GraphDefinitionResourceKind.Temporary;
            extractedResources.Add(resource with { Kind = portKind });
            if (portKind is GraphDefinitionResourceKind.Input
                or GraphDefinitionResourceKind.Output
                or GraphDefinitionResourceKind.InOut)
                callBindings.Add(new ExecutionGraphBinding(
                    resourceId.Value,
                    resourceId,
                    portKind switch
                    {
                        GraphDefinitionResourceKind.Input => GraphResourceAccess.Read,
                        GraphDefinitionResourceKind.Output => GraphResourceAccess.Write,
                        _ => GraphResourceAccess.ReadWrite,
                    }));
        }

        var extractedNodes = selectedNodes.Select(node =>
            node switch
            {
                ExecutionOperationNode operation => operation with
                {
                    Dependencies = operation.Dependencies
                        .Where(selected.Contains).ToArray(),
                    Source = operation.Source with
                    {
                        Transformation = "extract-nodes-to-graph",
                    },
                },
                _ => node,
            }).ToArray();
        var incoming = selectedNodes.SelectMany(node => node.Dependencies)
            .Where(dependency => !selected.Contains(dependency)).Distinct().ToArray();
        var call = new ExecutionGraphCallNode(
            callId,
            extractedGraphId,
            callBindings,
            incoming,
            boundary,
            new ExecutionSourceMap(
                selectedNodes.SelectMany(node => node.Source.LogicalNodes).Distinct().ToArray(),
                "extract-nodes-to-graph"),
            architecture);
        var remaining = source.Nodes.Where(node => !selected.Contains(node.Id))
            .Select(node => RewriteDependencies(node, selected, callId))
            .Append(call).ToArray();
        var extracted = new ExecutionGraph(
            extractedGraphId,
            extractedResources,
            extractedNodes,
            boundary,
            architecture);
        return new ExecutionGraphProgram(
            program.Graphs.Where(graph => graph.Id != sourceGraphId)
                .Append(CopyGraph(source, nodes: remaining))
                .Append(extracted));
    }

    /// <summary>Inlines one graph call into its caller and removes an unreferenced target graph.</summary>
    public static ExecutionGraphProgram InlineGraphCall(
        ExecutionGraphProgram program,
        ExecutionGraphId graphId,
        ExecutionNodeId callId)
    {
        ArgumentNullException.ThrowIfNull(program);
        var caller = RequiredGraph(program, graphId);
        var call = caller.Nodes.SingleOrDefault(node => node.Id == callId) as ExecutionGraphCallNode
            ?? throw new ArgumentException($"Graph '{graphId}' has no call '{callId}'.", nameof(callId));
        var target = RequiredGraph(program, call.Target);
        var portBindings = call.Bindings.ToDictionary(
            binding => new ResourceId(binding.Port),
            binding => binding);
        var prefix = call.Id.Value + "/";
        var internalIds = new Dictionary<ResourceId, ResourceId>();
        var callerResources = caller.Resources.ToList();
        foreach (var resource in target.Resources)
        {
            if (portBindings.ContainsKey(resource.Id))
                continue;
            var mapped = new ResourceId(prefix + resource.Id.Value);
            internalIds.Add(resource.Id, mapped);
            callerResources.Add(resource with
            {
                Id = mapped,
                Name = $"{call.Id.Value}/{resource.Name}",
            });
        }

        ResourceId MapResource(ResourceId id) =>
            portBindings.TryGetValue(id, out var bound) && bound.ResourceOwner.IsParent
                ? bound.Resource
                : internalIds.TryGetValue(id, out var mapped)
                    ? mapped
                    : throw new NotSupportedException(
                        $"Inlining call '{callId}' cannot flatten non-parent port binding '{id}'.");

        var idMap = target.Nodes.ToDictionary(
            node => node.Id,
            node => new ExecutionNodeId(prefix + node.Id.Value));
        var successors = target.Nodes.ToDictionary(node => node.Id, _ => 0);
        foreach (var node in target.Nodes)
            foreach (var dependency in node.Dependencies)
                successors[dependency]++;
        var entries = target.Nodes.Where(node => node.Dependencies.Count == 0)
            .Select(node => idMap[node.Id]).ToArray();
        var completions = target.Nodes.Where(node => successors[node.Id] == 0)
            .Select(node => idMap[node.Id]).ToArray();
        var inlined = target.Nodes.Select<ExecutionGraphNode, ExecutionGraphNode>(node => node switch
        {
            ExecutionOperationNode operation => operation with
            {
                Id = idMap[operation.Id],
                Bindings = operation.Bindings.Select(binding => binding with
                {
                    Resource = MapResource(binding.Resource),
                    Instance = binding.ResourceOwner.IsParent
                        ? GraphBindingOwner.Parent
                        : GraphBindingOwner.Instance(new GraphDefinitionNodeId(
                            prefix + binding.ResourceOwner.Call!.Value.Value)),
                }).ToArray(),
                Dependencies = operation.Dependencies.Select(id => idMap[id])
                    .Concat(entries.Contains(idMap[operation.Id]) ? call.Dependencies : [])
                    .Distinct().ToArray(),
                Source = operation.Source with { Transformation = "inline-graph-call" },
            },
            ExecutionGraphCallNode childCall => childCall with
            {
                Id = idMap[childCall.Id],
                Bindings = childCall.Bindings.Select(binding => binding with
                {
                    Resource = binding.ResourceOwner.IsParent
                        ? MapResource(binding.Resource)
                        : binding.Resource,
                    Instance = binding.ResourceOwner.IsParent
                        ? GraphBindingOwner.Parent
                        : GraphBindingOwner.Instance(new GraphDefinitionNodeId(
                            prefix + binding.ResourceOwner.Call!.Value.Value)),
                }).ToArray(),
                Dependencies = childCall.Dependencies.Select(id => idMap[id])
                    .Concat(entries.Contains(idMap[childCall.Id]) ? call.Dependencies : [])
                    .Distinct().ToArray(),
                Source = childCall.Source with { Transformation = "inline-graph-call" },
            },
            _ => throw new InvalidDataException(),
        }).ToArray();
        var completionSet = completions.ToHashSet();
        var remaining = caller.Nodes.Where(node => node.Id != callId)
            .Select(node => ReplaceDependency(node, callId, completionSet))
            .Concat(inlined).ToArray();
        var graphs = program.Graphs.Where(graph => graph.Id != graphId)
            .Append(CopyGraph(caller, callerResources, remaining)).ToList();
        var targetReferences = graphs.SelectMany(graph => graph.Nodes.OfType<ExecutionGraphCallNode>())
            .Count(node => node.Target == target.Id);
        if (targetReferences == 0)
            graphs.RemoveAll(graph => graph.Id == target.Id);
        return new ExecutionGraphProgram(graphs);
    }

    private static ExecutionGraph RequiredGraph(
        ExecutionGraphProgram program,
        ExecutionGraphId graphId) =>
        program.Graphs.SingleOrDefault(graph => graph.Id == graphId)
        ?? throw new ArgumentException($"Unknown execution graph '{graphId}'.", nameof(graphId));

    private static ExecutionGraphProgram ReplaceGraph(
        ExecutionGraphProgram program,
        ExecutionGraph replacement)
    {
        var graphs = program.Graphs.Select(graph =>
            graph.Id == replacement.Id ? replacement : graph).ToDictionary(graph => graph.Id);
        var reachable = new HashSet<ExecutionGraphId>();
        var pending = new Stack<ExecutionGraphId>();
        pending.Push(program.RootGraph);
        while (pending.TryPop(out var graphId))
        {
            if (!reachable.Add(graphId))
                continue;
            foreach (var call in graphs[graphId].Nodes.OfType<ExecutionGraphCallNode>())
                pending.Push(call.Target);
        }

        return new ExecutionGraphProgram(graphs.Values.Where(graph => reachable.Contains(graph.Id)));
    }

    private static ExecutionGraph CopyGraph(
        ExecutionGraph graph,
        IEnumerable<ExecutionGraphResource>? resources = null,
        IEnumerable<ExecutionGraphNode>? nodes = null) =>
        new(
            graph.Id,
            resources ?? graph.Resources,
            nodes ?? graph.Nodes,
            graph.DefaultCompileBoundary,
            graph.Architecture,
            graph.Source);

    private static ExecutionGraphNode RewriteDependencies(
        ExecutionGraphNode node,
        IReadOnlySet<ExecutionNodeId> removed,
        ExecutionNodeId replacement)
    {
        var dependencies = node.Dependencies.Select(dependency =>
            removed.Contains(dependency) ? replacement : dependency).Distinct().ToArray();
        return node switch
        {
            ExecutionOperationNode operation => operation with { Dependencies = dependencies },
            ExecutionGraphCallNode call => call with { Dependencies = dependencies },
            _ => node,
        };
    }

    private static ExecutionGraphNode ReplaceDependency(
        ExecutionGraphNode node,
        ExecutionNodeId removed,
        IReadOnlySet<ExecutionNodeId> replacements)
    {
        var dependencies = node.Dependencies.SelectMany(dependency =>
            dependency == removed
                ? replacements
                : (IEnumerable<ExecutionNodeId>)[dependency]).Distinct().ToArray();
        return node switch
        {
            ExecutionOperationNode operation => operation with { Dependencies = dependencies },
            ExecutionGraphCallNode call => call with { Dependencies = dependencies },
            _ => node,
        };
    }

    private static PrecisionRequirement MergeRequirements(
        IEnumerable<ExecutionOperationNode> operations) =>
        new(
            NumericTypeCompatibility.Maximum(
                operations.Select(operation => operation.Requirements.MinimumArithmeticType)),
            NumericTypeCompatibility.Maximum(
                operations.Select(operation => operation.Requirements.MinimumAccumulatorType)));
}
