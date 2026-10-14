using System.Collections.ObjectModel;
using SharpInference.Graphs.Core;

namespace SharpInference.Graphs;

/// <summary>Rewrites execution graphs while preserving model-level bindings and dependency boundaries.</summary>
public static class ExecutionGraphRewriter
{
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
        var call = graph.Nodes.SingleOrDefault(node => node.Id == callId)
            as ExecutionGraphCallNode
            ?? throw new ArgumentException(
                $"Graph '{graphId}' has no call '{callId}'.", nameof(callId));
        var replacement = new ExecutionOperationNode(
            call.Id,
            operation,
            Copy(attributes),
            requirements ?? new PrecisionRequirement(
                GraphElementType.Float32, GraphElementType.Float32),
            call.Source with
            {
                Transformation = transformation ??
                    "replace-graph-call-with-operation",
            },
            call.Architecture);
        var structure =
            new GraphRewritePlan<ExecutionGraphStructureNode, ExecutionGraphEdge>()
                .AddNode(replacement)
                .Apply(graph.Structure);
        return ReplaceGraph(
            program,
            CopyGraph(graph, structure.Nodes, structure.Edges));
    }

    public static ExecutionGraphProgram FuseNodesToOperation(
        ExecutionGraphProgram program,
        ExecutionGraphId graphId,
        IEnumerable<ExecutionNodeId> nodeIds,
        ExecutionNodeId replacementId,
        GraphOperationId operation,
        IEnumerable<ModelResourceBinding> bindings,
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
        {
            throw new ArgumentException(
                "Fusion requires at least two nodes.", nameof(nodeIds));
        }

        var sources = graph.Nodes.Where(node => selected.Contains(node.Id)).ToArray();
        if (sources.Length != selected.Count ||
            sources.Any(node => node is not ExecutionOperationNode))
        {
            throw new ArgumentException(
                "Every fused node must be an operation in the selected graph.",
                nameof(nodeIds));
        }

        if (graph.Nodes.Any(node =>
            node.Id == replacementId && !selected.Contains(node.Id)))
        {
            throw new ArgumentException(
                $"Replacement node '{replacementId}' already exists.",
                nameof(replacementId));
        }

        var target = GraphRewriteTarget<
            ExecutionGraphStructureNode,
            ExecutionGraphEdge>.Capture(
                graph.Structure,
                sources.Select(node => node.GraphId));
        var replacement = new ExecutionOperationNode(
            replacementId,
            operation,
            Copy(attributes),
            requirements ?? MergeRequirements(
                sources.Cast<ExecutionOperationNode>()),
            new ExecutionSourceMap(
                sources.SelectMany(node => node.Source.LogicalNodes)
                    .Distinct().ToArray(),
                transformation ?? "fuse-nodes-to-operation"),
            architecture);
        var replacementEdges = new List<ExecutionGraphEdge>();
        foreach (var source in target.IncomingEdges.OfType<ExecutionDependencyEdge>()
            .Select(edge => edge.Source).Distinct())
        {
            AddDependency(replacementEdges, source, replacement.GraphId);
        }

        foreach (var successor in target.OutgoingEdges.OfType<ExecutionDependencyEdge>()
            .Select(edge => edge.Target).Distinct())
        {
            AddDependency(replacementEdges, replacement.GraphId, successor);
        }

        AddBindings(replacementEdges, replacement, bindings);
        var plan = new GraphRewritePlan<
                ExecutionGraphStructureNode,
                ExecutionGraphEdge>()
            .Remove(target)
            .AddNode(replacement);
        foreach (var edge in replacementEdges)
        {
            plan.AddEdge(edge);
        }

        var rewritten = plan.Apply(graph.Structure);
        return ReplaceGraph(
            program,
            CopyGraph(graph, rewritten.Nodes, rewritten.Edges));
    }

    /// <summary>Extracts selected operations into a graph and replaces them with a call in the source.</summary>
    /// <param name="program">The program containing the source graph.</param>
    /// <param name="sourceGraphId">The identity of the graph containing the selected operations.</param>
    /// <param name="nodeIds">Operation identities to extract, consumed once and deduplicated.</param>
    /// <param name="extractedGraphId">The unused identity assigned to the extracted graph.</param>
    /// <param name="callId">The identity assigned to the replacement call node.</param>
    /// <param name="boundary">The compilation boundary assigned to the extracted graph.</param>
    /// <param name="architecture">Optional architecture metadata for the extracted graph.</param>
    /// <returns>A program containing the extracted graph and the rewritten source graph.</returns>
    /// <exception cref="ArgumentNullException">The program or node identity sequence is null.</exception>
    /// <exception cref="ArgumentException">An identity is unavailable or the selection is not a nonempty operation set.</exception>
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
        {
            throw new ArgumentException(
                $"Graph '{extractedGraphId}' already exists.",
                nameof(extractedGraphId));
        }

        var source = RequiredGraph(program, sourceGraphId);
        var selected = nodeIds?.ToHashSet()
            ?? throw new ArgumentNullException(nameof(nodeIds));
        var selectedNodes = source.Nodes
            .Where(node => selected.Contains(node.Id)).ToArray();
        if (selected.Count == 0 ||
            selectedNodes.Length != selected.Count ||
            selectedNodes.Any(node => node is not ExecutionOperationNode))
        {
            throw new ArgumentException(
                "Extraction requires operation nodes from one graph.",
                nameof(nodeIds));
        }

        if (source.Nodes.Any(node =>
            node.Id == callId && !selected.Contains(node.Id)))
        {
            throw new ArgumentException(
                $"Call node '{callId}' already exists.", nameof(callId));
        }

        var rewriteTarget = GraphRewriteTarget<
            ExecutionGraphStructureNode,
            ExecutionGraphEdge>.Capture(
                source.Structure,
                selectedNodes.Select(node => node.GraphId));
        var selectedGraphIds = rewriteTarget.Nodes
            .Select(node => node.GraphId).ToHashSet();
        var resourceNodes = source.Structure.NodesOfType<ExecutionResourceNode>()
            .Where(resource => source.Structure.Edges.Any(edge =>
                selectedGraphIds.Contains(edge.Source) &&
                edge.Target == resource.GraphId ||
                edge.Source == resource.GraphId &&
                selectedGraphIds.Contains(edge.Target)))
            .ToArray();
        if (resourceNodes.Any(resource => !resource.ResourceNodeId.Owner.IsParent))
        {
            throw new NotSupportedException(
                "Extraction cannot move direct call-instance resource accesses.");
        }

        var extractedResources = new List<ExecutionResourceNode>();
        var callBindings = new List<ModelResourceBinding>();
        foreach (var resourceNode in resourceNodes)
        {
            var selectedReads = source.Structure.EdgesFromNode(resourceNode.GraphId)
                .OfType<ExecutionResourceReadEdge>()
                .Any(edge => selectedGraphIds.Contains(edge.Target));
            var selectedWrites = source.Structure.EdgesToNode(resourceNode.GraphId)
                .OfType<ExecutionResourceWriteEdge>()
                .Any(edge => selectedGraphIds.Contains(edge.Source));
            var outsideReads = source.Structure.EdgesFromNode(resourceNode.GraphId)
                .OfType<ExecutionResourceReadEdge>()
                .Any(edge => !selectedGraphIds.Contains(edge.Target));
            var outsideWrites = source.Structure.EdgesToNode(resourceNode.GraphId)
                .OfType<ExecutionResourceWriteEdge>()
                .Any(edge => !selectedGraphIds.Contains(edge.Source));
            var kind = selectedReads && outsideWrites
                ? GraphDefinitionResourceKind.InOut
                : selectedReads && !selectedWrites
                    ? GraphDefinitionResourceKind.Input
                    : selectedWrites && (outsideReads ||
                        resourceNode.Kind is GraphDefinitionResourceKind.Output
                            or GraphDefinitionResourceKind.InOut)
                        ? GraphDefinitionResourceKind.Output
                        : GraphDefinitionResourceKind.Temporary;
            extractedResources.Add(resourceNode with
            {
                Resource = resourceNode.Resource with { Kind = kind },
            });
            if (kind is GraphDefinitionResourceKind.Input
                or GraphDefinitionResourceKind.Output
                or GraphDefinitionResourceKind.InOut)
            {
                callBindings.Add(new(
                    resourceNode.Resource.Id.Value,
                    resourceNode.ResourceNodeId,
                    kind switch
                    {
                        GraphDefinitionResourceKind.Input =>
                            GraphResourceAccess.Read,
                        GraphDefinitionResourceKind.Output =>
                            GraphResourceAccess.Write,
                        _ => GraphResourceAccess.ReadWrite,
                    },
                    $"{callId.Value}:{resourceNode.Resource.Id.Value}"));
            }
        }

        var extractedEdges = source.Structure.Edges.Where(edge =>
            edge switch
            {
                ExecutionDependencyEdge =>
                    selectedGraphIds.Contains(edge.Source) &&
                    selectedGraphIds.Contains(edge.Target),
                ExecutionResourceReadEdge =>
                    resourceNodes.Any(resource => resource.GraphId == edge.Source) &&
                    selectedGraphIds.Contains(edge.Target),
                ExecutionResourceWriteEdge =>
                    selectedGraphIds.Contains(edge.Source) &&
                    resourceNodes.Any(resource => resource.GraphId == edge.Target),
                _ => false,
            }).ToArray();
        var extractedNodes = extractedResources
            .Cast<ExecutionGraphStructureNode>()
            .Concat(selectedNodes.Select(node =>
                node is ExecutionOperationNode operation
                    ? operation with
                    {
                        Source = operation.Source with
                        {
                            Transformation = "extract-nodes-to-graph",
                        },
                    }
                    : node))
            .ToArray();
        var extracted = new ExecutionGraph(
            extractedGraphId,
            new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(
                extractedNodes, extractedEdges),
            boundary,
            architecture);

        var call = new ExecutionGraphCallNode(
            callId,
            extractedGraphId,
            boundary,
            new ExecutionSourceMap(
                selectedNodes.SelectMany(node => node.Source.LogicalNodes)
                    .Distinct().ToArray(),
                "extract-nodes-to-graph"),
            architecture);
        var replacementEdges = new List<ExecutionGraphEdge>();
        foreach (var incoming in rewriteTarget.IncomingEdges
            .OfType<ExecutionDependencyEdge>()
            .Select(edge => edge.Source).Distinct())
        {
            AddDependency(replacementEdges, incoming, call.GraphId);
        }

        foreach (var outgoing in rewriteTarget.OutgoingEdges
            .OfType<ExecutionDependencyEdge>()
            .Select(edge => edge.Target).Distinct())
        {
            AddDependency(replacementEdges, call.GraphId, outgoing);
        }

        AddBindings(replacementEdges, call, callBindings);
        var plan = new GraphRewritePlan<
                ExecutionGraphStructureNode,
                ExecutionGraphEdge>()
            .Remove(rewriteTarget)
            .AddNode(call);
        foreach (var edge in replacementEdges)
        {
            plan.AddEdge(edge);
        }

        var callerStructure = plan.Apply(source.Structure);
        var caller = CopyGraph(
            source, callerStructure.Nodes, callerStructure.Edges);
        return new ExecutionGraphProgram(
            program.Graphs.Where(graph => graph.Id != sourceGraphId)
                .Append(caller)
                .Append(extracted));
    }

    /// <summary>Replaces a graph call with mapped callee nodes, resources, and dependency connections.</summary>
    /// <param name="program">The program containing both the caller and callee graphs.</param>
    /// <param name="graphId">The identity of the caller graph.</param>
    /// <param name="callId">The identity of the graph call to inline.</param>
    /// <returns>A rewritten program retaining only graphs reachable from its root.</returns>
    /// <exception cref="ArgumentNullException">The program is null.</exception>
    /// <exception cref="ArgumentException">The caller, callee, or requested graph call cannot be found.</exception>
    public static ExecutionGraphProgram InlineGraphCall(
        ExecutionGraphProgram program,
        ExecutionGraphId graphId,
        ExecutionNodeId callId)
    {
        ArgumentNullException.ThrowIfNull(program);
        var caller = RequiredGraph(program, graphId);
        var call = caller.Nodes.SingleOrDefault(node => node.Id == callId)
            as ExecutionGraphCallNode
            ?? throw new ArgumentException(
                $"Graph '{graphId}' has no call '{callId}'.", nameof(callId));
        var rewriteTarget = GraphRewriteTarget<
            ExecutionGraphStructureNode,
            ExecutionGraphEdge>.Capture(
                caller.Structure,
                [call.GraphId]);
        var target = RequiredGraph(program, call.Target);
        var callBindings = caller.Bindings(call).ToDictionary(
            binding => new ResourceId(binding.Port));
        var prefix = call.Id.Value + "/";
        var nodeMap = target.Nodes.ToDictionary(
            node => node.GraphId,
            node => ExecutionGraphNode.NodeGraphId(
                new ExecutionNodeId(prefix + node.Id.Value)));
        var resourceMap = new Dictionary<GraphNodeId, GraphNodeId>();
        var addedResources = new List<ExecutionResourceNode>();
        foreach (var resource in target.Structure.NodesOfType<ExecutionResourceNode>())
        {
            if (resource.ResourceNodeId.Owner.IsParent &&
                callBindings.TryGetValue(
                    resource.Resource.Id, out var bound))
            {
                resourceMap.Add(resource.GraphId, bound.Resource.GraphId);
                continue;
            }
            var mappedOwner = resource.ResourceNodeId.Owner.IsParent
                ? GraphBindingOwner.Parent
                : GraphBindingOwner.Instance(new(
                    prefix + resource.ResourceNodeId.Owner.Call!.Value.Value));
            var mappedResource = resource.Resource with
            {
                Id = new(prefix + resource.Resource.Id.Value),
                Name = $"{call.Id.Value}/{resource.Resource.Name}",
            };
            var mappedNode = new ExecutionResourceNode(
                new ModelResourceNodeId(mappedOwner, mappedResource.Id),
                mappedResource);
            resourceMap.Add(resource.GraphId, mappedNode.GraphId);
            addedResources.Add(mappedNode);
        }

        GraphNodeId Map(GraphNodeId id) =>
            nodeMap.TryGetValue(id, out var node)
                ? node
                : resourceMap.TryGetValue(id, out var resource)
                    ? resource
                    : throw new InvalidDataException(
                        $"Call '{callId}' contains unknown target node '{id}'.");

        var inlinedNodes = target.Nodes.Select<ExecutionGraphNode, ExecutionGraphStructureNode>(
            node => node switch
        {
            ExecutionOperationNode operation => operation with
            {
                Id = new(prefix + operation.Id.Value),
            },
            ExecutionGraphCallNode child => child with
            {
                Id = new(prefix + child.Id.Value),
            },
            _ => throw new InvalidDataException(),
        }).ToArray();
        var inlinedEdges = target.Structure.Edges.Select<ExecutionGraphEdge, ExecutionGraphEdge>(
            edge => edge switch
        {
            ExecutionDependencyEdge dependency => new ExecutionDependencyEdge(
                new($"inline:{callId.Value}:{dependency.Id.Value}"),
                Map(dependency.Source),
                Map(dependency.Target)),
            ExecutionResourceReadEdge read => new ExecutionResourceReadEdge(
                new($"inline:{callId.Value}:{read.Id.Value}"),
                Map(read.Source),
                Map(read.Target),
                read.Port,
                $"{prefix}{read.BindingId}"),
            ExecutionResourceWriteEdge write => new ExecutionResourceWriteEdge(
                new($"inline:{callId.Value}:{write.Id.Value}"),
                Map(write.Source),
                Map(write.Target),
                write.Port,
                $"{prefix}{write.BindingId}"),
            _ => throw new InvalidDataException(),
        }).ToList();

        var targetDependency = target.Structure.DependencyView();
        var entries = target.Nodes
            .Where(node => !targetDependency.EdgesToNode(node.GraphId).Any())
            .Select(node => Map(node.GraphId)).ToArray();
        var completions = target.Nodes
            .Where(node => !targetDependency.EdgesFromNode(node.GraphId).Any())
            .Select(node => Map(node.GraphId)).ToArray();
        var callerIncoming = caller.Structure.EdgesToNode(call.GraphId)
            .OfType<ExecutionDependencyEdge>()
            .Select(edge => edge.Source).ToArray();
        var callerOutgoing = caller.Structure.EdgesFromNode(call.GraphId)
            .OfType<ExecutionDependencyEdge>()
            .Select(edge => edge.Target).ToArray();
        foreach (var incoming in callerIncoming)
        {
            foreach (var entry in entries)
            {
                AddDependency(inlinedEdges, incoming, entry);
            }
        }

        foreach (var completion in completions)
        {
            foreach (var outgoing in callerOutgoing)
            {
                AddDependency(inlinedEdges, completion, outgoing);
            }
        }

        var plan = new GraphRewritePlan<
                ExecutionGraphStructureNode,
                ExecutionGraphEdge>()
            .Remove(rewriteTarget);
        foreach (var resource in addedResources)
        {
            plan.AddNode(resource);
        }

        foreach (var node in inlinedNodes)
        {
            plan.AddNode(node);
        }

        foreach (var edge in inlinedEdges)
        {
            plan.AddEdge(edge);
        }

        var callerStructure = plan.Apply(caller.Structure);
        var rewritten = CopyGraph(
            caller, callerStructure.Nodes, callerStructure.Edges);
        return CreateReachableProgram(
            program.Graphs.Where(graph => graph.Id != graphId)
                .Append(rewritten),
            program.RootGraph);
    }

    private static void AddBindings(
        ICollection<ExecutionGraphEdge> edges,
        ExecutionGraphNode node,
        IEnumerable<ModelResourceBinding> bindings)
    {
        foreach (var binding in bindings)
        {
            if (binding.Access != GraphResourceAccess.Write)
            {
                edges.Add(new ExecutionResourceReadEdge(
                    new($"binding:{binding.BindingId}:read"),
                    binding.Resource.GraphId,
                    node.GraphId,
                    binding.Port,
                    binding.BindingId));
            }

            if (binding.Access != GraphResourceAccess.Read)
            {
                edges.Add(new ExecutionResourceWriteEdge(
                    new($"binding:{binding.BindingId}:write"),
                    node.GraphId,
                    binding.Resource.GraphId,
                    binding.Port,
                    binding.BindingId));
            }
        }
    }

    private static void AddDependency(
        ICollection<ExecutionGraphEdge> edges,
        GraphNodeId source,
        GraphNodeId target)
    {
        if (edges.OfType<ExecutionDependencyEdge>().Any(edge =>
            edge.Source == source && edge.Target == target))
        {
            return;
        }

        edges.Add(new ExecutionDependencyEdge(
            new($"dependency:{source.Value}:{target.Value}"),
            source,
            target));
    }

    private static ExecutionGraph RequiredGraph(
        ExecutionGraphProgram program,
        ExecutionGraphId id) =>
        program.Graphs.SingleOrDefault(graph => graph.Id == id)
        ?? throw new ArgumentException(
            $"Execution graph program has no graph '{id}'.", nameof(id));

    private static ExecutionGraph CopyGraph(
        ExecutionGraph source,
        IEnumerable<ExecutionGraphStructureNode> nodes,
        IEnumerable<ExecutionGraphEdge> edges) =>
        new(
            source.Id,
            new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(
                nodes, edges),
            source.DefaultCompileBoundary,
            source.Architecture,
            source.Source);

    private static ExecutionGraphProgram ReplaceGraph(
        ExecutionGraphProgram program,
        ExecutionGraph replacement) =>
        CreateReachableProgram(
            program.Graphs.Select(graph =>
                graph.Id == replacement.Id ? replacement : graph),
            program.RootGraph);

    private static ExecutionGraphProgram CreateReachableProgram(
        IEnumerable<ExecutionGraph> graphs,
        ExecutionGraphId root)
    {
        var values = graphs.ToArray();
        var map = values.ToDictionary(graph => graph.Id);
        var reachable = new HashSet<ExecutionGraphId>();
        var pending = new Stack<ExecutionGraphId>();
        pending.Push(root);
        while (pending.TryPop(out var id) && reachable.Add(id))
        {
            if (!map.TryGetValue(id, out var graph))
            {
                throw new InvalidDataException(
                    $"Execution graph program has no reachable graph '{id}'.");
            }

            foreach (var target in graph.Nodes.OfType<ExecutionGraphCallNode>()
                .Select(call => call.Target))
            {
                pending.Push(target);
            }
        }
        return new ExecutionGraphProgram(
            values.Where(graph => reachable.Contains(graph.Id)));
    }

    private static PrecisionRequirement MergeRequirements(
        IEnumerable<ExecutionOperationNode> nodes) =>
        new(
            NumericTypeCompatibility.Maximum(
                nodes.Select(node =>
                    node.Requirements.MinimumArithmeticType)),
            NumericTypeCompatibility.Maximum(
                nodes.Select(node =>
                    node.Requirements.MinimumAccumulatorType)));

    private static IReadOnlyDictionary<string, string> Copy(
        IReadOnlyDictionary<string, string>? values) =>
        new ReadOnlyDictionary<string, string>(
            new Dictionary<string, string>(
                values ?? new Dictionary<string, string>(),
                StringComparer.Ordinal));
}
