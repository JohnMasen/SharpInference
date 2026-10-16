using System.Collections.ObjectModel;
using SharpInference.Graphs.Core;
using static SharpInference.Graphs.Core.PatternElements;

namespace SharpInference.Graphs.Execution;

/// <summary>Rewrites execution graphs while preserving model-level bindings and dependency boundaries.</summary>
public static class ExecutionGraphRewriter
{
    /// <summary>Replaces a call node with an operation while retaining its identity, incident edges, and architecture.</summary>
    /// <remarks>The call's logical origins are retained with a new transformation label. Graphs not retained by
    /// root-based collection are removed. The call's compilation-boundary value does not restrict replacement.</remarks>
    /// <param name="program">The source program.</param>
    /// <param name="graphId">The graph containing the call.</param>
    /// <param name="callId">The local identity of the call to replace.</param>
    /// <param name="operation">The replacement operation name and version.</param>
    /// <param name="attributes">Optional attributes copied with ordinal keys; null produces no attributes.</param>
    /// <param name="requirements">Optional precision requirements; null selects Float32 arithmetic and accumulation.</param>
    /// <param name="transformation">An optional provenance label; null selects the replacement transformation name.</param>
    /// <returns>A newly validated program with the replacement and root-collected graphs.</returns>
    /// <exception cref="ArgumentNullException">The program is null.</exception>
    /// <exception cref="ArgumentException">The graph or call cannot be found.</exception>
    /// <exception cref="InvalidDataException">The rewritten structure or resulting program fails validation.</exception>
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

    /// <summary>Replaces at least two operations in one graph with a single operation and supplied bindings.</summary>
    /// <remarks>External dependency edges are redirected to the replacement, while selected nodes' access
    /// edges are replaced by the supplied bindings. Logical origins are combined distinctly. No convexity
    /// check is performed before rewriting; dependency cycles are rejected by graph construction.</remarks>
    /// <param name="program">The source program.</param>
    /// <param name="graphId">The graph containing every selected operation.</param>
    /// <param name="nodeIds">The operation identities, deduplicated into a selection.</param>
    /// <param name="replacementId">An identity unused outside the selection; a selected identity may be reused.</param>
    /// <param name="operation">The replacement operation name and version.</param>
    /// <param name="bindings">The complete resource bindings of the replacement.</param>
    /// <param name="attributes">Optional attributes copied with ordinal keys.</param>
    /// <param name="requirements">Optional requirements; null takes componentwise maximum selected precision requirements.</param>
    /// <param name="architecture">Optional architecture metadata for the replacement.</param>
    /// <param name="transformation">Optional source-map label; null selects the fusion transformation name.</param>
    /// <returns>A newly validated program with the fused operation and root-collected graphs.</returns>
    /// <exception cref="ArgumentNullException">The program or node identity sequence is null.</exception>
    /// <exception cref="ArgumentException">The graph is missing, fewer than two distinct operations are selected,
    /// selected identities are missing or are calls, or the replacement identity conflicts.</exception>
    /// <exception cref="InvalidDataException">The rewritten structure, bindings, or program fails validation.</exception>
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

        var sources = graph.Structure.Filter(node: node =>
                node is ExecutionGraphNode operation && selected.Contains(operation.Id))
            .Nodes.Cast<ExecutionGraphNode>().ToArray();
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
    /// <remarks>Only operations may be selected; directly accessed call-instance resources are rejected.
    /// Parent resources and selected operation identities are retained in the extracted graph. A selected
    /// read with an outside writer becomes InOut; a read without a selected writer becomes Input;
    /// selected writes visible to outside readers or existing Output/InOut resources become Output;
    /// other resources become Temporary. Non-temporary resources become call bindings named by resource identity.
    /// Internal dependency edges move with the selection and external dependency edges attach to the call.
    /// The supplied boundary is recorded on both graph and call, not used as a permission check.
    /// Operation provenance receives the extraction label, and no logical definition source is assigned to the new graph.</remarks>
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
    /// <exception cref="NotSupportedException">A selected operation directly accesses a call-instance resource.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The boundary value is undefined.</exception>
    /// <exception cref="InvalidDataException">The extracted or rewritten graph, bindings, or resulting program is invalid.</exception>
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
        var selectedNodes = source.Structure.Filter(node =>
                node is ExecutionGraphNode operation && selected.Contains(operation.Id))
            .Nodes.Cast<ExecutionGraphNode>().ToArray();
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
        var selectedConstraint = Node().Where<ExecutionGraphStructureNode>(node =>
            selectedGraphIds.Contains(node.GraphId));
        var resourceConstraint = Node().Where<ExecutionResourceNode>(_ => true);
        var resourceIds = source.Structure.Match(
                Pattern.Create(selectedConstraint) - Edge() - resourceConstraint)
            .Select(match => match.Nodes.Last().GraphId)
            .Concat(source.Structure.Match(
                    Pattern.Create(resourceConstraint) - Edge() - selectedConstraint)
                .Select(match => match.Nodes.First().GraphId))
            .ToHashSet();
        var resourceNodes = source.Structure.Filter(node =>
                node is ExecutionResourceNode && resourceIds.Contains(node.GraphId))
            .Nodes.Cast<ExecutionResourceNode>().ToArray();
        if (resourceNodes.Any(resource => !resource.ResourceNodeId.Owner.IsParent))
        {
            throw new NotSupportedException(
                "Extraction cannot move direct call-instance resource accesses.");
        }

        var extractedResources = new List<ExecutionResourceNode>();
        var callBindings = new List<ModelResourceBinding>();
        foreach (var resourceNode in resourceNodes)
        {
            var reader = Pattern.Create(Node(resourceNode.GraphId))
                - Edge().Where<ExecutionResourceReadEdge>(_ => true);
            var selectedReads = source.Structure.Match(reader
                - Node().Where<ExecutionGraphStructureNode>(node => selectedGraphIds.Contains(node.GraphId))).Any();
            var writers = source.Structure.Writers(resourceNode);
            var selectedWrites = writers.Any(node => selectedGraphIds.Contains(node.GraphId));
            var outsideReads = source.Structure.Match(reader
                - Node().Where<ExecutionGraphStructureNode>(node => !selectedGraphIds.Contains(node.GraphId))).Any();
            var outsideWrites = writers.Any(node => !selectedGraphIds.Contains(node.GraphId));
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
    /// <remarks>No compilation-boundary value prevents this explicit rewrite. Callee executable identities
    /// receive a call-identity-and-slash prefix; logical source maps and architecture metadata are retained.
    /// Parent resources whose resource identities match call binding port names map to the bound caller resources.
    /// Other resource identities and names are prefixed, and nested call-instance owner identities are prefixed.
    /// Callee access edge ports are retained and binding identities are prefixed. Incoming caller dependencies
    /// connect to all dependency-only entries, and all dependency-only completions connect to outgoing dependencies.
    /// An empty callee adds no bypass dependency edges. The result is rebuilt using root-based graph collection.</remarks>
    /// <param name="program">The program containing both the caller and callee graphs.</param>
    /// <param name="graphId">The identity of the caller graph.</param>
    /// <param name="callId">The identity of the graph call to inline.</param>
    /// <returns>A rewritten program retaining only graphs reachable from its root.</returns>
    /// <exception cref="ArgumentNullException">The program is null.</exception>
    /// <exception cref="ArgumentException">The caller, callee, or requested graph call cannot be found.</exception>
    /// <exception cref="InvalidDataException">Mapped identities, bindings, dependency topology, or program relationships are invalid.</exception>
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
            .Where(node => !targetDependency.Predecessors(node).Any())
            .Select(node => Map(node.GraphId)).ToArray();
        var completions = target.Nodes
            .Where(node => !targetDependency.Successors(node).Any())
            .Select(node => Map(node.GraphId)).ToArray();
        var callerIncoming = caller.Structure.Predecessors(call)
            .Select(node => node.GraphId).ToArray();
        var callerOutgoing = caller.Structure.Successors(call)
            .Select(node => node.GraphId).ToArray();
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

    /// <summary>Expands model bindings into read, write, or paired access edges with binding-derived identities.</summary>
    /// <param name="edges">The destination edge collection.</param>
    /// <param name="node">The operation or call using the bindings.</param>
    /// <param name="bindings">The complete binding sequence to append.</param>
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

    /// <summary>Adds a dependency unless that directed endpoint pair already has a dependency in the collection.</summary>
    /// <param name="edges">The destination edge collection.</param>
    /// <param name="source">The prerequisite node's structural identity.</param>
    /// <param name="target">The dependent node's structural identity.</param>
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

    /// <summary>Finds the unique graph with the requested program identity.</summary>
    /// <param name="program">The program to search.</param>
    /// <param name="id">The required graph identity.</param>
    /// <returns>The matching graph.</returns>
    /// <exception cref="ArgumentException">The graph identity is absent.</exception>
    private static ExecutionGraph RequiredGraph(
        ExecutionGraphProgram program,
        ExecutionGraphId id) =>
        program.Graphs.SingleOrDefault(graph => graph.Id == id)
        ?? throw new ArgumentException(
            $"Execution graph program has no graph '{id}'.", nameof(id));

    /// <summary>Reconstructs and validates a graph with replacement structure and unchanged graph metadata.</summary>
    /// <param name="source">The graph supplying identity, boundary, architecture, and logical provenance.</param>
    /// <param name="nodes">The replacement structural nodes.</param>
    /// <param name="edges">The replacement structural edges.</param>
    /// <returns>A newly constructed execution graph.</returns>
    /// <exception cref="InvalidDataException">The replacement structure or graph validation fails.</exception>
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

    /// <summary>Substitutes a graph by identity and rebuilds the program using collection from the original root.</summary>
    /// <param name="program">The source program.</param>
    /// <param name="replacement">The graph replacing the graph with the same identity.</param>
    /// <returns>The validated program built from root-collected definitions.</returns>
    /// <exception cref="InvalidDataException">Collection encounters a missing graph or resulting relationships are invalid.</exception>
    private static ExecutionGraphProgram ReplaceGraph(
        ExecutionGraphProgram program,
        ExecutionGraph replacement) =>
        CreateReachableProgram(
            program.Graphs.Select(graph =>
                graph.Id == replacement.Id ? replacement : graph),
            program.RootGraph);

    /// <summary>Collects graph identities from a root using a call-target stack and validates the retained program.</summary>
    /// <remarks>Collection stops when the stack is empty or a popped identity has already been collected.
    /// Retained graphs keep their input sequence order; program construction validates their call relationships.</remarks>
    /// <param name="graphs">The candidate graph definitions.</param>
    /// <param name="root">The graph identity initially placed on the stack.</param>
    /// <returns>A program containing the definitions whose identities were collected.</returns>
    /// <exception cref="ArgumentException">Candidate graph identities repeat.</exception>
    /// <exception cref="InvalidDataException">A collected graph is missing or the retained program is invalid.</exception>
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

            foreach (var target in graph.Structure.Filter(node: node => node is ExecutionGraphCallNode)
                .Nodes.Cast<ExecutionGraphCallNode>()
                .Select(call => call.Target))
            {
                pending.Push(target);
            }
        }
        return new ExecutionGraphProgram(
            values.Where(graph => reachable.Contains(graph.Id)));
    }

    /// <summary>Takes the numeric maximum independently across arithmetic and accumulator requirements.</summary>
    /// <param name="nodes">The selected operations supplying precision requirements.</param>
    /// <returns>The componentwise maximum precision requirements.</returns>
    private static PrecisionRequirement MergeRequirements(
        IEnumerable<ExecutionOperationNode> nodes) =>
        new(
            NumericTypeCompatibility.Maximum(
                nodes.Select(node =>
                    node.Requirements.MinimumArithmeticType)),
            NumericTypeCompatibility.Maximum(
                nodes.Select(node =>
                    node.Requirements.MinimumAccumulatorType)));

    /// <summary>Copies optional operation attributes into a read-only ordinal dictionary.</summary>
    /// <param name="values">The source attributes, or null for an empty dictionary.</param>
    /// <returns>A read-only snapshot independent of the supplied dictionary.</returns>
    private static IReadOnlyDictionary<string, string> Copy(
        IReadOnlyDictionary<string, string>? values) =>
        new ReadOnlyDictionary<string, string>(
            new Dictionary<string, string>(
                values ?? new Dictionary<string, string>(),
                StringComparer.Ordinal));
}
