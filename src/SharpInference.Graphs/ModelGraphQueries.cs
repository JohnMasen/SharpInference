using SharpInference.Graphs.Core;
using static SharpInference.Graphs.Core.PatternElements;

namespace SharpInference.Graphs;

/// <summary>Queries shared model-graph relationships.</summary>
public static class ModelGraphQueries
{
    /// <summary>Returns a view containing only operation nodes and dependency edges.</summary>
    /// <typeparam name="TNode">The model graph node type.</typeparam>
    /// <typeparam name="TEdge">The model graph edge type.</typeparam>
    /// <param name="graph">The source view limiting visible operations and dependencies.</param>
    /// <returns>A deferred dependency-only view preserving source order.</returns>
    public static GraphView<TNode, TEdge> DependencyView<TNode, TEdge>(
        this IGraphView<TNode, TEdge> graph)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        graph.Filter(node: node => node is IModelOperationNode, edge: edge => edge is IDependencyEdge);

    /// <summary>Returns a view containing operations, resources, and resource-access edges.</summary>
    /// <typeparam name="TNode">The model graph node type.</typeparam>
    /// <typeparam name="TEdge">The model graph edge type.</typeparam>
    /// <param name="graph">The source view limiting visible data-flow relationships.</param>
    /// <returns>A deferred view containing only visible read and write relationships.</returns>
    public static GraphView<TNode, TEdge> DataFlowView<TNode, TEdge>(
        this IGraphView<TNode, TEdge> graph)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        graph.Filter(
            node: node => node is IModelOperationNode or IModelResourceNode,
            edge: edge => edge is IResourceReadEdge or IResourceWriteEdge);

    /// <summary>Gets direct dependency predecessors of an operation.</summary>
    /// <typeparam name="TNode">The model graph node type.</typeparam>
    /// <typeparam name="TEdge">The model graph edge type.</typeparam>
    /// <param name="graph">The view containing the operation and its visible dependencies.</param>
    /// <param name="node">The operation whose incoming dependency edges are queried.</param>
    /// <returns>A deferred sequence of predecessors in incoming-edge order, without deduplication.</returns>
    /// <exception cref="KeyNotFoundException">The operation identity is unknown or hidden in the view.</exception>
    public static IEnumerable<TNode> Predecessors<TNode, TEdge>(
        this IGraphView<TNode, TEdge> graph,
        IModelOperationNode node)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        MatchAdjacentEdges<TNode, TEdge, IDependencyEdge>(graph, node.Id, incoming: true)
            .Select(edge => graph.GetNode(edge.Source));

    /// <summary>Gets direct dependency successors of an operation.</summary>
    /// <typeparam name="TNode">The model graph node type.</typeparam>
    /// <typeparam name="TEdge">The model graph edge type.</typeparam>
    /// <param name="graph">The view containing the operation and its visible dependencies.</param>
    /// <param name="node">The operation whose outgoing dependency edges are queried.</param>
    /// <returns>A deferred sequence of successors in outgoing-edge order, without deduplication.</returns>
    /// <exception cref="KeyNotFoundException">The operation identity is unknown or hidden in the view.</exception>
    public static IEnumerable<TNode> Successors<TNode, TEdge>(
        this IGraphView<TNode, TEdge> graph,
        IModelOperationNode node)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        MatchAdjacentEdges<TNode, TEdge, IDependencyEdge>(graph, node.Id)
            .Select(edge => graph.GetNode(edge.Target));

    /// <summary>Gets all resource bindings of an operation in edge declaration order.</summary>
    /// <typeparam name="TNode">The model graph node type.</typeparam>
    /// <typeparam name="TEdge">The model graph edge type.</typeparam>
    /// <param name="graph">The view containing visible resource-access edges and resource nodes.</param>
    /// <param name="node">The operation whose read and write bindings are reconstructed.</param>
    /// <returns>Bindings deduplicated by binding identity, combining paired reads and writes.</returns>
    /// <exception cref="KeyNotFoundException">The operation or a referenced resource is unknown or hidden.</exception>
    /// <exception cref="ArgumentException">Multiple read edges or multiple write edges share one binding identity.</exception>
    /// <exception cref="InvalidDataException">
    /// Paired accesses disagree about their port or resource, or an endpoint is not a resource node.
    /// </exception>
    public static IReadOnlyList<ModelResourceBinding> Bindings<TNode, TEdge>(
        this IGraphView<TNode, TEdge> graph,
        IModelOperationNode node)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge
    {
        var reads = MatchAdjacentEdges<TNode, TEdge, IResourceReadEdge>(graph, node.Id, incoming: true)
            .ToDictionary(edge => edge.BindingId, StringComparer.Ordinal);
        var writes = MatchAdjacentEdges<TNode, TEdge, IResourceWriteEdge>(graph, node.Id)
            .ToDictionary(edge => edge.BindingId, StringComparer.Ordinal);
        var order = graph.Edges
            .Where(edge =>
                edge is IResourceReadEdge read && read.Target == node.Id ||
                edge is IResourceWriteEdge write && write.Source == node.Id)
            .Select(edge => edge switch
            {
                IResourceReadEdge read => read.BindingId,
                IResourceWriteEdge write => write.BindingId,
                _ => throw new InvalidDataException(),
            })
            .Distinct(StringComparer.Ordinal);
        return order.Select(bindingId =>
        {
            reads.TryGetValue(bindingId, out var read);
            writes.TryGetValue(bindingId, out var write);
            var port = read?.Port ?? write!.Port;
            if (read is not null && write is not null && read.Port != write.Port)
            {
                throw new InvalidDataException(
                    $"Binding '{bindingId}' has inconsistent read and write ports.");
            }

            var resourceId = read?.Source ?? write!.Target;
            if (read is not null && write is not null && read.Source != write.Target)
            {
                throw new InvalidDataException(
                    $"Binding '{bindingId}' reads and writes different resources.");
            }

            var resource = graph.GetNode(resourceId) as IModelResourceNode
                ?? throw new InvalidDataException(
                    $"Binding '{bindingId}' does not reference a resource node.");
            return new ModelResourceBinding(
                port,
                resource.ResourceNodeId,
                read is not null && write is not null
                    ? GraphResourceAccess.ReadWrite
                    : read is not null
                        ? GraphResourceAccess.Read
                        : GraphResourceAccess.Write,
                bindingId);
        }).ToArray();
    }

    /// <summary>Gets operations that read one resource.</summary>
    /// <typeparam name="TNode">The model graph node type.</typeparam>
    /// <typeparam name="TEdge">The model graph edge type.</typeparam>
    /// <param name="graph">The view defining visible resource-read relationships.</param>
    /// <param name="resource">The resource whose outgoing read edges are queried.</param>
    /// <returns>Distinct readers in first-edge occurrence order.</returns>
    /// <exception cref="KeyNotFoundException">The resource or a referenced reader is unknown or hidden.</exception>
    public static IReadOnlyList<TNode> Readers<TNode, TEdge>(
        this IGraphView<TNode, TEdge> graph,
        IModelResourceNode resource)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        MatchAdjacentEdges<TNode, TEdge, IResourceReadEdge>(graph, resource.Id)
            .Select(edge => graph.GetNode(edge.Target))
            .DistinctBy(node => node.Id)
            .ToArray();

    /// <summary>Gets operations that write one resource.</summary>
    /// <typeparam name="TNode">The model graph node type.</typeparam>
    /// <typeparam name="TEdge">The model graph edge type.</typeparam>
    /// <param name="graph">The view defining visible resource-write relationships.</param>
    /// <param name="resource">The resource whose incoming write edges are queried.</param>
    /// <returns>Distinct writers in first-edge occurrence order.</returns>
    /// <exception cref="KeyNotFoundException">The resource or a referenced writer is unknown or hidden.</exception>
    public static IReadOnlyList<TNode> Writers<TNode, TEdge>(
        this IGraphView<TNode, TEdge> graph,
        IModelResourceNode resource)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        MatchAdjacentEdges<TNode, TEdge, IResourceWriteEdge>(graph, resource.Id, incoming: true)
            .Select(edge => graph.GetNode(edge.Source))
            .DistinctBy(node => node.Id)
            .ToArray();

    /// <summary>Gets all conflicting access pairs for one resource.</summary>
    /// <typeparam name="TNode">The model graph node type.</typeparam>
    /// <typeparam name="TEdge">The model graph edge type.</typeparam>
    /// <param name="graph">The view defining visible operations, accesses, and dependency paths.</param>
    /// <param name="resource">The resource whose read/write conflicts are classified.</param>
    /// <returns>Conflicting operation pairs and their dependency ordering in node declaration order.</returns>
    /// <remarks>
    /// First and Second follow node declaration order, not dependency order. Ordered means reachable
    /// in either direction. A shared endpoint-scoped reachability index is created only if a conflict is found.
    /// </remarks>
    public static IReadOnlyList<ModelResourceConflict<TNode>> ResourceConflicts<TNode, TEdge>(
        this IGraphView<TNode, TEdge> graph,
        IModelResourceNode resource)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge
    {
        var accesses = graph.Match(Pattern.Create(Node().Where<IModelOperationNode>(_ => true)))
            .Select(match => match.Nodes.Single())
            .Select(node =>
            {
                var bindings = graph.Bindings((IModelOperationNode)node)
                    .Where(binding => binding.Resource.GraphId == resource.Id)
                    .ToArray();
                return (
                    Node: node,
                    Reads: bindings.Any(binding =>
                        binding.Access != GraphResourceAccess.Write),
                    Writes: bindings.Any(binding =>
                        binding.Access != GraphResourceAccess.Read));
            })
            .Where(access => access.Reads || access.Writes)
            .ToArray();
        var result = new List<ModelResourceConflict<TNode>>();
        if (accesses.Length < 2)
            return result;
        var dependencyView = graph.DependencyView();
        var endpoints = accesses.Select(access => access.Node).ToArray();
        ReachabilityIndex<TNode>? reachability = null;
        for (var firstIndex = 0; firstIndex < accesses.Length; firstIndex++)
        {
            for (var secondIndex = firstIndex + 1;
                 secondIndex < accesses.Length;
                 secondIndex++)
            {
                var first = accesses[firstIndex];
                var second = accesses[secondIndex];
                var kind = ModelResourceConflictKind.None;
                if (first.Writes && second.Reads)
                {
                    kind |= ModelResourceConflictKind.ReadAfterWrite;
                }

                if (first.Reads && second.Writes)
                {
                    kind |= ModelResourceConflictKind.WriteAfterRead;
                }

                if (first.Writes && second.Writes)
                {
                    kind |= ModelResourceConflictKind.WriteAfterWrite;
                }

                if (kind == ModelResourceConflictKind.None)
                {
                    continue;
                }

                reachability ??= dependencyView.CreateReachabilityIndex(endpoints, endpoints);
                result.Add(new(
                    resource,
                    first.Node,
                    second.Node,
                    kind,
                    reachability.IsReachable(first.Node.Id, second.Node.Id) ||
                    reachability.IsReachable(second.Node.Id, first.Node.Id)));
            }
        }

        return result;
    }

    /// <summary>Gets conflicting accesses that have no dependency path in either direction.</summary>
    /// <typeparam name="TNode">The model graph node type.</typeparam>
    /// <typeparam name="TEdge">The model graph edge type.</typeparam>
    /// <param name="graph">The view defining resource conflicts and dependency paths.</param>
    /// <param name="resource">The resource whose unordered conflicts are requested.</param>
    /// <returns>Only conflicting operation pairs not ordered by a visible dependency path.</returns>
    public static IReadOnlyList<ModelResourceConflict<TNode>>
        UnorderedResourceConflicts<TNode, TEdge>(
            this IGraphView<TNode, TEdge> graph,
            IModelResourceNode resource)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        graph.ResourceConflicts(resource)
            .Where(conflict => !conflict.Ordered)
            .ToArray();

    /// <summary>Gets resource nodes crossing the boundary of selected operations.</summary>
    /// <typeparam name="TNode">The model graph node type.</typeparam>
    /// <typeparam name="TEdge">The model graph edge type.</typeparam>
    /// <param name="graph">The view defining visible operation/resource accesses and resource roles.</param>
    /// <param name="operations">Selected operation identities, consumed once and deduplicated.</param>
    /// <returns>Distinct boundary inputs and outputs, preserving first access-edge occurrence order.</returns>
    /// <remarks>
    /// Inputs are selected reads of input/in-out resources or resources with an outside writer.
    /// Outputs are selected writes of output/in-out resources or resources with an outside reader.
    /// Identities not represented by visible matching operations contribute no accesses.
    /// </remarks>
    /// <exception cref="ArgumentNullException">The operation identity sequence is null.</exception>
    public static ModelGraphBoundary<TNode> BoundaryResources<TNode, TEdge>(
        this IGraphView<TNode, TEdge> graph,
        IEnumerable<GraphNodeId> operations)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge
    {
        ArgumentNullException.ThrowIfNull(operations);
        var selected = operations.ToHashSet();
        var edgePositions = graph.Edges.Select((edge, position) => (edge.Id, position))
            .ToDictionary(item => item.Id, item => item.position);
        var readPattern = Pattern.Create(Node().Where<IModelResourceNode>(_ => true))
            - Edge().Where<IResourceReadEdge>(_ => true)
            - Node().Where<TNode>(node => selected.Contains(node.Id));
        var writePattern = Pattern.Create(Node().Where<TNode>(node => selected.Contains(node.Id)))
            - Edge().Where<IResourceWriteEdge>(_ => true)
            - Node().Where<IModelResourceNode>(_ => true);
        var inputs = graph.Match(readPattern)
            .OrderBy(match => edgePositions[match.Edges.Single().Id])
            .Select(match => (IModelResourceNode)match.Nodes.First())
            .Where(resource =>
                resource.Kind is GraphDefinitionResourceKind.Input
                    or GraphDefinitionResourceKind.InOut ||
                MatchAdjacentEdges<TNode, TEdge, IResourceWriteEdge>(graph, resource.Id, incoming: true)
                    .Any(edge => !selected.Contains(edge.Source)))
            .Cast<TNode>()
            .DistinctBy(node => node.Id)
            .ToArray();
        var outputs = graph.Match(writePattern)
            .OrderBy(match => edgePositions[match.Edges.Single().Id])
            .Select(match => (IModelResourceNode)match.Nodes.Last())
            .Where(resource =>
                resource.Kind is GraphDefinitionResourceKind.Output
                    or GraphDefinitionResourceKind.InOut ||
                MatchAdjacentEdges<TNode, TEdge, IResourceReadEdge>(graph, resource.Id)
                    .Any(edge => !selected.Contains(edge.Target)))
            .Cast<TNode>()
            .DistinctBy(node => node.Id)
            .ToArray();
        return new(inputs, outputs);
    }

    /// <summary>Matches typed adjacent edges while preserving consumer adjacency order and self-loop behavior.</summary>
    /// <typeparam name="TNode">The model graph node type.</typeparam>
    /// <typeparam name="TEdge">The model graph edge type.</typeparam>
    /// <typeparam name="TRequested">The edge contract to match.</typeparam>
    /// <param name="graph">The view defining visible endpoints and edges.</param>
    /// <param name="node">The visible endpoint identity, validated before returning the sequence.</param>
    /// <param name="incoming">Whether the endpoint is the target rather than the source.</param>
    /// <returns>Matching edges in declaration order, including parallel edges.</returns>
    /// <remarks>Simple-path Match excludes self-loops, so such adjacency queries use direct lookup.</remarks>
    /// <exception cref="KeyNotFoundException">The endpoint is unknown or hidden.</exception>
    private static IEnumerable<TRequested> MatchAdjacentEdges<TNode, TEdge, TRequested>(
        IGraphView<TNode, TEdge> graph, GraphNodeId node, bool incoming = false)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge
        where TRequested : IModelGraphEdge
    {
        var edges = incoming ? graph.EnumerateEdgesToNode(node) : graph.EnumerateEdgesFromNode(node);
        var typedEdges = edges.OfType<TRequested>().ToArray();
        if (typedEdges.Any(edge => edge.Source == edge.Target))
            return typedEdges;
        var pattern = Pattern.Create(incoming ? Node() : Node(node))
            - Edge().Where<TRequested>(_ => true) - (incoming ? Node(node) : Node());
        var matches = graph.Match(pattern).Select(match => match.Edges.Single()).Cast<TRequested>();
        if (!incoming)
            return matches;
        var positions = typedEdges.Select((edge, position) => (edge.Id, position))
            .ToDictionary(item => item.Id, item => item.position);
        return matches.OrderBy(edge => positions[edge.Id]);
    }
}
