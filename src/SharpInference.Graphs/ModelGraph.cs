using SharpInference.Graphs.Core;

namespace SharpInference.Graphs;

/// <summary>Identifies one resource node in a containing graph or direct call instance.</summary>
public readonly record struct ModelResourceNodeId(
    GraphBindingOwner Owner,
    ResourceId Resource)
{
    /// <summary>Gets the structural graph-node identity.</summary>
    public GraphNodeId GraphId => Owner.IsParent
        ? new($"resource:parent:{Resource.Value}")
        : new($"resource:instance:{Owner.Call!.Value.Value}:{Resource.Value}");

    /// <inheritdoc/>
    public override string ToString() => Owner.IsParent
        ? Resource.Value
        : $"{Owner}/{Resource.Value}";
}

/// <summary>Common behavior for model graph nodes.</summary>
public interface IModelGraphNode : IGraphNode;

/// <summary>Common behavior for model graph edges.</summary>
public interface IModelGraphEdge : IGraphEdge;

/// <summary>Common behavior for operation and graph-call nodes.</summary>
public interface IModelOperationNode : IModelGraphNode;

/// <summary>Common behavior for resource nodes.</summary>
public interface IModelResourceNode : IModelGraphNode
{
    /// <summary>Gets the scoped resource identity.</summary>
    ModelResourceNodeId ResourceNodeId { get; }

    /// <summary>Gets the tensor contract.</summary>
    TensorDescriptor Tensor { get; }

    /// <summary>Gets the resource role.</summary>
    GraphDefinitionResourceKind Kind { get; }
}

/// <summary>Common behavior for explicit execution dependency edges.</summary>
public interface IDependencyEdge : IModelGraphEdge;

/// <summary>Common behavior for resource-read edges.</summary>
public interface IResourceReadEdge : IModelGraphEdge
{
    /// <summary>Gets the bound operation port.</summary>
    string Port { get; }

    /// <summary>Gets the binding shared with a paired write edge.</summary>
    string BindingId { get; }
}

/// <summary>Common behavior for resource-write edges.</summary>
public interface IResourceWriteEdge : IModelGraphEdge
{
    /// <summary>Gets the bound operation port.</summary>
    string Port { get; }

    /// <summary>Gets the binding shared with a paired read edge.</summary>
    string BindingId { get; }
}

/// <summary>One resource access reconstructed from graph edges.</summary>
public sealed record ModelResourceBinding(
    string Port,
    ModelResourceNodeId Resource,
    GraphResourceAccess Access,
    string BindingId);

/// <summary>Classifies conflicting accesses to one mutable resource.</summary>
[Flags]
public enum ModelResourceConflictKind
{
    /// <summary>No conflicting access.</summary>
    None = 0,
    /// <summary>The first operation writes a value read by the second.</summary>
    ReadAfterWrite = 1,
    /// <summary>The first operation reads a value overwritten by the second.</summary>
    WriteAfterRead = 2,
    /// <summary>Both operations write the same resource.</summary>
    WriteAfterWrite = 4,
}

/// <summary>One pair of operations with conflicting access to a resource.</summary>
public sealed record ModelResourceConflict<TNode>(
    IModelResourceNode Resource,
    TNode First,
    TNode Second,
    ModelResourceConflictKind Kind,
    bool Ordered)
    where TNode : IModelGraphNode;

/// <summary>Resources crossing into or out of a selected operation set.</summary>
public sealed record ModelGraphBoundary<TNode>(
    IReadOnlyList<TNode> Inputs,
    IReadOnlyList<TNode> Outputs)
    where TNode : IModelGraphNode;

/// <summary>Queries shared model-graph relationships.</summary>
public static class ModelGraphQueries
{
    /// <summary>Returns a view containing only operation nodes and dependency edges.</summary>
    public static GraphView<TNode, TEdge> DependencyView<TNode, TEdge>(
        this Graph<TNode, TEdge> graph)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        graph.View(node => node is IModelOperationNode, edge => edge is IDependencyEdge);

    /// <summary>Returns a view containing operations, resources, and resource-access edges.</summary>
    public static GraphView<TNode, TEdge> DataFlowView<TNode, TEdge>(
        this Graph<TNode, TEdge> graph)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        graph.View(
            node => node is IModelOperationNode or IModelResourceNode,
            edge => edge is IResourceReadEdge or IResourceWriteEdge);

    /// <summary>Gets direct dependency predecessors of an operation.</summary>
    public static IEnumerable<TNode> Predecessors<TNode, TEdge>(
        this Graph<TNode, TEdge> graph,
        IModelOperationNode node)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        graph.InEdges(node.Id).OfType<IDependencyEdge>()
            .Select(edge => graph.GetNode(edge.Source));

    /// <summary>Gets direct dependency successors of an operation.</summary>
    public static IEnumerable<TNode> Successors<TNode, TEdge>(
        this Graph<TNode, TEdge> graph,
        IModelOperationNode node)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        graph.OutEdges(node.Id).OfType<IDependencyEdge>()
            .Select(edge => graph.GetNode(edge.Target));

    /// <summary>Gets all resource bindings of an operation in edge declaration order.</summary>
    public static IReadOnlyList<ModelResourceBinding> Bindings<TNode, TEdge>(
        this Graph<TNode, TEdge> graph,
        IModelOperationNode node)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge
    {
        var reads = graph.InEdges(node.Id).OfType<IResourceReadEdge>()
            .ToDictionary(edge => edge.BindingId, StringComparer.Ordinal);
        var writes = graph.OutEdges(node.Id).OfType<IResourceWriteEdge>()
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
    public static IReadOnlyList<TNode> Readers<TNode, TEdge>(
        this Graph<TNode, TEdge> graph,
        IModelResourceNode resource)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        graph.OutEdges(resource.Id).OfType<IResourceReadEdge>()
            .Select(edge => graph.GetNode(edge.Target))
            .DistinctBy(node => node.Id)
            .ToArray();

    /// <summary>Gets operations that write one resource.</summary>
    public static IReadOnlyList<TNode> Writers<TNode, TEdge>(
        this Graph<TNode, TEdge> graph,
        IModelResourceNode resource)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        graph.InEdges(resource.Id).OfType<IResourceWriteEdge>()
            .Select(edge => graph.GetNode(edge.Source))
            .DistinctBy(node => node.Id)
            .ToArray();

    /// <summary>Gets all conflicting access pairs for one resource.</summary>
    public static IReadOnlyList<ModelResourceConflict<TNode>> ResourceConflicts<TNode, TEdge>(
        this Graph<TNode, TEdge> graph,
        IModelResourceNode resource)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge
    {
        var accesses = graph.Nodes
            .Where(node => node is IModelOperationNode)
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
        var dependencyView = graph.DependencyView();
        var result = new List<ModelResourceConflict<TNode>>();
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

                result.Add(new(
                    resource,
                    first.Node,
                    second.Node,
                    kind,
                    dependencyView.IsReachable(first.Node.Id, second.Node.Id) ||
                    dependencyView.IsReachable(second.Node.Id, first.Node.Id)));
            }
        }

        return result;
    }

    /// <summary>Gets conflicting accesses that have no dependency path in either direction.</summary>
    public static IReadOnlyList<ModelResourceConflict<TNode>>
        UnorderedResourceConflicts<TNode, TEdge>(
            this Graph<TNode, TEdge> graph,
            IModelResourceNode resource)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge =>
        graph.ResourceConflicts(resource)
            .Where(conflict => !conflict.Ordered)
            .ToArray();

    /// <summary>Gets resource nodes crossing the boundary of selected operations.</summary>
    public static ModelGraphBoundary<TNode> BoundaryResources<TNode, TEdge>(
        this Graph<TNode, TEdge> graph,
        IEnumerable<GraphNodeId> operations)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge
    {
        ArgumentNullException.ThrowIfNull(operations);
        var selected = operations.ToHashSet();
        var inputs = graph.Edges.OfType<IResourceReadEdge>()
            .Where(edge => selected.Contains(edge.Target))
            .Select(edge => graph.GetNode(edge.Source))
            .OfType<IModelResourceNode>()
            .Where(resource =>
                resource.Kind is GraphDefinitionResourceKind.Input
                    or GraphDefinitionResourceKind.InOut ||
                graph.InEdges(resource.Id).OfType<IResourceWriteEdge>()
                    .Any(edge => !selected.Contains(edge.Source)))
            .Cast<TNode>()
            .DistinctBy(node => node.Id)
            .ToArray();
        var outputs = graph.Edges.OfType<IResourceWriteEdge>()
            .Where(edge => selected.Contains(edge.Source))
            .Select(edge => graph.GetNode(edge.Target))
            .OfType<IModelResourceNode>()
            .Where(resource =>
                resource.Kind is GraphDefinitionResourceKind.Output
                    or GraphDefinitionResourceKind.InOut ||
                graph.OutEdges(resource.Id).OfType<IResourceReadEdge>()
                    .Any(edge => !selected.Contains(edge.Target)))
            .Cast<TNode>()
            .DistinctBy(node => node.Id)
            .ToArray();
        return new(inputs, outputs);
    }
}
