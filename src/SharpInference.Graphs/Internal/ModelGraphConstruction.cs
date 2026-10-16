using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Internal;

internal static class ModelGraphConstruction
{
    internal static IEnumerable<TEdge> ExpandBindings<TEdge>(
        GraphNodeId node,
        IEnumerable<ModelResourceBinding> bindings,
        Func<GraphEdgeId, GraphNodeId, GraphNodeId, string, string, TEdge> read,
        Func<GraphEdgeId, GraphNodeId, GraphNodeId, string, string, TEdge> write)
    {
        foreach (var binding in bindings)
        {
            if (binding.Access != GraphResourceAccess.Write)
            {
                yield return read(new($"binding:{binding.BindingId}:read"),
                    binding.Resource.GraphId, node, binding.Port, binding.BindingId);
            }
            if (binding.Access != GraphResourceAccess.Read)
            {
                yield return write(new($"binding:{binding.BindingId}:write"),
                    node, binding.Resource.GraphId, binding.Port, binding.BindingId);
            }
        }
    }

    internal static IReadOnlyList<GraphDiagnostic> Validate<TNode, TEdge>(
        string graph,
        IReadOnlyList<TNode> nodes,
        IReadOnlyList<TEdge> edges,
        Func<TNode, bool> isCall)
        where TNode : IModelGraphNode
        where TEdge : IModelGraphEdge
    {
        var diagnostics = new List<GraphDiagnostic>();
        var map = new Dictionary<GraphNodeId, TNode>();
        var edgeIds = new HashSet<GraphEdgeId>();
        foreach (var node in nodes)
        {
            if (node is null)
            {
                Error("GR0001", "A graph node cannot be null.");
                continue;
            }
            if (string.IsNullOrWhiteSpace(node.Id.Value) || !map.TryAdd(node.Id, node))
                Error("GR0001", $"Invalid or duplicate node '{node.Id}'.", node.Id.Value);
            if (node is IModelOperationNode && node.Id.Value.StartsWith("operation:", StringComparison.Ordinal) &&
                string.IsNullOrWhiteSpace(node.Id.Value["operation:".Length..]))
                Error("GR0001", "An operation name is required.", node.Id.Value);
            if (node is IModelResourceNode resource &&
                (string.IsNullOrWhiteSpace(resource.ResourceNodeId.Resource.Value) ||
                 !Enum.IsDefined(resource.Kind) || !Enum.IsDefined(resource.Tensor.ElementType)))
                Error("GR0002", $"Invalid resource '{resource.ResourceNodeId}'.",
                    resource: resource.ResourceNodeId.Resource);
        }
        foreach (var node in nodes.OfType<IModelResourceNode>())
        {
            if (!node.ResourceNodeId.Owner.IsParent &&
                (!map.TryGetValue(new($"operation:{node.ResourceNodeId.Owner.Call!.Value.Value}"), out var owner) ||
                 !isCall(owner)))
                Error("GR0003", $"Unknown call scope '{node.ResourceNodeId.Owner}'.",
                    resource: node.ResourceNodeId.Resource);
        }
        foreach (var edge in edges)
        {
            if (edge is null)
            {
                Error("GR0004", "A graph edge cannot be null.");
                continue;
            }
            if (string.IsNullOrWhiteSpace(edge.Id.Value) || !edgeIds.Add(edge.Id))
                Error("GR0004", $"Invalid or duplicate edge '{edge.Id}'.");
            var sourceKnown = map.TryGetValue(edge.Source, out var source);
            var targetKnown = map.TryGetValue(edge.Target, out var target);
            if (!sourceKnown)
                Error("GR0005", $"Edge '{edge.Id}' references unknown source '{edge.Source}'.",
                    edge.Target.Value);
            if (!targetKnown)
                Error("GR0005", $"Edge '{edge.Id}' references unknown target '{edge.Target}'.",
                    edge.Source.Value);
            if (!sourceKnown || !targetKnown)
                continue;
            var valid = edge switch
            {
                IDependencyEdge => source is IModelOperationNode && target is IModelOperationNode,
                IResourceReadEdge => source is IModelResourceNode && target is IModelOperationNode,
                IResourceWriteEdge => source is IModelOperationNode && target is IModelResourceNode,
                _ => false,
            };
            if (!valid)
                Error("GR0006", $"Edge '{edge.Id}' has invalid relationship endpoints.");
        }

        var accesses = edges.Select(edge => edge switch
        {
            IResourceReadEdge read => (Node: read.Target, read.Port, read.BindingId,
                Resource: read.Source, Read: true),
            IResourceWriteEdge write => (Node: write.Source, write.Port, write.BindingId,
                Resource: write.Target, Read: false),
            _ => default,
        }).Where(access => access.Node != default).ToArray();
        foreach (var group in accesses.GroupBy(access => (access.Node, access.BindingId)))
        {
            if (string.IsNullOrWhiteSpace(group.Key.BindingId) ||
                group.Count(access => access.Read) > 1 ||
                group.Count(access => !access.Read) > 1 ||
                group.Select(access => (access.Port, access.Resource)).Distinct().Count() != 1)
                Error("GR0007", $"Inconsistent binding '{group.Key.BindingId}'.", group.Key.Node.Value);
        }
        foreach (var group in accesses.GroupBy(access => (access.Node, access.Port)))
        {
            if (string.IsNullOrWhiteSpace(group.Key.Port) ||
                group.Select(access => access.BindingId).Distinct(StringComparer.Ordinal).Count() != 1)
                Error("GR0008", $"Empty or duplicate port '{group.Key.Port}'.", group.Key.Node.Value,
                    port: group.Key.Port);
        }
        return diagnostics;

        void Error(string code, string message, string? node = null,
            ResourceId? resource = null, string? port = null) =>
            diagnostics.Add(new(GraphDiagnosticSeverity.Error, code, message, node, resource, port, graph));
    }

    internal static void ThrowIfErrors(IEnumerable<GraphDiagnostic> diagnostics)
    {
        var errors = diagnostics.Where(diagnostic => diagnostic.Severity == GraphDiagnosticSeverity.Error).ToArray();
        if (errors.Length != 0)
            throw new InvalidDataException(string.Join(Environment.NewLine,
                errors.Select(error => $"{error.Code} [{error.Graph}]: {error.Message}")));
    }

    internal static bool SameTensor(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType && left.Layout == right.Layout &&
        left.Dimensions.SequenceEqual(right.Dimensions);
}
