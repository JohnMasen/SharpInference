using SharpInference.Graphs.Core;
using SharpInference.Graphs.Internal;

namespace SharpInference.Graphs.Logical;

/// <summary>Builds a logical graph using names resolved only during validation or Build.</summary>
public sealed class GraphDefinitionBuilder
{
    private readonly GraphDefinitionId id;
    private readonly GraphCompileBoundary compileBoundary;
    private readonly GraphAnnotation? annotation;
    private readonly Dictionary<ResourceId, GraphDefinitionResource> resources = [];
    private readonly Dictionary<GraphDefinitionNodeId, GraphDefinitionNode> nodes = [];
    private readonly List<LogicalGraphEdge> edges = [];
    private readonly Dictionary<GraphDefinitionId, LogicalGraphDefinition> definitions = [];

    public GraphDefinitionBuilder(string id,
        GraphCompileBoundary compileBoundary = GraphCompileBoundary.Auto,
        GraphAnnotation? annotation = null)
    {
        this.id = new(LogicalGraphDefinition.RequireId(id, nameof(id)));
        this.compileBoundary = compileBoundary;
        this.annotation = annotation;
    }

    public ResourceId AddResource(string id, string name, TensorDescriptor tensor,
        GraphDefinitionResourceKind kind = GraphDefinitionResourceKind.Temporary, string? bindingKey = null)
    {
        var resource = new GraphDefinitionResource(
            new(LogicalGraphDefinition.RequireId(id, nameof(id))),
            LogicalGraphDefinition.RequireId(name, nameof(name)), kind,
            tensor ?? throw new ArgumentNullException(nameof(tensor)), this.id, bindingKey);
        if (!Enum.IsDefined(kind))
            throw new ArgumentOutOfRangeException(nameof(kind));
        if (!resources.TryAdd(resource.Id, resource))
            throw new InvalidOperationException($"Resource '{id}' is already declared in '{this.id}'.");
        return resource.Id;
    }

    public ResourceId AddInput(string id, TensorDescriptor tensor) =>
        AddResource(id, id, tensor, GraphDefinitionResourceKind.Input);
    public ResourceId AddOutput(string id, TensorDescriptor tensor) =>
        AddResource(id, id, tensor, GraphDefinitionResourceKind.Output);
    public ResourceId AddInOut(string id, TensorDescriptor tensor) =>
        AddResource(id, id, tensor, GraphDefinitionResourceKind.InOut);

    public void RegisterDefinition(LogicalGraphDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        if (definitions.TryGetValue(definition.Id, out var existing))
        {
            if (!ReferenceEquals(existing, definition) &&
                LogicalGraphXml.Serialize(existing) != LogicalGraphXml.Serialize(definition))
                throw new InvalidOperationException($"Conflicting graph definitions share '{definition.Id}'.");
            return;
        }
        definitions.Add(definition.Id, definition);
    }

    public TResult AddNode<TResult>(string id, GraphOperationId operation,
        Func<GraphOperationNodeBuilder, TResult> configure, GraphAnnotation? annotation = null)
    {
        ArgumentNullException.ThrowIfNull(configure);
        var builder = new GraphOperationNodeBuilder(LogicalGraphDefinition.RequireId(id, nameof(id)));
        var result = configure(builder);
        AddNode(builder.Build(operation, annotation), builder);
        return result;
    }

    public void AddNode(string id, GraphOperationId operation,
        Action<GraphOperationNodeBuilder> configure, GraphAnnotation? annotation = null)
    {
        ArgumentNullException.ThrowIfNull(configure);
        AddNode(id, operation, node => { configure(node); return true; }, annotation);
    }

    public GraphDefinitionNodeId AddCall(string id, GraphDefinitionId target,
        Action<GraphOperationNodeBuilder> bind, GraphAnnotation? annotation = null)
    {
        ArgumentNullException.ThrowIfNull(bind);
        var builder = new GraphOperationNodeBuilder(LogicalGraphDefinition.RequireId(id, nameof(id)));
        bind(builder);
        var call = new GraphCallNode(new(id), target, annotation);
        AddNode(call, builder);
        return call.Id;
    }

    public GraphDefinitionNodeId AddCall(string id, LogicalGraphDefinition definition,
        Action<GraphOperationNodeBuilder> bind, GraphAnnotation? annotation = null)
    {
        RegisterDefinition(definition);
        return AddCall(id, definition.Id, bind, annotation);
    }

    public GraphDefinitionNodeId AddCall(string id, LogicalGraphDefinition definition,
        IEnumerable<(string Port, ResourceId Resource, GraphResourceAccess Access)> bindings,
        IEnumerable<GraphDefinitionNodeId>? dependencies = null, GraphAnnotation? annotation = null) =>
        AddCall(id, definition, node =>
        {
            foreach (var binding in bindings)
            {
                switch (binding.Access)
                {
                    case GraphResourceAccess.Read: node.AddInput(binding.Port, binding.Resource); break;
                    case GraphResourceAccess.Write: node.AddOutput(binding.Port, binding.Resource); break;
                    case GraphResourceAccess.ReadWrite: node.AddInOut(binding.Port, binding.Resource); break;
                    default: throw new ArgumentOutOfRangeException(nameof(bindings));
                }
            }
            foreach (var dependency in dependencies ?? [])
                node.DependsOn(dependency);
        }, annotation);

    public IReadOnlyList<GraphDiagnostic> Validate()
    {
        var (structureNodes, structureEdges, diagnostics) = Prepare();
        if (diagnostics.Count != 0)
            return diagnostics;
        return Create(structureNodes, structureEdges, false).Diagnostics;
    }

    public LogicalGraphDefinition Build()
    {
        var (structureNodes, structureEdges, diagnostics) = Prepare();
        ModelGraphConstruction.ThrowIfErrors(diagnostics);
        return Create(structureNodes, structureEdges, true);
    }

    private LogicalGraphDefinition Create(
        LogicalGraphNode[] structureNodes, LogicalGraphEdge[] structureEdges, bool rejectErrors) =>
        new(id, compileBoundary, annotation,
            new Graph<LogicalGraphNode, LogicalGraphEdge>(structureNodes, structureEdges),
            nodes.Values.OfType<GraphCallNode>().Select(call => definitions[call.Definition])
                .DistinctBy(definition => definition.Id), rejectErrors);

    private (LogicalGraphNode[] Nodes, LogicalGraphEdge[] Edges, IReadOnlyList<GraphDiagnostic> Diagnostics) Prepare()
    {
        var scoped = resources.Values.ToDictionary(
            resource => new ModelResourceNodeId(GraphBindingOwner.Parent, resource.Id));
        var diagnostics = new List<GraphDiagnostic>();
        foreach (var call in nodes.Values.OfType<GraphCallNode>())
        {
            if (call.Definition == id)
                diagnostics.Add(new(GraphDiagnosticSeverity.Error, "LG0030",
                    $"Call '{call.Id}' recursively targets containing graph '{id}'.", call.Id.Value, Graph: id.Value));
            else if (!definitions.ContainsKey(call.Definition))
                diagnostics.Add(new(GraphDiagnosticSeverity.Error, "LG0014",
                    $"Call '{call.Id}' targets unknown graph '{call.Definition}'.", call.Id.Value, Graph: id.Value));
        }
        var references = edges.Select(edge => edge switch
        {
            LogicalResourceReadEdge read => read.Source,
            LogicalResourceWriteEdge write => write.Target,
            _ => default,
        }).Where(reference => reference != default).ToHashSet();
        foreach (var call in nodes.Values.OfType<GraphCallNode>())
        {
            if (!definitions.TryGetValue(call.Definition, out var definition))
                continue;
            foreach (var resource in definition.Resources)
            {
                var scopedId = new ModelResourceNodeId(GraphBindingOwner.Instance(call.Id), resource.Id);
                if (references.Contains(scopedId.GraphId))
                    scoped.TryAdd(scopedId, resource);
            }
        }
        var structureNodes = scoped.Select(pair => (LogicalGraphNode)new LogicalResourceNode(pair.Key, pair.Value))
            .Concat(nodes.Values).ToArray();
        var structureEdges = edges.ToList();
        diagnostics.AddRange(ModelGraphConstruction.Validate(id.Value, structureNodes,
            structureEdges, node => node is GraphCallNode));
        if (diagnostics.Count == 0)
            InferDependencies(structureNodes, structureEdges);
        return (structureNodes, structureEdges.ToArray(), diagnostics);
    }

    private void AddNode(GraphDefinitionNode node, GraphOperationNodeBuilder builder)
    {
        if (!nodes.TryAdd(node.Id, node))
            throw new InvalidOperationException($"Node '{node.Id}' is already declared in '{id}'.");
        edges.AddRange(ModelGraphConstruction.ExpandBindings<LogicalGraphEdge>(
            node.GraphId, builder.Bindings,
            (id, source, target, port, binding) => new LogicalResourceReadEdge(id, source, target, port, binding),
            (id, source, target, port, binding) => new LogicalResourceWriteEdge(id, source, target, port, binding)));
        foreach (var dependency in builder.Dependencies.Distinct())
            AddDependency(edges, GraphDefinitionNode.NodeGraphId(dependency), node.GraphId);
    }

    private static void AddDependency(List<LogicalGraphEdge> edges, GraphNodeId source, GraphNodeId target)
    {
        if (!edges.OfType<LogicalDependencyEdge>().Any(edge => edge.Source == source && edge.Target == target))
            edges.Add(new LogicalDependencyEdge(
                new($"dependency:{source.Value["operation:".Length..]}:{target.Value["operation:".Length..]}"),
                source, target));
    }

    private static void InferDependencies(LogicalGraphNode[] nodes, List<LogicalGraphEdge> edges)
    {
        var resources = nodes.OfType<LogicalResourceNode>().ToDictionary(node => node.GraphId);
        var writers = edges.OfType<LogicalResourceWriteEdge>().GroupBy(edge => edge.Target)
            .ToDictionary(group => group.Key, group => group.Select(edge => edge.Source).Distinct().ToArray());
        var pairedWrites = edges.OfType<LogicalResourceWriteEdge>()
            .Select(edge => edge.BindingId).ToHashSet(StringComparer.Ordinal);
        foreach (var read in edges.OfType<LogicalResourceReadEdge>().ToArray())
        {
            var resource = resources[read.Source];
            if (!resource.ResourceNodeId.Owner.IsParent &&
                GraphDefinitionNode.NodeGraphId(resource.ResourceNodeId.Owner.Call!.Value) is var owner &&
                owner != read.Target)
                AddDependency(edges, owner, read.Target);
            if (!pairedWrites.Contains(read.BindingId) &&
                writers.TryGetValue(read.Source, out var producers) &&
                producers.Where(producer => producer != read.Target).ToArray() is [var producer])
                AddDependency(edges, producer, read.Target);
        }
    }
}
