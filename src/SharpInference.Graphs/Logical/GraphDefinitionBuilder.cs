using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Logical;

/// <summary>Builds a logical definition from typed ports, resources, operations, and nested calls.</summary>
/// <typeparam name="TContract">The definition's self-creating public-port contract.</typeparam>
/// <remarks>
/// Resource access edges and execution dependencies are separate. During finalization, a read of another
/// call's instance-scoped resource depends on that owning call. An unpaired read also depends on its
/// sole distinct writer other than the reader, when exactly one exists. Paired read/write bindings
/// skip writer-based inference; multiple distinct other writers do not imply an ordering.
/// Explicit dependencies remain authoritative, and validation rejects dependency cycles.
/// </remarks>
public sealed class GraphDefinitionBuilder<TContract> where TContract : IGraphPortContract<TContract>
{
    /// <summary>The identity of the definition under construction.</summary>
    private readonly GraphDefinitionId id;
    /// <summary>The factory retaining public-port declarations, including any created after initialization.</summary>
    private readonly DefinitionPortFactory ports;
    /// <summary>The requested compilation-boundary policy.</summary>
    private readonly GraphCompileBoundary compileBoundary;
    /// <summary>Optional metadata for the resulting definition.</summary>
    private readonly GraphAnnotation? annotation;
    /// <summary>Referenced parent declarations, with conflicting identifiers rejected.</summary>
    private readonly List<GraphDefinitionResource> resources;
    /// <summary>Operations and calls in insertion order.</summary>
    private readonly List<GraphDefinitionNode> nodes = [];
    /// <summary>Resource-access and dependency edges accumulated for the structure.</summary>
    private readonly List<LogicalGraphEdge> edges = [];
    /// <summary>Resource declarations indexed by parent or call-instance scope and identifier.</summary>
    private readonly Dictionary<ModelResourceNodeId, GraphDefinitionResource> scopedResources = [];
    /// <summary>The first called definition registered for each definition identity.</summary>
    private readonly Dictionary<GraphDefinitionId, LogicalGraphDefinition> definitions = [];

    /// <summary>Initializes a definition builder from its already-created port contract.</summary>
    /// <param name="id">The containing definition identity.</param>
    /// <param name="contract">The typed public ports to expose.</param>
    /// <param name="ports">The factory supplying the port resource declarations.</param>
    /// <param name="compileBoundary">The compilation-boundary policy.</param>
    /// <param name="annotation">Optional definition metadata.</param>
    internal GraphDefinitionBuilder(
        GraphDefinitionId id,
        TContract contract,
        DefinitionPortFactory ports,
        GraphCompileBoundary compileBoundary,
        GraphAnnotation? annotation)
    {
        this.id = id;
        this.ports = ports;
        this.compileBoundary = compileBoundary;
        this.annotation = annotation;
        Ports = contract;
        resources = [.. ports.Resources];
    }

    /// <summary>Gets this definition's typed public ports.</summary>
    public TContract Ports { get; }
    /// <summary>Gets the currently registered parent resource declarations.</summary>
    internal IReadOnlyList<GraphDefinitionResource> Resources => resources;

    /// <summary>Adds a resource owned by this definition, excluding pure input and output declarations.</summary>
    /// <typeparam name="T">The resource handle's unchecked type marker.</typeparam>
    /// <param name="id">The nonblank definition-local resource identifier.</param>
    /// <param name="name">The nonblank display name.</param>
    /// <param name="tensor">The nonnull tensor contract.</param>
    /// <param name="kind">The resource kind; Input and Output must instead come from the port contract.</param>
    /// <param name="bindingKey">An optional external resource-binding key.</param>
    /// <returns>A typed handle over the registered declaration.</returns>
    /// <exception cref="ArgumentException">The identifier or name is blank, or the kind is Input or Output.</exception>
    /// <exception cref="ArgumentNullException">The tensor is null.</exception>
    /// <exception cref="InvalidOperationException">The identifier has a conflicting declaration.</exception>
    public GraphResource<T> AddResource<T>(
        string id,
        string name,
        TensorDescriptor tensor,
        GraphDefinitionResourceKind kind = GraphDefinitionResourceKind.Temporary,
        string? bindingKey = null)
    {
        if (kind is GraphDefinitionResourceKind.Input or GraphDefinitionResourceKind.Output)
        {
            throw new ArgumentException(
                "Use the graph port contract to declare inputs and outputs.", nameof(kind));
        }

        var resource = new GraphDefinitionResource(
            new ResourceId(LogicalGraphDefinition.RequireId(id, nameof(id))),
            string.IsNullOrWhiteSpace(name)
                ? throw new ArgumentException("A resource name is required.", nameof(name))
                : name,
            kind,
            tensor ?? throw new ArgumentNullException(nameof(tensor)),
            this.id,
            bindingKey);
        AddReferencedResource(resource);
        return new GraphResource<T>(resource);
    }

    /// <summary>Configures an operation, registers its accesses and explicit prerequisites, and returns a callback result.</summary>
    /// <typeparam name="TResult">The configuration callback's result type.</typeparam>
    /// <param name="id">The nonblank operation identifier.</param>
    /// <param name="operation">The operation name and version.</param>
    /// <param name="configure">The callback that configures parent-owned bindings and operation metadata.</param>
    /// <param name="annotation">Optional operation metadata.</param>
    /// <returns>The result returned by the configuration callback.</returns>
    /// <exception cref="ArgumentNullException">The configuration callback is null.</exception>
    /// <exception cref="ArgumentException">The operation identifier is blank.</exception>
    /// <exception cref="InvalidOperationException">A referenced resource conflicts with an existing declaration.</exception>
    public TResult AddNode<TResult>(
        string id,
        GraphOperationId operation,
        Func<GraphOperationNodeBuilder, TResult> configure,
        GraphAnnotation? annotation = null)
    {
        ArgumentNullException.ThrowIfNull(configure);
        var nodeId = new GraphDefinitionNodeId(LogicalGraphDefinition.RequireId(id, nameof(id)));
        var builder = new GraphOperationNodeBuilder(this.id, nodeId);
        var result = configure(builder);
        foreach (var resource in builder.Resources)
        {
            AddReferencedResource(resource);
        }

        var node = builder.Build(operation, annotation);
        AddNodeAndRelations(node, builder.Bindings, builder.Dependencies);
        return result;
    }

    /// <summary>Configures and adds an operation with its resource-access and explicit dependency edges.</summary>
    /// <param name="id">The nonblank operation identifier.</param>
    /// <param name="operation">The operation name and version.</param>
    /// <param name="configure">The callback that configures bindings, prerequisites, attributes, and precision.</param>
    /// <param name="annotation">Optional operation metadata.</param>
    /// <exception cref="ArgumentException">The operation identifier is blank.</exception>
    /// <exception cref="InvalidOperationException">A referenced resource conflicts with an existing declaration.</exception>
    public void AddNode(
        string id,
        GraphOperationId operation,
        Action<GraphOperationNodeBuilder> configure,
        GraphAnnotation? annotation = null) =>
        AddNode(id, operation, builder =>
        {
            configure(builder);
            return true;
        }, annotation);

    /// <summary>Adds a typed child call and returns its instantiated ports.</summary>
    /// <typeparam name="TChildContract">The called definition's port contract.</typeparam>
    /// <param name="id">The nonblank call identifier.</param>
    /// <param name="definition">The child definition to invoke.</param>
    /// <param name="bind">The callback binding required inputs and optional output targets.</param>
    /// <param name="annotation">Optional call metadata.</param>
    /// <returns>The child contract with ports remapped to call inputs and output targets.</returns>
    /// <exception cref="ArgumentNullException">The definition or binding callback is null.</exception>
    /// <exception cref="ArgumentException">The call identifier is blank or a configured binding is incompatible.</exception>
    /// <exception cref="InvalidOperationException">A required input is missing, a port conflicts, or resource declarations conflict.</exception>
    public TChildContract AddNode<TChildContract>(
        string id,
        GraphDefinition<TChildContract> definition,
        Action<GraphCallBindings<TChildContract>> bind,
        GraphAnnotation? annotation = null)
        where TChildContract : IGraphPortContract<TChildContract> =>
        AddCall(id, definition, bind, annotation).Ports;

    /// <summary>Adds a typed call, generating parent-owned output resources when no explicit target is supplied.</summary>
    /// <typeparam name="TChildContract">The called definition's port contract.</typeparam>
    /// <param name="id">The nonblank call identifier, also used to prefix generated outputs.</param>
    /// <param name="definition">The child graph definition to invoke.</param>
    /// <param name="bind">The callback binding every input and input/output port and any desired output targets.</param>
    /// <param name="annotation">Optional call metadata.</param>
    /// <returns>The call identity, child definition, and instantiated typed ports.</returns>
    /// <remarks>
    /// Input/output accesses are emitted as one read/write binding; pure outputs add write bindings.
    /// Only parent-scoped targets are added as parent resource declarations. Instance targets retain
    /// their call-qualified scope. No explicit prerequisites are added by this overload.
    /// </remarks>
    /// <exception cref="ArgumentNullException">The definition or binding callback is null.</exception>
    /// <exception cref="ArgumentException">The call identifier is blank or a configured binding is incompatible.</exception>
    /// <exception cref="InvalidOperationException">Required inputs are missing, contract ports conflict, or resource declarations conflict.</exception>
    public GraphCallInstance<TChildContract> AddCall<TChildContract>(
        string id,
        GraphDefinition<TChildContract> definition,
        Action<GraphCallBindings<TChildContract>> bind,
        GraphAnnotation? annotation = null)
        where TChildContract : IGraphPortContract<TChildContract>
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentNullException.ThrowIfNull(bind);
        var nodeId = new GraphDefinitionNodeId(LogicalGraphDefinition.RequireId(id, nameof(id)));
        var bindings = new GraphCallBindings<TChildContract>(this.id, definition.Ports);
        bind(bindings);
        var inputs = bindings.BuildInputs(definition.Definition);
        foreach (var resource in bindings.Resources)
        {
            AddReferencedResource(resource);
        }

        var instancePorts = new InstancePortFactory(
            this.id, nodeId, definition.Definition, inputs, bindings.Outputs);
        var result = TChildContract.Create(instancePorts);
        foreach (var resource in instancePorts.Resources)
        {
            AddReferencedResource(resource);
        }

        var values = inputs.Concat(instancePorts.OutputBindings).ToArray();
        AddNodeAndRelations(
            new GraphCallNode(nodeId, definition.Definition.Id, annotation),
            values,
            []);
        definitions.TryAdd(definition.Definition.Id, definition.Definition);
        return new GraphCallInstance<TChildContract>(nodeId, definition.Definition, result);
    }

    /// <summary>Adds an untyped call with exactly one explicit parent-scoped binding for every public child port.</summary>
    /// <param name="id">The nonblank call identifier.</param>
    /// <param name="definition">The definition to invoke.</param>
    /// <param name="bindings">Port names, resources, and exact required access modes.</param>
    /// <param name="dependencies">Explicit prerequisite identities, or null for none.</param>
    /// <param name="annotation">Optional call metadata.</param>
    /// <returns>The added call's definition-local identity.</returns>
    /// <remarks>
    /// Input, Output, and InOut require Read, Write, and ReadWrite respectively. Tensor descriptors
    /// must match exactly. This overload tags all accesses as parent-scoped without checking each
    /// declaration's graph owner and does not generate missing output targets.
    /// </remarks>
    /// <exception cref="ArgumentNullException">The definition or bindings sequence is null.</exception>
    /// <exception cref="ArgumentException">The call identifier is blank.</exception>
    /// <exception cref="InvalidOperationException">A port is unknown, duplicated, missing, incompatible, or has incorrect access, or declarations conflict.</exception>
    public GraphDefinitionNodeId AddCall(
        string id,
        LogicalGraphDefinition definition,
        IEnumerable<(string Port, GraphResource<object> Resource, GraphResourceAccess Access)> bindings,
        IEnumerable<GraphDefinitionNodeId>? dependencies = null,
        GraphAnnotation? annotation = null)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentNullException.ThrowIfNull(bindings);
        var nodeId = new GraphDefinitionNodeId(LogicalGraphDefinition.RequireId(id, nameof(id)));
        var ports = definition.Resources.Where(resource =>
                resource.Kind is GraphDefinitionResourceKind.Input
                    or GraphDefinitionResourceKind.Output
                    or GraphDefinitionResourceKind.InOut)
            .ToDictionary(resource => resource.Id.Value, StringComparer.Ordinal);
        var values = bindings.Select(binding =>
        {
            if (!ports.TryGetValue(binding.Port, out var port))
            {
                throw new InvalidOperationException(
                    $"Definition '{definition.Id}' has no public port '{binding.Port}'.");
            }

            if (!SameTensor(port.Tensor, binding.Resource.Tensor))
            {
                throw new InvalidOperationException(
                    $"Call '{nodeId}' binds incompatible resource '{binding.Resource.Id}' to '{binding.Port}'.");
            }

            var requiredAccess = AccessFor(port.Kind);
            if (binding.Access != requiredAccess)
            {
                throw new InvalidOperationException(
                    $"Call '{nodeId}' port '{binding.Port}' requires {requiredAccess} access.");
            }

            AddReferencedResource(binding.Resource.Resource);
            return new PendingBinding(
                binding.Port, binding.Resource.Resource, binding.Access, GraphBindingOwner.Parent);
        }).ToArray();
        if (values.Select(binding => binding.Port).Distinct(StringComparer.Ordinal).Count() != ports.Count ||
            values.Length != ports.Count)
        {
            throw new InvalidOperationException(
                $"Call '{nodeId}' must bind every public port of definition '{definition.Id}' exactly once.");
        }

        AddNodeAndRelations(
            new GraphCallNode(nodeId, definition.Id, annotation),
            values,
            dependencies ?? []);
        definitions.TryAdd(definition.Id, definition);
        return nodeId;
    }

    /// <summary>Registers all parent ports, infers dependencies, and constructs a validated logical graph.</summary>
    /// <returns>The validated definition paired with its original typed port contract.</returns>
    /// <exception cref="InvalidOperationException">A port conflicts with a registered resource declaration.</exception>
    /// <exception cref="InvalidDataException">The resulting graph structure or logical validation is invalid.</exception>
    internal GraphDefinition<TContract> Build()
    {
        foreach (var port in ports.Resources)
        {
            AddReferencedResource(port);
        }

        foreach (var resource in resources)
        {
            scopedResources.TryAdd(
                new ModelResourceNodeId(GraphBindingOwner.Parent, resource.Id),
                resource);
        }

        InferDependencies();
        var graphNodes = scopedResources
            .Select(pair => (LogicalGraphNode)new LogicalResourceNode(pair.Key, pair.Value))
            .Concat(nodes)
            .ToArray();
        var definition = new LogicalGraphDefinition(
            id,
            compileBoundary,
            annotation,
            new Graph<LogicalGraphNode, LogicalGraphEdge>(graphNodes, edges),
            definitions.Values);
        return new GraphDefinition<TContract>(definition, Ports);
    }

    /// <summary>Appends an operation, expands its bindings, and adds distinct explicit prerequisites.</summary>
    /// <param name="node">The operation or call to append.</param>
    /// <param name="bindings">The resource accesses to expand into edges.</param>
    /// <param name="dependencies">The prerequisite identities to deduplicate for this operation.</param>
    private void AddNodeAndRelations(
        GraphDefinitionNode node,
        IEnumerable<PendingBinding> bindings,
        IEnumerable<GraphDefinitionNodeId> dependencies)
    {
        nodes.Add(node);
        foreach (var binding in bindings)
        {
            AddBindingEdges(node, binding);
        }

        foreach (var dependency in dependencies.Distinct())
        {
            AddDependency(dependency, node.Id);
        }
    }

    /// <summary>Registers a scoped resource and emits read and/or write edges sharing one binding identity.</summary>
    /// <param name="node">The accessing operation.</param>
    /// <param name="binding">The scope, declaration, port, and access to represent.</param>
    /// <remarks>Binding identities combine operation and port names; paired edges reconstruct ReadWrite access.</remarks>
    private void AddBindingEdges(GraphDefinitionNode node, PendingBinding binding)
    {
        var scoped = new ModelResourceNodeId(binding.Owner, binding.Resource.Id);
        scopedResources.TryAdd(scoped, binding.Resource);
        var bindingId = $"{node.Id.Value}:{binding.Port}";
        if (binding.Access != GraphResourceAccess.Write)
        {
            edges.Add(new LogicalResourceReadEdge(
                new($"binding:{bindingId}:read"),
                scoped.GraphId,
                node.GraphId,
                binding.Port,
                bindingId));
        }

        if (binding.Access != GraphResourceAccess.Read)
        {
            edges.Add(new LogicalResourceWriteEdge(
                new($"binding:{bindingId}:write"),
                node.GraphId,
                scoped.GraphId,
                binding.Port,
                bindingId));
        }
    }

    /// <summary>Adds a prerequisite edge, suppressing repeated non-self edges but retaining self-dependencies.</summary>
    /// <param name="source">The prerequisite operation identity.</param>
    /// <param name="target">The dependent operation identity.</param>
    /// <remarks>Self-dependencies remain visible so final graph validation can reject their cycles.</remarks>
    private void AddDependency(GraphDefinitionNodeId source, GraphDefinitionNodeId target)
    {
        if (source == target)
        {
            edges.Add(new LogicalDependencyEdge(
                new($"dependency:{source.Value}:{target.Value}"),
                GraphDefinitionNode.NodeGraphId(source),
                GraphDefinitionNode.NodeGraphId(target)));
            return;
        }
        if (edges.OfType<LogicalDependencyEdge>().Any(edge =>
            edge.Source == GraphDefinitionNode.NodeGraphId(source) &&
            edge.Target == GraphDefinitionNode.NodeGraphId(target)))
        {
            return;
        }

        edges.Add(new LogicalDependencyEdge(
            new($"dependency:{source.Value}:{target.Value}"),
            GraphDefinitionNode.NodeGraphId(source),
            GraphDefinitionNode.NodeGraphId(target)));
    }

    /// <summary>Derives execution dependencies from instance ownership and uniquely determined resource producers.</summary>
    /// <remarks>
    /// Each read of an instance-scoped resource adds its owning call as a prerequisite unless that
    /// call is the reader itself. Reads whose binding identity also occurs on any write edge then
    /// skip producer inference. Other reads add a prerequisite only when the same scoped resource
    /// has exactly one distinct writer other than the reader. Writers are considered across the
    /// whole graph, not declaration order; ambiguous multi-writer accesses are not serialized here.
    /// </remarks>
    private void InferDependencies()
    {
        var writes = edges.OfType<LogicalResourceWriteEdge>()
            .GroupBy(edge => edge.Target)
            .ToDictionary(
                group => group.Key,
                group => group.Select(edge => edge.Source).Distinct().ToArray());
        var pairedWrites = edges.OfType<LogicalResourceWriteEdge>()
            .Select(edge => edge.BindingId).ToHashSet(StringComparer.Ordinal);
        foreach (var read in edges.OfType<LogicalResourceReadEdge>().ToArray())
        {
            var target = (GraphDefinitionNode)nodes.Single(node => node.GraphId == read.Target);
            var resource = scopedResources.Single(pair => pair.Key.GraphId == read.Source).Key;
            if (!resource.Owner.IsParent && resource.Owner.Call is { } ownerCall &&
                ownerCall != target.Id)
            {
                AddDependency(ownerCall, target.Id);
            }

            if (pairedWrites.Contains(read.BindingId))
            {
                continue;
            }

            if (writes.TryGetValue(read.Source, out var producers) &&
                producers.Where(producer => producer != read.Target).ToArray() is [var producer])
            {
                var source = (GraphDefinitionNode)nodes.Single(node => node.GraphId == producer);
                AddDependency(source.Id, target.Id);
            }
        }
    }

    /// <summary>Registers the first declaration for an identifier and rejects unequal subsequent declarations.</summary>
    /// <param name="resource">The parent resource declaration to register.</param>
    /// <exception cref="InvalidOperationException">The identifier is already associated with a different declaration.</exception>
    private void AddReferencedResource(GraphDefinitionResource resource)
    {
        var existing = resources.SingleOrDefault(value => value.Id == resource.Id);
        if (existing is null)
        {
            resources.Add(resource);
            return;
        }
        if (existing != resource)
        {
            throw new InvalidOperationException(
                $"Resource '{resource.Id}' has conflicting declarations.");
        }
    }

    /// <summary>Maps a public-port resource kind to its exact call access mode.</summary>
    /// <param name="kind">Input, Output, or InOut.</param>
    /// <returns>Read, Write, or ReadWrite respectively.</returns>
    /// <exception cref="InvalidDataException">The kind is not a public-port kind.</exception>
    private static GraphResourceAccess AccessFor(GraphDefinitionResourceKind kind) => kind switch
    {
        GraphDefinitionResourceKind.Input => GraphResourceAccess.Read,
        GraphDefinitionResourceKind.Output => GraphResourceAccess.Write,
        GraphDefinitionResourceKind.InOut => GraphResourceAccess.ReadWrite,
        _ => throw new InvalidDataException(),
    };

    /// <summary>Compares tensor element types, layouts, and ordered dimensions.</summary>
    /// <param name="left">The first descriptor.</param>
    /// <param name="right">The second descriptor.</param>
    /// <returns>True when all three parts of the tensor contract match.</returns>
    private static bool SameTensor(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType &&
        left.Layout == right.Layout &&
        left.Dimensions.SequenceEqual(right.Dimensions);
}
