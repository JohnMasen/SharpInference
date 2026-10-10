using System.Text.Json.Serialization;

namespace SharpInference.Graphs;

/// <summary>Computational structure, independent of backend patterns and fusion rules.</summary>
[JsonPolymorphic(TypeDiscriminatorPropertyName = "Kind")]
[JsonDerivedType(typeof(GraphNodeElement), "Node")]
[JsonDerivedType(typeof(GraphRegionBody), "Region")]
[JsonDerivedType(typeof(GraphCall), "Call")]
public abstract record GraphElement(string Id)
{
    public IReadOnlyList<string> DependsOn { get; init; } = [];
}

public sealed record GraphNodeElement(LogicalNode Node) : GraphElement(Node.Id.Value);

public sealed record GraphRegionBody(GraphRegion Region, IReadOnlyList<GraphElement> Children)
    : GraphElement(Region.Id.Value);

public sealed record GraphCall(string CallId, string Definition, IReadOnlyList<NodeResourceBinding> Bindings)
    : GraphElement(CallId);

public sealed record GraphDefinitionPort(string Id, GraphResourceAccess Access, TensorDescriptor Tensor,
    GraphResourceKind? ResourceKind = null);

public sealed record GraphLayerDefinition(string Id, IReadOnlyList<GraphDefinitionPort> Ports,
    IReadOnlyList<GraphResource> Resources, GraphRegionBody Body);

public sealed record GraphStructure(IReadOnlyList<GraphLayerDefinition> LayerDefinitions, GraphRegionBody Root)
{
    internal bool HasTensorViews => LayerDefinitions.Select(definition => definition.Body).Append(Root)
        .SelectMany(Walk).Any(element => element switch
        {
            GraphNodeElement node => node.Node.Resources.Any(binding => binding.View is not null),
            GraphCall call => call.Bindings.Any(binding => binding.View is not null),
            _ => false,
        });

    public static GraphStructure FromExpanded(IReadOnlyList<GraphRegion> regions, IReadOnlyList<LogicalNode> nodes)
    {
        var regionMap = Unique(regions, region => region.Id.Value, "region");
        var roots = regions.Where(region => region.ParentId is null).ToArray();
        if (roots.Length != 1 || roots[0].Type != GraphRegionTypes.Graph)
            throw new InvalidDataException("An ordered graph requires one graph root region.");
        foreach (var region in regions)
        {
            var ancestors = new HashSet<RegionId> { region.Id };
            var parent = region.ParentId;
            while (parent is { } parentId)
            {
                if (!regionMap.TryGetValue(parentId.Value, out var ancestor) || !ancestors.Add(parentId) || ancestors.Count > 64)
                    throw new InvalidDataException($"Invalid parent chain for region '{region.Id}'.");
                parent = ancestor.ParentId;
            }
        }
        Unique(nodes, node => node.Id.Value, "node");
        if (nodes.Any(node => !regionMap.ContainsKey(node.Region.Value)))
            throw new InvalidDataException("An ordered node references an unknown region.");
        var positions = nodes.Select((node, index) => (node.Id, index)).ToDictionary(x => x.Id, x => x.index);
        GraphRegionBody Build(GraphRegion region)
        {
            var children = nodes.Where(n => n.Region == region.Id)
                .Select(n => (GraphElement)new GraphNodeElement(n with { Dependencies = [] })
                    { DependsOn = n.Dependencies.Select(d => d.Value).ToArray() })
                .Concat(regions.Where(r => r.ParentId == region.Id).Select(Build))
                .OrderBy(First).ToArray();
            return new(region, children);
        }
        int First(GraphElement element) => element switch
        {
            GraphNodeElement node => positions[node.Node.Id],
            GraphRegionBody body => body.Children.Count == 0 ? int.MaxValue : body.Children.Min(First),
            _ => int.MaxValue,
        };
        var result = new GraphStructure([], Build(regions.Single(r => r.ParentId is null)));
        var ordered = Walk(result.Root).OfType<GraphNodeElement>().Select(n => n.Node.Id);
        if (!ordered.SequenceEqual(nodes.Select(n => n.Id)))
            throw new InvalidDataException("Region contents must be contiguous in execution order.");
        return result;
    }

    internal static IEnumerable<GraphElement> Walk(GraphElement element)
    {
        yield return element;
        if (element is GraphRegionBody body)
            foreach (var child in body.Children)
                foreach (var descendant in Walk(child)) yield return descendant;
    }

    internal (GraphResource[] Resources, GraphRegion[] Regions, LogicalNode[] Nodes) Expand(
        IReadOnlyList<GraphResource> externalResources)
    {
        const int maxExpandedNodes = 100_000;
        var definitions = Unique(LayerDefinitions, d => d.Id, "definition");
        // Validate every definition, including unused ones, before instantiation.
        var visited = new HashSet<string>(StringComparer.Ordinal);
        var active = new HashSet<string>(StringComparer.Ordinal);
        void ValidateDefinition(string id)
        {
            if (!definitions.TryGetValue(id, out var definition)) throw new InvalidDataException($"Unknown definition '{id}'.");
            if (active.Contains(id)) throw new InvalidDataException($"Recursive definition '{id}'.");
            if (!visited.Add(id)) return;
            if (active.Count >= 64) throw new InvalidDataException("Definition nesting exceeds limit.");
            active.Add(id);
            var ports = Unique(definition.Ports, p => p.Id, "port");
            var locals = Unique(definition.Resources, r => r.Id.Value, "local resource");
            foreach (var port in definition.Ports)
                if (!Enum.IsDefined(port.Access) || !Enum.IsDefined(port.Tensor.ElementType) ||
                    port.ResourceKind is { } kind && (!Enum.IsDefined(kind) || kind is GraphResourceKind.Weight or GraphResourceKind.Constant && port.Access != GraphResourceAccess.Read))
                    throw new InvalidDataException($"Invalid port '{port.Id}' in '{id}'.");
            foreach (var local in definition.Resources)
                if (string.IsNullOrWhiteSpace(local.Name) || !Enum.IsDefined(local.Kind) || !Enum.IsDefined(local.Lifetime) ||
                    !Enum.IsDefined(local.Tensor.ElementType) || local.Lifetime is GraphResourceLifetime.Model or GraphResourceLifetime.Session)
                    throw new InvalidDataException($"Invalid local resource '{local.Id}' in '{id}'.");
            if (locals.Keys.Any(ports.ContainsKey)) throw new InvalidDataException($"Port/local resource collision in '{id}'.");
            if (definition.Resources.Any(r => r.Scope != GraphResourceScope.Local || r.Kind is GraphResourceKind.Weight or GraphResourceKind.SessionState))
                throw new InvalidDataException($"Definition '{id}' cannot own weights or session state.");
            if (definition.Body.DependsOn.Count != 0) throw new InvalidDataException("A definition body cannot depend on its caller scope.");
            ValidateScope(definition.Body);
            var elements = Walk(definition.Body).ToArray();
            Unique(elements, e => e.Id, "definition element");
            var metadataResources = locals.Keys.Concat(ports.Keys).Select(key => new ResourceId(key)).ToHashSet();
            foreach (var element in elements)
            {
                if (element is GraphRegionBody { Region.Architecture: { } architecture } region)
                {
                    architecture.Validate(region.Id, metadataResources);
                    architecture.ValidateBindings(region.Id, Walk(region).SelectMany(item => item switch
                    {
                        GraphNodeElement node => node.Node.Resources,
                        GraphCall call => call.Bindings,
                        _ => Array.Empty<NodeResourceBinding>(),
                    }));
                }
                var bindings = element switch { GraphNodeElement n => n.Node.Resources, GraphCall c => c.Bindings, _ => [] };
                foreach (var binding in bindings)
                {
                    if (!locals.ContainsKey(binding.Resource.Value) && !ports.ContainsKey(binding.Resource.Value))
                        throw new InvalidDataException($"Resource '{binding.Resource}' escapes definition '{id}'.");
                    if (!Enum.IsDefined(binding.Access) || binding.InitializedBeforeRead)
                        throw new InvalidDataException($"Invalid binding in definition '{id}'.");
                    if (ports.TryGetValue(binding.Resource.Value, out var port) && !Allows(port.Access, binding.Access))
                        throw new InvalidDataException($"Access exceeds port '{port.Id}' in '{id}'.");
                    var tensor = locals.TryGetValue(binding.Resource.Value, out var local) ? local.Tensor : ports[binding.Resource.Value].Tensor;
                    binding.View?.Validate(tensor);
                    if (local?.Kind == GraphResourceKind.Constant && binding.Access != GraphResourceAccess.Read)
                        throw new InvalidDataException($"Definition '{id}' writes a constant.");
                }
                if (element is GraphCall call)
                {
                    ValidateDefinition(call.Definition);
                    var target = definitions[call.Definition];
                    var callBindings = Unique(call.Bindings, b => b.Port, "call binding");
                    if (callBindings.Count != target.Ports.Count || target.Ports.Any(p => !callBindings.ContainsKey(p.Id)))
                        throw new InvalidDataException($"Call '{call.Id}' must bind every port exactly once.");
                    foreach (var expected in target.Ports)
                    {
                        var binding = callBindings[expected.Id];
                        var tensor = locals.TryGetValue(binding.Resource.Value, out var resource) ? resource.Tensor : ports[binding.Resource.Value].Tensor;
                        var kind = resource?.Kind ?? ports[binding.Resource.Value].ResourceKind;
                        if (binding.View is not null || binding.Access != expected.Access || !SameTensor(tensor, expected.Tensor) || expected.ResourceKind is { } requiredKind && kind != requiredKind)
                            throw new InvalidDataException($"Incompatible nested call port '{expected.Id}'.");
                    }
                }
                if (element is GraphNodeElement node)
                {
                    Unique(node.Node.Resources, b => b.Port, "node binding");
                    if (string.IsNullOrWhiteSpace(node.Node.Operation.Name) || node.Node.Operation.Version <= 0 ||
                        node.Node.Requirements.MinimumArithmeticType is not (GraphElementType.Float16 or GraphElementType.Float32) ||
                        node.Node.Requirements.MinimumAccumulatorType is not (GraphElementType.Float16 or GraphElementType.Float32))
                        throw new InvalidDataException($"Invalid node contract in definition '{id}'.");
                    if (node.Node.Dependencies.Count != 0) throw new InvalidDataException("Definition nodes cannot use legacy dependencies.");
                }
            }
            // Nested Write contracts have already been proved above; ReadWrite calls may preserve their input.
            GraphViewAccessValidation.ValidateOutputs(
                definition.Ports.Select(port => new GraphResource(new(port.Id), port.Id,
                    GraphResourceKind.Temporary, GraphResourceLifetime.Invocation, port.Tensor)).ToArray(),
                definition.Ports.Where(port => port.Access == GraphResourceAccess.Write).Select(port => new ResourceId(port.Id)).ToArray(),
                elements.SelectMany(element => element switch
                {
                    GraphNodeElement node => node.Node.Resources,
                    GraphCall call => call.Bindings.Where(binding => binding.Access == GraphResourceAccess.Write).ToArray(),
                    _ => Array.Empty<NodeResourceBinding>(),
                }));
            active.Remove(id);
        }
        foreach (var definition in LayerDefinitions) ValidateDefinition(definition.Id);
        ValidateScope(Root);
        var externalIds = externalResources.Select(r => r.Id.Value).ToHashSet(StringComparer.Ordinal);
        var resources = externalResources.ToList();
        var resourceMap = Unique(resources, r => r.Id.Value, "resource");
        var regions = new List<GraphRegion>();
        var nodes = new List<LogicalNode>();
        var nodeAliases = new Dictionary<int, string>();
        var spans = new Dictionary<string, (string? Owner, int First, int Last)>(StringComparer.Ordinal);
        var dependencies = new List<(string Element, string Target)>();
        var documentOrder = 0;
        string Name(string scope, string id) => scope.Length == 0 ? id : scope + "::" + id;
        ResourceId Resolve(string scope, ResourceId id, IReadOnlyDictionary<string, ResourceId>? bindings)
        {
            if (bindings is null) return externalIds.Contains(id.Value) ? id : throw new InvalidDataException($"Unknown external resource '{id}'; definition locals cannot escape their call.");
            return bindings.TryGetValue(id.Value, out var value) ? value : throw new InvalidDataException($"Unknown scoped resource '{id}' in '{scope}'.");
        }
        void Emit(GraphElement element, string scope, string? owner, IReadOnlyDictionary<string, ResourceId>? bindings, int depth)
        {
            if (depth > 64) throw new InvalidDataException("Graph expansion exceeds nesting limit.");
            var id = Name(scope, element.Id);
            var first = documentOrder++;
            if (!spans.TryAdd(id, (owner, first, first))) throw new InvalidDataException($"Duplicate element '{id}'.");
            if (spans.Count > maxExpandedNodes) throw new InvalidDataException("Graph expansion exceeds element limit.");
            foreach (var target in element.DependsOn) dependencies.Add((id, Name(scope, target)));
            switch (element)
            {
                case GraphNodeElement item:
                    if (item.Node.Dependencies.Count != 0) throw new InvalidDataException("Structured nodes use DependsOn, not legacy Dependencies.");
                    if (owner is null) throw new InvalidDataException("Nodes must be contained in a region.");
                    if (item.Node.Resources.Any(binding => !Enum.IsDefined(binding.Access)))
                        throw new InvalidDataException($"Invalid access in node '{id}'.");
                    var resolved = item.Node.Resources.Select(b => b with { Resource = Resolve(scope, b.Resource, bindings) }).ToArray();
                    var outputs = resolved.Where(b => b.Access != GraphResourceAccess.Read &&
                        resourceMap.TryGetValue(b.Resource.Value, out var r) && r.Kind == GraphResourceKind.Temporary && b.Resource.Value.EndsWith("_output", StringComparison.Ordinal)).ToArray();
                    if (scope.Length != 0 && outputs.Length == 1)
                        nodeAliases.Add(nodes.Count, outputs[0].Resource.Value[..^7]);
                    nodes.Add(item.Node with { Id = new(id), Region = new(owner), Resources = resolved, Dependencies = [] });
                    if (nodes.Count > maxExpandedNodes) throw new InvalidDataException("Graph expansion exceeds node limit.");
                    break;
                case GraphRegionBody body:
                    regions.Add(body.Region with
                    {
                        Id = new(id), ParentId = owner is null ? null : new RegionId(owner),
                        Architecture = body.Region.Architecture?.MapResources(resource => Resolve(scope, resource, bindings)),
                    });
                    foreach (var child in body.Children) Emit(child, scope, id, bindings, depth + 1);
                    break;
                case GraphCall call:
                    if (owner is null) throw new InvalidDataException("Calls must be contained in a region.");
                    if (!definitions.TryGetValue(call.Definition, out var definition)) throw new InvalidDataException($"Unknown definition '{call.Definition}'.");
                    var callBindings = Unique(call.Bindings, b => b.Port, "call binding");
                    if (callBindings.Count != definition.Ports.Count || definition.Ports.Any(p => !callBindings.ContainsKey(p.Id)))
                        throw new InvalidDataException($"Call '{id}' must bind every port exactly once.");
                    var map = new Dictionary<string, ResourceId>(StringComparer.Ordinal);
                    foreach (var port in definition.Ports)
                    {
                        var binding = callBindings[port.Id];
                        var resourceId = Resolve(scope, binding.Resource, bindings);
                        if (!resourceMap.TryGetValue(resourceId.Value, out var resource)) throw new InvalidDataException($"Unknown call resource '{resourceId}'.");
                        var tensor = binding.View?.Tensor ?? resource.Tensor;
                        if (binding.View is not null) throw new InvalidDataException("Bind views inside definition nodes; call ports bind complete resources.");
                        if (binding.InitializedBeforeRead || binding.Access != port.Access || !SameTensor(port.Tensor, tensor) ||
                            resource.Kind is GraphResourceKind.Weight or GraphResourceKind.Constant && binding.Access != GraphResourceAccess.Read ||
                            port.ResourceKind is { } kind && resource.Kind != kind)
                            throw new InvalidDataException($"Call '{id}' has incompatible port '{port.Id}'.");
                        map.Add(port.Id, resourceId);
                    }
                    foreach (var local in definition.Resources)
                    {
                        var localId = Name(id, local.Id.Value);
                        var resource = local with { Id = new(localId), Name = localId };
                        if (!resourceMap.TryAdd(localId, resource)) throw new InvalidDataException($"Duplicate expanded resource '{localId}'.");
                        resources.Add(resource);
                        if (resources.Count > maxExpandedNodes) throw new InvalidDataException("Graph expansion exceeds resource limit.");
                        map.Add(local.Id.Value, resource.Id);
                    }
                    // A call is itself an optimization boundary; its body root becomes this region.
                    regions.Add(definition.Body.Region with
                    {
                        Id = new(id), ParentId = new RegionId(owner),
                        Architecture = definition.Body.Region.Architecture?.MapResources(resource => Resolve(id, resource, map)),
                    });
                    foreach (var child in definition.Body.Children) Emit(child, id, id, map, depth + 1);
                    break;
                default: throw new InvalidDataException("Unknown graph element.");
            }
            spans[id] = (owner, first, documentOrder++);
        }
        if (Root.DependsOn.Count != 0) throw new InvalidDataException("Root region cannot have dependencies.");
        Emit(Root, "", null, null, 0);
        foreach (var (element, target) in dependencies)
        {
            if (!spans.TryGetValue(target, out var dependency)) throw new InvalidDataException($"Unknown dependency '{target}'.");
            var current = spans[element];
            if (current.Owner == dependency.Owner) throw new InvalidDataException("DependsOn is forbidden between children of the same region.");
            if (dependency.Last >= current.First) throw new InvalidDataException("DependsOn conflicts with document order or creates a cycle.");
        }
        // Keep declared element IDs reserved, including future siblings and nested scopes.
        var reservedIds = spans.Keys.ToHashSet(StringComparer.Ordinal);
        foreach (var (index, alias) in nodeAliases)
            if (reservedIds.Add(alias)) nodes[index] = nodes[index] with { Id = new(alias) };
        // Materialize serial control edges for all existing lowering, allocator and execution consumers.
        for (var i = 1; i < nodes.Count; i++) nodes[i] = nodes[i] with { Dependencies = [nodes[i - 1].Id] };
        return (resources.ToArray(), regions.ToArray(), nodes.ToArray());
    }

    private static void ValidateScope(GraphRegionBody root)
    {
        var spans = new Dictionary<string, (string? Owner, int Start, int End)>(StringComparer.Ordinal);
        var order = 0;
        void Visit(GraphElement element, string? owner, int depth)
        {
            if (depth > 64) throw new InvalidDataException("Region nesting exceeds limit.");
            var start = order++;
            if (string.IsNullOrWhiteSpace(element.Id) || !spans.TryAdd(element.Id, (owner, start, start)))
                throw new InvalidDataException($"Missing or duplicate element '{element.Id}'.");
            if (element is GraphNodeElement node && element.Id != node.Node.Id.Value ||
                element is GraphRegionBody body && element.Id != body.Region.Id.Value ||
                element is GraphCall call && element.Id != call.CallId)
                throw new InvalidDataException($"Conflicting identity for element '{element.Id}'.");
            if (element is GraphRegionBody region)
            {
                if (string.IsNullOrWhiteSpace(region.Region.Name) || string.IsNullOrWhiteSpace(region.Region.Type))
                    throw new InvalidDataException("Regions require names and types.");
                foreach (var child in region.Children) Visit(child, element.Id, depth + 1);
            }
            spans[element.Id] = (owner, start, order++);
        }
        Visit(root, null, 0);
        foreach (var element in Walk(root))
        {
            if (element.DependsOn.Distinct(StringComparer.Ordinal).Count() != element.DependsOn.Count)
                throw new InvalidDataException($"Duplicate dependency for '{element.Id}'.");
            foreach (var target in element.DependsOn)
            {
                if (!spans.TryGetValue(target, out var previous)) throw new InvalidDataException($"Unknown dependency '{target}'.");
                var current = spans[element.Id];
                if (current.Owner == previous.Owner) throw new InvalidDataException("DependsOn is forbidden in the same region.");
                if (previous.End >= current.Start) throw new InvalidDataException("DependsOn conflicts with document order or creates a cycle.");
            }
        }
    }

    internal static bool SameTensor(TensorDescriptor a, TensorDescriptor b) =>
        a.ElementType == b.ElementType && a.Layout == b.Layout && a.Dimensions.SequenceEqual(b.Dimensions);

    private static bool Allows(GraphResourceAccess contract, GraphResourceAccess access) => contract == GraphResourceAccess.ReadWrite || contract == access;

    private static Dictionary<string, T> Unique<T>(IEnumerable<T> values, Func<T, string> key, string kind)
    {
        var result = new Dictionary<string, T>(StringComparer.Ordinal);
        foreach (var value in values)
        {
            var id = key(value);
            if (string.IsNullOrWhiteSpace(id) || !result.TryAdd(id, value)) throw new InvalidDataException($"Missing or duplicate {kind} '{id}'.");
        }
        return result;
    }
}
