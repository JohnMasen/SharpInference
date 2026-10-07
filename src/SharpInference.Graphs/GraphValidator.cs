namespace SharpInference.Graphs;

public static class GraphValidator
{
    public static void Validate(LogicalGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ValidateCommon(graph.Identity, graph.Model, graph.Resources, graph.Regions, graph.Inputs, graph.Outputs, graph.GraphState);
        var nodeIds = EnsureUnique(graph.Nodes.Select(node => node.Id), "logical node");
        var regionIds = graph.Regions.Select(region => region.Id).ToHashSet();
        var resourceMap = graph.Resources.ToDictionary(resource => resource.Id);
        foreach (var node in graph.Nodes)
        {
            ValidateNode(node.Id.Value, node.Operation, node.Region, node.Resources, node.Requirements, regionIds, resourceMap);
            foreach (var dependency in node.Dependencies)
            {
                if (!nodeIds.Contains(dependency))
                {
                    throw new InvalidDataException($"Logical node '{node.Id}' depends on unknown node '{dependency}'.");
                }
            }
        }

        ValidateAcyclic(
            graph.Nodes.Select(node => node.Id),
            id => graph.Nodes.First(node => node.Id == id).Dependencies,
            id => id.Value);
    }

    public static void Validate(ExecutionGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ValidateCommon(graph.Identity, graph.Model, graph.Resources, graph.Regions, graph.Inputs, graph.Outputs, graph.GraphState);
        var nodeIds = EnsureUnique(graph.Nodes.Select(node => node.Id), "execution node");
        var regionIds = graph.Regions.Select(region => region.Id).ToHashSet();
        var resourceMap = graph.Resources.ToDictionary(resource => resource.Id);
        var internalOwners = new Dictionary<ResourceId, ExecutionNodeId>();
        foreach (var node in graph.Nodes)
        {
            ValidateNode(node.Id.Value, node.Operation, node.Region, node.Resources, node.Requirements, regionIds, resourceMap);
            if (node.Source is null || node.Source.LogicalNodes is null)
            {
                throw new InvalidDataException($"Execution node '{node.Id}' has an invalid logical source mapping.");
            }
            if (node.FirstWriteResources is null ||
                node.FirstWriteResources.Distinct().Count() != node.FirstWriteResources.Count ||
                node.FirstWriteResources.Any(id => !node.Resources.Any(binding =>
                    binding.Resource == id && binding.Access != GraphResourceAccess.Read)))
            {
                throw new InvalidDataException($"Execution node '{node.Id}' has invalid first-write metadata.");
            }

            var externalPorts = node.Resources.Select(binding => binding.Port).ToHashSet(StringComparer.Ordinal);
            var internalPorts = new HashSet<string>(StringComparer.Ordinal);
            foreach (var binding in node.InternalResources)
            {
                if (string.IsNullOrWhiteSpace(binding.Port) ||
                    !internalPorts.Add(binding.Port) || externalPorts.Contains(binding.Port))
                {
                    throw new InvalidDataException($"Execution node '{node.Id}' has duplicate internal port '{binding.Port}'.");
                }
                if (binding.InitializedBeforeRead && binding.Access == GraphResourceAccess.Read)
                {
                    throw new InvalidDataException(
                        $"Execution node '{node.Id}' cannot prove initialization for read-only internal resource '{binding.Resource}'.");
                }
                if (!resourceMap.TryGetValue(binding.Resource, out var resource) ||
                    resource.Scope != GraphResourceScope.Local ||
                    resource.Kind is GraphResourceKind.Weight or GraphResourceKind.Constant)
                {
                    throw new InvalidDataException($"Execution node '{node.Id}' has invalid internal resource '{binding.Resource}'.");
                }
                if (!internalOwners.TryAdd(binding.Resource, node.Id))
                {
                    throw new InvalidDataException($"Resource '{binding.Resource}' is internal to multiple nodes.");
                }
            }

            foreach (var dependency in node.Dependencies)
            {
                if (!nodeIds.Contains(dependency))
                {
                    throw new InvalidDataException($"Execution node '{node.Id}' depends on unknown node '{dependency}'.");
                }
            }
        }
        foreach (var (resource, owner) in internalOwners)
        {
            if (graph.Inputs.Contains(resource) || graph.Outputs.Contains(resource) ||
                graph.GraphState?.Slots.Any(slot => slot.Resource == resource) == true ||
                graph.Nodes.Any(node => node.Resources.Any(binding => binding.Resource == resource) ||
                    node.Id != owner && node.InternalResources.Any(binding => binding.Resource == resource)))
            {
                throw new InvalidDataException($"Internal resource '{resource}' is consumed outside node '{owner}'.");
            }
        }

        ValidateAcyclic(
            graph.Nodes.Select(node => node.Id),
            id => graph.Nodes.First(node => node.Id == id).Dependencies,
            id => id.Value);
    }

    private static void ValidateCommon(
        GraphIdentity identity,
        GraphModelSignature model,
        IReadOnlyList<GraphResource> resources,
        IReadOnlyList<GraphRegion> regions,
        IReadOnlyList<ResourceId> inputs,
        IReadOnlyList<ResourceId> outputs,
        GraphState? graphState)
    {
        if (string.IsNullOrWhiteSpace(identity.ArchitectureId) ||
            string.IsNullOrWhiteSpace(identity.Name) ||
            identity.IrVersion <= 0)
        {
            throw new InvalidDataException("The graph identity is invalid.");
        }

        if (string.IsNullOrWhiteSpace(model.ModelType) ||
            string.IsNullOrWhiteSpace(model.StateAbiId) ||
            model.Dimensions.Any(value => string.IsNullOrWhiteSpace(value.Key) || value.Value <= 0) ||
            string.Equals(model.ModelType, "rwkv", StringComparison.Ordinal) && !model.IsRwkvCompatible)
        {
            throw new InvalidDataException("The graph model signature is invalid.");
        }

        var resourceIds = EnsureUnique(resources.Select(resource => resource.Id), "resource");
        EnsureUnique(regions.Select(region => region.Id), "region");
        ValidateRegions(regions);
        foreach (var resource in resources)
        {
            if (string.IsNullOrWhiteSpace(resource.Id.Value) || string.IsNullOrWhiteSpace(resource.Name))
            {
                throw new InvalidDataException("Graph resources require identifiers and names.");
            }
            if (resource.Tensor is null || !Enum.IsDefined(resource.Kind) || !Enum.IsDefined(resource.Lifetime) ||
                !Enum.IsDefined(resource.Tensor.ElementType) ||
                resource.DeclaredScope is GraphResourceScope declared && !Enum.IsDefined(declared) ||
                resource.DeviceId is not null && string.IsNullOrWhiteSpace(resource.DeviceId))
            {
                throw new InvalidDataException($"Resource '{resource.Id}' has invalid metadata.");
            }
            if (resource.Kind is GraphResourceKind.Input or GraphResourceKind.Output &&
                resource.Scope != GraphResourceScope.Local ||
                resource.Lifetime == GraphResourceLifetime.Model && resource.Scope != GraphResourceScope.Global ||
                resource.Lifetime == GraphResourceLifetime.Session && resource.Scope != GraphResourceScope.Session ||
                resource.Lifetime is GraphResourceLifetime.Token or GraphResourceLifetime.Invocation &&
                resource.Scope != GraphResourceScope.Local)
            {
                throw new InvalidDataException($"Resource '{resource.Id}' has an incompatible scope and lifetime.");
            }

            if (resource.Kind == GraphResourceKind.Weight &&
                (resource.Lifetime != GraphResourceLifetime.Model || string.IsNullOrWhiteSpace(resource.BindingKey)))
            {
                throw new InvalidDataException($"Weight resource '{resource.Id}' requires model lifetime and a binding key.");
            }

            if (resource.Kind == GraphResourceKind.SessionState && resource.Lifetime != GraphResourceLifetime.Session)
            {
                throw new InvalidDataException($"Session state resource '{resource.Id}' requires session lifetime.");
            }
        }

        foreach (var id in inputs.Concat(outputs))
        {
            if (!resourceIds.Contains(id))
            {
                throw new InvalidDataException($"Graph input/output references unknown resource '{id}'.");
            }
        }
        EnsureUnique(inputs, "graph input");
        EnsureUnique(outputs, "graph output");
        if (graphState is null || graphState.Entries.Count == 0) return;
        if (graphState.Schema is null || string.IsNullOrWhiteSpace(graphState.Schema.Name))
            throw new InvalidDataException("Graph state schema name is missing.");
        var resourceMap = resources.ToDictionary(resource => resource.Id);
        var names = new HashSet<string>(StringComparer.Ordinal);
        var usedResources = new HashSet<ResourceId>();
        for (var index = 0; index < graphState.Slots.Count; index++)
        {
            var slot = graphState.Slots[index];
            if (slot is null || string.IsNullOrWhiteSpace(slot.Name) ||
                !names.Add(slot.Name) || !usedResources.Add(slot.Resource) ||
                !resourceMap.TryGetValue(slot.Resource, out var resource) ||
                resource.Kind != GraphResourceKind.SessionState || resource.Scope != GraphResourceScope.Session ||
                resource.Lifetime != GraphResourceLifetime.Session ||
                inputs.Contains(slot.Resource) || outputs.Contains(slot.Resource) ||
                !Enum.IsDefined(resource.Tensor.ElementType) || string.IsNullOrWhiteSpace(resource.Tensor.Layout))
                throw new InvalidDataException($"Graph state slot '{slot?.Name}' does not match its session resource.");
        }
    }

    private static void ValidateRegions(IReadOnlyList<GraphRegion> regions)
    {
        var roots = regions.Where(region => region.ParentId is null).ToArray();
        if (roots.Length != 1 || !string.Equals(roots[0].Type, GraphRegionTypes.Graph, StringComparison.Ordinal))
        {
            throw new InvalidDataException("A graph requires exactly one root region of type 'graph'.");
        }

        var map = regions.ToDictionary(region => region.Id);
        foreach (var region in regions)
        {
            if (string.IsNullOrWhiteSpace(region.Id.Value) ||
                string.IsNullOrWhiteSpace(region.Type) ||
                string.IsNullOrWhiteSpace(region.Name))
            {
                throw new InvalidDataException("Regions require identifiers, types, and names.");
            }

            var visited = new HashSet<RegionId> { region.Id };
            var parent = region.ParentId;
            while (parent is RegionId parentId)
            {
                if (!map.TryGetValue(parentId, out var parentRegion))
                {
                    throw new InvalidDataException($"Region '{region.Id}' has unknown parent '{parentId}'.");
                }

                if (!visited.Add(parentId))
                {
                    throw new InvalidDataException($"Region '{region.Id}' participates in a parent cycle.");
                }

                parent = parentRegion.ParentId;
            }
        }
    }

    private static void ValidateNode(
        string nodeId,
        GraphOperationId operation,
        RegionId region,
        IReadOnlyList<NodeResourceBinding> bindings,
        PrecisionRequirement requirements,
        IReadOnlySet<RegionId> regionIds,
        IReadOnlyDictionary<ResourceId, GraphResource> resources)
    {
        if (string.IsNullOrWhiteSpace(nodeId) || string.IsNullOrWhiteSpace(operation.Name) || operation.Version <= 0)
        {
            throw new InvalidDataException("Graph nodes require valid identifiers and operations.");
        }

        if (!regionIds.Contains(region))
        {
            throw new InvalidDataException($"Node '{nodeId}' references unknown region '{region}'.");
        }

        ArgumentNullException.ThrowIfNull(requirements);
        NumericTypeCompatibility.RequireFloatingPoint(requirements.MinimumArithmeticType, nameof(requirements));
        NumericTypeCompatibility.RequireFloatingPoint(requirements.MinimumAccumulatorType, nameof(requirements));

        var ports = new HashSet<string>(StringComparer.Ordinal);
        foreach (var binding in bindings)
        {
            if (string.IsNullOrWhiteSpace(binding.Port) || !ports.Add(binding.Port))
            {
                throw new InvalidDataException($"Node '{nodeId}' binds port '{binding.Port}' more than once.");
            }
            if (binding.InitializedBeforeRead)
            {
                throw new InvalidDataException($"Node '{nodeId}' marks public port '{binding.Port}' as private scratch.");
            }

            if (!resources.TryGetValue(binding.Resource, out var resource))
            {
                throw new InvalidDataException($"Node '{nodeId}' references unknown resource '{binding.Resource}'.");
            }

            if (resource.Kind is GraphResourceKind.Weight or GraphResourceKind.Constant &&
                binding.Access != GraphResourceAccess.Read)
            {
                throw new InvalidDataException($"Node '{nodeId}' attempts to write read-only resource '{binding.Resource}'.");
            }
        }
    }

    private static HashSet<T> EnsureUnique<T>(IEnumerable<T> values, string kind)
        where T : notnull
    {
        var result = new HashSet<T>();
        foreach (var value in values)
        {
            if (!result.Add(value))
            {
                throw new InvalidDataException($"Duplicate {kind} identifier '{value}'.");
            }
        }

        return result;
    }

    private static void ValidateAcyclic<T>(
        IEnumerable<T> values,
        Func<T, IEnumerable<T>> dependencies,
        Func<T, string> display)
        where T : notnull
    {
        var states = new Dictionary<T, int>();
        foreach (var value in values)
        {
            Visit(value);
        }

        void Visit(T value)
        {
            if (states.TryGetValue(value, out var state))
            {
                if (state == 1)
                {
                    throw new InvalidDataException($"Graph dependency cycle detected at '{display(value)}'.");
                }

                return;
            }

            states[value] = 1;
            foreach (var dependency in dependencies(value))
            {
                Visit(dependency);
            }

            states[value] = 2;
        }
    }
}
