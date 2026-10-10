namespace SharpInference.Graphs;

/// <summary>Extracts model-owned regions as reusable, concrete-signature definitions. This is not fusion.</summary>
public static class GraphLayerReuse
{
    public static LogicalGraph Extract(LogicalGraph graph, Func<GraphRegion, string?> definitionFamily,
        Func<GraphRegion, GraphResource, GraphResourceAccess, string?>? portName = null)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(definitionFamily);
        var structure = graph.Structure ?? throw new ArgumentException("Extraction requires a structured graph.", nameof(graph));
        if (structure.LayerDefinitions.Count != 0) throw new ArgumentException("Extract regions before adding definitions.", nameof(graph));
        if (GraphStructure.Walk(structure.Root).Any(element => element.DependsOn.Count != 0))
            throw new ArgumentException("Extraction requires document-order-only regions; construct explicit definitions for supplemental dependencies.", nameof(graph));
        var resources = graph.Resources.ToDictionary(r => r.Id);
        var definitions = new List<GraphLayerDefinition>();
        var fingerprints = new Dictionary<string, string>(StringComparer.Ordinal);
        var removed = new HashSet<ResourceId>();
        GraphElement Convert(GraphElement element)
        {
            if (element is not GraphRegionBody region) return element;
            var family = definitionFamily(region.Region);
            if (family is null) return region with { Children = region.Children.Select(Convert).ToArray() };
            var elements = GraphStructure.Walk(region).ToArray();
            var nodeIds = elements.OfType<GraphNodeElement>().Select(n => n.Node.Id).ToHashSet();
            var nodes = elements.OfType<GraphNodeElement>().Select(n => n.Node).ToArray();
            var used = nodes.SelectMany(n => n.Resources).Select(b => b.Resource).Distinct().ToArray();
            var local = used.Where(id => resources[id].Kind == GraphResourceKind.Temporary &&
                !graph.Inputs.Contains(id) && !graph.Outputs.Contains(id) &&
                !graph.Nodes.Any(n => !nodeIds.Contains(n.Id) && n.Resources.Any(b => b.Resource == id))).ToHashSet();
            var names = new Dictionary<ResourceId, ResourceId>();
            var nodeNames = nodes.Select((node, index) => (node.Id, Name: $"n{index}")).ToDictionary(x => x.Id, x => x.Name);
            foreach (var id in local)
            {
                var producer = nodes.Last(n => n.Resources.Any(b => b.Resource == id && b.Access != GraphResourceAccess.Read));
                var outputCount = local.Count(resource => producer.Resources.Any(b => b.Resource == resource && b.Access != GraphResourceAccess.Read));
                var suffix = outputCount == 1 ? "" : "." + producer.Resources.First(b => b.Resource == id && b.Access != GraphResourceAccess.Read).Port;
                names.Add(id, new ResourceId(nodeNames[producer.Id] + suffix + "_output"));
                removed.Add(id);
            }
            var ports = new List<GraphDefinitionPort>();
            var bindings = new List<NodeResourceBinding>();
            foreach (var id in used.Where(id => !local.Contains(id)))
            {
                var uses = nodes.SelectMany(n => n.Resources).Where(b => b.Resource == id).ToArray();
                var read = uses.Any(b => b.Access != GraphResourceAccess.Write);
                var write = uses.Any(b => b.Access != GraphResourceAccess.Read);
                var access = read && write ? GraphResourceAccess.ReadWrite : write ? GraphResourceAccess.Write : GraphResourceAccess.Read;
                // Model weights/state keep meaningful roles; data ports remain independent of original node IDs.
                var resource = resources[id];
                var name = portName?.Invoke(region.Region, resource, access) ?? resource.Kind switch
                {
                    GraphResourceKind.Weight => "weight" + ports.Count,
                    GraphResourceKind.SessionState => "state" + ports.Count,
                    _ => (write ? "output" : "input") + ports.Count,
                };
                names.Add(id, new(name));
                ports.Add(new(name, access, resource.Tensor, resource.Kind is GraphResourceKind.Weight or GraphResourceKind.SessionState ? resource.Kind : null));
                bindings.Add(new(name, id, access));
            }
            var regionIndex = 0;
            GraphElement Normalize(GraphElement item, string? parent)
            {
                if (item is GraphNodeElement node)
                    return new GraphNodeElement(node.Node with { Id = new(nodeNames[node.Node.Id]), Region = new(parent!),
                        Dependencies = [], Resources = node.Node.Resources.Select(b => b with { Resource = names[b.Resource] }).ToArray() });
                if (item is not GraphRegionBody body) throw new InvalidDataException("Extraction accepts primitive regions only.");
                var id = parent is null ? "body" : "stage" + regionIndex++;
                return new GraphRegionBody(body.Region with
                    {
                        Id = new(id), ParentId = parent is null ? null : new RegionId(parent),
                        Name = body.Region.Architecture is null ? body.Region.Role ?? body.Region.Type : body.Region.Name,
                        Architecture = body.Region.Architecture?.MapResources(resource => names[resource]),
                    },
                    body.Children.Select(child => Normalize(child, id)).ToArray());
            }
            var body = (GraphRegionBody)Normalize(region, null);
            var locals = used.Where(local.Contains).Select(id => resources[id] with { Id = names[id], Name = names[id].Value }).ToArray();
            var definition = new GraphLayerDefinition(family, ports, locals, body);
            // Canonicalize actual ports/resources/body, including precision and views, not original names.
            var fingerprint = GraphXml.DefinitionFingerprint(definition);
            if (!fingerprints.TryGetValue(fingerprint, out var definitionId))
            {
                definitionId = definitions.Any(d => d.Id == family) ? family + "." + definitions.Count : family;
                definitions.Add(definition with { Id = definitionId });
                fingerprints.Add(fingerprint, definitionId);
            }
            return new GraphCall(region.Id, definitionId, bindings);
        }
        var root = (GraphRegionBody)Convert(structure.Root);
        return new LogicalGraph(graph.Identity, graph.Model, graph.DeclaredResources.Where(r => !removed.Contains(r.Id)),
            new GraphStructure(definitions, root), graph.Inputs, graph.Outputs, graph.GraphState);
    }
}
