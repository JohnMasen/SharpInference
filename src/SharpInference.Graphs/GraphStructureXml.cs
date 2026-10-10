using System.Xml.Linq;

namespace SharpInference.Graphs;

public static partial class GraphXml
{
    private static XElement TensorXml(TensorDescriptor tensor) => new("Tensor",
        new XAttribute("elementType", tensor.ElementType), new XAttribute("layout", tensor.Layout),
        tensor.Dimensions.Select(size => new XElement("Dimension", new XAttribute("size", size))));

    private static XElement ResourceXml(GraphResource resource) => new("Resource",
        new XAttribute("id", resource.Id.Value), new XAttribute("name", resource.Name),
        new XAttribute("kind", resource.Kind), new XAttribute("lifetime", resource.Lifetime),
        new XAttribute("scope", resource.Scope), Optional("bindingKey", resource.BindingKey),
        Optional("deviceId", resource.DeviceId), TensorXml(resource.Tensor));

    private static XElement ElementXml(GraphElement element)
    {
        XElement result;
        switch (element)
        {
            case GraphRegionBody body:
                var region = body.Region;
                result = new XElement("Region", new XAttribute("id", body.Id), new XAttribute("type", region.Type),
                    new XAttribute("name", region.Name), Optional("role", region.Role), RegionMetadataXml(region),
                    body.Children.Select(ElementXml));
                break;
            case GraphNodeElement node:
                result = CreateNode(node.Id, node.Node.Operation, node.Node.Region, node.Node.Resources, [], node.Node.Attributes, node.Node.Requirements);
                result.Attribute("region")!.Remove();
                result.Element("Dependencies")!.Remove();
                break;
            case GraphCall call:
                result = new XElement("Call", new XAttribute("id", call.Id), new XAttribute("definition", call.Definition),
                    CreateBindings("Bindings", call.Bindings, false));
                break;
            default: throw new InvalidDataException("Unknown structured element.");
        }
        if (element.DependsOn.Count > 0) result.Add(new XElement("DependsOn", element.DependsOn.Select(id => new XElement("ElementRef", new XAttribute("id", id)))));
        return result;
    }

    private static string SerializeStructure(LogicalGraph graph)
    {
        var structure = graph.Structure!;
        var root = CreateGraph("LogicalGraph", graph.Identity, graph.Model, graph.DeclaredResources, [], graph.Inputs, graph.Outputs, graph.GraphState);
        root.Element("Regions")!.Remove();
        root.Add(new XElement("LayerDefinitions", structure.LayerDefinitions.Select(DefinitionXml)));
        root.Add(ElementXml(structure.Root));
        if (structure.HasTensorViews) root.SetAttributeValue("version", 2);
        return Write(root);
    }

    private static XElement DefinitionXml(GraphLayerDefinition definition) => new("LayerDefinition",
        new XAttribute("id", definition.Id), new XElement("Ports", definition.Ports.Select(port => new XElement("Port",
            new XAttribute("id", port.Id), new XAttribute("access", port.Access), Optional("resourceKind", port.ResourceKind?.ToString()), TensorXml(port.Tensor)))),
        new XElement("LocalResources", definition.Resources.Select(ResourceXml)), ElementXml(definition.Body));

    internal static string DefinitionFingerprint(GraphLayerDefinition definition) => DefinitionXml(definition).ToString(SaveOptions.DisableFormatting);

    private static TensorDescriptor ReadStructureTensor(XElement tensor)
    {
        Check(tensor, "Tensor", "elementType", "layout");
        return new(EnumValue<GraphElementType>(tensor, "elementType"), Children(tensor, "Dimension").Select(d =>
        {
            Check(d, "Dimension", "size");
            return Number(d, "size");
        }), Required(tensor, "layout"));
    }

    private static GraphResource ReadStructureResource(XElement item)
    {
        Check(item, "Resource", "id", "name", "kind", "lifetime", "scope", "bindingKey", "deviceId");
        return new(new(Required(item, "id")), Required(item, "name"), EnumValue<GraphResourceKind>(item, "kind"),
            EnumValue<GraphResourceLifetime>(item, "lifetime"), ReadStructureTensor(Ordered(item, "Tensor")[0]),
            (string?)item.Attribute("bindingKey"), (string?)item.Attribute("deviceId"),
            item.Attribute("scope") is null ? null : EnumValue<GraphResourceScope>(item, "scope"));
    }

    private static GraphElement ReadStructureElement(XElement element, string? parent)
    {
        var copy = new XElement(element);
        var depends = copy.Element("DependsOn");
        string[] dependencies = [];
        if (depends is not null)
        {
            Check(depends, "DependsOn");
            dependencies = Children(depends, "ElementRef").Select(e => { Check(e, "ElementRef", "id"); return Required(e, "id"); }).ToArray();
            depends.Remove();
        }
        GraphElement result;
        switch (copy.Name.LocalName)
        {
            case "Region":
                var region = ReadRegion(copy, parent, structured: true);
                result = new GraphRegionBody(region,
                    copy.Elements().Where(e => e.Name != "Attributes" && e.Name != "Architecture")
                        .Select(e => ReadStructureElement(e, Required(copy, "id"))).ToArray());
                break;
            case "Node":
                Check(copy, "Node", "id", "operation", "operationVersion");
                if (parent is null) throw new InvalidDataException("Node must belong to a region.");
                var nodeParts = Ordered(copy, "Bindings", "Attributes", "Precision");
                copy.SetAttributeValue("region", parent);
                nodeParts[0].AddAfterSelf(new XElement("Dependencies"));
                var fields = ReadNode(copy, false);
                result = new GraphNodeElement(new(new(fields.Id), fields.Operation, fields.Region, fields.Resources, [], fields.Attributes, fields.Requirements));
                break;
            case "Call":
                Check(copy, "Call", "id", "definition");
                var bindings = Ordered(copy, "Bindings")[0];
                Check(bindings, "Bindings");
                result = new GraphCall(Required(copy, "id"), Required(copy, "definition"), ReadBindings(bindings, false));
                break;
            default: throw new InvalidDataException($"Unknown executable element '{copy.Name}'.");
        }
        return result with { DependsOn = dependencies };
    }

    private static LogicalGraph DeserializeStructure(XElement root, IGraphModelSignatureReader? legacyReader)
    {
        var names = new List<string> { "Identity", "Model", "Resources", "Inputs", "Outputs" };
        if (root.Element("GraphState") is not null) names.Add("GraphState");
        names.AddRange(["LayerDefinitions", "Region"]);
        Ordered(root, names.ToArray());
        var identity = root.Element("Identity")!;
        Check(identity, "Identity", "architectureId", "irVersion", "name");
        var resourcesElement = root.Element("Resources")!;
        Check(resourcesElement, "Resources");
        Check(root.Element("Inputs")!, "Inputs");
        Check(root.Element("Outputs")!, "Outputs");
        var definitionsElement = root.Element("LayerDefinitions")!;
        Check(definitionsElement, "LayerDefinitions");
        var definitions = Children(definitionsElement, "LayerDefinition").Select(item =>
        {
            Check(item, "LayerDefinition", "id");
            var parts = Ordered(item, "Ports", "LocalResources", "Region");
            Check(parts[0], "Ports"); Check(parts[1], "LocalResources");
            var ports = Children(parts[0], "Port").Select(port =>
            {
                Check(port, "Port", "id", "access", "resourceKind");
                return new GraphDefinitionPort(Required(port, "id"), EnumValue<GraphResourceAccess>(port, "access"),
                    ReadStructureTensor(Ordered(port, "Tensor")[0]), port.Attribute("resourceKind") is null ? null : EnumValue<GraphResourceKind>(port, "resourceKind"));
            }).ToArray();
            return new GraphLayerDefinition(Required(item, "id"), ports, Children(parts[1], "Resource").Select(ReadStructureResource).ToArray(),
                (GraphRegionBody)ReadStructureElement(parts[2], null));
        }).ToArray();
        var stateElement = root.Element("GraphState");
        if (stateElement is not null) Check(stateElement, "GraphState");
        return new(new(Required(identity, "architectureId"), Number(identity, "irVersion"), Required(identity, "name")),
            ReadModel(root.Element("Model")!, legacyReader), Children(resourcesElement, "Resource").Select(ReadStructureResource),
            new GraphStructure(definitions, (GraphRegionBody)ReadStructureElement(root.Element("Region")!, null)),
            ReadRefs(root.Element("Inputs")!), ReadRefs(root.Element("Outputs")!), stateElement is null ? null : ReadGraphState(stateElement));
    }
}
