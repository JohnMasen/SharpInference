using System.Xml.Linq;

namespace SharpInference.Graphs;

public static partial class GraphXml
{
    private static IEnumerable<XElement> RegionMetadataXml(GraphRegion region)
    {
        if (region.Architecture is { } architecture)
        {
            yield return new XElement("Architecture",
                Optional("role", architecture.Role),
                Optional("defaultCollapsed", architecture.DefaultCollapsed?.ToString().ToLowerInvariant()),
                Optional("repeatGroup", architecture.RepeatGroup?.ToString().ToLowerInvariant()),
                Optional("defaultView", architecture.DefaultView?.ToString().ToLowerInvariant()),
                Optional("step", architecture.Step?.ToString().ToLowerInvariant()),
                architecture.Description is null ? null : new XElement("Description", architecture.Description),
                architecture.Formula is null ? null : new XElement("Formula", architecture.Formula),
                architecture.Ports.Count == 0 ? null : new XElement("Ports", architecture.Ports.Select(port =>
                    new XElement("Port", new XAttribute("resource", port.Resource.Value),
                        new XAttribute("direction", port.Direction.ToString().ToLowerInvariant()),
                        new XAttribute("name", port.Name)))));
        }
        if (region.Architecture is null || region.Attributes.Count != 0)
            yield return CreateAttributes(region.Attributes);
    }

    private static GraphRegion ReadRegion(XElement element, string? parent, bool structured)
    {
        Check(element, "Region", structured ? ["id", "type", "name", "role"] : ["id", "parentId", "type", "name", "role"]);
        var metadata = element.Elements().TakeWhile(item => item.Name == "Attributes" || item.Name == "Architecture").ToArray();
        if (element.Elements("Attributes").Count() > 1 || element.Elements("Architecture").Count() > 1 ||
            element.Elements().Skip(metadata.Length).Any(item => item.Name == "Attributes" || item.Name == "Architecture") ||
            !structured && element.Elements().Skip(metadata.Length).Any())
            throw new InvalidDataException("Region metadata must occur once, before executable children.");
        var attributes = element.Element("Attributes");
        return new GraphRegion(new(Required(element, "id")), parent is null ? null : new RegionId(parent),
            Required(element, "type"), (string?)element.Attribute("role"), Required(element, "name"),
            attributes is null ? new Dictionary<string, string>(StringComparer.Ordinal) : ReadAttributes(attributes))
        {
            Architecture = ReadArchitecture(element.Element("Architecture")),
        };
    }

    private static GraphRegionArchitecture? ReadArchitecture(XElement? element)
    {
        if (element is null) return null;
        Check(element, "Architecture", "role", "defaultCollapsed", "repeatGroup", "defaultView", "step");
        var elements = element.Elements().ToArray();
        if (elements.Any(item => item.Name != "Description" && item.Name != "Formula" && item.Name != "Ports") ||
            elements.GroupBy(item => item.Name).Any(group => group.Count() > 1))
            throw new InvalidDataException("Unknown or duplicate Architecture child.");
        string? Content(string tag)
        {
            var item = element.Element(tag);
            if (item is null) return null;
            if (item.HasAttributes || item.Nodes().Any(node => node is not XText))
                throw new InvalidDataException($"Architecture {tag} must contain text only.");
            return item.Value;
        }
        bool? Flag(string name) => element.Attribute(name) is null ? null : Required(element, name) switch
        {
            "true" => true,
            "false" => false,
            _ => throw new InvalidDataException($"Invalid Architecture '{name}'."),
        };
        var ports = element.Element("Ports");
        if (ports is not null) Check(ports, "Ports");
        return new((string?)element.Attribute("role"), Content("Description"), Content("Formula"),
            Flag("defaultCollapsed"), Flag("repeatGroup"), ArchitectureEnum<GraphArchitectureView>(element, "defaultView"),
            ArchitectureEnum<GraphArchitectureStep>(element, "step"))
        {
            Ports = ports is null ? [] : Children(ports, "Port").Select(port =>
            {
                Check(port, "Port", "resource", "direction", "name");
                Ordered(port);
                return new GraphArchitecturePort(new(Required(port, "resource")),
                    ArchitectureEnum<GraphArchitecturePortDirection>(port, "direction") ??
                        throw new InvalidDataException("Architecture port requires a direction."),
                    Required(port, "name"));
            }).ToArray(),
        };
    }

    private static T? ArchitectureEnum<T>(XElement element, string name) where T : struct, Enum
    {
        if (element.Attribute(name) is null) return null;
        var value = Required(element, name);
        if (!Enum.TryParse<T>(value, true, out var parsed) || !Enum.IsDefined(parsed) ||
            parsed.ToString().ToLowerInvariant() != value)
            throw new InvalidDataException($"Invalid Architecture '{name}': '{value}'.");
        return parsed;
    }
}
