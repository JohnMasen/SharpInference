using System.Collections.ObjectModel;
using System.Globalization;
using System.Xml;
using System.Xml.Linq;
using SharpInference.Graphs.Core;

namespace SharpInference.Graphs;

public static partial class GraphXml
{
    private const int MaxCharacters = 16 * 1024 * 1024;
    private const int MaxElements = 100_000;
    private const int MaxDepth = 64;

    public static string Serialize(LogicalGraphDefinition graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var definitions = CollectDefinitions(graph);
        return Write(new XElement(
            "Graph",
            new XAttribute("kind", "LogicalGraphDefinition"),
            new XAttribute("version", 2),
            new XElement("Root", new XAttribute("definition", graph.Id.Value)),
            new XElement(
                "Definitions",
                definitions.Values
                    .OrderBy(definition => definition.Id.Value, StringComparer.Ordinal)
                    .Select(WriteDefinition))));
    }

    public static LogicalGraphDefinition DeserializeDefinition(string xml)
    {
        var root = Read(xml, "LogicalGraphDefinition", 2);
        var rootElement = RequiredChild(root, "Root");
        Check(rootElement, "Root", "definition");
        var rootId = Required(rootElement, "definition");
        var records = Children(RequiredChild(root, "Definitions"), "Definition")
            .Select(ReadDefinitionRecord)
            .ToDictionary(record => record.Id, StringComparer.Ordinal);
        var completed = new Dictionary<string, LogicalGraphDefinition>(StringComparer.Ordinal);
        var active = new HashSet<string>(StringComparer.Ordinal);
        var result = Build(rootId);
        if (completed.Count != records.Count)
        {
            throw new InvalidDataException(
                "Logical graph XML contains unreachable definitions.");
        }

        return result;

        LogicalGraphDefinition Build(string id)
        {
            if (completed.TryGetValue(id, out var existing))
            {
                return existing;
            }

            if (!records.TryGetValue(id, out var record))
            {
                throw new InvalidDataException($"Unknown graph definition '{id}'.");
            }

            if (!active.Add(id))
            {
                throw new InvalidDataException($"Recursive graph definition '{id}'.");
            }

            var children = record.Structure.Nodes.OfType<GraphCallNode>()
                .Select(call => Build(call.Definition.Value))
                .DistinctBy(definition => definition.Id)
                .ToArray();
            var definition = new LogicalGraphDefinition(
                new(record.Id),
                record.Boundary,
                record.Annotation,
                record.Structure,
                children);
            active.Remove(id);
            completed.Add(id, definition);
            return definition;
        }
    }

    private static IReadOnlyDictionary<GraphDefinitionId, LogicalGraphDefinition>
        CollectDefinitions(LogicalGraphDefinition root)
    {
        var result = new Dictionary<GraphDefinitionId, LogicalGraphDefinition>();
        var active = new HashSet<GraphDefinitionId>();
        Visit(root);
        return result;

        void Visit(LogicalGraphDefinition definition)
        {
            if (result.ContainsKey(definition.Id))
            {
                return;
            }

            if (!active.Add(definition.Id))
            {
                throw new InvalidDataException(
                    $"Recursive graph definition '{definition.Id}'.");
            }

            foreach (var child in definition.Definitions)
            {
                Visit(child);
            }

            active.Remove(definition.Id);
            result.Add(definition.Id, definition);
        }
    }

    private static XElement WriteDefinition(LogicalGraphDefinition definition) =>
        new(
            "Definition",
            new XAttribute("id", definition.Id.Value),
            new XAttribute("compileBoundary", definition.CompileBoundary),
            WriteAnnotation(definition.Annotation),
            new XElement("Nodes", definition.Structure.Nodes.Select(WriteLogicalNode)),
            new XElement("Edges", definition.Structure.Edges.Select(WriteLogicalEdge)));

    private static XElement WriteLogicalNode(LogicalGraphNode node) => node switch
    {
        LogicalResourceNode resource => new XElement(
            "Resource",
            new XAttribute("id", resource.Resource.Id.Value),
            new XAttribute("name", resource.Resource.Name),
            new XAttribute("kind", resource.Resource.Kind),
            new XAttribute("owner", resource.Resource.Owner.Value),
            resource.ResourceNodeId.Owner.IsParent
                ? null
                : new XAttribute(
                    "instance", resource.ResourceNodeId.Owner.Call!.Value.Value),
            Optional("bindingKey", resource.Resource.BindingKey),
            WriteTensor(resource.Resource.Tensor)),
        GraphCallNode call => new XElement(
            "Call",
            new XAttribute("id", call.Id.Value),
            new XAttribute("definition", call.Definition.Value),
            WriteAnnotation(call.Annotation)),
        GraphOperationNode operation => new XElement(
            "Operation",
            new XAttribute("id", operation.Id.Value),
            new XAttribute("operation", operation.Operation.Name),
            new XAttribute("operationVersion", operation.Operation.Version),
            WriteAnnotation(operation.Annotation),
            WriteAttributes(operation.Attributes),
            WritePrecision(operation.Requirements)),
        _ => throw new InvalidDataException(
            $"Unsupported logical node type '{node.GetType().Name}'."),
    };

    private static XElement WriteLogicalEdge(LogicalGraphEdge edge) => edge switch
    {
        LogicalDependencyEdge dependency => new XElement(
            "Dependency",
            new XAttribute("id", dependency.Id.Value),
            new XAttribute("source", dependency.Source.Value),
            new XAttribute("target", dependency.Target.Value)),
        LogicalResourceReadEdge read => new XElement(
            "Read",
            new XAttribute("id", read.Id.Value),
            new XAttribute("source", read.Source.Value),
            new XAttribute("target", read.Target.Value),
            new XAttribute("port", read.Port),
            new XAttribute("binding", read.BindingId)),
        LogicalResourceWriteEdge write => new XElement(
            "Write",
            new XAttribute("id", write.Id.Value),
            new XAttribute("source", write.Source.Value),
            new XAttribute("target", write.Target.Value),
            new XAttribute("port", write.Port),
            new XAttribute("binding", write.BindingId)),
        _ => throw new InvalidDataException(
            $"Unsupported logical edge type '{edge.GetType().Name}'."),
    };

    private static DefinitionRecord ReadDefinitionRecord(XElement element)
    {
        Check(element, "Definition", "id", "compileBoundary");
        var nodes = DefinitionChildren(
                RequiredChild(element, "Nodes"), "Resource", "Operation", "Call")
            .Select(ReadLogicalNode).ToArray();
        var edges = DefinitionChildren(
                RequiredChild(element, "Edges"), "Dependency", "Read", "Write")
            .Select(ReadLogicalEdge).ToArray();
        return new(
            Required(element, "id"),
            DefinitionEnumValue<GraphCompileBoundary>(element, "compileBoundary"),
            ReadAnnotation(element.Element("Annotation")),
            new Graph<LogicalGraphNode, LogicalGraphEdge>(nodes, edges));
    }

    private static LogicalGraphNode ReadLogicalNode(XElement element)
    {
        if (element.Name == "Resource")
        {
            Check(
                element, "Resource", "id", "name", "kind", "owner", "instance",
                "bindingKey");
            var resource = new GraphDefinitionResource(
                new(Required(element, "id")),
                Required(element, "name"),
                DefinitionEnumValue<GraphDefinitionResourceKind>(element, "kind"),
                ReadTensor(RequiredChild(element, "Tensor")),
                new(Required(element, "owner")),
                (string?)element.Attribute("bindingKey"));
            var instance = (string?)element.Attribute("instance");
            var owner = string.IsNullOrWhiteSpace(instance)
                ? GraphBindingOwner.Parent
                : GraphBindingOwner.Instance(new(instance));
            return new LogicalResourceNode(
                new ModelResourceNodeId(owner, resource.Id), resource);
        }

        if (element.Name == "Call")
        {
            Check(element, "Call", "id", "definition");
            return new GraphCallNode(
                new(Required(element, "id")),
                new(Required(element, "definition")),
                ReadAnnotation(element.Element("Annotation")));
        }

        Check(element, "Operation", "id", "operation", "operationVersion");
        return new GraphOperationNode(
            new(Required(element, "id")),
            new(Required(element, "operation"), Number(element, "operationVersion")),
            ReadAttributes(RequiredChild(element, "Attributes")),
            ReadPrecision(RequiredChild(element, "Precision")),
            ReadAnnotation(element.Element("Annotation")));
    }

    private static LogicalGraphEdge ReadLogicalEdge(XElement element)
    {
        Check(
            element,
            element.Name.LocalName,
            element.Name == "Dependency"
                ? ["id", "source", "target"]
                : ["id", "source", "target", "port", "binding"]);
        var id = new GraphEdgeId(Required(element, "id"));
        var source = new GraphNodeId(Required(element, "source"));
        var target = new GraphNodeId(Required(element, "target"));
        return element.Name.LocalName switch
        {
            "Dependency" => new LogicalDependencyEdge(id, source, target),
            "Read" => new LogicalResourceReadEdge(
                id, source, target, Required(element, "port"),
                Required(element, "binding")),
            "Write" => new LogicalResourceWriteEdge(
                id, source, target, Required(element, "port"),
                Required(element, "binding")),
            _ => throw new InvalidDataException(
                $"Unsupported logical edge element <{element.Name}>."),
        };
    }

    private static XElement WriteAttributes(IReadOnlyDictionary<string, string> attributes) =>
        new(
            "Attributes",
            attributes.Select(attribute => new XElement(
                "Attribute",
                new XAttribute("name", attribute.Key),
                new XAttribute("value", attribute.Value))));

    private static IReadOnlyDictionary<string, string> ReadAttributes(XElement element) =>
        new ReadOnlyDictionary<string, string>(
            Children(element, "Attribute").ToDictionary(attribute =>
            {
                Check(attribute, "Attribute", "name", "value");
                return Required(attribute, "name");
            }, attribute => Required(attribute, "value"), StringComparer.Ordinal));

    private static XElement WritePrecision(PrecisionRequirement precision) =>
        new(
            "Precision",
            new XAttribute("arithmetic", precision.MinimumArithmeticType),
            new XAttribute("accumulator", precision.MinimumAccumulatorType));

    private static PrecisionRequirement ReadPrecision(XElement element)
    {
        Check(element, "Precision", "arithmetic", "accumulator");
        return new(
            DefinitionEnumValue<GraphElementType>(element, "arithmetic"),
            DefinitionEnumValue<GraphElementType>(element, "accumulator"));
    }

    private static XElement? WriteAnnotation(GraphAnnotation? annotation)
    {
        if (annotation is null)
        {
            return null;
        }

        return new XElement(
            "Annotation",
            Optional("name", annotation.Name),
            Optional("role", annotation.Role),
            Optional("description", annotation.Description),
            (annotation.Attributes ?? new Dictionary<string, string>())
                .Select(attribute => new XElement(
                    "Attribute",
                    new XAttribute("name", attribute.Key),
                    new XAttribute("value", attribute.Value))));
    }

    private static GraphAnnotation? ReadAnnotation(XElement? element)
    {
        if (element is null)
        {
            return null;
        }

        Check(element, "Annotation", "name", "role", "description");
        return new(
            (string?)element.Attribute("name"),
            (string?)element.Attribute("role"),
            (string?)element.Attribute("description"),
            ReadAttributes(element));
    }

    private static XElement WriteTensor(TensorDescriptor tensor) =>
        new(
            "Tensor",
            new XAttribute("elementType", tensor.ElementType),
            new XAttribute("layout", tensor.Layout),
            tensor.Dimensions.Select(dimension =>
                new XElement("Dimension", new XAttribute("value", dimension))));

    private static TensorDescriptor ReadTensor(XElement element)
    {
        Check(element, "Tensor", "elementType", "layout");
        return new(
            DefinitionEnumValue<GraphElementType>(element, "elementType"),
            Children(element, "Dimension").Select(dimension =>
            {
                Check(dimension, "Dimension", "value");
                return Number(dimension, "value");
            }),
            Required(element, "layout"));
    }

    private static T DefinitionEnumValue<T>(XElement element, string attribute)
        where T : struct, Enum
    {
        var value = Required(element, attribute);
        if (!Enum.TryParse<T>(value, false, out var result) || !Enum.IsDefined(result))
        {
            throw new InvalidDataException($"Invalid {attribute} '{value}'.");
        }

        return result;
    }

    private static XElement RequiredChild(XElement parent, string name)
    {
        var values = parent.Elements(name).ToArray();
        if (values.Length != 1)
        {
            throw new InvalidDataException(
                $"Element <{parent.Name}> requires exactly one <{name}> child.");
        }

        return values[0];
    }

    private static IEnumerable<XElement> DefinitionChildren(
        XElement parent,
        params string[] names)
    {
        if (parent.Elements().Any(element =>
            !names.Contains(element.Name.LocalName, StringComparer.Ordinal) ||
            element.Name.Namespace != XNamespace.None))
        {
            throw new InvalidDataException($"Unexpected children in <{parent.Name}>.");
        }

        return parent.Elements();
    }

    private static string Write(XElement root)
    {
        var xml = root.ToString(SaveOptions.DisableFormatting);
        if (xml.Length > MaxCharacters ||
            root.DescendantsAndSelf().Take(MaxElements + 1).Count() > MaxElements)
        {
            throw new InvalidDataException("Graph XML exceeds the size limit.");
        }

        return xml;
    }

    private static XElement Read(string xml, string kind, int version)
    {
        ArgumentNullException.ThrowIfNull(xml);
        if (xml.Length > MaxCharacters)
        {
            throw new InvalidDataException("Graph XML exceeds the size limit.");
        }

        try
        {
            var settings = new XmlReaderSettings
            {
                DtdProcessing = DtdProcessing.Prohibit,
                XmlResolver = null,
                MaxCharactersInDocument = MaxCharacters,
                MaxCharactersFromEntities = 0,
            };
            using var reader = XmlReader.Create(new StringReader(xml), settings);
            var document = XDocument.Load(reader, LoadOptions.None);
            var root = document.Root
                ?? throw new InvalidDataException("Graph XML has no root.");
            if (root.DescendantsAndSelf().Take(MaxElements + 1).Count() > MaxElements ||
                root.DescendantsAndSelf().Any(element =>
                    element.Ancestors().Take(MaxDepth + 1).Count() > MaxDepth))
            {
                throw new InvalidDataException(
                    "Graph XML exceeds the structural limit.");
            }

            Check(root, "Graph", "kind", "version");
            if (Required(root, "kind") != kind)
            {
                throw new InvalidDataException($"Graph XML kind must be '{kind}'.");
            }

            if (Number(root, "version") != version)
            {
                throw new InvalidDataException(
                    $"Unsupported graph XML version '{Required(root, "version")}'.");
            }

            return root;
        }
        catch (InvalidDataException)
        {
            throw;
        }
        catch (Exception exception) when (
            exception is XmlException or InvalidOperationException or FormatException)
        {
            throw new InvalidDataException("Invalid graph XML.", exception);
        }
    }

    private static void Check(
        XElement element,
        string name,
        params string[] allowedAttributes)
    {
        if (element.Name != name || element.Name.Namespace != XNamespace.None)
        {
            throw new InvalidDataException($"Expected <{name}>.");
        }

        if (element.Attributes().Any(attribute =>
            attribute.IsNamespaceDeclaration ||
            !allowedAttributes.Contains(
                attribute.Name.LocalName, StringComparer.Ordinal) ||
            attribute.Name.Namespace != XNamespace.None))
        {
            throw new InvalidDataException($"Unexpected attribute on <{name}>.");
        }
    }

    private static IEnumerable<XElement> Children(XElement parent, string name)
    {
        if (parent.Elements().Any(element =>
            element.Name != name || element.Name.Namespace != XNamespace.None))
        {
            throw new InvalidDataException($"Unexpected children in <{parent.Name}>.");
        }

        return parent.Elements(name);
    }

    private static string Required(XElement element, string attribute) =>
        string.IsNullOrWhiteSpace((string?)element.Attribute(attribute))
            ? throw new InvalidDataException(
                $"Element <{element.Name}> requires attribute '{attribute}'.")
            : (string)element.Attribute(attribute)!;

    private static int Number(XElement element, string attribute) =>
        int.TryParse(
            Required(element, attribute),
            NumberStyles.None,
            CultureInfo.InvariantCulture,
            out var value)
            ? value
            : throw new InvalidDataException(
                $"Attribute '{attribute}' on <{element.Name}> must be an integer.");

    private static XAttribute? Optional(string name, string? value) =>
        value is null ? null : new XAttribute(name, value);

    private sealed record DefinitionRecord(
        string Id,
        GraphCompileBoundary Boundary,
        GraphAnnotation? Annotation,
        Graph<LogicalGraphNode, LogicalGraphEdge> Structure);
}
