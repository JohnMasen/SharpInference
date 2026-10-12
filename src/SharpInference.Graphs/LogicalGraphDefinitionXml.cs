using System.Globalization;
using System.Xml;
using System.Xml.Linq;

namespace SharpInference.Graphs;

public static partial class GraphXml
{
    private const int MaxCharacters = 16 * 1024 * 1024;
    private const int MaxElements = 100_000;
    private const int MaxDepth = 64;

    /// <summary>Serializes a reusable logical graph definition and all directly or transitively called definitions.</summary>
    public static string Serialize(LogicalGraphDefinition graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var definitions = CollectDefinitions(graph);
        var root = new XElement("Graph",
            new XAttribute("kind", "LogicalGraphDefinition"),
            new XAttribute("version", 1),
            new XElement("Root", new XAttribute("definition", graph.Id.Value)),
            new XElement("Definitions", definitions.Values
                .OrderBy(definition => definition.Id.Value, StringComparer.Ordinal)
                .Select(WriteDefinition)));
        return Write(root);
    }

    /// <summary>Deserializes and validates a reusable logical graph definition.</summary>
    public static LogicalGraphDefinition DeserializeDefinition(string xml)
    {
        var root = Read(xml, "LogicalGraphDefinition");
        Check(root, "Graph", "kind", "version");
        var rootElement = RequiredChild(root, "Root");
        Check(rootElement, "Root", "definition");
        var rootId = Required(rootElement, "definition");
        var container = RequiredChild(root, "Definitions");
        var records = Children(container, "Definition")
            .Select(ReadDefinitionRecord)
            .ToDictionary(record => record.Id, StringComparer.Ordinal);
        var completed = new Dictionary<string, LogicalGraphDefinition>(StringComparer.Ordinal);
        var active = new HashSet<string>(StringComparer.Ordinal);
        var result = Build(rootId);
        if (completed.Count != records.Count)
            throw new InvalidDataException("Logical graph XML contains unreachable definitions.");
        return result;

        LogicalGraphDefinition Build(string id)
        {
            if (completed.TryGetValue(id, out var existing)) return existing;
            if (!records.TryGetValue(id, out var record))
                throw new InvalidDataException($"Unknown graph definition '{id}'.");
            if (!active.Add(id))
                throw new InvalidDataException($"Recursive graph definition '{id}'.");
            var children = record.Nodes.OfType<GraphCallNode>()
                .Select(call => Build(call.Definition.Value))
                .DistinctBy(definition => definition.Id)
                .ToArray();
            var definition = new LogicalGraphDefinition(
                new GraphDefinitionId(record.Id),
                record.Boundary,
                record.Annotation,
                record.Resources,
                LogicalGraphDefinitionValidator.InferDependencies(record.Nodes),
                children);
            active.Remove(id);
            completed.Add(id, definition);
            return definition;
        }
    }

    private static IReadOnlyDictionary<GraphDefinitionId, LogicalGraphDefinition> CollectDefinitions(
        LogicalGraphDefinition root)
    {
        var result = new Dictionary<GraphDefinitionId, LogicalGraphDefinition>();
        var active = new HashSet<GraphDefinitionId>();
        Visit(root);
        return result;

        void Visit(LogicalGraphDefinition definition)
        {
            if (result.ContainsKey(definition.Id)) return;
            if (!active.Add(definition.Id))
                throw new InvalidDataException($"Recursive graph definition '{definition.Id}'.");
            foreach (var child in definition.Definitions) Visit(child);
            active.Remove(definition.Id);
            result.Add(definition.Id, definition);
        }
    }

    private static XElement WriteDefinition(LogicalGraphDefinition definition) =>
        new("Definition",
            new XAttribute("id", definition.Id.Value),
            new XAttribute("compileBoundary", definition.CompileBoundary),
            WriteAnnotation(definition.Annotation),
            new XElement("Resources", definition.Resources.Select(WriteDefinitionResource)),
            new XElement("Nodes", definition.Nodes.Select(WriteDefinitionNode)));

    private static XElement WriteDefinitionResource(GraphDefinitionResource resource) =>
        new("Resource",
            new XAttribute("id", resource.Id.Value),
            new XAttribute("name", resource.Name),
            new XAttribute("kind", resource.Kind),
            new XAttribute("owner", resource.Owner.Value),
            Optional("bindingKey", resource.BindingKey),
            WriteTensor(resource.Tensor));

    private static XElement WriteDefinitionNode(GraphDefinitionNode node)
    {
        var result = new XElement(node is GraphCallNode ? "Call" : "Operation",
            new XAttribute("id", node.Id.Value));
        if (node is GraphCallNode call)
            result.Add(new XAttribute("definition", call.Definition.Value));
        else
        {
            var operationNode = (GraphOperationNode)node;
            result.Add(
                new XAttribute("operation", operationNode.Operation.Name),
                new XAttribute("operationVersion", operationNode.Operation.Version));
        }
        result.Add(WriteAnnotation(node.Annotation),
            new XElement("Bindings", node.Bindings.Select(binding =>
                new XElement("Binding",
                    new XAttribute("port", binding.Port),
                    new XAttribute("resource", binding.Resource.Value),
                    binding.ResourceOwner.IsParent
                        ? null
                        : new XAttribute("instance", binding.ResourceOwner.Call!.Value.Value),
                    new XAttribute("access", binding.Access)))),
            new XElement("Dependencies", node.Dependencies.Select(dependency =>
                new XElement("NodeRef", new XAttribute("id", dependency.Value)))));
        if (node is GraphOperationNode operation)
        {
            result.Add(new XElement("Attributes", operation.Attributes.Select(attribute =>
                    new XElement("Attribute",
                        new XAttribute("name", attribute.Key),
                        new XAttribute("value", attribute.Value)))),
                new XElement("Precision",
                    new XAttribute("arithmetic", operation.Requirements.MinimumArithmeticType),
                    new XAttribute("accumulator", operation.Requirements.MinimumAccumulatorType)));
        }
        return result;
    }

    private static XElement? WriteAnnotation(GraphAnnotation? annotation)
    {
        if (annotation is null) return null;
        return new XElement("Annotation",
            Optional("name", annotation.Name),
            Optional("role", annotation.Role),
            Optional("description", annotation.Description),
            (annotation.Attributes ?? new Dictionary<string, string>()).Select(attribute =>
                new XElement("Attribute",
                    new XAttribute("name", attribute.Key),
                    new XAttribute("value", attribute.Value))));
    }

    private static XElement WriteTensor(TensorDescriptor tensor) =>
        new("Tensor",
            new XAttribute("elementType", tensor.ElementType),
            new XAttribute("layout", tensor.Layout),
            tensor.Dimensions.Select(dimension =>
                new XElement("Dimension", new XAttribute("value", dimension))));

    private static DefinitionRecord ReadDefinitionRecord(XElement element)
    {
        Check(element, "Definition", "id", "compileBoundary");
        var id = Required(element, "id");
        var boundary = DefinitionEnumValue<GraphCompileBoundary>(element, "compileBoundary");
        var annotation = ReadAnnotation(element.Element("Annotation"));
        var resources = Children(RequiredChild(element, "Resources"), "Resource")
            .Select(ReadDefinitionResource).ToArray();
        var nodes = DefinitionChildren(RequiredChild(element, "Nodes"), "Operation", "Call")
            .Select(ReadDefinitionNode).ToArray();
        return new DefinitionRecord(id, boundary, annotation, resources, nodes);
    }

    private static GraphDefinitionResource ReadDefinitionResource(XElement element)
    {
        Check(element, "Resource", "id", "name", "kind", "owner", "bindingKey");
        return new GraphDefinitionResource(
            new ResourceId(Required(element, "id")),
            Required(element, "name"),
            DefinitionEnumValue<GraphDefinitionResourceKind>(element, "kind"),
            ReadTensor(RequiredChild(element, "Tensor")),
            new GraphDefinitionId(Required(element, "owner")),
            (string?)element.Attribute("bindingKey"));
    }

    private static GraphDefinitionNode ReadDefinitionNode(XElement element)
    {
        var operation = element.Name == "Operation";
        Check(element, operation ? "Operation" : "Call",
            operation
                ? ["id", "operation", "operationVersion"]
                : ["id", "definition"]);
        var id = new GraphDefinitionNodeId(Required(element, "id"));
        var annotation = ReadAnnotation(element.Element("Annotation"));
        var bindings = Children(RequiredChild(element, "Bindings"), "Binding").Select(binding =>
        {
            Check(binding, "Binding", "port", "resource", "instance", "access");
            var instance = (string?)binding.Attribute("instance");
            return new GraphDefinitionBinding(
                Required(binding, "port"),
                new ResourceId(Required(binding, "resource")),
                DefinitionEnumValue<GraphResourceAccess>(binding, "access"),
                string.IsNullOrWhiteSpace(instance)
                    ? GraphBindingOwner.Parent
                    : GraphBindingOwner.Instance(new GraphDefinitionNodeId(instance)));
        }).ToArray();
        var dependencies = Children(RequiredChild(element, "Dependencies"), "NodeRef").Select(reference =>
        {
            Check(reference, "NodeRef", "id");
            return new GraphDefinitionNodeId(Required(reference, "id"));
        }).ToArray();
        if (!operation)
            return new GraphCallNode(
                id,
                new GraphDefinitionId(Required(element, "definition")),
                bindings,
                dependencies,
                annotation);
        var attributes = Children(RequiredChild(element, "Attributes"), "Attribute").ToDictionary(attribute =>
        {
            Check(attribute, "Attribute", "name", "value");
            return Required(attribute, "name");
        }, attribute => Required(attribute, "value"), StringComparer.Ordinal);
        var precision = RequiredChild(element, "Precision");
        Check(precision, "Precision", "arithmetic", "accumulator");
        return new GraphOperationNode(
            id,
            new GraphOperationId(
                Required(element, "operation"),
                Number(element, "operationVersion")),
            bindings,
            dependencies,
            new System.Collections.ObjectModel.ReadOnlyDictionary<string, string>(attributes),
            new PrecisionRequirement(
                DefinitionEnumValue<GraphElementType>(precision, "arithmetic"),
                DefinitionEnumValue<GraphElementType>(precision, "accumulator")),
            annotation);
    }

    private static GraphAnnotation? ReadAnnotation(XElement? element)
    {
        if (element is null) return null;
        Check(element, "Annotation", "name", "role", "description");
        var attributes = Children(element, "Attribute").ToDictionary(attribute =>
        {
            Check(attribute, "Attribute", "name", "value");
            return Required(attribute, "name");
        }, attribute => Required(attribute, "value"), StringComparer.Ordinal);
        return new GraphAnnotation(
            (string?)element.Attribute("name"),
            (string?)element.Attribute("role"),
            (string?)element.Attribute("description"),
            new System.Collections.ObjectModel.ReadOnlyDictionary<string, string>(attributes));
    }

    private static TensorDescriptor ReadTensor(XElement element)
    {
        Check(element, "Tensor", "elementType", "layout");
        var dimensions = Children(element, "Dimension").Select(dimension =>
        {
            Check(dimension, "Dimension", "value");
            return Number(dimension, "value");
        }).ToArray();
        return new TensorDescriptor(
            DefinitionEnumValue<GraphElementType>(element, "elementType"),
            dimensions,
            Required(element, "layout"));
    }

    private static T DefinitionEnumValue<T>(XElement element, string attribute) where T : struct, Enum
    {
        var value = Required(element, attribute);
        if (!Enum.TryParse<T>(value, false, out var result) || !Enum.IsDefined(result))
            throw new InvalidDataException($"Invalid {attribute} '{value}'.");
        return result;
    }

    private static XElement RequiredChild(XElement parent, string name)
    {
        var values = parent.Elements(name).ToArray();
        if (values.Length != 1)
            throw new InvalidDataException($"Element <{parent.Name}> requires exactly one <{name}> child.");
        return values[0];
    }

    private static IEnumerable<XElement> DefinitionChildren(XElement parent, params string[] names)
    {
        if (parent.Elements().Any(element =>
            !names.Contains(element.Name.LocalName, StringComparer.Ordinal) ||
            element.Name.Namespace != XNamespace.None))
            throw new InvalidDataException($"Unexpected children in <{parent.Name}>.");
        return parent.Elements();
    }

    private static string Write(XElement root)
    {
        var xml = root.ToString(SaveOptions.DisableFormatting);
        if (xml.Length > MaxCharacters ||
            root.DescendantsAndSelf().Take(MaxElements + 1).Count() > MaxElements)
            throw new InvalidDataException("Graph XML exceeds the size limit.");
        return xml;
    }

    private static XElement Read(string xml, string kind)
    {
        ArgumentNullException.ThrowIfNull(xml);
        if (xml.Length > MaxCharacters)
            throw new InvalidDataException("Graph XML exceeds the size limit.");
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
            var root = document.Root ?? throw new InvalidDataException("Graph XML has no root.");
            if (root.DescendantsAndSelf().Take(MaxElements + 1).Count() > MaxElements ||
                root.DescendantsAndSelf().Any(element =>
                    element.Ancestors().Take(MaxDepth + 1).Count() > MaxDepth))
                throw new InvalidDataException("Graph XML exceeds the structural limit.");
            Check(root, "Graph", "kind", "version");
            if (Required(root, "kind") != kind || Required(root, "version") != "1")
                throw new InvalidDataException($"Expected {kind} XML schema version 1.");
            return root;
        }
        catch (XmlException exception)
        {
            throw new InvalidDataException("Invalid graph XML.", exception);
        }
    }

    private static IEnumerable<XElement> Children(XElement parent, string name)
    {
        if (parent.Elements().Any(element =>
            element.Name != name || element.Name.Namespace != XNamespace.None))
            throw new InvalidDataException($"Unexpected children in <{parent.Name}>.");
        return parent.Elements();
    }

    private static void Check(XElement element, string name, params string[] attributes)
    {
        if (element.Name != name ||
            element.Attributes().Any(attribute =>
                attribute.IsNamespaceDeclaration ||
                attribute.Name.Namespace != XNamespace.None ||
                !attributes.Contains(attribute.Name.LocalName, StringComparer.Ordinal)) ||
            element.Nodes().Any(node =>
                node is XText text && !string.IsNullOrWhiteSpace(text.Value) ||
                node is not XElement && node is not XText))
            throw new InvalidDataException($"Invalid <{name}> element.");
    }

    private static string Required(XElement element, string attribute)
    {
        var value = (string?)element.Attribute(attribute);
        return string.IsNullOrWhiteSpace(value)
            ? throw new InvalidDataException($"Element <{element.Name}> requires '{attribute}'.")
            : value;
    }

    private static int Number(XElement element, string attribute)
    {
        var value = Required(element, attribute);
        return int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var result)
            ? result
            : throw new InvalidDataException($"Invalid integer '{value}'.");
    }

    private static XAttribute? Optional(string name, string? value) =>
        value is null ? null : new XAttribute(name, value);

    private sealed record DefinitionRecord(
        string Id,
        GraphCompileBoundary Boundary,
        GraphAnnotation? Annotation,
        IReadOnlyList<GraphDefinitionResource> Resources,
        IReadOnlyList<GraphDefinitionNode> Nodes);
}
