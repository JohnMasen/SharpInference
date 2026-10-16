using System.Xml.Linq;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Internal;
using static SharpInference.Graphs.Internal.GraphXmlSupport;

namespace SharpInference.Graphs.Logical;

/// <summary>Reads and writes version-three logical definitions with named bindings.</summary>
public static class LogicalGraphXml
{

    /// <summary>Serializes a logical graph and its reachable definitions.</summary>
    /// <param name="graph">The root definition to serialize.</param>
    /// <returns>The versioned logical graph XML document.</returns>
    /// <remarks>
    /// Uses schema version 3, sorts definitions by ordinal identifier, and preserves each structure's
    /// node and edge order. Traversal follows the supplied child-definition collections and deduplicates
    /// by definition identity. Parent versus instance resource scopes are encoded separately from
    /// declaration graph owners. Serialization enforces character and element-count limits.
    /// </remarks>
    /// <exception cref="ArgumentNullException">The root definition is null.</exception>
    /// <exception cref="InvalidDataException">The definitions are recursive, exceed XML limits, or contain unsupported logical node or edge types.</exception>
    public static string Serialize(LogicalGraphDefinition graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var definitions = CollectDefinitions(graph);
        return Write(new XElement(
            "Graph",
            new XAttribute("kind", "LogicalGraphDefinition"),
            new XAttribute("version", 3),
            new XElement("Root", new XAttribute("definition", graph.Id.Value)),
            new XElement(
                "Definitions",
                definitions.Values
                    .OrderBy(definition => definition.Id.Value, StringComparer.Ordinal)
                    .Select(WriteDefinition))));
    }

    /// <summary>Reads and validates a versioned logical graph XML document.</summary>
    /// <param name="xml">The serialized root and reachable definitions.</param>
    /// <returns>The reconstructed root logical graph.</returns>
    /// <remarks>
    /// Requires the version-3 logical graph envelope and exactly one Root and Definitions container.
    /// DTDs are prohibited; character, element-count, and nesting limits are enforced during parsing.
    /// Definition records are resolved recursively from call nodes, rejecting missing, recursive, and
    /// unreachable definitions. Each reconstructed definition runs normal logical validation.
    /// Unknown attributes and unexpected node or edge element names are rejected at checked locations;
    /// this is not a blanket rejection of all extra elements. Only the initial XML-reader failures are
    /// normalized by the shared reader, so later duplicate-key and descriptor errors may propagate.
    /// </remarks>
    /// <exception cref="ArgumentNullException">The XML is null.</exception>
    /// <exception cref="InvalidDataException">The XML or graph relationships are invalid.</exception>
    /// <exception cref="ArgumentException">Parsed definition or attribute keys are duplicated, or a parsed descriptor rejects its values.</exception>
    /// <exception cref="InvalidOperationException">Malformed graph relationships violate a required uniqueness assumption.</exception>
    public static LogicalGraphDefinition DeserializeDefinition(string xml)
    {
        var root = Read(xml, "LogicalGraphDefinition", 3);
        _ = DefinitionChildren(root, "Root", "Definitions");
        var rootElement = RequiredChild(root, "Root");
        Check(rootElement, "Root", "definition");
        var rootId = Required(rootElement, "definition");
        var records = new Dictionary<string, LogicalGraphDefinitionRecord>(StringComparer.Ordinal);
        foreach (var element in Children(RequiredChild(root, "Definitions"), "Definition"))
        {
            var record = ReadDefinitionRecord(element);
            if (!records.TryAdd(record.Id, record))
                throw new InvalidDataException($"Duplicate logical definition '{record.Id}'.");
        }
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

            var children = record.Structure.Filter(node: node => node is GraphCallNode)
                .Nodes.Cast<GraphCallNode>()
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

    /// <summary>Traverses declared child definitions and collects one entry per identity.</summary>
    /// <param name="root">The root of the definition traversal.</param>
    /// <returns>The identity-indexed definitions to serialize.</returns>
    /// <exception cref="InvalidDataException">An active definition identity is encountered recursively.</exception>
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

    /// <summary>Writes one definition with its structure's existing node and edge order.</summary>
    /// <param name="definition">The definition to encode.</param>
    /// <returns>A Definition element containing metadata, nodes, and edges.</returns>
    private static XElement WriteDefinition(LogicalGraphDefinition definition) =>
        new(
            "Definition",
            new XAttribute("id", definition.Id.Value),
            new XAttribute("compileBoundary", definition.CompileBoundary),
            WriteAnnotation(definition.Annotation),
            new XElement("Nodes", definition.Structure.Nodes.Select(WriteLogicalNode)),
            ModelGraphXml.WriteRelations(definition.Structure));

    /// <summary>Writes a scoped resource, child call, or operation node.</summary>
    /// <param name="node">The logical node to encode.</param>
    /// <returns>The corresponding Resource, Call, or Operation element.</returns>
    /// <remarks>Resource graph owners are always written; an instance attribute is emitted only for non-parent scopes.</remarks>
    /// <exception cref="InvalidDataException">The node has an unsupported runtime type.</exception>
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

    /// <summary>Parses one definition's metadata and exact structure before resolving child definitions.</summary>
    /// <param name="element">The Definition element with Nodes, Bindings, and Dependencies containers.</param>
    /// <returns>The parsed record used during recursive reconstruction.</returns>
    /// <exception cref="InvalidDataException">Required attributes or containers, enum values, child node names, or graph structure are invalid.</exception>
    private static LogicalGraphDefinitionRecord ReadDefinitionRecord(XElement element)
    {
        Check(element, "Definition", "id", "compileBoundary");
        _ = DefinitionChildren(element, "Annotation", "Nodes", "Bindings", "Dependencies");
        var nodes = DefinitionChildren(
                RequiredChild(element, "Nodes"), "Resource", "Operation", "Call")
            .Select(ReadLogicalNode).ToArray();
        var edges = ModelGraphXml.ReadRelations<LogicalGraphEdge>(element,
            (id, source, target, port, binding) => new LogicalResourceReadEdge(id, source, target, port, binding),
            (id, source, target, port, binding) => new LogicalResourceWriteEdge(id, source, target, port, binding),
            (id, source, target) => new LogicalDependencyEdge(id, source, target));
        ModelGraphConstruction.ThrowIfErrors(ModelGraphConstruction.Validate(
            Required(element, "id"), nodes, edges, node => node is GraphCallNode));
        return new(
            Required(element, "id"),
            DefinitionEnumValue<GraphCompileBoundary>(element, "compileBoundary"),
            ReadAnnotation(element.Element("Annotation")),
            new Graph<LogicalGraphNode, LogicalGraphEdge>(nodes, edges));
    }

    /// <summary>Reconstructs one resource, call, or operation from its serialized attributes and children.</summary>
    /// <param name="element">The logical node element.</param>
    /// <returns>The parsed logical node without inferring any additional relationships.</returns>
    /// <remarks>A missing or whitespace instance attribute selects parent scope; a nonblank value selects that call's scope.</remarks>
    /// <exception cref="InvalidDataException">The element, attributes, required children, enum values, or numbers are invalid.</exception>
    /// <exception cref="ArgumentException">Attribute keys are duplicated or tensor descriptor values are invalid.</exception>
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

    /// <summary>Writes optional metadata and its attributes directly as Annotation children.</summary>
    /// <param name="annotation">The metadata to encode, or null.</param>
    /// <returns>An Annotation element, or null when metadata is absent.</returns>
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

    /// <summary>Reads optional Annotation metadata and its direct Attribute children.</summary>
    /// <param name="element">The Annotation element, or null.</param>
    /// <returns>The parsed metadata, or null when the element is absent.</returns>
    /// <exception cref="InvalidDataException">The element, allowed attributes, or Attribute children are invalid.</exception>
    /// <exception cref="ArgumentException">An attribute name is duplicated.</exception>
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
}
