using System.Xml.Linq;
using SharpInference.Graphs.Core;
using static SharpInference.Graphs.Internal.GraphXmlSupport;
using SharpInference.Graphs.Logical;

namespace SharpInference.Graphs.Execution;

/// <summary>Reads and writes version-2 execution graph programs with explicit typed nodes and edges.</summary>
/// <remarks>Serialization sorts graph definitions by ordinal graph identity while retaining each graph's
/// node and edge declaration order. Deserialization reconstructs and validates graphs and call relationships,
/// then checks the declared root against the inferred root; it does not perform compilation or inlining.</remarks>
public static class ExecutionGraphXml
{
    /// <summary>Serializes a complete execution graph program.</summary>
    /// <remarks>Writes an unformatted Graph document with kind ExecutionGraphProgram and version 2.
    /// Graphs are sorted by ordinal identity; source maps, boundary values, and optional metadata are emitted.
    /// Output character and element counts are limited; the writer does not enforce the reader's depth limit.</remarks>
    /// <param name="program">The program and its graph definitions.</param>
    /// <returns>The versioned execution graph XML document.</returns>
    /// <exception cref="ArgumentNullException">The program is null.</exception>
    /// <exception cref="InvalidDataException">The serialized structure exceeds XML size limits or contains an unsupported node or edge type.</exception>
    public static string Serialize(ExecutionGraphProgram program)
    {
        ArgumentNullException.ThrowIfNull(program);
        return Write(new XElement(
            "Graph",
            new XAttribute("kind", "ExecutionGraphProgram"),
            new XAttribute("version", 2),
            new XElement("Root", new XAttribute("graph", program.RootGraph.Value)),
            new XElement(
                "Graphs",
                program.Graphs.OrderBy(
                        graph => graph.Id.Value, StringComparer.Ordinal)
                    .Select(WriteExecutionGraph))));
    }

    /// <summary>Reads and validates a versioned execution graph program.</summary>
    /// <remarks>Requires kind ExecutionGraphProgram and version 2, prohibits DTDs, disables external resolution,
    /// and enforces character, element-count, and depth limits. Defined enum values are parsed case-sensitively.
    /// Node and edge containers reject unsupported element names. Optional resource scopes, architecture,
    /// and logical provenance are reconstructed; no call-interface compatibility check is added.</remarks>
    /// <param name="xml">The serialized execution graph program.</param>
    /// <returns>The reconstructed execution graph program.</returns>
    /// <exception cref="ArgumentNullException">The XML is null.</exception>
    /// <exception cref="InvalidDataException">The XML or declared root is invalid.</exception>
    /// <exception cref="ArgumentException">A reconstructed identifier or value is invalid, the program is empty,
    /// or an attribute or binding dictionary contains duplicate keys.</exception>
    /// <exception cref="KeyNotFoundException">A reconstructed scoped resource owner is missing.</exception>
    public static ExecutionGraphProgram DeserializeProgram(string xml)
    {
        var root = Read(xml, "ExecutionGraphProgram", 2);
        var rootElement = RequiredChild(root, "Root");
        Check(rootElement, "Root", "graph");
        var rootId = new ExecutionGraphId(Required(rootElement, "graph"));
        var result = new ExecutionGraphProgram(
            Children(RequiredChild(root, "Graphs"), "ExecutionGraph")
                .Select(ReadExecutionGraph));
        if (result.RootGraph != rootId)
        {
            throw new InvalidDataException(
                $"Declared execution root '{rootId}' does not match inferred root " +
                $"'{result.RootGraph}'.");
        }

        return result;
    }

    /// <summary>Writes one graph with metadata and nodes and edges in structural declaration order.</summary>
    /// <param name="graph">The graph definition to write.</param>
    /// <returns>The ExecutionGraph element.</returns>
    /// <exception cref="InvalidDataException">A node or edge type cannot be serialized.</exception>
    private static XElement WriteExecutionGraph(ExecutionGraph graph) =>
        new(
            "ExecutionGraph",
            new XAttribute("id", graph.Id.Value),
            new XAttribute("compileBoundary", graph.DefaultCompileBoundary),
            graph.Source is null
                ? null
                : new XElement(
                    "Source",
                    new XAttribute("definition", graph.Source.Definition.Value)),
            WriteArchitecture(graph.Architecture),
            new XElement(
                "Nodes", graph.Structure.Nodes.Select(WriteExecutionNode)),
            new XElement(
                "Edges", graph.Structure.Edges.Select(WriteExecutionEdge)));

    /// <summary>Writes a resource, call, or operation using its concrete execution node type.</summary>
    /// <param name="node">The node whose resource scope or executable metadata is written.</param>
    /// <returns>The Resource, GraphCall, or Operation element.</returns>
    /// <exception cref="InvalidDataException">The execution node type is unsupported.</exception>
    private static XElement WriteExecutionNode(
        ExecutionGraphStructureNode node) => node switch
        {
            ExecutionResourceNode resource => new XElement(
                "Resource",
                new XAttribute("id", resource.Resource.Id.Value),
                new XAttribute("name", resource.Resource.Name),
                new XAttribute("kind", resource.Resource.Kind),
                resource.ResourceNodeId.Owner.IsParent
                    ? null
                    : new XAttribute(
                        "instance", resource.ResourceNodeId.Owner.Call!.Value.Value),
                Optional("bindingKey", resource.Resource.BindingKey),
                WriteTensor(resource.Resource.Tensor)),
            ExecutionGraphCallNode call => new XElement(
                "GraphCall",
                new XAttribute("id", call.Id.Value),
                new XAttribute("target", call.Target.Value),
                new XAttribute("compileBoundary", call.CompileBoundary),
                WriteArchitecture(call.Architecture),
                WriteSourceMap(call.Source)),
            ExecutionOperationNode operation => new XElement(
                "Operation",
                new XAttribute("id", operation.Id.Value),
                new XAttribute("operation", operation.Operation.Name),
                new XAttribute("operationVersion", operation.Operation.Version),
                WriteArchitecture(operation.Architecture),
                WriteSourceMap(operation.Source),
                WriteAttributes(operation.Attributes),
                WritePrecision(operation.Requirements)),
            _ => throw new InvalidDataException(
                $"Unsupported execution node type '{node.GetType().Name}'."),
        };

    /// <summary>Writes a dependency or resource-access edge with its structural endpoint identities.</summary>
    /// <param name="edge">The typed edge to serialize.</param>
    /// <returns>The Dependency, Read, or Write element, including binding metadata for access edges.</returns>
    /// <exception cref="InvalidDataException">The execution edge type is unsupported.</exception>
    private static XElement WriteExecutionEdge(ExecutionGraphEdge edge) => edge switch
    {
        ExecutionDependencyEdge dependency => new XElement(
            "Dependency",
            new XAttribute("id", dependency.Id.Value),
            new XAttribute("source", dependency.Source.Value),
            new XAttribute("target", dependency.Target.Value)),
        ExecutionResourceReadEdge read => new XElement(
            "Read",
            new XAttribute("id", read.Id.Value),
            new XAttribute("source", read.Source.Value),
            new XAttribute("target", read.Target.Value),
            new XAttribute("port", read.Port),
            new XAttribute("binding", read.BindingId)),
        ExecutionResourceWriteEdge write => new XElement(
            "Write",
            new XAttribute("id", write.Id.Value),
            new XAttribute("source", write.Source.Value),
            new XAttribute("target", write.Target.Value),
            new XAttribute("port", write.Port),
            new XAttribute("binding", write.BindingId)),
        _ => throw new InvalidDataException(
            $"Unsupported execution edge type '{edge.GetType().Name}'."),
    };

    /// <summary>Reads one graph definition and constructs its validated typed structure.</summary>
    /// <param name="element">The ExecutionGraph element with required nodes, edges, identity, and boundary.</param>
    /// <returns>The reconstructed graph with optional logical source and architecture.</returns>
    /// <exception cref="InvalidDataException">Required XML structure, enum values, endpoints, or graph validation is invalid.</exception>
    /// <exception cref="ArgumentException">An identifier or binding identity is invalid or duplicated.</exception>
    /// <exception cref="KeyNotFoundException">A scoped resource owner does not identify an existing node.</exception>
    private static ExecutionGraph ReadExecutionGraph(XElement element)
    {
        Check(element, "ExecutionGraph", "id", "compileBoundary");
        var sourceElement = element.Element("Source");
        ExecutionGraphSource? source = null;
        if (sourceElement is not null)
        {
            Check(sourceElement, "Source", "definition");
            source = new(new(Required(sourceElement, "definition")));
        }
        var nodes = DefinitionChildren(
                RequiredChild(element, "Nodes"),
                "Resource", "Operation", "GraphCall")
            .Select(ReadExecutionNode).ToArray();
        var edges = DefinitionChildren(
                RequiredChild(element, "Edges"),
                "Dependency", "Read", "Write")
            .Select(ReadExecutionEdge).ToArray();
        return new(
            new(Required(element, "id")),
            new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(nodes, edges),
            DefinitionEnumValue<GraphCompileBoundary>(
                element, "compileBoundary"),
            ReadArchitecture(element.Element("Architecture")),
            source);
    }

    /// <summary>Reads a resource, graph call, or operation including its type-specific metadata.</summary>
    /// <remarks>A missing or whitespace resource instance attribute denotes parent ownership.
    /// Executable nodes require a source map; operation attributes and precision are also required.</remarks>
    /// <param name="element">An element admitted by the execution graph's node container.</param>
    /// <returns>The corresponding execution structure node.</returns>
    /// <exception cref="InvalidDataException">The node's XML shape, required values, or enum values are invalid.</exception>
    /// <exception cref="ArgumentException">Resource, operation, tensor, or attribute construction rejects a value or duplicate key.</exception>
    private static ExecutionGraphStructureNode ReadExecutionNode(XElement element)
    {
        if (element.Name == "Resource")
        {
            Check(
                element, "Resource", "id", "name", "kind", "instance",
                "bindingKey");
            var resource = new ExecutionGraphResource(
                new(Required(element, "id")),
                Required(element, "name"),
                DefinitionEnumValue<GraphDefinitionResourceKind>(element, "kind"),
                ReadTensor(RequiredChild(element, "Tensor")),
                (string?)element.Attribute("bindingKey"));
            var instance = (string?)element.Attribute("instance");
            var owner = string.IsNullOrWhiteSpace(instance)
                ? GraphBindingOwner.Parent
                : GraphBindingOwner.Instance(new(instance));
            return new ExecutionResourceNode(
                new ModelResourceNodeId(owner, resource.Id), resource);
        }

        var source = ReadSourceMap(RequiredChild(element, "SourceMap"));
        var architecture = ReadArchitecture(element.Element("Architecture"));
        if (element.Name == "GraphCall")
        {
            Check(element, "GraphCall", "id", "target", "compileBoundary");
            return new ExecutionGraphCallNode(
                new(Required(element, "id")),
                new(Required(element, "target")),
                DefinitionEnumValue<GraphCompileBoundary>(
                    element, "compileBoundary"),
                source,
                architecture);
        }

        Check(element, "Operation", "id", "operation", "operationVersion");
        return new ExecutionOperationNode(
            new(Required(element, "id")),
            new(Required(element, "operation"), Number(element, "operationVersion")),
            ReadAttributes(RequiredChild(element, "Attributes")),
            ReadPrecision(RequiredChild(element, "Precision")),
            source,
            architecture);
    }

    /// <summary>Reads a typed edge, requiring port and binding attributes only for resource-access edges.</summary>
    /// <param name="element">A Dependency, Read, or Write element.</param>
    /// <returns>The reconstructed directed edge.</returns>
    /// <exception cref="InvalidDataException">The element type, attributes, or required values are invalid.</exception>
    private static ExecutionGraphEdge ReadExecutionEdge(XElement element)
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
            "Dependency" => new ExecutionDependencyEdge(id, source, target),
            "Read" => new ExecutionResourceReadEdge(
                id, source, target, Required(element, "port"),
                Required(element, "binding")),
            "Write" => new ExecutionResourceWriteEdge(
                id, source, target, Required(element, "port"),
                Required(element, "binding")),
            _ => throw new InvalidDataException(
                $"Unsupported execution edge element <{element.Name}>."),
        };
    }

    /// <summary>Writes optional architecture metadata and its attributes in dictionary enumeration order.</summary>
    /// <param name="architecture">The architecture metadata, or null.</param>
    /// <returns>An Architecture element, or null when metadata is absent.</returns>
    private static XElement? WriteArchitecture(GraphArchitecture? architecture)
    {
        if (architecture is null)
        {
            return null;
        }

        return new XElement(
            "Architecture",
            Optional("name", architecture.Name),
            Optional("role", architecture.Role),
            Optional("description", architecture.Description),
            (architecture.Attributes ?? new Dictionary<string, string>())
                .Select(attribute => new XElement(
                    "Attribute",
                    new XAttribute("name", attribute.Key),
                    new XAttribute("value", attribute.Value))));
    }

    /// <summary>Reads optional architecture fields and an ordinal read-only attribute dictionary.</summary>
    /// <param name="element">The Architecture element, or null.</param>
    /// <returns>The reconstructed metadata, or null when the element is absent.</returns>
    /// <exception cref="InvalidDataException">The element or its attribute entries are malformed.</exception>
    /// <exception cref="ArgumentException">Architecture attribute names repeat.</exception>
    private static GraphArchitecture? ReadArchitecture(XElement? element)
    {
        if (element is null)
        {
            return null;
        }

        Check(element, "Architecture", "name", "role", "description");
        return new(
            (string?)element.Attribute("name"),
            (string?)element.Attribute("role"),
            (string?)element.Attribute("description"),
            ReadAttributes(element));
    }

    /// <summary>Writes logical origin identities in list order and an optional transformation label.</summary>
    /// <param name="source">The execution node's logical provenance.</param>
    /// <returns>The SourceMap element.</returns>
    private static XElement WriteSourceMap(ExecutionSourceMap source) =>
        new(
            "SourceMap",
            Optional("transformation", source.Transformation),
            source.LogicalNodes.Select(node =>
                new XElement(
                    "LogicalNode", new XAttribute("id", node.Value))));

    /// <summary>Reads ordered logical node identities and an optional transformation label.</summary>
    /// <param name="element">The SourceMap element.</param>
    /// <returns>The reconstructed logical provenance, retaining repeated origin identities.</returns>
    /// <exception cref="InvalidDataException">The source map or a logical node entry is malformed.</exception>
    private static ExecutionSourceMap ReadSourceMap(XElement element)
    {
        Check(element, "SourceMap", "transformation");
        return new(
            Children(element, "LogicalNode").Select(node =>
            {
                Check(node, "LogicalNode", "id");
                return new GraphDefinitionNodeId(Required(node, "id"));
            }).ToArray(),
            (string?)element.Attribute("transformation"));
    }
}
