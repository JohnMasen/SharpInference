using System.Collections.ObjectModel;
using System.Xml.Linq;
using SharpInference.Graphs.Core;

namespace SharpInference.Graphs;

public static partial class GraphXml
{
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

    private static XElement WriteSourceMap(ExecutionSourceMap source) =>
        new(
            "SourceMap",
            Optional("transformation", source.Transformation),
            source.LogicalNodes.Select(node =>
                new XElement(
                    "LogicalNode", new XAttribute("id", node.Value))));

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
