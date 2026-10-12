using System.Collections.ObjectModel;
using System.Xml.Linq;

namespace SharpInference.Graphs;

public static partial class GraphXml
{
    /// <summary>Serializes a validated execution graph program.</summary>
    public static string Serialize(ExecutionGraphProgram program)
    {
        ArgumentNullException.ThrowIfNull(program);
        return Write(new XElement(
            "Graph",
            new XAttribute("kind", "ExecutionGraphProgram"),
            new XAttribute("version", 1),
            new XElement("Root", new XAttribute("graph", program.RootGraph.Value)),
            new XElement("Graphs", program.Graphs
                .OrderBy(graph => graph.Id.Value, StringComparer.Ordinal)
                .Select(WriteExecutionGraph))));
    }

    /// <summary>Deserializes and validates an execution graph program.</summary>
    public static ExecutionGraphProgram DeserializeProgram(string xml)
    {
        var root = Read(xml, "ExecutionGraphProgram");
        Check(root, "Graph", "kind", "version");
        var rootElement = RequiredChild(root, "Root");
        Check(rootElement, "Root", "graph");
        var rootId = new ExecutionGraphId(Required(rootElement, "graph"));
        var graphs = Children(RequiredChild(root, "Graphs"), "ExecutionGraph")
            .Select(ReadExecutionGraph).ToArray();
        var result = new ExecutionGraphProgram(graphs);
        if (result.RootGraph != rootId)
            throw new InvalidDataException(
                $"Declared execution root '{rootId}' does not match inferred root '{result.RootGraph}'.");
        return result;
    }

    private static XElement WriteExecutionGraph(ExecutionGraph graph) =>
        new(
            "ExecutionGraph",
            new XAttribute("id", graph.Id.Value),
            new XAttribute("compileBoundary", graph.DefaultCompileBoundary),
            graph.Source is null
                ? null
                : new XElement("Source", new XAttribute("definition", graph.Source.Definition.Value)),
            WriteArchitecture(graph.Architecture),
            new XElement("Resources", graph.Resources.Select(WriteExecutionResource)),
            new XElement("Nodes", graph.Nodes.Select(WriteExecutionNode)));

    private static XElement WriteExecutionResource(ExecutionGraphResource resource) =>
        new(
            "Resource",
            new XAttribute("id", resource.Id.Value),
            new XAttribute("name", resource.Name),
            new XAttribute("kind", resource.Kind),
            Optional("bindingKey", resource.BindingKey),
            WriteTensor(resource.Tensor));

    private static XElement WriteExecutionNode(ExecutionGraphNode node)
    {
        var element = new XElement(
            node is ExecutionGraphCallNode ? "GraphCall" : "Operation",
            new XAttribute("id", node.Id.Value));
        if (node is ExecutionGraphCallNode call)
        {
            element.Add(
                new XAttribute("target", call.Target.Value),
                new XAttribute("compileBoundary", call.CompileBoundary));
        }
        else
        {
            var operation = (ExecutionOperationNode)node;
            element.Add(
                new XAttribute("operation", operation.Operation.Name),
                new XAttribute("operationVersion", operation.Operation.Version));
        }
        element.Add(
            WriteArchitecture(node.Architecture),
            WriteSourceMap(node.Source),
            new XElement("Bindings", node.Bindings.Select(binding =>
                new XElement(
                    "Binding",
                    new XAttribute("port", binding.Port),
                    new XAttribute("resource", binding.Resource.Value),
                    new XAttribute("access", binding.Access),
                    binding.ResourceOwner.IsParent
                        ? null
                        : new XAttribute("instance", binding.ResourceOwner.Call!.Value.Value)))),
            new XElement("Dependencies", node.Dependencies.Select(dependency =>
                new XElement("NodeRef", new XAttribute("id", dependency.Value)))));
        if (node is ExecutionOperationNode operationNode)
        {
            element.Add(
                new XElement("Attributes", operationNode.Attributes.Select(attribute =>
                    new XElement(
                        "Attribute",
                        new XAttribute("name", attribute.Key),
                        new XAttribute("value", attribute.Value)))),
                new XElement(
                    "Precision",
                    new XAttribute("arithmetic", operationNode.Requirements.MinimumArithmeticType),
                    new XAttribute("accumulator", operationNode.Requirements.MinimumAccumulatorType)));
        }
        return element;
    }

    private static XElement? WriteArchitecture(GraphArchitecture? architecture)
    {
        if (architecture is null)
            return null;
        return new XElement(
            "Architecture",
            Optional("name", architecture.Name),
            Optional("role", architecture.Role),
            Optional("description", architecture.Description),
            (architecture.Attributes ?? new Dictionary<string, string>()).Select(attribute =>
                new XElement(
                    "Attribute",
                    new XAttribute("name", attribute.Key),
                    new XAttribute("value", attribute.Value))));
    }

    private static XElement WriteSourceMap(ExecutionSourceMap source) =>
        new(
            "SourceMap",
            Optional("transformation", source.Transformation),
            source.LogicalNodes.Select(node =>
                new XElement("LogicalNode", new XAttribute("id", node.Value))));

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
        return new ExecutionGraph(
            new(Required(element, "id")),
            Children(RequiredChild(element, "Resources"), "Resource")
                .Select(ReadExecutionResource),
            DefinitionChildren(RequiredChild(element, "Nodes"), "Operation", "GraphCall")
                .Select(ReadExecutionNode),
            DefinitionEnumValue<GraphCompileBoundary>(element, "compileBoundary"),
            ReadArchitecture(element.Element("Architecture")),
            source);
    }

    private static ExecutionGraphResource ReadExecutionResource(XElement element)
    {
        Check(element, "Resource", "id", "name", "kind", "bindingKey");
        return new(
            new(Required(element, "id")),
            Required(element, "name"),
            DefinitionEnumValue<GraphDefinitionResourceKind>(element, "kind"),
            ReadTensor(RequiredChild(element, "Tensor")),
            (string?)element.Attribute("bindingKey"));
    }

    private static ExecutionGraphNode ReadExecutionNode(XElement element)
    {
        var isCall = element.Name == "GraphCall";
        Check(
            element,
            isCall ? "GraphCall" : "Operation",
            isCall
                ? ["id", "target", "compileBoundary"]
                : ["id", "operation", "operationVersion"]);
        var id = new ExecutionNodeId(Required(element, "id"));
        var architecture = ReadArchitecture(element.Element("Architecture"));
        var source = ReadSourceMap(RequiredChild(element, "SourceMap"));
        var bindings = Children(RequiredChild(element, "Bindings"), "Binding").Select(binding =>
        {
            Check(binding, "Binding", "port", "resource", "access", "instance");
            var instance = (string?)binding.Attribute("instance");
            return new ExecutionGraphBinding(
                Required(binding, "port"),
                new(Required(binding, "resource")),
                DefinitionEnumValue<GraphResourceAccess>(binding, "access"),
                string.IsNullOrWhiteSpace(instance)
                    ? GraphBindingOwner.Parent
                    : GraphBindingOwner.Instance(new(instance)));
        }).ToArray();
        var dependencies = Children(RequiredChild(element, "Dependencies"), "NodeRef")
            .Select(reference =>
            {
                Check(reference, "NodeRef", "id");
                return new ExecutionNodeId(Required(reference, "id"));
            }).ToArray();
        if (isCall)
            return new ExecutionGraphCallNode(
                id,
                new(Required(element, "target")),
                bindings,
                dependencies,
                DefinitionEnumValue<GraphCompileBoundary>(element, "compileBoundary"),
                source,
                architecture);
        var attributes = Children(RequiredChild(element, "Attributes"), "Attribute")
            .ToDictionary(attribute =>
            {
                Check(attribute, "Attribute", "name", "value");
                return Required(attribute, "name");
            }, attribute => Required(attribute, "value"), StringComparer.Ordinal);
        var precision = RequiredChild(element, "Precision");
        Check(precision, "Precision", "arithmetic", "accumulator");
        return new ExecutionOperationNode(
            id,
            new(Required(element, "operation"), Number(element, "operationVersion")),
            bindings,
            dependencies,
            new ReadOnlyDictionary<string, string>(attributes),
            new(
                DefinitionEnumValue<GraphElementType>(precision, "arithmetic"),
                DefinitionEnumValue<GraphElementType>(precision, "accumulator")),
            source,
            architecture);
    }

    private static GraphArchitecture? ReadArchitecture(XElement? element)
    {
        if (element is null)
            return null;
        Check(element, "Architecture", "name", "role", "description");
        var attributes = Children(element, "Attribute").ToDictionary(attribute =>
        {
            Check(attribute, "Attribute", "name", "value");
            return Required(attribute, "name");
        }, attribute => Required(attribute, "value"), StringComparer.Ordinal);
        return new(
            (string?)element.Attribute("name"),
            (string?)element.Attribute("role"),
            (string?)element.Attribute("description"),
            new ReadOnlyDictionary<string, string>(attributes));
    }

    private static ExecutionSourceMap ReadSourceMap(XElement element)
    {
        Check(element, "SourceMap", "transformation");
        return new(
            Children(element, "LogicalNode")
                .Select(node =>
                {
                    Check(node, "LogicalNode", "id");
                    return new GraphDefinitionNodeId(Required(node, "id"));
                }).ToArray(),
            (string?)element.Attribute("transformation"));
    }
}
