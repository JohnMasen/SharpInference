using System.Xml.Linq;
using SharpInference.Graphs.Internal;

namespace SharpInference.Graphs;

/// <summary>The version-one XML entry point for business graph packages.</summary>
public static class GraphPackageXml
{
    public static string Serialize(GraphPackage<Graph> package) =>
        new BusinessSerializer<Graph>().Serialize(package);

    /// <summary>Reads a business package; only <see cref="Graph"/> is supported as the graph type.</summary>
    /// <exception cref="NotSupportedException">The graph type is derived from Graph.</exception>
    public static GraphPackage<TGraph> Deserialize<TGraph>(string xml) where TGraph : Graph, new()
    {
        if (typeof(TGraph) != typeof(Graph))
        {
            throw new NotSupportedException("The business XML entry point supports only Graph. Use a specialized serializer for derived graph types.");
        }

        return new BusinessSerializer<TGraph>().Deserialize(xml);
    }

    private sealed class BusinessSerializer<TGraph> : GraphPackageXmlSerializer<TGraph, GraphPackage<TGraph>>
        where TGraph : Graph, new();
}

/// <summary>
/// Implements the version-one common graph schema. Derived serializers declare their stage's
/// additional attributes and children, then read and write those fields through the protected hooks.
/// </summary>
public class GraphPackageXmlSerializer<TGraph, TPackage>
    where TGraph : Graph, new()
    where TPackage : GraphPackage<TGraph>, new()
{
    protected virtual string Stage => "business";
    protected virtual IEnumerable<string> AdditionalAttributes(string elementName) => [];
    protected virtual IEnumerable<string> AdditionalChildren(string elementName) => [];
    protected virtual bool IncludeMetadata(string key) => true;
    protected virtual void WritePackageFields(TPackage package, XElement element) { }
    protected virtual void ReadPackageFields(TPackage package, XElement element) { }
    protected virtual void WriteGraphFields(TGraph graph, XElement element) { }
    protected virtual void ReadGraphFields(TGraph graph, XElement element) { }
    protected virtual void WriteResourceFields(TGraph graph, Resource resource, XElement element) { }
    protected virtual void ReadResourceFields(TGraph graph, Resource resource, XElement element) { }
    protected virtual void WriteActionFields(GraphAction action, XElement element) { }
    protected virtual void ReadActionFields(GraphAction action, XElement element) { }

    public string Serialize(TPackage package)
    {
        ArgumentNullException.ThrowIfNull(package);
        var root = new XElement("GraphPackage", new XAttribute("version", 1),
            new XAttribute("stage", Stage), Optional("main", package.MainGraphId));
        WritePackageFields(package, root);
        root.Add(WriteMetadata(package.Metadata.Where(pair => IncludeMetadata(pair.Key))
                .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal)),
            new XElement("Graphs", package.Select(WriteGraph)));
        return GraphXmlSupport.Write(root);
    }

    public TPackage Deserialize(string xml)
    {
        var root = GraphXmlSupport.ReadDocument(xml);
        CheckFields(root, "GraphPackage", "version", "stage", "main");
        if (Number(root, "version") != 1 || Required(root, "stage") != Stage)
        {
            throw new InvalidDataException("Unsupported GraphPackage version or stage.");
        }

        CheckChildren(root, "Metadata", "Graphs");
        var package = new TPackage { MainGraphId = (string?)root.Attribute("main") };
        ReadMetadata(root, package.Metadata);
        ReadPackageFields(package, root);
        var graphs = RequiredChild(root, "Graphs");
        Check(graphs, "Graphs");
        foreach (var element in Children(graphs, "Graph"))
        {
            package.Add(ReadGraph(element));
        }

        return package;
    }

    private XElement WriteGraph(TGraph graph)
    {
        var element = new XElement("Graph", new XAttribute("id", graph.Id),
            Optional("name", graph.Name), Optional("description", graph.Description));
        WriteGraphFields(graph, element);
        element.Add(WriteMetadata(graph.Metadata),
            new XElement("Resources", graph.Resources.Select(resource =>
            {
                var value = new XElement("Resource", new XAttribute("id", resource.Id),
                    new XAttribute("category", resource.Category),
                    resource.GraphIO != GraphIO.Internal ? new XAttribute("graphIO", resource.GraphIO) : null,
                    resource.Scope != ResourceScope.Call ? new XAttribute("scope", resource.Scope) : null,
                    resource.Usage is { } usage ? new XAttribute("usage", usage) : null,
                    resource.IsState ? new XAttribute("isState", true) : null,
                    resource.Permission != ResourcePermission.ReadWrite ? new XAttribute("permission", resource.Permission) : null,
                    Optional("bindingKey", resource.BindingKey), GraphXmlSupport.WriteTensor(resource.Tensor));
                WriteResourceFields(graph, resource, value);
                return value;
            })),
            new XElement("Actions", graph.Actions.Select(WriteAction)),
            new XElement("GraphFlow", graph.GraphFlow.Select(flow =>
                new XElement("Then", new XAttribute("fromGraph", flow.Source.Graph),
                    new XAttribute("from", flow.Source.Action), new XAttribute("toGraph", flow.Target.Graph),
                    new XAttribute("to", flow.Target.Action)))));
        return element;
    }

    private XElement WriteAction(GraphAction action)
    {
        var element = action switch
        {
            OperationAction operation => new XElement("Operation",
                new XAttribute("operation", operation.Operation.Name),
                new XAttribute("version", operation.Operation.Version),
                GraphXmlSupport.WriteAttributes(operation.Attributes),
                operation.Precision is { } precision ? GraphXmlSupport.WritePrecision(precision) : null),
            CallAction call => new XElement("Call", new XAttribute("target", call.Target)),
            _ => throw new NotSupportedException($"Cannot serialize action type '{action.GetType().Name}'."),
        };
        element.Add(new XAttribute("id", action.Id), new XAttribute("category", action.Category),
            WriteMetadata(action.Metadata),
            action.Bindings.Select(binding =>
                new XElement("Binding", new XAttribute("property", binding.SourceProperty),
                    new XAttribute("targetKind", binding.Target.TargetKind),
                    Optional("target", binding.Target.Target),
                    new XAttribute("targetProperty", binding.Target.TargetProperty),
                    new XAttribute("access", binding.Access))));
        WriteActionFields(action, element);
        return element;
    }

    private TGraph ReadGraph(XElement element)
    {
        CheckFields(element, "Graph", "id", "name", "description");
        CheckChildren(element, "Metadata", "Resources", "Actions", "GraphFlow");
        var graph = new TGraph
        {
            Id = Required(element, "id"),
            Name = (string?)element.Attribute("name"),
            Description = (string?)element.Attribute("description"),
        };
        ReadGraphFields(graph, element);
        ReadMetadata(element, graph.Metadata);
        var resources = RequiredChild(element, "Resources");
        Check(resources, "Resources");
        foreach (var value in Children(resources, "Resource"))
        {
            CheckFields(value, "Resource", "id", "category", "graphIO", "scope", "usage", "isState", "permission", "bindingKey");
            CheckChildren(value, "Tensor");
            var resource = graph.Resource(Required(value, "id"), GraphXmlSupport.ReadTensor(RequiredChild(value, "Tensor")),
                EnumOr(value, "graphIO", GraphIO.Internal), EnumOr(value, "scope", ResourceScope.Call),
                value.Attribute("usage") is null ? null : DefinitionEnumValue<ResourceUsage>(value, "usage"),
                BoolOr(value, "isState"), EnumOr(value, "permission", ResourcePermission.ReadWrite),
                (string?)value.Attribute("bindingKey"));
            resource.Category = (string?)value.Attribute("category") ?? "resource";
            ReadResourceFields(graph, resource, value);
        }

        var actions = RequiredChild(element, "Actions");
        Check(actions, "Actions");
        foreach (var value in DefinitionChildren(actions, "Operation", "Call"))
        {
            GraphAction action;
            if (value.Name == "Operation")
            {
                CheckFields(value, "Operation", "id", "category", "operation", "version");
                CheckChildren(value, "Attributes", "Precision", "Metadata", "Binding");
                var operation = graph.Operation(Required(value, "id"),
                    new OperationId(Required(value, "operation"), Number(value, "version")));
                foreach (var pair in GraphXmlSupport.ReadAttributes(RequiredChild(value, "Attributes")))
                {
                    operation.Attributes.Add(pair.Key, pair.Value);
                }

                var precision = OptionalChild(value, "Precision");
                operation.Precision = precision is null ? null : GraphXmlSupport.ReadPrecision(precision);
                action = operation;
            }
            else
            {
                CheckFields(value, "Call", "id", "category", "target");
                CheckChildren(value, "Metadata", "Binding");
                action = graph.Call(Required(value, "id"), Required(value, "target"));
            }

            action.Category = (string?)value.Attribute("category") ?? (action is OperationAction ? "operation" : "call");
            ReadMetadata(value, action.Metadata);
            ReadActionFields(action, value);
            foreach (var binding in value.Elements("Binding"))
            {
                Check(binding, "Binding", "property", "targetKind", "target", "targetProperty", "access");
                DefinitionChildren(binding);
                action.Bind(Required(binding, "property"),
                    new ResourceRef(DefinitionEnumValue<ResourceTarget>(binding, "targetKind"),
                        (string?)binding.Attribute("target"), Required(binding, "targetProperty")),
                    DefinitionEnumValue<ResourceAccess>(binding, "access"));
            }
        }

        var flows = RequiredChild(element, "GraphFlow");
        Check(flows, "GraphFlow");
        foreach (var flow in Children(flows, "Then"))
        {
            Check(flow, "Then", "fromGraph", "from", "toGraph", "to");
            DefinitionChildren(flow);
            graph.GraphFlow.Add(new(
                new((string?)flow.Attribute("fromGraph") ?? graph.Id, Required(flow, "from")),
                new((string?)flow.Attribute("toGraph") ?? graph.Id, Required(flow, "to"))));
        }

        return graph;
    }

    private void CheckFields(XElement element, string name, params string[] attributes) =>
        Check(element, name, attributes.Concat(AdditionalAttributes(name)).ToArray());

    private void CheckChildren(XElement element, params string[] names) =>
        DefinitionChildren(element, names.Concat(AdditionalChildren(element.Name.LocalName)).ToArray());

    private static XElement WriteMetadata(IReadOnlyDictionary<string, string> metadata) =>
        new("Metadata", metadata.Select(pair =>
            new XElement("Entry", new XAttribute("key", pair.Key), new XAttribute("value", pair.Value))));

    private static void ReadMetadata(XElement parent, Dictionary<string, string> metadata)
    {
        var container = OptionalChild(parent, "Metadata");
        if (container is null)
        {
            return;
        }

        Check(container, "Metadata");
        foreach (var entry in Children(container, "Entry"))
        {
            Check(entry, "Entry", "key", "value");
            if (!metadata.TryAdd(Required(entry, "key"), (string?)entry.Attribute("value")
                ?? throw new InvalidDataException("Metadata requires a value.")))
            {
                throw new InvalidDataException("Duplicate metadata key.");
            }
        }
    }

    // Stage serializers share these validators without exposing the internal XML implementation.
    protected static void Check(XElement element, string name, params string[] attributes) =>
        GraphXmlSupport.Check(element, name, attributes);

    protected static IEnumerable<XElement> DefinitionChildren(XElement element, params string[] names) =>
        GraphXmlSupport.DefinitionChildren(element, names);

    protected static IEnumerable<XElement> Children(XElement element, string name) =>
        GraphXmlSupport.Children(element, name);

    protected static XElement RequiredChild(XElement element, string name) =>
        GraphXmlSupport.RequiredChild(element, name);

    protected static XElement? OptionalChild(XElement parent, string name)
    {
        var children = parent.Elements(name).ToArray();
        if (children.Length > 1)
        {
            throw new InvalidDataException($"Duplicate <{name}>.");
        }

        return children.SingleOrDefault();
    }

    protected static string Required(XElement element, string attribute) => GraphXmlSupport.Required(element, attribute);
    protected static int Number(XElement element, string attribute) => GraphXmlSupport.Number(element, attribute);
    protected static XAttribute? Optional(string name, string? value) => GraphXmlSupport.Optional(name, value);
    protected static T DefinitionEnumValue<T>(XElement element, string attribute) where T : struct, Enum =>
        GraphXmlSupport.DefinitionEnumValue<T>(element, attribute);

    protected static T EnumOr<T>(XElement element, string name, T fallback) where T : struct, Enum =>
        element.Attribute(name) is null ? fallback : DefinitionEnumValue<T>(element, name);

    private static bool BoolOr(XElement element, string name)
    {
        var attribute = element.Attribute(name);
        if (attribute is null)
        {
            return false;
        }

        return attribute.Value switch
        {
            "true" => true,
            "false" => false,
            _ => throw new InvalidDataException($"Invalid boolean '{name}'."),
        };
    }
}
