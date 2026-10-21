using System.Xml.Linq;
using SharpInference.Graphs.Internal;
using SharpInference.Graphs.Logical;
using SharpInference.Graphs.Execution;
using SharpInference.Instructions;
using static SharpInference.Graphs.Internal.GraphXmlSupport;

namespace SharpInference.Graphs;

/// <summary>The version-one GraphPackage format for editable computation sources.</summary>
public static class GraphPackageXml
{
    public static string Serialize<TGraph>(GraphPackage<TGraph> package) where TGraph : Graph, new()
    {
        ArgumentNullException.ThrowIfNull(package);
        var root = new XElement("GraphPackage",
            new XAttribute("version", 1), new XAttribute("stage", Stage(new TGraph())),
            Optional("main", package.MainGraphId),
            package is GraphPackage<ExecutionGraph> execution && execution.GetTarget() is { } backend
                ? new XElement("Target", new XAttribute("backend", backend)) : null,
            WriteMetadata(package.Metadata.Where(pair => package is not GraphPackage<ExecutionGraph> ||
                    pair.Key != ExecutionPackage.BackendMetadataKey)
                .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal)),
            new XElement("Graphs", package.Select(WriteGraph)));
        return Write(root);
    }

    public static GraphPackage<TGraph> Deserialize<TGraph>(string xml) where TGraph : Graph, new()
    {
        var root = ReadDocument(xml);
        Check(root, "GraphPackage", "version", "stage", "main");
        if (Number(root, "version") != 1 || Required(root, "stage") != Stage(new TGraph()))
            throw new InvalidDataException("Unsupported GraphPackage version or stage.");
        DefinitionChildren(root, "Target", "Metadata", "Graphs");
        var package = new GraphPackage<TGraph> { MainGraphId = (string?)root.Attribute("main") };
        ReadMetadata(root, package.Metadata);
        if (OptionalChild(root, "Target") is { } target)
        {
            Check(target, "Target", "backend");
            DefinitionChildren(target);
            if (package is not GraphPackage<ExecutionGraph> execution)
                throw new InvalidDataException("A target belongs to an execution package.");
            if (package.Metadata.ContainsKey(ExecutionPackage.BackendMetadataKey))
                throw new InvalidDataException("Duplicate execution target.");
            execution.SetTarget(Required(target, "backend"));
        }
        var graphs = RequiredChild(root, "Graphs");
        Check(graphs, "Graphs");
        foreach (var element in Children(graphs, "Graph"))
            package.Add(ReadGraph<TGraph>(element));
        return package;
    }

    private static string Stage(Graph graph) => graph switch
    {
        LogicalGraph => "logical",
        ExecutionGraph => "execution",
        _ => "business",
    };

    private static XElement WriteGraph(Graph graph) =>
        new("Graph", new XAttribute("id", graph.Id),
            Optional("name", graph.Name), Optional("description", graph.Description),
            graph is LogicalGraph logical ? new XAttribute("compileBoundary", logical.CompileBoundary) : null,
            WriteMetadata(graph.Metadata),
            new XElement("Resources", graph.Resources.Select(resource =>
                new XElement("Resource", new XAttribute("id", resource.Id),
                    new XAttribute("category", resource.Category),
                    resource.GraphIO != GraphIO.Internal ? new XAttribute("graphIO", resource.GraphIO) : null,
                    resource.Scope != ResourceScope.Call ? new XAttribute("scope", resource.Scope) : null,
                    resource.Usage is { } usage ? new XAttribute("usage", usage) : null,
                    resource.IsState ? new XAttribute("isState", true) : null,
                    resource.Permission != ResourcePermission.ReadWrite ? new XAttribute("permission", resource.Permission) : null,
                    Optional("bindingKey", resource.BindingKey), WriteTensor(resource.Tensor),
                    graph is ExecutionGraph execution && execution.Storage.TryGetValue(resource.Id, out var storage)
                        ? new XElement("Storage", new XAttribute("kind", storage.Kind), Optional("slot", storage.Slot)) : null))),
            new XElement("Actions", graph.Actions.Select(WriteAction)),
            new XElement("GraphFlow", graph.GraphFlow.Select(flow =>
                new XElement("Then", new XAttribute("fromGraph", flow.Source.Graph),
                    new XAttribute("from", flow.Source.Action), new XAttribute("toGraph", flow.Target.Graph),
                    new XAttribute("to", flow.Target.Action)))));

    private static XElement WriteAction(GraphAction action)
    {
        var element = action switch
        {
            OperationAction operation => new XElement("Operation",
                new XAttribute("operation", operation.Operation.Name),
                new XAttribute("version", operation.Operation.Version),
                WriteAttributes(operation.Attributes),
                operation.Precision is { } precision ? WritePrecision(precision) : null),
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
        if (action is ExecutionOperation { SelectedImplementation: { } implementation })
            element.Add(new XElement("Implementation",
                new XAttribute("collection", implementation.CollectionId),
                new XAttribute("instruction", implementation.InstructionName),
                new XAttribute("target", implementation.Target),
                new XAttribute("arithmetic", implementation.Precision.ArithmeticType),
                new XAttribute("accumulator", implementation.Precision.AccumulatorType),
                implementation.Dispatch is { } dispatch
                    ? new XElement("Dispatch", new XAttribute("threadsX", dispatch.ThreadsX),
                        new XAttribute("threadsY", dispatch.ThreadsY), new XAttribute("threadsZ", dispatch.ThreadsZ),
                        new XAttribute("groupsX", dispatch.GroupsX), new XAttribute("groupsY", dispatch.GroupsY),
                        new XAttribute("groupsZ", dispatch.GroupsZ)) : null));
        var sources = action switch
        {
            ExecutionOperation operation => operation.Sources,
            ExecutionCall call => call.Sources,
            _ => null,
        };
        if (sources is { Count: > 0 })
            element.Add(new XElement("Sources", sources.Select(source => new XElement("Source",
                new XAttribute("document", source.Document), new XAttribute("graph", source.Graph),
                new XAttribute("node", source.Node)))));
        return element;
    }

    private static TGraph ReadGraph<TGraph>(XElement element) where TGraph : Graph, new()
    {
        Check(element, "Graph", "id", "name", "description", "compileBoundary");
        DefinitionChildren(element, "Metadata", "Resources", "Actions", "GraphFlow");
        var graph = new TGraph
        {
            Id = Required(element, "id"), Name = (string?)element.Attribute("name"),
            Description = (string?)element.Attribute("description"),
        };
        if (graph is LogicalGraph logical)
            logical.CompileBoundary = EnumOr(element, "compileBoundary", GraphCompileBoundary.Auto);
        else if (element.Attribute("compileBoundary") is not null)
            throw new InvalidDataException("Compilation intent belongs to logical graphs.");
        ReadMetadata(element, graph.Metadata);
        var resources = RequiredChild(element, "Resources");
        Check(resources, "Resources");
        foreach (var value in Children(resources, "Resource"))
        {
            Check(value, "Resource", "id", "category", "graphIO", "scope", "usage", "isState", "permission", "bindingKey");
            DefinitionChildren(value, "Tensor", "Storage");
            var resource = graph.Resource(Required(value, "id"), ReadTensor(RequiredChild(value, "Tensor")),
                EnumOr(value, "graphIO", GraphIO.Internal), EnumOr(value, "scope", ResourceScope.Call),
                value.Attribute("usage") is null ? null : DefinitionEnumValue<ResourceUsage>(value, "usage"),
                BoolOr(value, "isState"), EnumOr(value, "permission", ResourcePermission.ReadWrite),
                (string?)value.Attribute("bindingKey"));
            resource.Category = (string?)value.Attribute("category") ?? "resource";
            if (OptionalChild(value, "Storage") is { } storage)
            {
                Check(storage, "Storage", "kind", "slot");
                DefinitionChildren(storage);
                if (graph is not ExecutionGraph execution)
                    throw new InvalidDataException("Storage belongs to an execution graph.");
                if (!execution.Storage.TryAdd(resource.Id, new(
                    DefinitionEnumValue<ResourceStorageKind>(storage, "kind"), (string?)storage.Attribute("slot"))))
                    throw new InvalidDataException($"Duplicate storage for '{resource.Id}'.");
            }
        }
        var actions = RequiredChild(element, "Actions");
        Check(actions, "Actions");
        foreach (var value in DefinitionChildren(actions, "Operation", "Call"))
        {
            GraphAction action;
            if (value.Name == "Operation")
            {
                Check(value, "Operation", "id", "category", "operation", "version");
                DefinitionChildren(value, "Attributes", "Precision", "Metadata", "Binding", "Implementation", "Sources");
                var operation = graph.Operation(Required(value, "id"),
                    new OperationId(Required(value, "operation"), Number(value, "version")));
                foreach (var pair in ReadAttributes(RequiredChild(value, "Attributes")))
                    operation.Attributes.Add(pair.Key, pair.Value);
                var precision = OptionalChild(value, "Precision");
                operation.Precision = precision is null ? null : ReadPrecision(precision);
                action = operation;
            }
            else
            {
                Check(value, "Call", "id", "category", "target");
                DefinitionChildren(value, "Metadata", "Binding", "Sources");
                action = graph.Call(Required(value, "id"), Required(value, "target"));
            }
            action.Category = (string?)value.Attribute("category") ?? (action is OperationAction ? "operation" : "call");
            ReadMetadata(value, action.Metadata);
            if (OptionalChild(value, "Implementation") is { } implementation)
            {
                if (action is not ExecutionOperation operation)
                    throw new InvalidDataException("An implementation belongs to an execution operation.");
                Check(implementation, "Implementation", "collection", "instruction", "target", "arithmetic", "accumulator");
                DefinitionChildren(implementation, "Dispatch");
                if (!Guid.TryParse(Required(implementation, "collection"), out var collection))
                    throw new InvalidDataException("Invalid instruction collection identifier.");
                operation.SelectedImplementation = new(collection, Required(implementation, "instruction"),
                    new InstructionTarget(Required(implementation, "target")),
                    new KernelPrecisionProfile(DefinitionEnumValue<ElementType>(implementation, "arithmetic"),
                        DefinitionEnumValue<ElementType>(implementation, "accumulator")),
                    ReadDispatch(OptionalChild(implementation, "Dispatch")));
            }
            if (OptionalChild(value, "Sources") is { } sources)
            {
                var destination = action switch
                {
                    ExecutionOperation operation => operation.Sources,
                    ExecutionCall call => call.Sources,
                    _ => throw new InvalidDataException("Sources belong to execution actions."),
                };
                Check(sources, "Sources");
                foreach (var source in Children(sources, "Source"))
                {
                    Check(source, "Source", "document", "graph", "node");
                    DefinitionChildren(source);
                    destination.Add(new(Required(source, "document"), Required(source, "graph"), Required(source, "node")));
                }
            }
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

    private static XElement WriteMetadata(IReadOnlyDictionary<string, string> metadata) =>
        new("Metadata", metadata.Select(pair =>
            new XElement("Entry", new XAttribute("key", pair.Key), new XAttribute("value", pair.Value))));

    private static void ReadMetadata(XElement parent, Dictionary<string, string> metadata)
    {
        var container = OptionalChild(parent, "Metadata");
        if (container is null)
            return;
        Check(container, "Metadata");
        foreach (var entry in Children(container, "Entry"))
        {
            Check(entry, "Entry", "key", "value");
            if (!metadata.TryAdd(Required(entry, "key"), (string?)entry.Attribute("value")
                ?? throw new InvalidDataException("Metadata requires a value.")))
                throw new InvalidDataException("Duplicate metadata key.");
        }
    }

    private static XElement? OptionalChild(XElement parent, string name)
    {
        var children = parent.Elements(name).ToArray();
        if (children.Length > 1)
            throw new InvalidDataException($"Duplicate <{name}>.");
        return children.SingleOrDefault();
    }

    private static GraphInstructionDispatch? ReadDispatch(XElement? element)
    {
        if (element is null)
            return null;
        Check(element, "Dispatch", "threadsX", "threadsY", "threadsZ", "groupsX", "groupsY", "groupsZ");
        DefinitionChildren(element);
        return new((uint)Number(element, "threadsX"), (uint)Number(element, "threadsY"),
            (uint)Number(element, "threadsZ"), (uint)Number(element, "groupsX"),
            (uint)Number(element, "groupsY"), (uint)Number(element, "groupsZ"));
    }

    private static T EnumOr<T>(XElement element, string name, T fallback) where T : struct, Enum =>
        element.Attribute(name) is null ? fallback : DefinitionEnumValue<T>(element, name);

    private static bool BoolOr(XElement element, string name)
    {
        var attribute = element.Attribute(name);
        if (attribute is null)
            return false;
        return attribute.Value switch
        {
            "true" => true,
            "false" => false,
            _ => throw new InvalidDataException($"Invalid boolean '{name}'."),
        };
    }
}
