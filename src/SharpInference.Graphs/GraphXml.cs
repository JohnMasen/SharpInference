using System.Globalization;
using System.Xml;
using System.Xml.Linq;

namespace SharpInference.Graphs;

public static class GraphXml
{
    private const int SchemaVersion = 1;
    private const int MaxCharacters = 16 * 1024 * 1024;
    private const int MaxElements = 100_000;
    private const int MaxDepth = 64;

    public static string Serialize(LogicalGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var root = CreateGraph("LogicalGraph", graph.Identity, graph.Model, graph.Resources, graph.Regions,
            graph.Inputs, graph.Outputs, graph.GraphState);
        if (graph.Nodes.Any(node => node.Resources.Any(binding => binding.View is not null)))
            root.SetAttributeValue("version", 2);
        root.Add(new XElement("Nodes", graph.Nodes.Select(node =>
            CreateNode(node.Id.Value, node.Operation, node.Region, node.Resources, node.Dependencies.Select(id => id.Value),
                node.Attributes, node.Requirements))));
        return Write(root);
    }

    public static string Serialize(ExecutionGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var root = CreateGraph("ExecutionGraph", graph.Identity, graph.Model, graph.Resources, graph.Regions,
            graph.Inputs, graph.Outputs, graph.GraphState);
        if (graph.Nodes.Any(node => node.Resources.Any(binding => binding.View is not null)))
            root.SetAttributeValue("version", 2);
        root.Add(new XElement("Nodes", graph.Nodes.Select(node =>
        {
            var result = CreateNode(node.Id.Value, node.Operation, node.Region, node.Resources,
                node.Dependencies.Select(id => id.Value), node.Attributes, node.Requirements);
            if (node.InternalResources.Count > 0)
            {
                result.Element("Bindings")!.AddAfterSelf(CreateBindings("InternalResources", node.InternalResources, isPrivate: true));
            }
            if (node.FirstWriteResources.Count > 0)
            {
                (result.Element("InternalResources") ?? result.Element("Bindings")!).AddAfterSelf(
                    new XElement("FirstWrites", node.FirstWriteResources.Select(id =>
                        new XElement("ResourceRef", new XAttribute("id", id.Value)))));
            }

            if (node.Source.LogicalNodes.Count > 0 || node.Source.AppliedRuleId is not null)
            {
                result.Add(new XElement("Source", Optional("appliedRuleId", node.Source.AppliedRuleId),
                    node.Source.LogicalNodes.Select(id => new XElement("LogicalNode", new XAttribute("id", id.Value)))));
            }

            return result;
        })));
        return Write(root);
    }

    public static LogicalGraph DeserializeLogical(string xml) => DeserializeLogical(xml, null);

    public static LogicalGraph DeserializeLogical(string xml, IGraphModelSignatureReader? legacyReader)
    {
        var root = Read(xml, "LogicalGraph");
        try
        {
            var parts = ReadParts(root, legacyReader);
            var nodes = Children(parts.Nodes, "Node").Select(node =>
            {
                var fields = ReadNode(node, execution: false);
                return new LogicalNode(new LogicalNodeId(fields.Id), fields.Operation, fields.Region,
                    fields.Resources, fields.Dependencies.Select(id => new LogicalNodeId(id)).ToArray(),
                    fields.Attributes, fields.Requirements);
            }).ToArray();
            return new LogicalGraph(parts.Identity, parts.Model, parts.Resources, parts.Regions, nodes,
                parts.Inputs, parts.Outputs, parts.GraphState);
        }
        catch (ArgumentException exception)
        {
            throw new InvalidDataException("Invalid logical graph XML.", exception);
        }
    }

    public static ExecutionGraph DeserializeExecution(string xml) => DeserializeExecution(xml, null);

    public static ExecutionGraph DeserializeExecution(string xml, IGraphModelSignatureReader? legacyReader)
    {
        var root = Read(xml, "ExecutionGraph");
        try
        {
            var parts = ReadParts(root, legacyReader);
            var nodes = Children(parts.Nodes, "Node").Select(node =>
            {
                var fields = ReadNode(node, execution: true);
                var sourceElement = node.Element("Source");
                var source = ExecutionSourceMap.XmlOnly;
                if (sourceElement is not null)
                {
                    Check(sourceElement, "Source", "appliedRuleId");
                    source = new ExecutionSourceMap(
                        Children(sourceElement, "LogicalNode").Select(item =>
                        {
                            Check(item, "LogicalNode", "id");
                            return new LogicalNodeId(Required(item, "id"));
                        }).ToArray(),
                        (string?)sourceElement.Attribute("appliedRuleId"));
                }

                return new ExecutionNode(new ExecutionNodeId(fields.Id), fields.Operation, fields.Region,
                    fields.Resources, fields.Dependencies.Select(id => new ExecutionNodeId(id)).ToArray(),
                    fields.Attributes, fields.Requirements, source)
                {
                    InternalResources = fields.InternalResources,
                    FirstWriteResources = fields.FirstWrites,
                };
            }).ToArray();
            return new ExecutionGraph(parts.Identity, parts.Model, parts.Resources, parts.Regions, nodes,
                parts.Inputs, parts.Outputs, parts.GraphState);
        }
        catch (ArgumentException exception)
        {
            throw new InvalidDataException("Invalid execution graph XML.", exception);
        }
    }

    private static XElement CreateGraph(string kind, GraphIdentity identity, GraphModelSignature model,
        IReadOnlyList<GraphResource> resources, IReadOnlyList<GraphRegion> regions,
        IReadOnlyList<ResourceId> inputs, IReadOnlyList<ResourceId> outputs,
        GraphState? graphState) =>
        new("Graph", new XAttribute("kind", kind), new XAttribute("version", SchemaVersion),
            new XElement("Identity", new XAttribute("architectureId", identity.ArchitectureId),
                new XAttribute("irVersion", identity.IrVersion), new XAttribute("name", identity.Name)),
            CreateModel(model),
            new XElement("Resources", resources.Select(resource =>
                new XElement("Resource", new XAttribute("id", resource.Id.Value), new XAttribute("name", resource.Name),
                    new XAttribute("kind", resource.Kind), new XAttribute("lifetime", resource.Lifetime),
                    new XAttribute("scope", resource.Scope), Optional("bindingKey", resource.BindingKey),
                    Optional("deviceId", resource.DeviceId),
                    new XElement("Tensor", new XAttribute("elementType", resource.Tensor.ElementType),
                        new XAttribute("layout", resource.Tensor.Layout),
                        resource.Tensor.Dimensions.Select(size => new XElement("Dimension", new XAttribute("size", size))))))),
            new XElement("Regions", regions.Select(region =>
                new XElement("Region", new XAttribute("id", region.Id.Value),
                    Optional("parentId", region.ParentId?.Value), new XAttribute("type", region.Type),
                    Optional("role", region.Role), new XAttribute("name", region.Name),
                    CreateAttributes(region.Attributes)))),
            new XElement("Inputs", inputs.Select(id => new XElement("ResourceRef", new XAttribute("id", id.Value)))),
            new XElement("Outputs", outputs.Select(id => new XElement("ResourceRef", new XAttribute("id", id.Value)))),
            graphState is null || graphState.Entries.Count == 0 ? null : new XElement("GraphState",
                new XElement("Schema", new XAttribute("name", graphState.Schema.Name)),
                new XElement("Slots", graphState.Slots.Select(slot => new XElement("Slot",
                    new XAttribute("name", slot.Name), new XAttribute("resource", slot.Resource.Value))))));

    private static XElement CreateModel(GraphModelSignature model)
    {
        return new XElement("Model",
            new XAttribute("modelType", model.ModelType),
            new XAttribute("stateAbiId", model.StateAbiId),
            new XElement("Dimensions", model.Dimensions.OrderBy(value => value.Key, StringComparer.Ordinal)
                .Select(value => new XElement("Dimension",
                    new XAttribute("name", value.Key), new XAttribute("size", value.Value)))),
            new XElement("Attributes", model.Attributes.OrderBy(value => value.Key, StringComparer.Ordinal)
                .Select(value => new XElement("Attribute",
                    new XAttribute("name", value.Key), new XAttribute("value", value.Value)))));
    }

    private static XElement CreateNode(string id, GraphOperationId operation, RegionId region,
        IReadOnlyList<NodeResourceBinding> resources, IEnumerable<string> dependencies,
        IReadOnlyDictionary<string, string> attributes, PrecisionRequirement requirements) =>
        new("Node", new XAttribute("id", id), new XAttribute("operation", operation.Name),
            new XAttribute("operationVersion", operation.Version), new XAttribute("region", region.Value),
            CreateBindings("Bindings", resources, isPrivate: false),
            new XElement("Dependencies", dependencies.Select(dependency =>
                new XElement("Dependency", new XAttribute("id", dependency)))),
            CreateAttributes(attributes),
            new XElement("Precision", new XAttribute("minimumArithmeticType", requirements.MinimumArithmeticType),
                new XAttribute("minimumAccumulatorType", requirements.MinimumAccumulatorType)));

    private static XElement CreateBindings(string name, IReadOnlyList<NodeResourceBinding> resources, bool isPrivate) =>
        new(name, resources.Select(binding =>
        {
            if (!isPrivate && binding.InitializedBeforeRead)
            {
                throw new InvalidDataException("Public bindings cannot carry private initialization metadata.");
            }

            return new XElement("Binding", new XAttribute("port", binding.Port),
                new XAttribute("resource", binding.Resource.Value), new XAttribute("access", binding.Access),
                isPrivate && binding.InitializedBeforeRead
                    ? new XAttribute("initializedBeforeRead", "true")
                    : null,
                binding.View is { } view ? new XElement("View", new XAttribute("byteOffset", view.ByteOffset),
                    new XElement("Tensor", new XAttribute("elementType", view.Tensor.ElementType),
                        new XAttribute("layout", view.Tensor.Layout), view.Tensor.Dimensions.Select(size =>
                            new XElement("Dimension", new XAttribute("size", size))))) : null);
        }));

    private static XElement CreateAttributes(IReadOnlyDictionary<string, string> attributes) =>
        new("Attributes", attributes.Select(pair =>
            new XElement("Attribute", new XAttribute("key", pair.Key), new XAttribute("value", pair.Value))));

    private static XAttribute? Optional(string key, string? value) =>
        value is null ? null : new XAttribute(key, value);

    private static string Write(XElement root)
    {
        var xml = root.ToString(SaveOptions.DisableFormatting);
        if (xml.Length > MaxCharacters || root.DescendantsAndSelf().Take(MaxElements + 1).Count() > MaxElements)
        {
            throw new InvalidDataException("Graph XML exceeds the size limit.");
        }

        return xml;
    }

    private static XElement Read(string xml, string kind)
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
            if (document.Nodes().Any(node => node is not XElement))
            {
                throw new InvalidDataException("Graph XML contains unexpected document content.");
            }

            var root = document.Root ?? throw new InvalidDataException("Graph XML has no root.");
            if (root.DescendantsAndSelf().Take(MaxElements + 1).Count() > MaxElements ||
                root.DescendantsAndSelf().Any(element => element.Ancestors().Take(MaxDepth + 1).Count() > MaxDepth))
            {
                throw new InvalidDataException("Graph XML exceeds the structural limit.");
            }

            Check(root, "Graph", "kind", "version");
            var version = Required(root, "version");
            if (Required(root, "kind") != kind || version is not ("1" or "2"))
            {
                throw new InvalidDataException($"Expected {kind} XML schema version 1 or 2.");
            }
            if (version == "1" && root.Descendants("View").Any())
                throw new InvalidDataException("Legacy graph XML cannot contain tensor views.");
            if (version == "2" && (root.Element("Identity") is not { } identity || Number(identity, "irVersion") < 2))
                throw new InvalidDataException("Graph XML version 2 requires graph IR version 2 or newer.");

            return root;
        }
        catch (XmlException exception)
        {
            throw new InvalidDataException("Malformed or unsafe graph XML.", exception);
        }
    }

    private static GraphParts ReadParts(XElement root, IGraphModelSignatureReader? legacyReader)
    {
        var hasState = root.Element("GraphState") is not null;
        var sections = Ordered(root, hasState
            ? ["Identity", "Model", "Resources", "Regions", "Inputs", "Outputs", "GraphState", "Nodes"]
            : ["Identity", "Model", "Resources", "Regions", "Inputs", "Outputs", "Nodes"]);
        Check(sections[2], "Resources");
        Check(sections[3], "Regions");
        Check(sections[4], "Inputs");
        Check(sections[5], "Outputs");
        if (hasState) Check(sections[6], "GraphState");
        Check(sections[hasState ? 7 : 6], "Nodes");
        var identity = sections[0];
        Check(identity, "Identity", "architectureId", "irVersion", "name");
        var model = sections[1];
        var resources = Children(sections[2], "Resource").Select(item =>
        {
            Check(item, "Resource", "id", "name", "kind", "lifetime", "scope", "bindingKey", "deviceId");
            var tensor = Ordered(item, "Tensor")[0];
            Check(tensor, "Tensor", "elementType", "layout");
            var dimensions = Children(tensor, "Dimension").Select(dimension =>
            {
                Check(dimension, "Dimension", "size");
                return Number(dimension, "size");
            }).ToArray();
            var resource = new GraphResource(new ResourceId(Required(item, "id")), Required(item, "name"),
                EnumValue<GraphResourceKind>(item, "kind"), EnumValue<GraphResourceLifetime>(item, "lifetime"),
                new TensorDescriptor(EnumValue<GraphElementType>(tensor, "elementType"), dimensions,
                    Required(tensor, "layout")), (string?)item.Attribute("bindingKey"),
                (string?)item.Attribute("deviceId"),
                item.Attribute("scope") is null ? null : EnumValue<GraphResourceScope>(item, "scope"));
            return resource;
        }).ToArray();
        var regions = Children(sections[3], "Region").Select(item =>
        {
            Check(item, "Region", "id", "parentId", "type", "role", "name");
            var attributes = Ordered(item, "Attributes")[0];
            return new GraphRegion(new RegionId(Required(item, "id")),
                item.Attribute("parentId") is XAttribute parent ? new RegionId(parent.Value) : null,
                Required(item, "type"), (string?)item.Attribute("role"), Required(item, "name"),
                ReadAttributes(attributes));
        }).ToArray();
        return new GraphParts(
            new GraphIdentity(Required(identity, "architectureId"), Number(identity, "irVersion"), Required(identity, "name")),
            ReadModel(model, legacyReader),
            resources, regions, ReadRefs(sections[4]), ReadRefs(sections[5]),
            hasState ? ReadGraphState(sections[6]) : null, sections[hasState ? 7 : 6]);
    }

    private static GraphModelSignature ReadModel(XElement model, IGraphModelSignatureReader? legacyReader)
    {
        if (model.Attribute("modelType") is null)
        {
            return legacyReader?.ReadXml(model) ??
                throw new InvalidDataException("A legacy model signature requires an explicitly supplied model reader.");
        }

        Check(model, "Model", "modelType", "stateAbiId");
        var sections = Ordered(model, "Dimensions", "Attributes");
        Check(sections[0], "Dimensions");
        Check(sections[1], "Attributes");
        var dimensions = Children(sections[0], "Dimension").ToDictionary(value =>
        {
            Check(value, "Dimension", "name", "size");
            return Required(value, "name");
        }, value => Number(value, "size"), StringComparer.Ordinal);
        var attributes = Children(sections[1], "Attribute").ToDictionary(value =>
        {
            Check(value, "Attribute", "name", "value");
            return Required(value, "name");
        }, value => Required(value, "value"), StringComparer.Ordinal);
        return new GraphModelSignature(
            Required(model, "modelType"), Required(model, "stateAbiId"), dimensions, attributes);
    }

    private static GraphState ReadGraphState(XElement element)
    {
        var parts = Ordered(element, "Schema", "Slots");
        Check(parts[0], "Schema", "name");
        if (parts[0].HasElements) throw new InvalidDataException("A state schema cannot contain nested elements.");
        Check(parts[1], "Slots");
        var slots = Children(parts[1], "Slot").Select(slot =>
        {
            Check(slot, "Slot", "name", "resource");
            if (slot.HasElements) throw new InvalidDataException("State slots cannot contain nested elements.");
            return new GraphStateSlot(Required(slot, "name"), new ResourceId(Required(slot, "resource")));
        });
        return new GraphState(new StateSchema(Required(parts[0], "name")), slots);
    }

    private static NodeParts ReadNode(XElement node, bool execution)
    {
        Check(node, "Node", "id", "operation", "operationVersion", "region");
        var hasInternal = execution && node.Element("InternalResources") is not null;
        var hasFirstWrites = execution && node.Element("FirstWrites") is not null;
        var hasSource = execution && node.Element("Source") is not null;
        var names = new List<string> { "Bindings" };
        if (hasInternal) names.Add("InternalResources");
        if (hasFirstWrites) names.Add("FirstWrites");
        names.AddRange(["Dependencies", "Attributes", "Precision"]);
        if (hasSource) names.Add("Source");
        var sections = Ordered(node, names.ToArray());
        Check(sections[0], "Bindings");
        var bindings = ReadBindings(sections[0], isPrivate: false);
        var internalResources = hasInternal ? ReadBindings(sections[1], isPrivate: true) : [];
        if (hasInternal) Check(sections[1], "InternalResources");
        var offset = hasInternal ? 1 : 0;
        var firstWrites = hasFirstWrites ? ReadRefs(sections[offset + 1]) : [];
        if (hasFirstWrites)
        {
            Check(sections[offset + 1], "FirstWrites");
            offset++;
        }
        Check(sections[offset + 1], "Dependencies");
        var dependencies = Children(sections[offset + 1], "Dependency").Select(item =>
        {
            Check(item, "Dependency", "id");
            return Required(item, "id");
        }).ToArray();
        var attributes = ReadAttributes(sections[offset + 2]);
        var precision = sections[offset + 3];
        Check(precision, "Precision", "minimumArithmeticType", "minimumAccumulatorType");
        return new NodeParts(Required(node, "id"),
            new GraphOperationId(Required(node, "operation"), Number(node, "operationVersion")),
            new RegionId(Required(node, "region")), bindings, internalResources, firstWrites, dependencies, attributes,
            new PrecisionRequirement(EnumValue<GraphElementType>(precision, "minimumArithmeticType"),
                EnumValue<GraphElementType>(precision, "minimumAccumulatorType")));
    }

    private static NodeResourceBinding[] ReadBindings(XElement element, bool isPrivate) =>
        Children(element, "Binding").Select(item =>
        {
            Check(item, "Binding", "port", "resource", "access", "initializedBeforeRead");
            var initialization = (string?)item.Attribute("initializedBeforeRead");
            if (initialization is not null && (!isPrivate || initialization is not ("true" or "false")))
            {
                throw new InvalidDataException("Invalid private binding initialization metadata.");
            }
            var views = Children(item, "View").ToArray();
            if (views.Length > 1) throw new InvalidDataException("A binding may declare only one tensor view.");
            GraphTensorView? view = null;
            if (views.Length == 1)
            {
                var element = views[0];
                Check(element, "View", "byteOffset");
                if (!ulong.TryParse(Required(element, "byteOffset"), NumberStyles.None, CultureInfo.InvariantCulture, out var offset))
                    throw new InvalidDataException("Invalid tensor-view byte offset.");
                var tensor = Ordered(element, "Tensor")[0];
                Check(tensor, "Tensor", "elementType", "layout");
                var dimensions = Children(tensor, "Dimension").Select(dimension =>
                {
                    Check(dimension, "Dimension", "size");
                    return Number(dimension, "size");
                });
                view = new(offset, new TensorDescriptor(EnumValue<GraphElementType>(tensor, "elementType"), dimensions,
                    Required(tensor, "layout")));
            }

            return new NodeResourceBinding(Required(item, "port"), new ResourceId(Required(item, "resource")),
                EnumValue<GraphResourceAccess>(item, "access"), initialization == "true", view);
        }).ToArray();

    private static IReadOnlyDictionary<string, string> ReadAttributes(XElement attributes)
    {
        Check(attributes, "Attributes");
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var item in Children(attributes, "Attribute"))
        {
            Check(item, "Attribute", "key", "value");
            if (!result.TryAdd(Required(item, "key"), Required(item, "value")))
            {
                throw new InvalidDataException("Duplicate graph attribute key.");
            }
        }

        return result;
    }

    private static ResourceId[] ReadRefs(XElement parent) =>
        Children(parent, "ResourceRef").Select(item =>
        {
            Check(item, "ResourceRef", "id");
            return new ResourceId(Required(item, "id"));
        }).ToArray();

    private static XElement[] Ordered(XElement parent, params string[] names)
    {
        var elements = parent.Elements().ToArray();
        if (elements.Length != names.Length || elements.Where((element, index) => element.Name != names[index]).Any())
        {
            throw new InvalidDataException($"Unexpected children in <{parent.Name}>.");
        }

        return elements;
    }

    private static IEnumerable<XElement> Children(XElement parent, string name)
    {
        if (parent.Elements().Any(element => element.Name != name))
        {
            throw new InvalidDataException($"Unexpected children in <{parent.Name}>.");
        }

        return parent.Elements();
    }

    private static void Check(XElement element, string name, params string[] attributes)
    {
        if (element.Name != name || element.Attributes().Any(attribute =>
                attribute.IsNamespaceDeclaration || !attributes.Contains(attribute.Name.LocalName, StringComparer.Ordinal) ||
                attribute.Name.Namespace != XNamespace.None) ||
            element.Nodes().Any(node => node is XText text && !string.IsNullOrWhiteSpace(text.Value) ||
                node is not XText and not XElement))
        {
            throw new InvalidDataException($"Invalid <{name}> element.");
        }
    }

    private static string Required(XElement element, string attribute) =>
        (string?)element.Attribute(attribute) ?? throw new InvalidDataException($"Missing '{attribute}' in <{element.Name}>.");

    private static int Number(XElement element, string attribute) =>
        int.TryParse(Required(element, attribute), NumberStyles.None, CultureInfo.InvariantCulture, out var value)
            ? value
            : throw new InvalidDataException($"Invalid '{attribute}' in <{element.Name}>.");

    private static T EnumValue<T>(XElement element, string attribute) where T : struct, Enum
    {
        var value = Required(element, attribute);
        if (!Enum.TryParse<T>(value, ignoreCase: false, out var parsed) ||
            !Enum.IsDefined(parsed) || parsed.ToString() != value)
        {
            throw new InvalidDataException($"Invalid '{attribute}' in <{element.Name}>.");
        }

        return parsed;
    }

    private sealed record GraphParts(GraphIdentity Identity, GraphModelSignature Model, GraphResource[] Resources,
        GraphRegion[] Regions, ResourceId[] Inputs, ResourceId[] Outputs, GraphState? GraphState, XElement Nodes);

    private sealed record NodeParts(string Id, GraphOperationId Operation, RegionId Region,
        NodeResourceBinding[] Resources, NodeResourceBinding[] InternalResources, ResourceId[] FirstWrites,
        string[] Dependencies,
        IReadOnlyDictionary<string, string> Attributes,
        PrecisionRequirement Requirements);
}
