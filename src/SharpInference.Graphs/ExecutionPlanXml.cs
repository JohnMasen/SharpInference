using System.Globalization;
using System.Xml;
using System.Xml.Linq;

namespace SharpInference.Graphs;

public static class ExecutionPlanXml
{
    public static string Serialize(TargetExecutionPlan plan)
    {
        TargetExecutionPlanValidator.Validate(plan);
        return new XElement("ExecutionPlan", new XAttribute("schema", "sharp-inference"),
            new XAttribute("version", 1), new XAttribute("kind", plan.Target == ExecutionPlanTarget.Cpu ? "cpu" : "gpu"),
            new XAttribute("name", plan.Name), new XAttribute("abi", plan.Abi),
            plan.Bindings is null ? null : new XElement("Bindings", XElement.Parse(GraphXml.Serialize(plan.Bindings))),
            new XElement("Defaults", new XAttribute("order", "document-sequential"),
                new XAttribute("allocation", "unified-manager")),
            new XElement("Slots", plan.Slots.Select(slot => new XElement("Slot",
                new XAttribute("id", slot.Id), new XAttribute("scope", slot.Scope),
                new XAttribute("access", slot.Access), Tensor(slot.Tensor),
                slot.BindingKey is null ? null : new XAttribute("bindingKey", slot.BindingKey)))),
            new XElement("ResourcePlan", plan.ResourceBindings.Select(binding => new XElement("Map",
                new XAttribute("resource", binding.Resource), new XAttribute("slot", binding.Slot),
                new XAttribute("byteOffset", binding.ByteOffset), new XAttribute("view", binding.IsView),
                Tensor(binding.Tensor)))),
            new XElement("State", new XAttribute("schema", plan.StateSchema), new XAttribute("version", plan.StateVersion),
                plan.State.Select(entry => new XElement("Entry", new XAttribute("name", entry.Name), new XAttribute("slot", entry.Slot)))),
            new XElement("Definitions", plan.Definitions.Select(definition => new XElement("Definition",
                new XAttribute("id", definition.Id), new XAttribute("kind", definition.Kind),
                definition.Threads is null ? null : Grid("Threads", definition.Threads),
                new XElement("Parameters", definition.Parameters.Select(parameter => new XElement("Parameter",
                    new XAttribute("name", parameter.Name), new XAttribute("access", parameter.Access), Tensor(parameter.Tensor)))),
                new XElement("Body", definition.Nodes.Select(Node))))),
            new XElement("EntryPoints", plan.Entries.Select(entry => new XElement("Entry",
                new XAttribute("name", entry.Name), new XAttribute("definition", entry.Definition), Arguments(entry.Arguments)))))
            .ToString();
    }

    public static TargetExecutionPlan DeserializeTarget(string xml)
    {
        try
        {
            var root = Read(xml);
            Check(root, "ExecutionPlan", ["schema", "version", "kind", "name", "abi"],
                ["Defaults", "Slots", "State", "Definitions", "EntryPoints", "Bindings", "ResourcePlan"]);
            Header(root);
            var target = Required(root, "kind") switch
            {
                "cpu" => ExecutionPlanTarget.Cpu,
                "gpu" => ExecutionPlanTarget.Direct3D12,
                _ => throw new InvalidDataException("A CPU or GPU execution plan is required."),
            };
            var defaults = One(root, "Defaults");
            Check(defaults, "Defaults", ["order", "allocation"], []);
            if (Required(defaults, "order") != "document-sequential" ||
                Required(defaults, "allocation") != "unified-manager")
                throw new InvalidDataException("Unsupported execution plan defaults.");
            var slots = Container(root, "Slots", "Slot").Select(slot =>
            {
                Check(slot, "Slot", ["id", "scope", "access", "elementType", "shape", "bindingKey"], []);
                return new ExecutionPlanSlot(Required(slot, "id"), EnumValue<GraphResourceScope>(slot, "scope"),
                    EnumValue<GraphResourceAccess>(slot, "access"), ReadTensor(slot), (string?)slot.Attribute("bindingKey"));
            }).ToArray();
            var resourceBindings = Container(root, "ResourcePlan", "Map").Select(binding =>
            {
                Check(binding, "Map", ["resource", "slot", "byteOffset", "view", "elementType", "shape"], []);
                return new ExecutionPlanResourceBinding(Required(binding, "resource"), Required(binding, "slot"),
                    ReadTensor(binding), Unsigned(binding, "byteOffset"), bool.Parse(Required(binding, "view")));
            }).ToArray();
            var state = One(root, "State");
            Check(state, "State", ["schema", "version"], ["Entry"]);
            var stateEntries = state.Elements().Select(entry =>
            {
                Check(entry, "Entry", ["name", "slot"], []);
                return new ExecutionPlanStateEntry(Required(entry, "name"), Required(entry, "slot"));
            }).ToArray();
            var definitions = Container(root, "Definitions", "Definition").Select(definition =>
            {
                Check(definition, "Definition", ["id", "kind"], ["Threads", "Parameters", "Body"]);
                var parameters = Container(definition, "Parameters", "Parameter").Select(parameter =>
                {
                    Check(parameter, "Parameter", ["name", "access", "elementType", "shape"], []);
                    return new ExecutionPlanParameter(Required(parameter, "name"),
                        EnumValue<GraphResourceAccess>(parameter, "access"), ReadTensor(parameter));
                }).ToArray();
                var threads = OptionalOne(definition, "Threads");
                return new ExecutionPlanDefinition(Required(definition, "id"),
                    EnumValue<ExecutionPlanDefinitionKind>(definition, "kind"), parameters,
                    Container(definition, "Body", "Node").Select(ReadNode),
                    threads is null ? null : ReadGrid(threads));
            }).ToArray();
            var entries = Container(root, "EntryPoints", "Entry").Select(entry =>
            {
                Check(entry, "Entry", ["name", "definition"], ["Arguments"]);
                return new ExecutionPlanEntry(Required(entry, "name"), Required(entry, "definition"), ReadArguments(entry));
            }).ToArray();
            var bindings = OptionalOne(root, "Bindings");
            if (bindings is not null) Check(bindings, "Bindings", [], ["Graph"]);
            return new(Required(root, "name"), Required(root, "abi"), target, slots, definitions, entries,
                Required(state, "schema"), Integer(state, "version"), stateEntries,
                bindings is null ? null : GraphXml.DeserializeLogical(One(bindings, "Graph").ToString()), resourceBindings);
        }
        catch (Exception error) when (error is XmlException or ArgumentException or FormatException or OverflowException)
        {
            throw new InvalidDataException("Invalid target execution plan XML.", error);
        }
    }

    private static IEnumerable<XAttribute> Tensor(TensorDescriptor tensor) =>
        [new("elementType", tensor.ElementType), new("shape", string.Join(",", tensor.Dimensions))];

    private static TensorDescriptor ReadTensor(XElement element) => new(EnumValue<GraphElementType>(element, "elementType"),
        Required(element, "shape").Split(',').Select(value => int.Parse(value, NumberStyles.None, CultureInfo.InvariantCulture)));

    private static XElement Arguments(IEnumerable<ExecutionPlanArgument> arguments) => new("Arguments", arguments.Select(argument =>
        new XElement("Argument", new XAttribute("parameter", argument.Parameter), new XAttribute("source", argument.Source),
            new XAttribute("byteOffset", argument.ByteOffset))));

    private static ExecutionPlanArgument[] ReadArguments(XElement owner) =>
        Container(owner, "Arguments", "Argument").Select(argument =>
        {
            Check(argument, "Argument", ["parameter", "source", "byteOffset"], []);
            return new ExecutionPlanArgument(Required(argument, "parameter"), Required(argument, "source"), Unsigned(argument, "byteOffset"));
        }).ToArray();

    private static XElement Grid(string name, ExecutionPlanGrid grid) => new(name,
        new XAttribute("x", grid.X), new XAttribute("y", grid.Y), new XAttribute("z", grid.Z));

    private static ExecutionPlanGrid ReadGrid(XElement grid)
    {
        Check(grid, grid.Name.LocalName, ["x", "y", "z"], []);
        return new(checked((uint)Unsigned(grid, "x")), checked((uint)Unsigned(grid, "y")), checked((uint)Unsigned(grid, "z")));
    }

    private static XElement Node(ExecutionPlanNode node)
    {
        XElement instruction = node.Instruction switch
        {
            ExecutionPlanOperator operation => new XElement("Operation",
                new XAttribute("collection", operation.Collection), new XAttribute("name", operation.Name),
                new XAttribute("arithmetic", operation.Precision.MinimumArithmeticType),
                new XAttribute("accumulator", operation.Precision.MinimumAccumulatorType),
                operation.Implementation is null ? null : new XAttribute("implementation", operation.Implementation),
                operation.CooperativeThreads is null ? null : new XAttribute("cooperativeThreads", operation.CooperativeThreads),
                operation.ResolvedPrecision is null ? null : new XElement("ResolvedPrecision",
                    new XAttribute("arithmetic", operation.ResolvedPrecision.Arithmetic),
                    new XAttribute("accumulator", operation.ResolvedPrecision.Accumulator)),
                Arguments(operation.Arguments),
                new XElement("Attributes", operation.Attributes.OrderBy(pair => pair.Key, StringComparer.Ordinal).Select(pair =>
                    new XElement("Attribute", new XAttribute("name", pair.Key), new XAttribute("value", pair.Value)))),
                new XElement("Accesses", operation.ParameterAccesses.OrderBy(pair => pair.Key, StringComparer.Ordinal).Select(pair =>
                    new XElement("Access", new XAttribute("port", pair.Key), new XAttribute("value", pair.Value)))),
                new XElement("IndexBounds", operation.IndexBounds.Select(bound => new XElement("Bound",
                    new XAttribute("indexPort", bound.IndexPort), new XAttribute("tensorPort", bound.TensorPort),
                    new XAttribute("axis", bound.Axis), new XAttribute("minimum", bound.MinimumIndex))))),
            ExecutionPlanCall call => new XElement("Call", new XAttribute("definition", call.Definition), Arguments(call.Arguments)),
            ExecutionPlanDispatch dispatch => new XElement("Dispatch", new XAttribute("definition", dispatch.Definition),
                Grid("Groups", dispatch.Groups), Arguments(dispatch.Arguments)),
            ExecutionPlanBarrier barrier => new XElement("Barrier", new XAttribute("kind", "uav"),
                barrier.Resources.Select(resource => new XElement("ResourceRef", new XAttribute("id", resource)))),
            _ => throw new InvalidDataException($"Unknown instruction at '{node.Id}'."),
        };
        return new XElement("Node", new XAttribute("id", node.Id),
            new XElement("DependsOn", node.Dependencies.Select(id => new XElement("NodeRef", new XAttribute("id", id)))),
            new XElement("Source", node.Sources.Select(id => new XElement("NodeRef", new XAttribute("id", id)))),
            instruction);
    }

    private static ExecutionPlanNode ReadNode(XElement node)
    {
        Check(node, "Node", ["id"], ["DependsOn", "Source", "Operation", "Call", "Dispatch", "Barrier"]);
        var body = node.Elements().Where(element => element.Name != "DependsOn" && element.Name != "Source").ToArray();
        if (body.Length != 1) throw new InvalidDataException("A plan node requires exactly one instruction.");
        var element = body[0];
        ExecutionPlanInstruction instruction;
        switch (element.Name.LocalName)
        {
            case "Operation":
                Check(element, "Operation", ["collection", "name", "arithmetic", "accumulator", "implementation", "cooperativeThreads"],
                    ["Arguments", "Attributes", "Accesses", "IndexBounds", "ResolvedPrecision"]);
                var attributes = Container(element, "Attributes", "Attribute").Select(attribute =>
                {
                    Check(attribute, "Attribute", ["name", "value"], []);
                    return new KeyValuePair<string, string>(Required(attribute, "name"), (string?)attribute.Attribute("value") ??
                        throw new InvalidDataException("Missing attribute value."));
                }).ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
                var accesses = Container(element, "Accesses", "Access").Select(access =>
                {
                    Check(access, "Access", ["port", "value"], []);
                    return new KeyValuePair<string, GraphResourceAccess>(Required(access, "port"), EnumValue<GraphResourceAccess>(access, "value"));
                }).ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
                var bounds = Container(element, "IndexBounds", "Bound").Select(bound =>
                {
                    Check(bound, "Bound", ["indexPort", "tensorPort", "axis", "minimum"], []);
                    return new ExecutionPlanIndexBound(Required(bound, "indexPort"), Required(bound, "tensorPort"),
                        Integer(bound, "axis"), Integer(bound, "minimum"));
                }).ToArray();
                var resolved = OptionalOne(element, "ResolvedPrecision");
                if (resolved is not null) Check(resolved, "ResolvedPrecision", ["arithmetic", "accumulator"], []);
                instruction = new ExecutionPlanOperator(new(EnumValue<GraphElementType>(element, "arithmetic"),
                    EnumValue<GraphElementType>(element, "accumulator")), Guid.Parse(Required(element, "collection")),
                    Required(element, "name"), ReadArguments(element), attributes, (string?)element.Attribute("implementation"),
                    element.Attribute("cooperativeThreads") is null ? null : checked((uint)Unsigned(element, "cooperativeThreads")),
                    bounds, accesses, resolved is null ? null : new(EnumValue<GraphElementType>(resolved, "arithmetic"),
                        EnumValue<GraphElementType>(resolved, "accumulator")));
                break;
            case "Call":
                Check(element, "Call", ["definition"], ["Arguments"]);
                instruction = new ExecutionPlanCall(Required(element, "definition"), ReadArguments(element));
                break;
            case "Dispatch":
                Check(element, "Dispatch", ["definition"], ["Groups", "Arguments"]);
                instruction = new ExecutionPlanDispatch(Required(element, "definition"), ReadArguments(element), ReadGrid(One(element, "Groups")));
                break;
            case "Barrier":
                Check(element, "Barrier", ["kind"], ["ResourceRef"]);
                if (Required(element, "kind") != "uav") throw new InvalidDataException("Unsupported barrier kind.");
                instruction = new ExecutionPlanBarrier(element.Elements().Select(resource =>
                {
                    Check(resource, "ResourceRef", ["id"], []);
                    return Required(resource, "id");
                }));
                break;
            default:
                throw new InvalidDataException("Unknown execution plan instruction.");
        }
        var dependencies = Container(node, "DependsOn", "NodeRef").Select(reference =>
        {
            Check(reference, "NodeRef", ["id"], []);
            return Required(reference, "id");
        });
        var sources = Container(node, "Source", "NodeRef").Select(reference =>
        {
            Check(reference, "NodeRef", ["id"], []);
            return Required(reference, "id");
        });
        return new(Required(node, "id"), instruction, dependencies, sources);
    }

    internal static XElement Read(string xml)
    {
        ArgumentNullException.ThrowIfNull(xml);
        var settings = new XmlReaderSettings
        {
            DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null, MaxCharactersInDocument = 16 * 1024 * 1024,
            IgnoreComments = true, IgnoreProcessingInstructions = true,
        };
        using (var probe = XmlReader.Create(new StringReader(xml), settings))
        {
            var count = 0;
            while (probe.Read())
                if (probe.NodeType == XmlNodeType.Element && (++count > 100_000 || probe.Depth > 64))
                    throw new InvalidDataException("Execution plan XML exceeds element or nesting limits.");
        }
        using var reader = XmlReader.Create(new StringReader(xml), settings);
        return XDocument.Load(reader).Root ?? throw new InvalidDataException("Missing execution plan.");
    }

    internal static void Header(XElement root)
    {
        if (root.Name != "ExecutionPlan" || Required(root, "schema") != "sharp-inference" || Required(root, "version") != "1")
            throw new InvalidDataException("Unsupported execution plan schema/version.");
    }

    private static void Check(XElement element, string name, string[] attributes, string[] children)
    {
        if (element.Name != name || element.Attributes().Any(attribute => !attributes.Contains(attribute.Name.ToString())) ||
            element.Elements().Any(child => !children.Contains(child.Name.ToString())) ||
            element.Nodes().OfType<XText>().Any(text => !string.IsNullOrWhiteSpace(text.Value)))
            throw new InvalidDataException($"Unknown content in execution plan '{element.Name}'.");
    }
    private static XElement One(XElement owner, string name) =>
        OptionalOne(owner, name) ?? throw new InvalidDataException($"Missing '{name}' in '{owner.Name}'.");
    private static XElement? OptionalOne(XElement owner, string name)
    {
        var values = owner.Elements(name).ToArray();
        if (values.Length > 1) throw new InvalidDataException($"Duplicate '{name}' in '{owner.Name}'.");
        return values.SingleOrDefault();
    }
    private static IEnumerable<XElement> Container(XElement owner, string name, string child)
    {
        var container = One(owner, name);
        Check(container, name, [], [child]);
        return container.Elements(child);
    }
    private static string Required(XElement element, string name) =>
        (string?)element.Attribute(name) is { } value && !string.IsNullOrWhiteSpace(value) ? value :
            throw new InvalidDataException($"Missing '{name}' on '{element.Name}'.");
    private static int Integer(XElement element, string name) => int.Parse(Required(element, name), CultureInfo.InvariantCulture);
    private static ulong Unsigned(XElement element, string name) => ulong.Parse(Required(element, name), NumberStyles.None, CultureInfo.InvariantCulture);
    private static T EnumValue<T>(XElement element, string name) where T : struct, Enum =>
        Enum.TryParse<T>(Required(element, name), out var result) && Enum.IsDefined(result) ? result :
            throw new InvalidDataException($"Invalid '{name}' on '{element.Name}'.");
}
