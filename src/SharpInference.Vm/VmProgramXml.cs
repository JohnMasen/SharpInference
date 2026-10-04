using System.Globalization;
using System.Xml;
using System.Xml.Linq;
using SharpInference.Instructions;
using SharpInference.Graphs;

namespace SharpInference.Vm;

public static class VmProgramXml
{
    public const int FormatVersion = 2;
    private const int MaximumCharacters = 16 * 1024 * 1024;

    public static string Serialize(VmProgram program)
    {
        VmProgramValidator.Validate(program);
        if (program.Definitions.SelectMany(definition => definition.Nodes).Select(node => node.Instruction)
            .OfType<VmOperator>().Any(operation => operation.Version != 1))
            throw new NotSupportedException("Instruction versions cannot be serialized; use a distinct IC GUID.");
        var root = new XElement("VmProgram",
            new XAttribute("version", FormatVersion), new XAttribute("name", program.Name),
            new XAttribute("abi", program.Abi), new XAttribute("target", program.Target),
            new XElement("Slots", program.Slots.Select(slot => new XElement("Slot",
                new XAttribute("id", slot.Id), new XAttribute("scope", slot.Scope),
                new XAttribute("access", slot.Access),
                slot.BindingKey is null ? null : new XAttribute("bindingKey", slot.BindingKey),
                Tensor(slot.Tensor)))),
            new XElement("State", new XAttribute("schema", program.State.Schema),
                new XAttribute("version", program.State.Version),
                program.State.Entries.Select(entry => new XElement("Entry",
                    new XAttribute("name", entry.Name), new XAttribute("slot", entry.Slot)))),
            new XElement("Definitions", program.Definitions.Select(definition => new XElement("Definition",
                new XAttribute("id", definition.Id), new XAttribute("kind", definition.Kind),
                definition.Threads is null ? null : Grid("Threads", definition.Threads),
                new XElement("Parameters", definition.Parameters.Select(parameter => new XElement("Parameter",
                    new XAttribute("name", parameter.Name), new XAttribute("access", parameter.Access),
                    Tensor(parameter.Tensor)))),
                new XElement("Nodes", definition.Nodes.Select(Node))))),
            new XElement("Entries", program.Entries.Select(entry => new XElement("Entry",
                new XAttribute("name", entry.Name), new XAttribute("definition", entry.Definition),
                Arguments(entry.Arguments)))));
        return root.ToString();
    }

    public static VmProgram Deserialize(string xml)
    {
        ArgumentNullException.ThrowIfNull(xml);
        var settings = new XmlReaderSettings
        {
            DtdProcessing = DtdProcessing.Prohibit,
            XmlResolver = null,
            MaxCharactersInDocument = MaximumCharacters,
            IgnoreComments = true,
            IgnoreProcessingInstructions = true,
        };
        try
        {
            using (var probe = XmlReader.Create(new StringReader(xml), settings))
            {
                var elements = 0;
                while (probe.Read())
                    if (probe.NodeType == XmlNodeType.Element && (++elements > 100_000 || probe.Depth > 64))
                        throw new InvalidDataException("VM XML exceeds element or nesting limits.");
            }
            using var reader = XmlReader.Create(new StringReader(xml), settings);
            var root = XDocument.Load(reader).Root ?? throw new InvalidDataException("Missing VM program.");
            Check(root, "VmProgram", ["version", "name", "abi", "target"],
                ["Slots", "State", "Definitions", "Entries"]);
            if (Int(root, "version") != FormatVersion)
                throw new InvalidDataException("Unsupported VM program XML version.");
            var slots = Container(root, "Slots", "Slot").Select(slot =>
            {
                Check(slot, "Slot", ["id", "scope", "access", "bindingKey"], ["Tensor"]);
                return new VmSlot(Required(slot, "id"), EnumValue<VmSlotScope>(slot, "scope"),
                    EnumValue<VmAccess>(slot, "access"), ReadTensor(One(slot, "Tensor")),
                    (string?)slot.Attribute("bindingKey"));
            }).ToArray();
            var state = One(root, "State");
            Check(state, "State", ["schema", "version"], ["Entry"]);
            var stateEntries = state.Elements().Select(entry =>
            {
                Check(entry, "Entry", ["name", "slot"], []);
                return new VmStateEntry(Required(entry, "name"), Required(entry, "slot"));
            }).ToArray();
            var definitions = Container(root, "Definitions", "Definition").Select(definition =>
            {
                Check(definition, "Definition", ["id", "kind"], ["Threads", "Parameters", "Nodes"]);
                var parameters = Container(definition, "Parameters", "Parameter").Select(parameter =>
                {
                    Check(parameter, "Parameter", ["name", "access"], ["Tensor"]);
                    return new VmParameter(Required(parameter, "name"),
                        EnumValue<VmAccess>(parameter, "access"), ReadTensor(One(parameter, "Tensor")));
                }).ToArray();
                var threads = OptionalOne(definition, "Threads");
                return new VmDefinition(Required(definition, "id"),
                    EnumValue<VmDefinitionKind>(definition, "kind"), parameters,
                    Container(definition, "Nodes", "Node").Select(ReadNode),
                    threads is null ? null : ReadGrid(threads, "Threads"));
            }).ToArray();
            var entries = Container(root, "Entries", "Entry").Select(entry =>
            {
                Check(entry, "Entry", ["name", "definition"], ["Arguments"]);
                return new VmEntry(Required(entry, "name"), Required(entry, "definition"), ReadArguments(entry));
            }).ToArray();
            return new VmProgram(Required(root, "name"), Required(root, "abi"),
                EnumValue<VmTarget>(root, "target"), slots, definitions, entries,
                new VmState(Required(state, "schema"), Int(state, "version"), stateEntries));
        }
        catch (Exception error) when (error is XmlException or ArgumentException or OverflowException or FormatException)
        {
            throw new InvalidDataException("Invalid VM program XML.", error);
        }
    }

    private static XElement Tensor(VmTensor tensor) => new("Tensor",
        new XAttribute("elementType", tensor.ElementType),
        tensor.Dimensions.Select(size => new XElement("Dimension", new XAttribute("size", size))));

    private static VmTensor ReadTensor(XElement element)
    {
        Check(element, "Tensor", ["elementType"], ["Dimension"]);
        return new VmTensor(EnumValue<VmElementType>(element, "elementType"), element.Elements().Select(dimension =>
        {
            Check(dimension, "Dimension", ["size"], []);
            return Int(dimension, "size");
        }));
    }

    private static XElement Arguments(IEnumerable<VmArgument> arguments) => new("Arguments",
        arguments.Select(argument => new XElement("Argument", new XAttribute("parameter", argument.Parameter),
            new XAttribute("source", argument.Source), new XAttribute("byteOffset", argument.ByteOffset))));

    private static VmArgument[] ReadArguments(XElement owner) =>
        Container(owner, "Arguments", "Argument").Select(argument =>
        {
            Check(argument, "Argument", ["parameter", "source", "byteOffset"], []);
            return new VmArgument(Required(argument, "parameter"), Required(argument, "source"),
                ULong(argument, "byteOffset"));
        }).ToArray();

    private static XElement Grid(string name, VmThreadGroup grid) => new(name,
        new XAttribute("x", grid.X), new XAttribute("y", grid.Y), new XAttribute("z", grid.Z));

    private static VmThreadGroup ReadGrid(XElement element, string name)
    {
        Check(element, name, ["x", "y", "z"], []);
        return new VmThreadGroup(checked((uint)ULong(element, "x")),
            checked((uint)ULong(element, "y")), checked((uint)ULong(element, "z")));
    }

    private static XElement Node(VmNode node)
    {
        XElement instruction = node.Instruction switch
        {
            VmOperator operation => new XElement("Operator", new XAttribute("name", operation.InstructionName),
                new XAttribute("collection", operation.InstructionCollectionId.ToString("D")), Arguments(operation.Arguments),
                new XElement("Attributes", operation.Attributes.OrderBy(pair => pair.Key, StringComparer.Ordinal)
                    .Select(pair => new XElement("Attribute",
                        new XAttribute("name", pair.Key), new XAttribute("value", pair.Value)))),
                new XElement("IndexBounds", operation.IndexBounds.Select(bound => new XElement("Bound",
                    new XAttribute("index", bound.IndexPort), new XAttribute("tensor", bound.TensorPort),
                    new XAttribute("axis", bound.Axis)))),
                new XElement("ParameterAccesses", operation.ParameterAccesses.OrderBy(pair => pair.Key, StringComparer.Ordinal)
                    .Select(pair => new XElement("Port", new XAttribute("name", pair.Key), new XAttribute("access", pair.Value))))),
            VmCall call => new XElement("Call", new XAttribute("definition", call.Definition),
                Arguments(call.Arguments)),
            VmDispatch dispatch => new XElement("Dispatch", new XAttribute("definition", dispatch.Definition),
                Arguments(dispatch.Arguments), Grid("Groups", dispatch.Groups)),
            VmBarrier barrier => new XElement("Barrier", barrier.Resources.Select(resource =>
                new XElement("Resource", new XAttribute("parameter", resource)))),
            _ => throw new InvalidDataException("Unknown VM instruction."),
        };
        return new XElement("Node", new XAttribute("id", node.Id),
            new XElement("Dependencies", node.Dependencies.Select(id => new XElement("NodeRef",
                new XAttribute("id", id)))), instruction);
    }

    private static VmNode ReadNode(XElement node)
    {
        Check(node, "Node", ["id"], ["Dependencies", "Operator", "Call", "Dispatch", "Barrier"]);
        var dependencies = Container(node, "Dependencies", "NodeRef").Select(reference =>
        {
            Check(reference, "NodeRef", ["id"], []);
            return Required(reference, "id");
        }).ToArray();
        var instructions = node.Elements().Where(element => element.Name != "Dependencies").ToArray();
        if (instructions.Length != 1)
            throw new InvalidDataException("A node must contain exactly one instruction.");
        var instruction = instructions[0];
        VmInstruction result;
        switch (instruction.Name.LocalName)
        {
            case "Operator":
                Check(instruction, "Operator", ["name", "collection"], ["Arguments", "Attributes", "IndexBounds", "ParameterAccesses"]);
                var attributes = new Dictionary<string, string>(StringComparer.Ordinal);
                foreach (var attribute in Container(instruction, "Attributes", "Attribute"))
                {
                    Check(attribute, "Attribute", ["name", "value"], []);
                    if (!attributes.TryAdd(Required(attribute, "name"), Required(attribute, "value")))
                        throw new InvalidDataException("Duplicate operator attribute.");
                }
                if (!Guid.TryParseExact(Required(instruction, "collection"), "D", out var collection))
                    throw new InvalidDataException("Instruction collection must be a canonical GUID.");
                result = new VmOperator(collection, Required(instruction, "name"),
                    ReadArguments(instruction), attributes,
                    OptionalOne(instruction, "IndexBounds")?.Elements().Select(bound =>
                    {
                        Check(bound, "Bound", ["index", "tensor", "axis"], []);
                        return new InstructionIndexBound(Required(bound, "index"), Required(bound, "tensor"), Int(bound, "axis"));
                    }), OptionalOne(instruction, "ParameterAccesses")?.Elements().ToDictionary(port =>
                    {
                        Check(port, "Port", ["name", "access"], []);
                        return Required(port, "name");
                    }, port => EnumValue<GraphResourceAccess>(port, "access"), StringComparer.Ordinal));
                break;
            case "Call":
                Check(instruction, "Call", ["definition"], ["Arguments"]);
                result = new VmCall(Required(instruction, "definition"), ReadArguments(instruction));
                break;
            case "Dispatch":
                Check(instruction, "Dispatch", ["definition"], ["Arguments", "Groups"]);
                result = new VmDispatch(Required(instruction, "definition"), ReadArguments(instruction),
                    ReadGrid(One(instruction, "Groups"), "Groups"));
                break;
            case "Barrier":
                Check(instruction, "Barrier", [], ["Resource"]);
                result = new VmBarrier(instruction.Elements().Select(resource =>
                {
                    Check(resource, "Resource", ["parameter"], []);
                    return Required(resource, "parameter");
                }));
                break;
            default:
                throw new InvalidDataException("Unknown VM instruction.");
        }
        return new VmNode(Required(node, "id"), result, dependencies);
    }

    private static void Check(XElement element, string name, string[] attributes, string[] children)
    {
        if (element.Name != name ||
            element.Attributes().Any(attribute => attribute.Name.Namespace != XNamespace.None ||
                !attributes.Contains(attribute.Name.LocalName, StringComparer.Ordinal)) ||
            element.Elements().Any(child => child.Name.Namespace != XNamespace.None ||
                !children.Contains(child.Name.LocalName, StringComparer.Ordinal)) ||
            element.Nodes().OfType<XText>().Any(text => !string.IsNullOrWhiteSpace(text.Value)))
            throw new InvalidDataException($"Unexpected XML content in '{name}'.");
    }

    private static IEnumerable<XElement> Container(XElement parent, string name, string child)
    {
        var element = One(parent, name);
        Check(element, name, [], [child]);
        return element.Elements();
    }

    private static XElement One(XElement parent, string name) => OptionalOne(parent, name) ??
        throw new InvalidDataException($"Missing '{name}'.");

    private static XElement? OptionalOne(XElement parent, string name)
    {
        var elements = parent.Elements(name).ToArray();
        if (elements.Length > 1)
            throw new InvalidDataException($"Duplicate '{name}'.");
        return elements.SingleOrDefault();
    }

    private static string Required(XElement element, string attribute) =>
        (string?)element.Attribute(attribute) ?? throw new InvalidDataException($"Missing '{attribute}'.");

    private static int Int(XElement element, string attribute) =>
        int.Parse(Required(element, attribute), NumberStyles.Integer, CultureInfo.InvariantCulture);

    private static ulong ULong(XElement element, string attribute) =>
        ulong.Parse(Required(element, attribute), NumberStyles.None, CultureInfo.InvariantCulture);

    private static T EnumValue<T>(XElement element, string attribute) where T : struct, Enum
    {
        var text = Required(element, attribute);
        if (!Enum.GetNames<T>().Contains(text, StringComparer.Ordinal))
            throw new InvalidDataException($"Invalid '{attribute}' value '{text}'.");
        return Enum.Parse<T>(text);
    }
}
