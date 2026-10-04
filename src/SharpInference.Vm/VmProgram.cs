using System.Collections.ObjectModel;
using SharpInference.Instructions;
using SharpInference.Graphs;

namespace SharpInference.Vm;

public enum VmTarget { Cpu, Direct3D12 }
public enum VmElementType { Byte, Int32, UInt32, Float16, Float32 }
public enum VmSlotScope { Global, Session, Local }
public enum VmAccess { ReadOnly, ReadWrite }
public enum VmDefinitionKind { Function, Kernel, Orchestration }

public sealed class VmTensor
{
    public VmTensor(VmElementType elementType, IEnumerable<int> dimensions)
    {
        if (!Enum.IsDefined(elementType))
            throw new ArgumentOutOfRangeException(nameof(elementType));
        ArgumentNullException.ThrowIfNull(dimensions);
        var shape = dimensions.ToArray();
        if (shape.Length == 0 || shape.Any(value => value <= 0))
            throw new ArgumentException("A tensor requires positive dimensions.", nameof(dimensions));
        ElementType = elementType;
        Dimensions = Array.AsReadOnly(shape);
        ElementCount = shape.Aggregate(1UL, (count, dimension) => checked(count * (ulong)dimension));
        ByteLength = checked(ElementCount * (ulong)ElementSize);
    }

    public VmElementType ElementType { get; }
    public IReadOnlyList<int> Dimensions { get; }
    public ulong ElementCount { get; }
    public ulong ByteLength { get; }
    public int ElementSize => ElementType switch
    {
        VmElementType.Byte => 1,
        VmElementType.Float16 => 2,
        _ => 4,
    };
}

public sealed record VmSlot(
    string Id, VmSlotScope Scope, VmAccess Access, VmTensor Tensor, string? BindingKey = null);

public sealed record VmParameter(string Name, VmAccess Access, VmTensor Tensor);
public sealed record VmArgument(string Parameter, string Source, ulong ByteOffset = 0);
public sealed record VmStateEntry(string Name, string Slot);
public sealed record VmThreadGroup(uint X, uint Y = 1, uint Z = 1);

public sealed class VmState
{
    public VmState(string schema, int version, IEnumerable<VmStateEntry> entries)
    {
        Schema = schema;
        Version = version;
        Entries = Array.AsReadOnly(entries.ToArray());
    }

    public string Schema { get; }
    public int Version { get; }
    public IReadOnlyList<VmStateEntry> Entries { get; }
}

public abstract class VmInstruction;

public sealed class VmOperator : VmInstruction
{
    public VmOperator(Guid instructionCollectionId, string instructionName, IEnumerable<VmArgument> arguments,
        IReadOnlyDictionary<string, string>? attributes = null, IEnumerable<InstructionIndexBound>? indexBounds = null,
        IReadOnlyDictionary<string, GraphResourceAccess>? parameterAccesses = null)
        : this(instructionName, 1, arguments, attributes)
    {
        InstructionCollectionId = instructionCollectionId;
        IsLegacy = false;
        IndexBounds = Array.AsReadOnly(indexBounds?.ToArray() ?? []);
        ParameterAccesses = new ReadOnlyDictionary<string, GraphResourceAccess>(
            parameterAccesses is null ? new Dictionary<string, GraphResourceAccess>(StringComparer.Ordinal) :
                new Dictionary<string, GraphResourceAccess>(parameterAccesses, StringComparer.Ordinal));
    }

    public VmOperator(string operation, int version, IEnumerable<VmArgument> arguments,
        IReadOnlyDictionary<string, string>? attributes = null)
    {
        Operation = operation;
        Version = version;
        Arguments = Array.AsReadOnly(arguments.ToArray());
        Attributes = new ReadOnlyDictionary<string, string>(
            attributes is null ? new Dictionary<string, string>(StringComparer.Ordinal) :
            new Dictionary<string, string>(attributes, StringComparer.Ordinal));
    }

    public string Operation { get; }
    public int Version { get; }
    public Guid InstructionCollectionId { get; } = InstructionCollectionIds.TierZeroFloat32;
    public string InstructionName => Operation;
    internal bool IsLegacy { get; private set; } = true;
    public IReadOnlyList<VmArgument> Arguments { get; }
    public IReadOnlyDictionary<string, string> Attributes { get; }
    public IReadOnlyList<InstructionIndexBound> IndexBounds { get; } = [];
    public IReadOnlyDictionary<string, GraphResourceAccess> ParameterAccesses { get; } =
        new ReadOnlyDictionary<string, GraphResourceAccess>(new Dictionary<string, GraphResourceAccess>());
}

public sealed class VmCall : VmInstruction
{
    public VmCall(string definition, IEnumerable<VmArgument> arguments)
    {
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
    }

    public string Definition { get; }
    public IReadOnlyList<VmArgument> Arguments { get; }
}

public sealed class VmDispatch : VmInstruction
{
    public VmDispatch(string definition, IEnumerable<VmArgument> arguments, VmThreadGroup groups)
    {
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
        Groups = groups;
    }

    public string Definition { get; }
    public IReadOnlyList<VmArgument> Arguments { get; }
    public VmThreadGroup Groups { get; }
}

public sealed class VmBarrier : VmInstruction
{
    public VmBarrier(IEnumerable<string> resources) => Resources = Array.AsReadOnly(resources.ToArray());
    public IReadOnlyList<string> Resources { get; }
}

public sealed class VmNode
{
    public VmNode(string id, VmInstruction instruction, IEnumerable<string>? dependencies = null)
    {
        Id = id;
        Instruction = instruction ?? throw new ArgumentNullException(nameof(instruction));
        Dependencies = Array.AsReadOnly(dependencies?.ToArray() ?? []);
    }

    public string Id { get; }
    public VmInstruction Instruction { get; }
    public IReadOnlyList<string> Dependencies { get; }
}

public sealed class VmDefinition
{
    public VmDefinition(string id, VmDefinitionKind kind, IEnumerable<VmParameter> parameters,
        IEnumerable<VmNode> nodes, VmThreadGroup? threads = null)
    {
        Id = id;
        Kind = kind;
        Parameters = Array.AsReadOnly(parameters.ToArray());
        Nodes = Array.AsReadOnly(nodes.Select(node =>
        {
            if (node.Instruction is not VmOperator { IsLegacy: true, Version: 1 } operation) return node;
            var output = operation.Arguments.SingleOrDefault(argument => argument.Parameter == "output");
            if (output is null || Parameters.SingleOrDefault(parameter => parameter.Name == output.Source)
                    ?.Tensor.ElementType != VmElementType.Float16) return node;
            return new VmNode(node.Id, new VmOperator(InstructionCollectionIds.TierZeroFloat16,
                operation.InstructionName, operation.Arguments, operation.Attributes), node.Dependencies);
        }).ToArray());
        Threads = threads;
    }

    public string Id { get; }
    public VmDefinitionKind Kind { get; }
    public IReadOnlyList<VmParameter> Parameters { get; }
    public IReadOnlyList<VmNode> Nodes { get; }
    public VmThreadGroup? Threads { get; }
}

public sealed class VmEntry
{
    public VmEntry(string name, string definition, IEnumerable<VmArgument> arguments)
    {
        Name = name;
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
    }

    public string Name { get; }
    public string Definition { get; }
    public IReadOnlyList<VmArgument> Arguments { get; }
}

public sealed class VmProgram
{
    public VmProgram(string name, string abi, VmTarget target, IEnumerable<VmSlot> slots,
        IEnumerable<VmDefinition> definitions, IEnumerable<VmEntry> entries, VmState state)
    {
        Name = name;
        Abi = abi;
        Target = target;
        Slots = Array.AsReadOnly(slots.ToArray());
        Definitions = Array.AsReadOnly(definitions.ToArray());
        Entries = Array.AsReadOnly(entries.ToArray());
        State = state ?? throw new ArgumentNullException(nameof(state));
        VmProgramValidator.Validate(this);
    }

    public string Name { get; }
    public string Abi { get; }
    public VmTarget Target { get; }
    public IReadOnlyList<VmSlot> Slots { get; }
    public IReadOnlyList<VmDefinition> Definitions { get; }
    public IReadOnlyList<VmEntry> Entries { get; }
    public VmState State { get; }
}
