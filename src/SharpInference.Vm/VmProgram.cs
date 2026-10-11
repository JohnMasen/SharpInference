using System.Collections.ObjectModel;
using SharpInference.Instructions;
using SharpInference.Graphs;

namespace SharpInference.Vm;

/// <summary>Identifies the execution backend of a VM program.</summary>
public enum VmTarget
{
    /// <summary>Executes on the CPU backend.</summary>
    Cpu,
    /// <summary>Executes on the Direct3D12 GPU backend.</summary>
    Direct3D12
}
/// <summary>Identifies the scalar representation stored in a VM tensor.</summary>
public enum VmElementType
{
    /// <summary>Stores unsigned 8-bit integers.</summary>
    Byte,
    /// <summary>Stores signed 32-bit integers.</summary>
    Int32,
    /// <summary>Stores unsigned 32-bit integers.</summary>
    UInt32,
    /// <summary>Stores IEEE 754 half-precision values.</summary>
    Float16,
    /// <summary>Stores IEEE 754 single-precision values.</summary>
    Float32
}
/// <summary>Defines how a VM resource is shared across programs and sessions.</summary>
public enum VmSlotScope
{
    /// <summary>Shares model-level storage by binding key.</summary>
    Global,
    /// <summary>Shares storage within one inference session.</summary>
    Session,
    /// <summary>Uses per-worker temporary storage.</summary>
    Local
}
/// <summary>Defines the access permitted through a VM resource binding.</summary>
public enum VmAccess
{
    /// <summary>Permits reads without writes.</summary>
    ReadOnly,
    /// <summary>Permits both reads and writes.</summary>
    ReadWrite
}
/// <summary>Identifies a callable body, GPU kernel, or orchestration definition.</summary>
public enum VmDefinitionKind
{
    /// <summary>Defines a callable computation body.</summary>
    Function,
    /// <summary>Defines a GPU kernel with thread-group dimensions.</summary>
    Kernel,
    /// <summary>Defines entry-point calls, dispatches, and barriers.</summary>
    Orchestration
}

/// <summary>Describes a dense tensor's element type, dimensions, and checked byte capacity.</summary>
public sealed class VmTensor
{
    /// <summary>Creates a dense tensor descriptor with positive dimensions and checked element and byte counts.</summary>
    /// <param name="elementType">The scalar storage representation of each tensor element.</param>
    /// <param name="dimensions">The ordered positive dimensions of the tensor.</param>
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

    /// <summary>Gets the scalar storage representation of each tensor element.</summary>
    public VmElementType ElementType { get; }
    /// <summary>Gets the ordered positive dimensions of the tensor.</summary>
    public IReadOnlyList<int> Dimensions { get; }
    /// <summary>Gets the checked product of all tensor dimensions.</summary>
    public ulong ElementCount { get; }
    /// <summary>Gets the total storage capacity in bytes.</summary>
    public ulong ByteLength { get; }
    /// <summary>Gets the storage size of one tensor element in bytes.</summary>
    public int ElementSize => ElementType switch
    {
        VmElementType.Byte => 1,
        VmElementType.Float16 => 2,
        _ => 4,
    };
}

/// <summary>Declares a program resource slot and its binding contract.</summary>
/// <param name="Id">The program slot identifier referenced by entry-point arguments.</param>
/// <param name="Scope">The resource sharing scope.</param>
/// <param name="Access">The permitted resource access.</param>
/// <param name="Tensor">The tensor descriptor defining element representation and shape.</param>
/// <param name="BindingKey">The shared-resource lookup key, or null to use the slot identifier.</param>
public sealed record VmSlot(
    string Id, VmSlotScope Scope, VmAccess Access, VmTensor Tensor, string? BindingKey = null);

/// <summary>Declares a tensor parameter accepted by a VM definition.</summary>
/// <param name="Name">The definition parameter name referenced by argument bindings.</param>
/// <param name="Access">The permitted resource access.</param>
/// <param name="Tensor">The tensor descriptor defining element representation and shape.</param>
public sealed record VmParameter(string Name, VmAccess Access, VmTensor Tensor);
/// <summary>Binds a definition parameter to a source resource and byte offset.</summary>
/// <param name="Parameter">The name of the destination definition parameter.</param>
/// <param name="Source">The source resource identity.</param>
/// <param name="ByteOffset">The byte offset within the source resource.</param>
public sealed record VmArgument(string Parameter, string Source, ulong ByteOffset = 0);
/// <summary>Associates a named persistent state value with a program slot.</summary>
/// <param name="Name">The canonical state-entry name used during serialization.</param>
/// <param name="Slot">The program slot containing the state value.</param>
public sealed record VmStateEntry(string Name, string Slot);
/// <summary>Specifies the dimensions of a GPU thread group or dispatch grid.</summary>
/// <param name="X">The number of threads or groups along the X axis.</param>
/// <param name="Y">The number of threads or groups along the Y axis.</param>
/// <param name="Z">The number of threads or groups along the Z axis.</param>
public sealed record VmThreadGroup(uint X, uint Y = 1, uint Z = 1);

/// <summary>Describes the versioned schema and ordered entries of persistent VM state.</summary>
public sealed class VmState
{
    /// <summary>Creates a versioned state schema with a snapshot of its ordered entries.</summary>
    /// <param name="schema">The persistent-state schema identifier.</param>
    /// <param name="version">The persistent-state schema version required for compatible state transfers.</param>
    /// <param name="entries">The ordered persistent-state entries.</param>
    public VmState(string schema, int version, IEnumerable<VmStateEntry> entries)
    {
        Schema = schema;
        Version = version;
        Entries = Array.AsReadOnly(entries.ToArray());
    }

    /// <summary>Gets the persistent-state schema identifier.</summary>
    public string Schema { get; }
    /// <summary>Gets the version of this serialized contract.</summary>
    public int Version { get; }
    /// <summary>Gets the ordered persistent-state entries.</summary>
    public IReadOnlyList<VmStateEntry> Entries { get; }
}

/// <summary>Represents an operation, call, dispatch, or barrier in a VM definition.</summary>
public abstract class VmInstruction;

/// <summary>Invokes a registered instruction with tensor arguments, attributes, and precision requirements.</summary>
public sealed class VmOperator : VmInstruction
{
    /// <summary>Creates a registered collection instruction invocation with optional resolved access, index, and precision contracts.</summary>
    /// <param name="precision">The minimum arithmetic and accumulator precision required by the invocation.</param>
    /// <param name="instructionCollectionId">The identity of the collection providing the instruction.</param>
    /// <param name="instructionName">The registered instruction name.</param>
    /// <param name="arguments">The ordered parameter-to-resource bindings.</param>
    /// <param name="attributes">The instruction attributes supplied to code generation.</param>
    /// <param name="indexBounds">The index-port bounds embedded in the instruction contract.</param>
    /// <param name="parameterAccesses">The resource access required by each bound instruction port.</param>
    /// <param name="resolvedPrecision">The bound internal instruction precision, or null before binding.</param>
    /// <param name="executionConfiguration">The selected execution implementation, or null when no configuration is required.</param>
    public VmOperator(PrecisionRequirement precision, Guid instructionCollectionId, string instructionName, IEnumerable<VmArgument> arguments,
        IReadOnlyDictionary<string, string>? attributes = null, IEnumerable<InstructionIndexBound>? indexBounds = null,
        IReadOnlyDictionary<string, GraphResourceAccess>? parameterAccesses = null,
        KernelPrecisionProfile? resolvedPrecision = null,
        InstructionExecutionConfiguration? executionConfiguration = null)
        : this(precision, instructionName, 1, arguments, attributes)
    {
        InstructionCollectionId = instructionCollectionId;
        IsLegacy = false;
        IndexBounds = Array.AsReadOnly(indexBounds?.ToArray() ?? []);
        ParameterAccesses = new ReadOnlyDictionary<string, GraphResourceAccess>(
            parameterAccesses is null ? new Dictionary<string, GraphResourceAccess>(StringComparer.Ordinal) :
                new Dictionary<string, GraphResourceAccess>(parameterAccesses, StringComparer.Ordinal));
        ResolvedPrecision = resolvedPrecision;
        ExecutionConfiguration = executionConfiguration;
    }

    /// <summary>Creates a legacy versioned operation invocation in the baseline FP32 instruction collection.</summary>
    /// <param name="precision">The minimum arithmetic and accumulator precision required by the invocation.</param>
    /// <param name="operation">The registered instruction name or legacy operation name.</param>
    /// <param name="version">The legacy operation contract version.</param>
    /// <param name="arguments">The ordered parameter-to-resource bindings.</param>
    /// <param name="attributes">The instruction attributes supplied to code generation.</param>
    public VmOperator(PrecisionRequirement precision, string operation, int version, IEnumerable<VmArgument> arguments,
        IReadOnlyDictionary<string, string>? attributes = null)
    {
        ArgumentNullException.ThrowIfNull(precision);
        Precision = precision;
        Operation = operation;
        Version = version;
        Arguments = Array.AsReadOnly(arguments.ToArray());
        Attributes = new ReadOnlyDictionary<string, string>(
            attributes is null ? new Dictionary<string, string>(StringComparer.Ordinal) :
            new Dictionary<string, string>(attributes, StringComparer.Ordinal));
    }

    /// <summary>Gets the registered instruction name or legacy operation name.</summary>
    public string Operation { get; }
    /// <summary>Gets the minimum arithmetic and accumulator precision required by the invocation.</summary>
    public PrecisionRequirement Precision { get; }
    /// <summary>Gets the bound instruction's internal precision, or null before contract resolution.</summary>
    public KernelPrecisionProfile? ResolvedPrecision { get; }
    /// <summary>Gets the selected execution implementation, or null when no configuration is required.</summary>
    public InstructionExecutionConfiguration? ExecutionConfiguration { get; }
    /// <summary>Gets the legacy operation contract version.</summary>
    public int Version { get; }
    /// <summary>Gets the identity of the collection providing the instruction.</summary>
    public Guid InstructionCollectionId { get; } = InstructionCollectionIds.TierZeroFloat32;
    /// <summary>Gets the registered instruction name.</summary>
    public string InstructionName => Operation;
    internal bool IsLegacy { get; private set; } = true;
    /// <summary>Gets the ordered parameter-to-resource bindings.</summary>
    public IReadOnlyList<VmArgument> Arguments { get; }
    /// <summary>Gets the instruction attributes supplied to code generation.</summary>
    public IReadOnlyDictionary<string, string> Attributes { get; }
    /// <summary>Gets the index-port bounds required by the instruction.</summary>
    public IReadOnlyList<InstructionIndexBound> IndexBounds { get; } = [];
    /// <summary>Gets the resource access required for each instruction port after binding.</summary>
    public IReadOnlyDictionary<string, GraphResourceAccess> ParameterAccesses { get; } =
        new ReadOnlyDictionary<string, GraphResourceAccess>(new Dictionary<string, GraphResourceAccess>());
}

/// <summary>Invokes another VM definition with parameter bindings.</summary>
public sealed class VmCall : VmInstruction
{
    /// <summary>Creates a definition call with a snapshot of its argument bindings.</summary>
    /// <param name="definition">The identifier of the VM definition to invoke.</param>
    /// <param name="arguments">The ordered parameter-to-resource bindings.</param>
    public VmCall(string definition, IEnumerable<VmArgument> arguments)
    {
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
    }

    /// <summary>Gets the identifier of the VM definition to invoke.</summary>
    public string Definition { get; }
    /// <summary>Gets the ordered parameter-to-resource bindings.</summary>
    public IReadOnlyList<VmArgument> Arguments { get; }
}

/// <summary>Launches a VM kernel over a GPU dispatch grid.</summary>
public sealed class VmDispatch : VmInstruction
{
    /// <summary>Creates a kernel dispatch with argument bindings and a dispatch grid.</summary>
    /// <param name="definition">The identifier of the VM definition to invoke.</param>
    /// <param name="arguments">The ordered parameter-to-resource bindings.</param>
    /// <param name="groups">The number of GPU thread groups to dispatch along each axis.</param>
    public VmDispatch(string definition, IEnumerable<VmArgument> arguments, VmThreadGroup groups)
    {
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
        Groups = groups;
    }

    /// <summary>Gets the identifier of the VM definition to invoke.</summary>
    public string Definition { get; }
    /// <summary>Gets the ordered parameter-to-resource bindings.</summary>
    public IReadOnlyList<VmArgument> Arguments { get; }
    /// <summary>Gets the number of GPU thread groups to dispatch along each axis.</summary>
    public VmThreadGroup Groups { get; }
}

/// <summary>Synchronizes access to the specified VM resources.</summary>
public sealed class VmBarrier : VmInstruction
{
    /// <summary>Creates a barrier for a snapshot of the specified resource identifiers.</summary>
    /// <param name="resources">The resources whose accesses must be synchronized.</param>
    public VmBarrier(IEnumerable<string> resources) => Resources = Array.AsReadOnly(resources.ToArray());
    /// <summary>Gets the resources whose accesses must be synchronized.</summary>
    public IReadOnlyList<string> Resources { get; }
}

/// <summary>Associates a VM instruction with an identifier and execution dependencies.</summary>
public sealed class VmNode
{
    /// <summary>Creates an instruction node with an identifier and optional dependency identifiers.</summary>
    /// <param name="id">The node identifier referenced by dependency edges within the definition.</param>
    /// <param name="instruction">The instruction executed by this node.</param>
    /// <param name="dependencies">The identifiers of nodes that must complete before this node.</param>
    public VmNode(string id, VmInstruction instruction, IEnumerable<string>? dependencies = null)
    {
        Id = id;
        Instruction = instruction ?? throw new ArgumentNullException(nameof(instruction));
        Dependencies = Array.AsReadOnly(dependencies?.ToArray() ?? []);
    }

    /// <summary>Gets the node identifier referenced by dependency edges within the definition.</summary>
    public string Id { get; }
    /// <summary>Gets the instruction executed by this node.</summary>
    public VmInstruction Instruction { get; }
    /// <summary>Gets the identifiers of nodes that must complete before this node.</summary>
    public IReadOnlyList<string> Dependencies { get; }
}

/// <summary>Describes a callable VM body, its parameters, and its dependency-ordered nodes.</summary>
public sealed class VmDefinition
{
    /// <summary>Creates a definition and snapshots its parameters and nodes, adapting legacy FP16-output operators.</summary>
    /// <param name="id">The definition identifier referenced by calls, dispatches, and entry points.</param>
    /// <param name="kind">The execution role of this definition.</param>
    /// <param name="parameters">The tensor parameters accepted by the definition.</param>
    /// <param name="nodes">The instruction nodes forming the definition body.</param>
    /// <param name="threads">The GPU thread-group dimensions, or null for non-kernel definitions.</param>
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
            return new VmNode(node.Id, new VmOperator(operation.Precision, InstructionCollectionIds.TierZeroFloat16,
                operation.InstructionName, operation.Arguments, operation.Attributes), node.Dependencies);
        }).ToArray());
        Threads = threads;
    }

    /// <summary>Gets the definition identifier referenced by calls, dispatches, and entry points.</summary>
    public string Id { get; }
    /// <summary>Gets the execution role of this definition.</summary>
    public VmDefinitionKind Kind { get; }
    /// <summary>Gets the tensor parameters accepted by the definition.</summary>
    public IReadOnlyList<VmParameter> Parameters { get; }
    /// <summary>Gets the instruction nodes forming the definition body.</summary>
    public IReadOnlyList<VmNode> Nodes { get; }
    /// <summary>Gets the GPU thread-group dimensions, or null for non-kernel definitions.</summary>
    public VmThreadGroup? Threads { get; }
}

/// <summary>Exposes a named VM entry point and its program-slot argument bindings.</summary>
public sealed class VmEntry
{
    /// <summary>Creates a named entry point with a snapshot of its program-slot bindings.</summary>
    /// <param name="name">The externally callable entry-point name.</param>
    /// <param name="definition">The identifier of the VM definition to invoke.</param>
    /// <param name="arguments">The ordered parameter-to-resource bindings.</param>
    public VmEntry(string name, string definition, IEnumerable<VmArgument> arguments)
    {
        Name = name;
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
    }

    /// <summary>Gets the externally callable entry-point name.</summary>
    public string Name { get; }
    /// <summary>Gets the identifier of the VM definition to invoke.</summary>
    public string Definition { get; }
    /// <summary>Gets the ordered parameter-to-resource bindings.</summary>
    public IReadOnlyList<VmArgument> Arguments { get; }
}

/// <summary>Contains a validated VM artifact with resource, instruction, entry-point, and state contracts.</summary>
public sealed class VmProgram
{
    /// <summary>Snapshots the supplied artifact contracts and validates the resulting VM program.</summary>
    /// <param name="name">The VM artifact name.</param>
    /// <param name="abi">The model/application binary interface identifier.</param>
    /// <param name="target">The CPU or Direct3D12 VM execution backend.</param>
    /// <param name="slots">The declared program resources in executable buffer order.</param>
    /// <param name="definitions">The callable definitions contained in the program.</param>
    /// <param name="entries">The named entry points and their program-slot argument bindings.</param>
    /// <param name="state">The persistent-state schema and slot mapping.</param>
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

    /// <summary>Gets the VM artifact name.</summary>
    public string Name { get; }
    /// <summary>Gets the model/application binary interface identifier.</summary>
    public string Abi { get; }
    /// <summary>Gets the execution architecture required by this contract.</summary>
    public VmTarget Target { get; }
    /// <summary>Gets the resource slots in executable buffer order.</summary>
    public IReadOnlyList<VmSlot> Slots { get; }
    /// <summary>Gets the callable definitions contained in the program.</summary>
    public IReadOnlyList<VmDefinition> Definitions { get; }
    /// <summary>Gets the named entry points and their program-slot argument bindings.</summary>
    public IReadOnlyList<VmEntry> Entries { get; }
    /// <summary>Gets the persistent-state schema and slot mapping.</summary>
    public VmState State { get; }
}
