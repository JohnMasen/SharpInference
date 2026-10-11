using System.Collections.ObjectModel;

namespace SharpInference.Graphs;

public enum ExecutionPlanTarget { Cpu, Direct3D12 }
public enum ExecutionPlanDefinitionKind { Function, Kernel, Orchestration }

public sealed record ExecutionPlanSlot(
    string Id, GraphResourceScope Scope, GraphResourceAccess Access, TensorDescriptor Tensor, string? BindingKey = null);
public sealed record ExecutionPlanParameter(string Name, GraphResourceAccess Access, TensorDescriptor Tensor);
public sealed record ExecutionPlanArgument(string Parameter, string Source, ulong ByteOffset = 0);
public sealed record ExecutionPlanGrid(uint X, uint Y = 1, uint Z = 1);

public static class ExecutionPlanGpuContracts
{
    public const uint CooperativeMatVecThreads = 64;

    public static ExecutionPlanGrid CooperativeMatVecGroups(ulong rows)
    {
        if (rows == 0 || rows > uint.MaxValue / CooperativeMatVecThreads)
            throw new ArgumentOutOfRangeException(nameof(rows), "Cooperative MatVec exceeds 32-bit thread indexing.");
        var y = checked((uint)((rows + 65534) / 65535));
        var x = checked((uint)((rows + y - 1) / y));
        if ((ulong)x * y * CooperativeMatVecThreads > uint.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(rows), "Padded cooperative grid exceeds 32-bit thread indexing.");
        return new(x, y);
    }
}
public sealed record ExecutionPlanStateEntry(string Name, string Slot);
public sealed record ExecutionPlanResourceBinding(
    string Resource, string Slot, TensorDescriptor Tensor, ulong ByteOffset = 0, bool IsView = false);
public sealed record ExecutionPlanIndexBound(string IndexPort, string TensorPort, int Axis, int MinimumIndex);
public sealed record ExecutionPlanPrecision(GraphElementType Arithmetic, GraphElementType Accumulator);

public abstract record ExecutionPlanInstruction;

public sealed record ExecutionPlanOperator : ExecutionPlanInstruction
{
    public ExecutionPlanOperator(PrecisionRequirement precision, Guid collection, string name,
        IEnumerable<ExecutionPlanArgument> arguments, IReadOnlyDictionary<string, string>? attributes = null,
        string? implementation = null, uint? cooperativeThreads = null,
        IEnumerable<ExecutionPlanIndexBound>? indexBounds = null,
        IReadOnlyDictionary<string, GraphResourceAccess>? parameterAccesses = null,
        ExecutionPlanPrecision? resolvedPrecision = null)
    {
        Precision = precision;
        Collection = collection;
        Name = name;
        Arguments = Array.AsReadOnly(arguments.ToArray());
        Attributes = new ReadOnlyDictionary<string, string>(
            new Dictionary<string, string>(attributes ?? new Dictionary<string, string>(), StringComparer.Ordinal));
        Implementation = implementation;
        CooperativeThreads = cooperativeThreads;
        IndexBounds = Array.AsReadOnly(indexBounds?.ToArray() ?? []);
        ParameterAccesses = new ReadOnlyDictionary<string, GraphResourceAccess>(
            new Dictionary<string, GraphResourceAccess>(parameterAccesses ??
                new Dictionary<string, GraphResourceAccess>(), StringComparer.Ordinal));
        ResolvedPrecision = resolvedPrecision;
    }

    public PrecisionRequirement Precision { get; }
    public Guid Collection { get; }
    public string Name { get; }
    public IReadOnlyList<ExecutionPlanArgument> Arguments { get; }
    public IReadOnlyDictionary<string, string> Attributes { get; }
    public string? Implementation { get; }
    public uint? CooperativeThreads { get; }
    public IReadOnlyList<ExecutionPlanIndexBound> IndexBounds { get; }
    public IReadOnlyDictionary<string, GraphResourceAccess> ParameterAccesses { get; }
    public ExecutionPlanPrecision? ResolvedPrecision { get; }
}

public sealed record ExecutionPlanCall : ExecutionPlanInstruction
{
    public ExecutionPlanCall(string definition, IEnumerable<ExecutionPlanArgument> arguments)
    {
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
    }
    public string Definition { get; }
    public IReadOnlyList<ExecutionPlanArgument> Arguments { get; }
}

public sealed record ExecutionPlanDispatch : ExecutionPlanInstruction
{
    public ExecutionPlanDispatch(string definition, IEnumerable<ExecutionPlanArgument> arguments, ExecutionPlanGrid groups)
    {
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
        Groups = groups;
    }
    public string Definition { get; }
    public IReadOnlyList<ExecutionPlanArgument> Arguments { get; }
    public ExecutionPlanGrid Groups { get; }
}

public sealed record ExecutionPlanBarrier : ExecutionPlanInstruction
{
    public ExecutionPlanBarrier(IEnumerable<string> resources) => Resources = Array.AsReadOnly(resources.ToArray());
    public IReadOnlyList<string> Resources { get; }
}

public sealed record ExecutionPlanNode
{
    public ExecutionPlanNode(string id, ExecutionPlanInstruction instruction, IEnumerable<string>? dependencies = null,
        IEnumerable<string>? sources = null)
    {
        Id = id;
        Instruction = instruction;
        Dependencies = Array.AsReadOnly(dependencies?.ToArray() ?? []);
        Sources = Array.AsReadOnly(sources?.ToArray() ?? []);
    }
    public string Id { get; }
    public ExecutionPlanInstruction Instruction { get; }
    public IReadOnlyList<string> Dependencies { get; }
    public IReadOnlyList<string> Sources { get; }
}

public sealed record ExecutionPlanDefinition
{
    public ExecutionPlanDefinition(string id, ExecutionPlanDefinitionKind kind,
        IEnumerable<ExecutionPlanParameter> parameters, IEnumerable<ExecutionPlanNode> nodes,
        ExecutionPlanGrid? threads = null)
    {
        Id = id;
        Kind = kind;
        Parameters = Array.AsReadOnly(parameters.ToArray());
        Nodes = Array.AsReadOnly(nodes.ToArray());
        Threads = threads;
    }
    public string Id { get; }
    public ExecutionPlanDefinitionKind Kind { get; }
    public IReadOnlyList<ExecutionPlanParameter> Parameters { get; }
    public IReadOnlyList<ExecutionPlanNode> Nodes { get; }
    public ExecutionPlanGrid? Threads { get; }
}

public sealed record ExecutionPlanEntry
{
    public ExecutionPlanEntry(string name, string definition, IEnumerable<ExecutionPlanArgument> arguments)
    {
        Name = name;
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
    }
    public string Name { get; }
    public string Definition { get; }
    public IReadOnlyList<ExecutionPlanArgument> Arguments { get; }
}

/// <summary>A target execution plan contains all scheduling and storage decisions, without VM dependencies.</summary>
public sealed class TargetExecutionPlan
{
    public TargetExecutionPlan(string name, string abi, ExecutionPlanTarget target,
        IEnumerable<ExecutionPlanSlot> slots, IEnumerable<ExecutionPlanDefinition> definitions,
        IEnumerable<ExecutionPlanEntry> entries, string stateSchema, int stateVersion,
        IEnumerable<ExecutionPlanStateEntry> state, LogicalGraph? bindings = null,
        IEnumerable<ExecutionPlanResourceBinding>? resourceBindings = null)
    {
        Name = name;
        Abi = abi;
        Target = target;
        Slots = Array.AsReadOnly(slots.ToArray());
        Definitions = Array.AsReadOnly(definitions.ToArray());
        Entries = Array.AsReadOnly(entries.ToArray());
        StateSchema = stateSchema;
        StateVersion = stateVersion;
        State = Array.AsReadOnly(state.ToArray());
        Bindings = bindings;
        ResourceBindings = Array.AsReadOnly(resourceBindings?.ToArray() ?? []);
        TargetExecutionPlanValidator.Validate(this);
    }

    public string Name { get; }
    public string Abi { get; }
    public ExecutionPlanTarget Target { get; }
    public IReadOnlyList<ExecutionPlanSlot> Slots { get; }
    public IReadOnlyList<ExecutionPlanDefinition> Definitions { get; }
    public IReadOnlyList<ExecutionPlanEntry> Entries { get; }
    public string StateSchema { get; }
    public int StateVersion { get; }
    public IReadOnlyList<ExecutionPlanStateEntry> State { get; }
    public LogicalGraph? Bindings { get; }
    public IReadOnlyList<ExecutionPlanResourceBinding> ResourceBindings { get; }

    public static LogicalGraph CreateBindings(IGraphBindingMetadata graph) => new(graph.Identity, graph.Model, graph.Resources,
        [new GraphRegion(new("bindings"), null, GraphRegionTypes.Graph, null, "Bindings", new Dictionary<string, string>())],
        [], graph.Inputs, graph.Outputs, graph.GraphState);
}
