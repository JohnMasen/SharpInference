using System.Collections.ObjectModel;

namespace SharpInference.Graphs;

/// <summary>Identifies the execution target of a fully lowered plan.</summary>
public enum ExecutionPlanTarget
{
    /// <summary>Executes the plan on the CPU.</summary>
    Cpu,
    /// <summary>Executes the plan using Direct3D 12.</summary>
    Direct3D12
}
/// <summary>Identifies whether a definition is a function, GPU kernel, or orchestration body.</summary>
public enum ExecutionPlanDefinitionKind
{
    /// <summary>A callable CPU function definition.</summary>
    Function,
    /// <summary>A GPU kernel definition.</summary>
    Kernel,
    /// <summary>A definition coordinating calls, dispatches, and barriers.</summary>
    Orchestration
}

/// <summary>Describes a named storage slot and its tensor, scope, and access contract.</summary>
/// <param name="Id">The storage slot identifier.</param>
/// <param name="Scope">The resource sharing scope.</param>
/// <param name="Access">The permitted read and write access.</param>
/// <param name="Tensor">The tensor element type, dimensions, and layout.</param>
/// <param name="BindingKey">The external resource binding key, or null for an unbound resource.</param>
public sealed record ExecutionPlanSlot(
    string Id, GraphResourceScope Scope, GraphResourceAccess Access, TensorDescriptor Tensor, string? BindingKey = null);
/// <summary>Describes a named tensor parameter and its access contract.</summary>
/// <param name="Name">The definition parameter name.</param>
/// <param name="Access">The permitted read and write access.</param>
/// <param name="Tensor">The tensor element type, dimensions, and layout.</param>
public sealed record ExecutionPlanParameter(string Name, GraphResourceAccess Access, TensorDescriptor Tensor);
/// <summary>Binds a definition parameter to a source slot or parameter at a byte offset.</summary>
/// <param name="Parameter">The target parameter name.</param>
/// <param name="Source">The source storage slot or enclosing definition parameter name.</param>
/// <param name="ByteOffset">The byte offset into the backing source.</param>
public sealed record ExecutionPlanArgument(string Parameter, string Source, ulong ByteOffset = 0);
/// <summary>Describes a three-dimensional GPU group or thread grid.</summary>
/// <param name="X">The grid size along the X axis.</param>
/// <param name="Y">The grid size along the Y axis.</param>
/// <param name="Z">The grid size along the Z axis.</param>
public sealed record ExecutionPlanGrid(uint X, uint Y = 1, uint Z = 1);

/// <summary>Defines cooperative GPU matrix-vector thread and dispatch-grid contracts.</summary>
public static class ExecutionPlanGpuContracts
{
    /// <summary>The number of cooperative GPU threads used per matrix-vector row.</summary>
    public const uint CooperativeMatVecThreads = 64;

    /// <summary>Computes a two-dimensional cooperative matrix-vector dispatch grid within 32-bit thread indexing limits.</summary>
    /// <param name="rows">The number of matrix rows.</param>
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
/// <summary>Maps a named state entry to an execution-plan storage slot.</summary>
/// <param name="Name">The state entry name.</param>
/// <param name="Slot">The execution-plan storage slot identifier.</param>
public sealed record ExecutionPlanStateEntry(string Name, string Slot);
/// <summary>Maps a graph resource or view to a plan slot and byte offset.</summary>
/// <param name="Resource">The graph resource identifier.</param>
/// <param name="Slot">The execution-plan storage slot identifier.</param>
/// <param name="Tensor">The tensor element type, dimensions, and layout.</param>
/// <param name="ByteOffset">The byte offset into the backing source.</param>
/// <param name="IsView">Whether the resource binding represents a tensor view.</param>
public sealed record ExecutionPlanResourceBinding(
    string Resource, string Slot, TensorDescriptor Tensor, ulong ByteOffset = 0, bool IsView = false);
/// <summary>Specifies a lower index bound for an index port relative to a tensor axis.</summary>
/// <param name="IndexPort">The parameter supplying tensor indices.</param>
/// <param name="TensorPort">The tensor parameter whose axis constrains the indices.</param>
/// <param name="Axis">The zero-based tensor axis.</param>
/// <param name="MinimumIndex">The inclusive minimum allowed index.</param>
public sealed record ExecutionPlanIndexBound(string IndexPort, string TensorPort, int Axis, int MinimumIndex);
/// <summary>Describes resolved arithmetic and accumulation element types for a plan operator.</summary>
/// <param name="Arithmetic">The arithmetic element type.</param>
/// <param name="Accumulator">The accumulation element type.</param>
public sealed record ExecutionPlanPrecision(GraphElementType Arithmetic, GraphElementType Accumulator);

/// <summary>Represents an operator, call, dispatch, or barrier in an execution plan.</summary>
public abstract record ExecutionPlanInstruction;

/// <summary>Describes a backend operator invocation and its precision, arguments, and implementation metadata.</summary>
public sealed record ExecutionPlanOperator : ExecutionPlanInstruction
{
    /// <summary>Creates an operator instruction with argument bindings, precision, and optional backend metadata.</summary>
    /// <param name="precision">The minimum arithmetic and accumulation precision requirement.</param>
    /// <param name="collection">The operator collection identifier.</param>
    /// <param name="name">The backend operator name.</param>
    /// <param name="arguments">The parameter-to-source argument bindings.</param>
    /// <param name="attributes">The operation or model attributes keyed by name.</param>
    /// <param name="implementation">The optional selected backend implementation identifier.</param>
    /// <param name="cooperativeThreads">The optional cooperative GPU thread count.</param>
    /// <param name="indexBounds">The index-port bounds required by the operator.</param>
    /// <param name="parameterAccesses">The access contracts keyed by operator parameter name.</param>
    /// <param name="resolvedPrecision">The optional resolved arithmetic and accumulation precision.</param>
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

    /// <summary>Gets the minimum arithmetic and accumulation precision requirement.</summary>
    public PrecisionRequirement Precision { get; }
    /// <summary>Gets the backend operator collection identifier.</summary>
    public Guid Collection { get; }
    /// <summary>Gets the name of this element.</summary>
    public string Name { get; }
    /// <summary>Gets the ordered argument bindings.</summary>
    public IReadOnlyList<ExecutionPlanArgument> Arguments { get; }
    /// <summary>Gets attributes keyed by name.</summary>
    public IReadOnlyDictionary<string, string> Attributes { get; }
    /// <summary>Gets the optional selected backend implementation identifier.</summary>
    public string? Implementation { get; }
    /// <summary>Gets the optional cooperative GPU thread count.</summary>
    public uint? CooperativeThreads { get; }
    /// <summary>Gets the index-port bounds required by the operator.</summary>
    public IReadOnlyList<ExecutionPlanIndexBound> IndexBounds { get; }
    /// <summary>Gets access contracts keyed by operator parameter name.</summary>
    public IReadOnlyDictionary<string, GraphResourceAccess> ParameterAccesses { get; }
    /// <summary>Gets the optional resolved arithmetic and accumulation element types.</summary>
    public ExecutionPlanPrecision? ResolvedPrecision { get; }
}

/// <summary>Calls another execution-plan definition with explicit arguments.</summary>
public sealed record ExecutionPlanCall : ExecutionPlanInstruction
{
    /// <summary>Creates a definition call with a snapshot of its argument bindings.</summary>
    /// <param name="definition">The reusable definition identifier.</param>
    /// <param name="arguments">The parameter-to-source argument bindings.</param>
    public ExecutionPlanCall(string definition, IEnumerable<ExecutionPlanArgument> arguments)
    {
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
    }
    /// <summary>Gets the identifier of the invoked definition.</summary>
    public string Definition { get; }
    /// <summary>Gets the ordered argument bindings.</summary>
    public IReadOnlyList<ExecutionPlanArgument> Arguments { get; }
}

/// <summary>Dispatches a GPU definition using explicit arguments and group dimensions.</summary>
public sealed record ExecutionPlanDispatch : ExecutionPlanInstruction
{
    /// <summary>Creates a GPU dispatch with argument bindings and group dimensions.</summary>
    /// <param name="definition">The reusable definition identifier.</param>
    /// <param name="arguments">The parameter-to-source argument bindings.</param>
    /// <param name="groups">The GPU dispatch group dimensions.</param>
    public ExecutionPlanDispatch(string definition, IEnumerable<ExecutionPlanArgument> arguments, ExecutionPlanGrid groups)
    {
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
        Groups = groups;
    }
    /// <summary>Gets the identifier of the invoked definition.</summary>
    public string Definition { get; }
    /// <summary>Gets the ordered argument bindings.</summary>
    public IReadOnlyList<ExecutionPlanArgument> Arguments { get; }
    /// <summary>Gets the GPU dispatch group dimensions.</summary>
    public ExecutionPlanGrid Groups { get; }
}

/// <summary>Synchronizes accesses to a set of execution-plan resources.</summary>
public sealed record ExecutionPlanBarrier : ExecutionPlanInstruction
{
    /// <summary>Creates a barrier with a snapshot of the resource names to synchronize.</summary>
    /// <param name="resources">The names of the resources synchronized by the barrier.</param>
    public ExecutionPlanBarrier(IEnumerable<string> resources) => Resources = Array.AsReadOnly(resources.ToArray());
    /// <summary>Gets the resource names synchronized by the barrier.</summary>
    public IReadOnlyList<string> Resources { get; }
}

/// <summary>Contains an instruction, scheduling dependencies, and source-node provenance.</summary>
public sealed record ExecutionPlanNode
{
    /// <summary>Creates an instruction node with dependency and source-node snapshots.</summary>
    /// <param name="id">The instruction node identifier within its definition.</param>
    /// <param name="instruction">The instruction executed by the plan node.</param>
    /// <param name="dependencies">The predecessor node identifiers.</param>
    /// <param name="sources">The source-node identifiers retained for provenance.</param>
    public ExecutionPlanNode(string id, ExecutionPlanInstruction instruction, IEnumerable<string>? dependencies = null,
        IEnumerable<string>? sources = null)
    {
        Id = id;
        Instruction = instruction;
        Dependencies = Array.AsReadOnly(dependencies?.ToArray() ?? []);
        Sources = Array.AsReadOnly(sources?.ToArray() ?? []);
    }
    /// <summary>Gets the identifier within the containing graph or plan.</summary>
    public string Id { get; }
    /// <summary>Gets the instruction executed by this plan node.</summary>
    public ExecutionPlanInstruction Instruction { get; }
    /// <summary>Gets the predecessor node identifiers.</summary>
    public IReadOnlyList<string> Dependencies { get; }
    /// <summary>Gets source-node identifiers retained for provenance.</summary>
    public IReadOnlyList<string> Sources { get; }
}

/// <summary>Contains a function, kernel, or orchestration body with parameters and ordered nodes.</summary>
public sealed record ExecutionPlanDefinition
{
    /// <summary>Creates a definition with parameter and node snapshots and optional GPU thread dimensions.</summary>
    /// <param name="id">The reusable definition identifier.</param>
    /// <param name="kind">The resource role or execution definition category.</param>
    /// <param name="parameters">The definition&apos;s tensor parameters.</param>
    /// <param name="nodes">The graph or plan operation nodes.</param>
    /// <param name="threads">The optional GPU thread-group dimensions.</param>
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
    /// <summary>Gets the identifier within the containing graph or plan.</summary>
    public string Id { get; }
    /// <summary>Gets the execution definition category.</summary>
    public ExecutionPlanDefinitionKind Kind { get; }
    /// <summary>Gets the definition&apos;s tensor parameters.</summary>
    public IReadOnlyList<ExecutionPlanParameter> Parameters { get; }
    /// <summary>Gets the operation nodes in execution order.</summary>
    public IReadOnlyList<ExecutionPlanNode> Nodes { get; }
    /// <summary>Gets the optional GPU thread-group dimensions.</summary>
    public ExecutionPlanGrid? Threads { get; }
}

/// <summary>Exposes a named entry point that invokes a definition with explicit arguments.</summary>
public sealed record ExecutionPlanEntry
{
    /// <summary>Creates a named entry point invoking a definition with argument bindings.</summary>
    /// <param name="name">The public entry point name.</param>
    /// <param name="definition">The reusable definition identifier.</param>
    /// <param name="arguments">The parameter-to-source argument bindings.</param>
    public ExecutionPlanEntry(string name, string definition, IEnumerable<ExecutionPlanArgument> arguments)
    {
        Name = name;
        Definition = definition;
        Arguments = Array.AsReadOnly(arguments.ToArray());
    }
    /// <summary>Gets the name of this element.</summary>
    public string Name { get; }
    /// <summary>Gets the identifier of the invoked definition.</summary>
    public string Definition { get; }
    /// <summary>Gets the ordered argument bindings.</summary>
    public IReadOnlyList<ExecutionPlanArgument> Arguments { get; }
}

/// <summary>A target execution plan contains all scheduling and storage decisions, without VM dependencies.</summary>
public sealed class TargetExecutionPlan
{
    /// <summary>Creates and validates a target plan with explicit storage, definitions, entry points, and state.</summary>
    /// <param name="name">The name identifying this graph element.</param>
    /// <param name="abi">The execution plan ABI identifier.</param>
    /// <param name="target">The execution target of the plan.</param>
    /// <param name="slots">The execution-plan storage slots.</param>
    /// <param name="definitions">The reusable execution-plan definitions.</param>
    /// <param name="entries">The ordered named entries.</param>
    /// <param name="stateSchema">The execution plan state schema name.</param>
    /// <param name="stateVersion">The execution plan state schema version.</param>
    /// <param name="state">The named state-to-slot bindings.</param>
    /// <param name="bindings">The optional logical graph containing external resource binding metadata.</param>
    /// <param name="resourceBindings">The mappings from graph resources to plan storage.</param>
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

    /// <summary>Gets the name of this element.</summary>
    public string Name { get; }
    /// <summary>Gets the execution plan ABI identifier.</summary>
    public string Abi { get; }
    /// <summary>Gets the execution target of the plan.</summary>
    public ExecutionPlanTarget Target { get; }
    /// <summary>Gets the ordered storage slots.</summary>
    public IReadOnlyList<ExecutionPlanSlot> Slots { get; }
    /// <summary>Gets the reusable execution-plan definitions.</summary>
    public IReadOnlyList<ExecutionPlanDefinition> Definitions { get; }
    /// <summary>Gets the ordered named entries.</summary>
    public IReadOnlyList<ExecutionPlanEntry> Entries { get; }
    /// <summary>Gets the execution plan state schema name.</summary>
    public string StateSchema { get; }
    /// <summary>Gets the execution plan state schema version.</summary>
    public int StateVersion { get; }
    /// <summary>Gets the named execution-plan state entries.</summary>
    public IReadOnlyList<ExecutionPlanStateEntry> State { get; }
    /// <summary>Gets the optional logical graph containing external resource binding metadata.</summary>
    public LogicalGraph? Bindings { get; }
    /// <summary>Gets the mappings from graph resources and views to plan storage slots.</summary>
    public IReadOnlyList<ExecutionPlanResourceBinding> ResourceBindings { get; }

    /// <summary>Creates a binding-only logical graph containing the supplied identity, resources, IO, and state.</summary>
    /// <param name="graph">The graph supplying operations and resource binding metadata.</param>
    public static LogicalGraph CreateBindings(IGraphBindingMetadata graph) => new(graph.Identity, graph.Model, graph.Resources,
        [new GraphRegion(new("bindings"), null, GraphRegionTypes.Graph, null, "Bindings", new Dictionary<string, string>())],
        [], graph.Inputs, graph.Outputs, graph.GraphState);
}
