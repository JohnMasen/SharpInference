using System.Diagnostics.CodeAnalysis;

namespace SharpInference.Vm;

public static class VmProgramValidator
{
    public static void Validate(VmProgram program)
    {
        ArgumentNullException.ThrowIfNull(program);
        RequireName(program.Name, "program");
        RequireName(program.Abi, "ABI");
        if (!Enum.IsDefined(program.Target))
            Fail("Invalid program target.");
        var slots = Unique(program.Slots, slot => slot.Id, "slot");
        var definitions = Unique(program.Definitions, definition => definition.Id, "definition");
        var entries = Unique(program.Entries, entry => entry.Name, "entry");
        if (entries.Count == 0)
            Fail("The program requires at least one entry.");

        foreach (var slot in program.Slots)
        {
            if (!Enum.IsDefined(slot.Scope) || !Enum.IsDefined(slot.Access) || slot.Tensor is null)
                Fail($"Slot '{slot.Id}' has an invalid descriptor.");
            if (slot.BindingKey is not null)
                RequireName(slot.BindingKey, "binding key");
        }
        RequireName(program.State.Schema, "state schema");
        if (program.State.Version <= 0)
            Fail("State schema version must be positive.");
        Unique(program.State.Entries, entry => entry.Name, "state entry");
        var stateSlots = new HashSet<string>(StringComparer.Ordinal);
        foreach (var state in program.State.Entries)
        {
            RequireName(state.Slot, "state slot");
            if (!slots.TryGetValue(state.Slot, out var slot) || slot.Scope != VmSlotScope.Session ||
                !stateSlots.Add(state.Slot))
                Fail($"State '{state.Name}' must reference a distinct session slot.");
        }

        foreach (var definition in program.Definitions)
        {
            if (!Enum.IsDefined(definition.Kind))
                Fail($"Definition '{definition.Id}' has an invalid kind.");
            if (definition.Kind == VmDefinitionKind.Kernel)
            {
                if (program.Target != VmTarget.Direct3D12 || definition.Threads is null)
                    Fail($"Kernel '{definition.Id}' requires Direct3D12 and a thread group.");
                ValidateGrid(definition.Threads!, threads: true);
            }
            else if (definition.Threads is not null)
                Fail($"Non-kernel '{definition.Id}' cannot declare a thread group.");

            var parameters = Unique(definition.Parameters, parameter => parameter.Name, "parameter");
            foreach (var parameter in parameters.Values)
                if (!Enum.IsDefined(parameter.Access) || parameter.Tensor is null)
                    Fail($"Parameter '{parameter.Name}' has an invalid descriptor.");
            Unique(definition.Nodes, node => node.Id, "node");
            var preceding = new HashSet<string>(StringComparer.Ordinal);
            foreach (var node in definition.Nodes)
            {
                if (node.Dependencies.Distinct(StringComparer.Ordinal).Count() != node.Dependencies.Count ||
                    node.Dependencies.Any(id => !preceding.Contains(id)))
                    Fail($"Node '{node.Id}' dependencies must reference distinct preceding nodes.");
                switch (node.Instruction)
                {
                    case VmOperator operation:
                        if (definition.Kind == VmDefinitionKind.Orchestration)
                            Fail($"Orchestration '{definition.Id}' cannot contain tensor operators.");
                        RequireName(operation.Operation, "operation");
                        if (operation.Version <= 0)
                            Fail($"Node '{node.Id}' has an invalid operation version.");
                        if (operation.ResolvedPrecision is { } precision &&
                            !SharpInference.Graphs.NumericTypeCompatibility.Satisfies(precision, operation.Precision))
                            Fail($"Node '{node.Id}' resolved precision does not satisfy its requirement.");
                        Unique(operation.Arguments, argument => argument.Parameter, "operator argument");
                        if (operation.ParameterAccesses.Count != 0 &&
                            (operation.ParameterAccesses.Count != operation.Arguments.Count ||
                            operation.ParameterAccesses.Any(pair => !Enum.IsDefined(pair.Value) ||
                                !operation.Arguments.Any(argument => argument.Parameter == pair.Key))))
                            Fail($"Node '{node.Id}' has invalid parameter access contracts.");
                        foreach (var bound in operation.IndexBounds)
                        {
                            var indexArgument = operation.Arguments.SingleOrDefault(argument => argument.Parameter == bound.IndexPort);
                            var tensorArgument = operation.Arguments.SingleOrDefault(argument => argument.Parameter == bound.TensorPort);
                            if (indexArgument is null || tensorArgument is null ||
                                !parameters.TryGetValue(indexArgument.Source, out var indexParameter) ||
                                indexParameter.Tensor.ElementType != VmElementType.Int32 || indexParameter.Tensor.ElementCount != 1 ||
                                !parameters.TryGetValue(tensorArgument.Source, out var tensorParameter) ||
                                bound.Axis < 0 || bound.Axis >= tensorParameter.Tensor.Dimensions.Count)
                                Fail($"Node '{node.Id}' has invalid index bounds.");
                        }
                        foreach (var argument in operation.Arguments)
                        {
                            RequireName(argument.Source, "operator source");
                            if (!parameters.TryGetValue(argument.Source, out var source) ||
                                argument.ByteOffset >= source.Tensor.ByteLength ||
                                argument.ByteOffset % (ulong)source.Tensor.ElementSize != 0)
                                Fail($"Operator '{node.Id}' has an invalid source view.");
                            if (operation.ParameterAccesses.TryGetValue(argument.Parameter, out var portAccess) &&
                                portAccess != SharpInference.Graphs.GraphResourceAccess.Read && source.Access == VmAccess.ReadOnly)
                                Fail($"Node '{node.Id}' writes a read-only parameter.");
                        }
                        foreach (var (name, value) in operation.Attributes)
                        {
                            RequireName(name, "attribute");
                            if (value is null)
                                Fail($"Operator '{node.Id}' has a null attribute.");
                        }
                        break;
                    case VmCall call:
                        var callee = Resolve(definitions, call.Definition);
                        var valid = definition.Kind == VmDefinitionKind.Orchestration
                            ? callee.Kind == VmDefinitionKind.Orchestration ||
                              program.Target == VmTarget.Cpu && callee.Kind == VmDefinitionKind.Function
                            : callee.Kind == VmDefinitionKind.Function;
                        if (!valid)
                            Fail($"Call '{node.Id}' has incompatible execution domains.");
                        ValidateArguments(call.Arguments, callee.Parameters,
                            parameters.ToDictionary(pair => pair.Key,
                                pair => (pair.Value.Tensor, pair.Value.Access), StringComparer.Ordinal));
                        break;
                    case VmDispatch dispatch:
                        if (program.Target != VmTarget.Direct3D12 ||
                            definition.Kind != VmDefinitionKind.Orchestration)
                            Fail($"Dispatch '{node.Id}' requires GPU orchestration.");
                        var kernel = Resolve(definitions, dispatch.Definition);
                        if (kernel.Kind != VmDefinitionKind.Kernel)
                            Fail($"Dispatch '{node.Id}' must target a kernel.");
                        ValidateGrid(dispatch.Groups, threads: false);
                        ValidateArguments(dispatch.Arguments, kernel.Parameters,
                            parameters.ToDictionary(pair => pair.Key,
                                pair => (pair.Value.Tensor, pair.Value.Access), StringComparer.Ordinal));
                        break;
                    case VmBarrier barrier:
                        if (program.Target != VmTarget.Direct3D12 ||
                            definition.Kind != VmDefinitionKind.Orchestration || barrier.Resources.Count == 0 ||
                            barrier.Resources.Distinct(StringComparer.Ordinal).Count() != barrier.Resources.Count ||
                            barrier.Resources.Any(resource => !parameters.TryGetValue(resource, out var parameter) ||
                                parameter.Access != VmAccess.ReadWrite))
                            Fail($"Barrier '{node.Id}' requires distinct writable GPU orchestration parameters.");
                        break;
                    default:
                        Fail($"Node '{node.Id}' contains an unknown instruction.");
                        break;
                }
                preceding.Add(node.Id);
            }
        }
        ValidateCalls(definitions);
        foreach (var entry in entries.Values)
        {
            var definition = Resolve(definitions, entry.Definition);
            ValidateArguments(entry.Arguments, definition.Parameters,
                slots.ToDictionary(pair => pair.Key,
                    pair => (pair.Value.Tensor, pair.Value.Access), StringComparer.Ordinal));
        }
    }

    private static void ValidateArguments(IReadOnlyList<VmArgument> arguments,
        IReadOnlyList<VmParameter> parameters,
        IReadOnlyDictionary<string, (VmTensor Tensor, VmAccess Access)> sources)
    {
        var bindings = Unique(arguments, argument => argument.Parameter, "argument");
        foreach (var argument in arguments)
            RequireName(argument.Source, "argument source");
        if (bindings.Count != parameters.Count)
            Fail("Call arguments must exactly match the definition parameters.");
        foreach (var parameter in parameters)
        {
            if (!bindings.TryGetValue(parameter.Name, out var argument) ||
                !sources.TryGetValue(argument.Source, out var source))
                Fail($"Parameter '{parameter.Name}' has no valid binding.");
            else if (parameter.Access == VmAccess.ReadWrite && source.Access != VmAccess.ReadWrite ||
                source.Tensor.ElementType != VmElementType.Byte &&
                    source.Tensor.ElementType != parameter.Tensor.ElementType ||
                argument.ByteOffset % (ulong)parameter.Tensor.ElementSize != 0 ||
                argument.ByteOffset > source.Tensor.ByteLength ||
                parameter.Tensor.ByteLength > source.Tensor.ByteLength - argument.ByteOffset)
                Fail($"Parameter '{parameter.Name}' binding violates access, type, alignment or capacity.");
        }
    }

    private static void ValidateGrid(VmThreadGroup grid, bool threads)
    {
        if (grid is null || grid.X == 0 || grid.Y == 0 || grid.Z == 0 ||
            (threads ? grid.X > 1024 || grid.Y > 1024 || grid.Z > 64 ||
                (ulong)grid.X * grid.Y * grid.Z > 1024 :
                grid.X > 65535 || grid.Y > 65535 || grid.Z > 65535))
            Fail(threads ? "Invalid D3D12 thread group." : "Invalid D3D12 dispatch dimensions.");
    }

    private static void ValidateCalls(IReadOnlyDictionary<string, VmDefinition> definitions)
    {
        var heights = new Dictionary<string, int>(StringComparer.Ordinal);
        var active = new HashSet<string>(StringComparer.Ordinal);
        foreach (var id in definitions.Keys)
            Visit(id, 0);

        void Visit(string id, int depth)
        {
            if (depth > 64)
                Fail("Definition nesting exceeds 64.");
            if (heights.ContainsKey(id)) return;
            if (!active.Add(id))
                Fail($"Recursive definition '{id}' is not supported.");
            var height = 0;
            foreach (var node in definitions[id].Nodes)
            {
                var child = node.Instruction switch
                {
                    VmCall call => call.Definition,
                    VmDispatch dispatch => dispatch.Definition,
                    _ => null,
                };
                if (child is not null)
                {
                    Visit(child, depth + 1);
                    height = Math.Max(height, heights[child] + 1);
                }
            }
            if (height > 64)
                Fail("Definition nesting exceeds 64.");
            active.Remove(id);
            heights.Add(id, height);
        }
    }

    private static VmDefinition Resolve(IReadOnlyDictionary<string, VmDefinition> definitions, string id) =>
        !string.IsNullOrWhiteSpace(id) && definitions.TryGetValue(id, out var definition) ? definition :
        throw new InvalidDataException($"Unknown definition '{id}'.");

    private static Dictionary<string, T> Unique<T>(IEnumerable<T> values, Func<T, string> key, string kind)
        where T : class
    {
        var result = new Dictionary<string, T>(StringComparer.Ordinal);
        foreach (var value in values)
        {
            if (value is null)
                Fail($"Null {kind}.");
            var id = key(value!);
            RequireName(id, kind);
            if (!result.TryAdd(id, value!))
                Fail($"Duplicate {kind} '{id}'.");
        }
        return result;
    }

    private static void RequireName(string value, string kind)
    {
        if (string.IsNullOrWhiteSpace(value))
            Fail($"A {kind} name is required.");
    }

    [DoesNotReturn]
    private static void Fail(string message) => throw new InvalidDataException(message);
}
