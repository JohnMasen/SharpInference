using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Instructions;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Vm.Optimization;

/// <summary>Configures the stable VM artifact identity emitted from an execution graph program.</summary>
public sealed record VmExecutionGraphEmitterOptions(
    string? Name = null,
    string Abi = "vm:execution-graph:v1",
    string Entry = "forward",
    string StateSchema = "execution-graph",
    int StateVersion = 1,
    uint DefaultGpuThreads = 64);

/// <summary>Emits a VM program directly from the new execution-graph IR without legacy plan adapters.</summary>
public sealed class VmExecutionGraphEmitter
{
    private readonly VmTarget target;
    private readonly InstructionTarget instructionTarget;
    private readonly InstructionRegistry registry;
    private readonly IReadOnlyDictionary<OperationId, GraphInstructionBinding> bindings;

    public VmExecutionGraphEmitter(
        VmTarget target,
        IEnumerable<IInstructionCollectionProvider> instructionCollections)
    {
        this.target = target;
        instructionTarget = target switch
        {
            VmTarget.Cpu => InstructionTarget.Cpu,
            VmTarget.Direct3D12 => InstructionTarget.Direct3D12,
            _ => throw new NotSupportedException($"No VM emitter is installed for target '{target}'."),
        };
        registry = new InstructionRegistry(instructionCollections);
        bindings = registry.QueryGraphInstructionBindings()
            .Where(binding => binding.Target == instructionTarget)
            .ToDictionary(binding => binding.Operation);
    }

    /// <summary>Gets the instruction collections used to select concrete backend operations.</summary>
    public IInstructionCollectionProvider InstructionCollections => registry;

    /// <summary>Emits and validates a CPU or Direct3D12 VM program.</summary>
    public VmProgram Emit(
        ExecutionGraphProgram program,
        VmExecutionGraphEmitterOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(program);
        options ??= new();
        if (string.IsNullOrWhiteSpace(options.Abi) ||
            string.IsNullOrWhiteSpace(options.Entry) ||
            string.IsNullOrWhiteSpace(options.StateSchema) ||
            options.StateVersion <= 0)
        {
            throw new ArgumentException("VM artifact, entry, and state identities must be nonempty and versioned.", nameof(options));
        }

        if (options.DefaultGpuThreads is 0 or > 1024)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "GPU thread count must be between 1 and 1024.");
        }

        var graphs = program.Graphs.ToDictionary(graph => graph.Id);
        var frames = new Dictionary<ExecutionGraphId, Frame>();
        Frame BuildFrame(ExecutionGraphId id)
        {
            if (frames.TryGetValue(id, out var existing))
            {
                return existing;
            }

            var graph = graphs[id];
            var parameters = graph.Resources.Select(resource =>
                new FrameParameter(resource.Id.Value, resource.Tensor, resource.Kind, resource.BindingKey)).ToList();
            foreach (var call in StableOrder(graph).OfType<ExecutionGraphCallNode>())
            {
                var child = BuildFrame(call.Target);
                var boundPorts = graph.Bindings(call).Select(binding => binding.Port)
                    .ToHashSet(StringComparer.Ordinal);
                foreach (var parameter in child.Parameters.Where(parameter => !boundPorts.Contains(parameter.Name)))
                {
                    parameters.Add(parameter with
                    {
                        Name = InstanceName(call.Id, parameter.Name),
                        BindingKey = null,
                    });
                }
            }
            EnsureUnique(parameters.Select(parameter => parameter.Name),
                $"graph '{graph.Id}' VM parameters");
            var frame = new Frame(graph, parameters);
            frames.Add(id, frame);
            return frame;
        }

        var root = BuildFrame(program.RootGraph);
        foreach (var graph in program.Graphs)
        {
            BuildFrame(graph.Id);
        }

        var definitions = new List<VmDefinition>();
        foreach (var graph in program.Graphs)
        {
            EmitGraph(frames[graph.Id], frames, definitions, options);
        }

        var slots = root.Parameters.Select(parameter => new VmSlot(
            parameter.Name,
            Scope(parameter.Kind),
            Access(parameter.Kind),
            Tensor(parameter.Tensor),
            parameter.BindingKey)).ToArray();
        var entryArguments = root.Parameters.Select(parameter =>
            new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var state = root.Parameters
            .Where(parameter => parameter.Kind == GraphDefinitionResourceKind.State)
            .Select(parameter => new VmStateEntry(parameter.BindingKey ?? parameter.Name, parameter.Name));
        var result = new VmProgram(
            options.Name ?? root.Graph.Id.Value,
            options.Abi,
            target,
            slots,
            definitions,
            [new VmEntry(options.Entry, DefinitionId(root.Graph.Id), entryArguments)],
            new VmState(options.StateSchema, options.StateVersion, state));
        return VmInstructionContracts.Bind(result, registry);
    }

    private void EmitGraph(
        Frame frame,
        IReadOnlyDictionary<ExecutionGraphId, Frame> frames,
        ICollection<VmDefinition> definitions,
        VmExecutionGraphEmitterOptions options)
    {
        var parameters = frame.Parameters.Select(Parameter).ToArray();
        var nodes = new List<VmNode>();
        var completion = new Dictionary<ExecutionNodeId, string>();
        foreach (var node in StableOrder(frame.Graph))
        {
            var dependencies = frame.Graph.Dependencies(node)
                .Select(dependency => completion[dependency]).ToArray();
            switch (node)
            {
                case ExecutionOperationNode operation:
                    var selected = Resolve(operation, frame);
                    var operationBindings = frame.Graph.Bindings(operation);
                    if (target == VmTarget.Cpu)
                    {
                        nodes.Add(new VmNode(operation.Id.Value,
                            Operator(operation, selected, frame), dependencies));
                        completion.Add(operation.Id, operation.Id.Value);
                    }
                    else
                    {
                        var kernelId = KernelId(frame.Graph.Id, operation.Id);
                        var operationParameters = operationBindings.Select(binding =>
                            new VmParameter(binding.Port, Access(binding.Access),
                                Tensor(ResolveTensor(frame, binding)))).ToArray();
                        EnsureUnique(operationParameters.Select(parameter => parameter.Name),
                            $"operation '{frame.Graph.Id}/{operation.Id}' ports");
                        definitions.Add(new VmDefinition(
                            kernelId,
                            VmDefinitionKind.Kernel,
                            operationParameters,
                            [new VmNode("body", Operator(operation, selected, operationParameters))],
                            Dispatch(selected, operation, frame, options).Threads));
                        var arguments = operationBindings.Select(binding =>
                            new VmArgument(binding.Port, ResolveSource(frame, binding))).ToArray();
                        var launch = Dispatch(selected, operation, frame, options);
                        nodes.Add(new VmNode(operation.Id.Value,
                            new VmDispatch(kernelId, arguments, launch.Groups), dependencies));
                        var writes = operationBindings
                            .Where(binding => binding.Access != ResourceAccess.Read)
                            .Select(binding => ResolveSource(frame, binding))
                            .Distinct(StringComparer.Ordinal).ToArray();
                        if (writes.Length == 0)
                        {
                            completion.Add(operation.Id, operation.Id.Value);
                        }
                        else
                        {
                            var barrierId = $"{operation.Id.Value}.barrier";
                            nodes.Add(new VmNode(barrierId, new VmBarrier(writes), [operation.Id.Value]));
                            completion.Add(operation.Id, barrierId);
                        }
                    }
                    break;
                case ExecutionGraphCallNode call:
                    var child = frames[call.Target];
                    var callArguments = child.Parameters.Select(parameter =>
                        new VmArgument(parameter.Name, ResolveCallSource(frame, call, parameter.Name))).ToArray();
                    nodes.Add(new VmNode(call.Id.Value,
                        new VmCall(DefinitionId(call.Target), callArguments), dependencies));
                    completion.Add(call.Id, call.Id.Value);
                    break;
                default:
                    throw new NotSupportedException(
                        $"Execution graph node '{frame.Graph.Id}/{node.Id}' has unsupported type '{node.GetType().Name}'.");
            }
        }
        definitions.Add(new VmDefinition(
            DefinitionId(frame.Graph.Id),
            target == VmTarget.Cpu ? VmDefinitionKind.Function : VmDefinitionKind.Orchestration,
            parameters,
            nodes));
    }

    private GraphInstructionBinding Resolve(ExecutionOperationNode operation, Frame frame)
    {
        if (bindings.TryGetValue(operation.Operation, out var binding))
        {
            return binding;
        }

        var output = frame.Graph.Bindings(operation).FirstOrDefault(binding =>
            binding.Access != ResourceAccess.Read);
        var collection = output is not null
            ? CollectionForFallback(frame, output)
            : InstructionCollectionIds.TierZeroFloat32;
        try
        {
            registry.Resolve(collection, operation.Operation.Name, instructionTarget);
            return new GraphInstructionBinding(
                operation.Operation, collection, operation.Operation.Name, instructionTarget);
        }
        catch (NotSupportedException error)
        {
            throw new NotSupportedException(
                $"Backend '{instructionTarget}' does not support execution operation '{operation.Operation}'.",
                error);
        }
    }

    private static Guid CollectionForFallback(
        Frame frame,
        ModelResourceBinding output)
    {
        var tensor = ResolveTensor(frame, output);
        return tensor.ElementType == ElementType.Float16
            ? InstructionCollectionIds.TierZeroFloat16
            : InstructionCollectionIds.TierZeroFloat32;
    }

    private VmOperator Operator(
        ExecutionOperationNode operation,
        GraphInstructionBinding selected,
        Frame frame) =>
        new(
            operation.Requirements,
            selected.CollectionId,
            selected.InstructionName,
            frame.Graph.Bindings(operation).Select(binding =>
                new VmArgument(binding.Port, ResolveSource(frame, binding))),
            operation.Attributes);

    private static VmOperator Operator(
        ExecutionOperationNode operation,
        GraphInstructionBinding selected,
        IReadOnlyList<VmParameter> parameters) =>
        new(
            operation.Requirements,
            selected.CollectionId,
            selected.InstructionName,
            parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)),
            operation.Attributes);

    private static DispatchContract Dispatch(
        GraphInstructionBinding selected,
        ExecutionOperationNode operation,
        Frame frame,
        VmExecutionGraphEmitterOptions options)
    {
        var operationBindings = frame.Graph.Bindings(operation);
        var tensors = operationBindings.ToDictionary(
            binding => binding.Port,
            binding => ResolveTensor(frame, binding),
            StringComparer.Ordinal);
        var dispatch = selected.Dispatch?.Invoke(new(tensors, operation.Attributes));
        if (dispatch is not null)
        {
            ValidateDispatch(dispatch, operation);
            return new(
                new VmThreadGroup(dispatch.ThreadsX, dispatch.ThreadsY, dispatch.ThreadsZ),
                new VmThreadGroup(dispatch.GroupsX, dispatch.GroupsY, dispatch.GroupsZ));
        }
        var tensor = operationBindings
            .Where(binding => binding.Access != ResourceAccess.Read)
            .Select(binding => tensors[binding.Port])
            .FirstOrDefault() ?? tensors.Values.FirstOrDefault()
            ?? throw new InvalidDataException($"Operation '{operation.Id}' has no tensor bindings.");
        var count = tensor.Dimensions.Aggregate(
            1UL, (value, dimension) => checked(value * (ulong)dimension));
        var groups = checked((uint)((count + options.DefaultGpuThreads - 1) /
            options.DefaultGpuThreads));
        if (groups is 0 or > 65535)
        {
            throw new NotSupportedException(
                $"Operation '{operation.Id}' exceeds the default one-dimensional D3D12 dispatch grid.");
        }

        return new(new VmThreadGroup(options.DefaultGpuThreads), new VmThreadGroup(groups));
    }

    private static void ValidateDispatch(
        GraphInstructionDispatch dispatch,
        ExecutionOperationNode operation)
    {
        if (dispatch.ThreadsX is 0 or > 1024 ||
            dispatch.ThreadsY is 0 or > 1024 ||
            dispatch.ThreadsZ is 0 or > 64 ||
            (ulong)dispatch.ThreadsX * dispatch.ThreadsY * dispatch.ThreadsZ > 1024 ||
            dispatch.GroupsX is 0 or > 65535 ||
            dispatch.GroupsY is 0 or > 65535 ||
            dispatch.GroupsZ is 0 or > 65535)
        {
            throw new InvalidDataException(
                $"Operation '{operation.Id}' has an invalid registered D3D12 dispatch configuration.");
        }
    }

    private static string ResolveCallSource(
        Frame frame,
        ExecutionGraphCallNode call,
        string childParameter)
    {
        var binding = frame.Graph.Bindings(call).SingleOrDefault(binding =>
            string.Equals(binding.Port, childParameter, StringComparison.Ordinal));
        return binding is null
            ? InstanceName(call.Id, childParameter)
            : ResolveSource(frame, binding);
    }

    private static string ResolveSource(Frame frame, ModelResourceBinding binding) =>
        binding.Resource.Owner.IsParent
            ? binding.Resource.Resource.Value
            : InstanceName(
                new ExecutionNodeId(binding.Resource.Owner.Call!.Value.Value),
                binding.Resource.Resource.Value);

    private static TensorDescriptor ResolveTensor(
        Frame frame,
        ModelResourceBinding binding)
    {
        var source = ResolveSource(frame, binding);
        return frame.Parameters.SingleOrDefault(parameter => parameter.Name == source)?.Tensor
            ?? throw new InvalidDataException(
                $"Binding '{frame.Graph.Id}/{binding.Port}' resolves to unknown VM resource '{source}'.");
    }

    private static IReadOnlyList<ExecutionGraphNode> StableOrder(ExecutionGraph graph) =>
        graph.Structure.DependencyView().TopologicalOrder()
            .OfType<ExecutionGraphNode>().ToArray();

    private static VmParameter Parameter(FrameParameter parameter) =>
        new(parameter.Name, Access(parameter.Kind), Tensor(parameter.Tensor));

    private static VmAccess Access(GraphDefinitionResourceKind kind) =>
        kind is GraphDefinitionResourceKind.Input
            or GraphDefinitionResourceKind.Weight
            or GraphDefinitionResourceKind.Constant
            ? VmAccess.ReadOnly
            : VmAccess.ReadWrite;

    private static VmAccess Access(ResourceAccess access) =>
        access == ResourceAccess.Read ? VmAccess.ReadOnly : VmAccess.ReadWrite;

    private static VmSlotScope Scope(GraphDefinitionResourceKind kind) => kind switch
    {
        GraphDefinitionResourceKind.Weight or GraphDefinitionResourceKind.Constant => VmSlotScope.Global,
        GraphDefinitionResourceKind.State or GraphDefinitionResourceKind.Input
            or GraphDefinitionResourceKind.Output or GraphDefinitionResourceKind.InOut => VmSlotScope.Session,
        GraphDefinitionResourceKind.Temporary => VmSlotScope.Local,
        _ => throw new InvalidDataException($"Unsupported execution resource kind '{kind}'."),
    };

    private static VmTensor Tensor(TensorDescriptor tensor)
    {
        if (!string.Equals(tensor.Layout, "dense", StringComparison.Ordinal))
        {
            throw new NotSupportedException($"VM emission does not support tensor layout '{tensor.Layout}'.");
        }

        return new VmTensor(tensor.ElementType switch
        {
            ElementType.Byte => VmElementType.Byte,
            ElementType.Int32 => VmElementType.Int32,
            ElementType.UInt32 => VmElementType.UInt32,
            ElementType.Float16 => VmElementType.Float16,
            ElementType.Float32 => VmElementType.Float32,
            _ => throw new NotSupportedException($"VM emission does not support element type '{tensor.ElementType}'."),
        }, tensor.Dimensions);
    }

    private static string DefinitionId(ExecutionGraphId id) => $"graph.{id.Value}";
    private static string KernelId(ExecutionGraphId graph, ExecutionNodeId node) =>
        $"kernel.{graph.Value}.{node.Value}";
    private static string InstanceName(ExecutionNodeId call, string parameter) =>
        $"{call.Value}/{parameter}";

    private static void EnsureUnique(IEnumerable<string> values, string description)
    {
        var duplicate = values.GroupBy(value => value, StringComparer.Ordinal)
            .FirstOrDefault(group => group.Count() != 1);
        if (duplicate is not null)
        {
            throw new InvalidDataException($"Duplicate {description} identifier '{duplicate.Key}'.");
        }
    }

    private sealed record Frame(ExecutionGraph Graph, IReadOnlyList<FrameParameter> Parameters);
    private sealed record FrameParameter(
        string Name,
        TensorDescriptor Tensor,
        GraphDefinitionResourceKind Kind,
        string? BindingKey);
    private sealed record DispatchContract(VmThreadGroup Threads, VmThreadGroup Groups);
}
