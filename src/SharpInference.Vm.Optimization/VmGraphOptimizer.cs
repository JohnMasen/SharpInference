using System.Xml.Linq;
using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

public sealed record VmOptimizationOptions(
    bool ReuseLocalStorage = true, uint ThreadsPerGroup = 64, int PrefillCapacity = 64,
    bool NativeHalfWeights = true, bool WeightViews = true, TierOneOptimizationSettings? TierOne = null,
    GpuMatVecOptimizationSettings? GpuMatVec = null);

public static class VmGraphOptimizer
{
    public static VmProgram Optimize(LogicalGraph logical, VmTarget target, VmOptimizationOptions? options = null)
        => OptimizeWithReport(logical, target, options).Program;

    public static VmOptimizationResult OptimizeWithReport(LogicalGraph logical, VmTarget target,
        VmOptimizationOptions? options = null, IReadOnlyList<InstructionOptimizationCapability>? capabilities = null,
        IReadOnlyList<GraphInstructionBinding>? operationBindings = null)
    {
        ArgumentNullException.ThrowIfNull(logical);
        options ??= new VmOptimizationOptions();
        if (options.ThreadsPerGroup is 0 or > 1024)
            throw new ArgumentOutOfRangeException(nameof(options), "Thread group size must be between 1 and 1024.");
        if (options.PrefillCapacity is < 1 or > 1024)
            throw new ArgumentOutOfRangeException(nameof(options), "Prefill capacity must be between 1 and 1024.");
        if (target != VmTarget.Direct3D12 && options.GpuMatVec is not null)
            throw new NotSupportedException("GPU MatVec settings require a Direct3D12 target.");
        GraphValidator.Validate(logical);
        var hasViews = logical.Nodes.Any(node => node.Resources.Any(binding => binding.View is not null));
        if (hasViews && options.TierOne is not null)
            throw new NotSupportedException("Tier-one instruction selection does not declare tensor-view support.");
        var architecture = target switch
        {
            VmTarget.Cpu => InstructionTarget.Cpu,
            VmTarget.Direct3D12 => InstructionTarget.Direct3D12,
            _ => throw new NotSupportedException($"No VM lowering is installed for '{target}'."),
        };
        var bindings = (operationBindings ?? []).Where(binding => binding.Target == architecture)
            .ToDictionary(binding => binding.Operation);
        if (bindings.Keys.Any(operation => TierZeroOperationContracts.Contracts.Any(contract => contract.Operation == operation)))
            throw new InvalidDataException("Model instruction bindings cannot override standard tensor operation contracts.");
        var resourceDescriptions = logical.Resources.ToDictionary(resource => resource.Id);
        foreach (var node in logical.Nodes)
            if (!bindings.ContainsKey(node.Operation)) TierZeroOperationContracts.ValidateNode(node, resourceDescriptions);
        var execution = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded));
        var descriptors = logical.Resources.ToDictionary(resource => resource.Id);
        var aliases = new Dictionary<ResourceId, ResourceId>();
        IReadOnlyList<ExecutionNode> nodes = execution.Nodes;
        if (options.WeightViews && !hasViews)
            (nodes, aliases) = LowerWeightViews(execution, descriptors, options.NativeHalfWeights);
        execution = new ExecutionGraph(execution.Identity, execution.Model, descriptors.Values, execution.Regions,
            nodes, execution.Inputs, execution.Outputs, execution.GraphState);
        var selection = TierOneGraphSelection.Apply(execution, target, options, capabilities ?? [], aliases);
        execution = selection.Graph;
        nodes = execution.Nodes;
        var retainedResources = execution.Resources.Select(resource => resource.Id).ToHashSet();
        var physical = new Dictionary<ResourceId, string>();
        var slots = new List<VmSlot>();
        if (options.ReuseLocalStorage)
        {
            var allocationGraph = aliases.Count == 0 ? execution : new ExecutionGraph(execution.Identity, execution.Model,
                execution.Resources.Where(resource => !aliases.ContainsKey(resource.Id)), execution.Regions,
                nodes.Select(node => node with
                {
                    Resources = node.Resources.Select(binding => aliases.TryGetValue(binding.Resource, out var source)
                        ? binding with { Resource = source } : binding).ToArray(),
                }), execution.Inputs, execution.Outputs, execution.GraphState);
            var allocation = GraphResourceAllocator.PlanLocal(allocationGraph);
            foreach (var slot in allocation.Slots)
                slots.Add(new VmSlot($"local.{slot.Id}", VmSlotScope.Local,
                    VmAccess.ReadWrite, Tensor(slot.Tensor)));
            foreach (var (resource, slot) in allocation.SlotByResource)
                physical.Add(resource, $"local.{slot}");
        }
        foreach (var resource in logical.Resources)
        {
            if (!retainedResources.Contains(resource.Id)) continue;
            if (aliases.ContainsKey(resource.Id)) continue;
            if (physical.ContainsKey(resource.Id)) continue;
            physical.Add(resource.Id, resource.Id.Value);
            var scope = Scope(resource.Scope);
            if (resource.Tensor.Layout != "dense")
                throw new NotSupportedException($"Resource '{resource.Id}' requires unsupported layout '{resource.Tensor.Layout}'.");
            slots.Add(new VmSlot(resource.Id.Value, scope,
                resource.Kind is GraphResourceKind.Weight or GraphResourceKind.Constant or GraphResourceKind.Input
                    ? VmAccess.ReadOnly : VmAccess.ReadWrite,
                Tensor(resource.Tensor), resource.BindingKey));
        }
        foreach (var (alias, source) in aliases) physical.Add(alias, physical[source]);
        var definitions = new List<VmDefinition>();
        var keys = new Dictionary<string, VmDefinition>(StringComparer.Ordinal);
        var sequence = new List<VmNode>();
        var rootParameters = slots.Select(slot => new VmParameter(slot.Id, slot.Access, slot.Tensor)).ToArray();
        string? previous = null;
        var matVecCalls = 0;
        var cooperativeCalls = 0;
        var matVecDiagnostics = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in nodes)
        {
            bindings.TryGetValue(node.Operation, out var instructionBinding);
            selection.Implementations.TryGetValue(node.Id, out var implementation);
            if (!node.Resources.Where(binding => binding.Access == GraphResourceAccess.ReadWrite).Select(binding => binding.Port)
                    .ToHashSet(StringComparer.Ordinal).SetEquals(instructionBinding?.ReadWritePorts ?? []))
                throw new NotSupportedException($"Node '{node.Id}' requires an explicit read/write operator contract.");
            var parameters = node.Resources.Select(binding => new VmParameter(binding.Port,
                binding.Access == GraphResourceAccess.Read ? VmAccess.ReadOnly : VmAccess.ReadWrite,
                Tensor(binding.View?.Tensor ?? descriptors[binding.Resource].Tensor))).ToArray();
            InstructionExecutionConfiguration? configuration = implementation?.Configuration;
            var threads = options.ThreadsPerGroup;
            var dispatch = target == VmTarget.Direct3D12 ? instructionBinding?.Dispatch?.Invoke(
                new(node.Resources.ToDictionary(binding => binding.Port,
                    binding => binding.View?.Tensor ?? descriptors[binding.Resource].Tensor, StringComparer.Ordinal), node.Attributes)) : null;
            if (dispatch is not null)
            {
                if (dispatch.ThreadsX == 0 || dispatch.ThreadsY == 0 || dispatch.ThreadsZ == 0 ||
                    dispatch.ThreadsX > 1024 || dispatch.ThreadsY > 1024 || dispatch.ThreadsZ > 1024 ||
                    (ulong)dispatch.ThreadsX * dispatch.ThreadsY * dispatch.ThreadsZ > 1024 ||
                    dispatch.GroupsX is 0 or > 65535 || dispatch.GroupsY is 0 or > 65535 || dispatch.GroupsZ is 0 or > 65535)
                    throw new InvalidDataException($"Node '{node.Id}' has an invalid registered dispatch configuration.");
                threads = dispatch.ThreadsX;
            }
            var cooperative = false;
            if (target == VmTarget.Direct3D12 && node.Operation.Name == "core.mat-vec")
            {
                matVecCalls++;
                var choice = options.GpuMatVec is { } settings ? settings.Select(parameters, options) :
                    (Cooperative: true, Diagnostic: "Default GPU MatVec implementation: cooperative.");
                cooperative = choice.Cooperative;
                matVecDiagnostics.Add(choice.Diagnostic);
                configuration = cooperative ? GpuMatVecExecution.Cooperative : GpuMatVecExecution.Serial;
                if (cooperative) { threads = GpuMatVecExecution.Threads; cooperativeCalls++; }
            }
            var key = new XElement("Operator",
                new XAttribute("name", node.Operation.Name), new XAttribute("version", node.Operation.Version),
                new XAttribute("target", target), new XAttribute("threads", threads),
                dispatch is null ? null : new XAttribute("threadsY", dispatch.ThreadsY),
                dispatch is null ? null : new XAttribute("threadsZ", dispatch.ThreadsZ),
                configuration is null ? null : new XAttribute("implementation", configuration.Implementation.Value),
                new XAttribute("minimumArithmeticType", node.Requirements.MinimumArithmeticType),
                new XAttribute("minimumAccumulatorType", node.Requirements.MinimumAccumulatorType),
                parameters.Select(parameter => new XElement("Parameter",
                    new XAttribute("name", parameter.Name), new XAttribute("access", parameter.Access),
                    new XAttribute("type", parameter.Tensor.ElementType),
                    new XAttribute("dimensions", string.Join(",", parameter.Tensor.Dimensions)))),
                node.Attributes.OrderBy(pair => pair.Key, StringComparer.Ordinal).Select(pair =>
                    new XElement("Attribute", new XAttribute("name", pair.Key), new XAttribute("value", pair.Value))))
                .ToString(SaveOptions.DisableFormatting);
            if (!keys.TryGetValue(key, out var definition))
            {
                definition = new VmDefinition($"op.{definitions.Count:D4}",
                    target == VmTarget.Cpu ? VmDefinitionKind.Function : VmDefinitionKind.Kernel,
                    parameters, [new VmNode("body", new VmOperator(node.Requirements,
                        instructionBinding?.CollectionId ?? implementation?.CollectionId ?? (parameters.Single(parameter => parameter.Name == "output").Tensor.ElementType == VmElementType.Float16
                            ? InstructionCollectionIds.TierZeroFloat16 : InstructionCollectionIds.TierZeroFloat32),
                        instructionBinding?.InstructionName ?? implementation?.Name ?? node.Operation.Name,
                        parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)),
                        node.Attributes, executionConfiguration: configuration))],
                    target == VmTarget.Direct3D12 ? new VmThreadGroup(threads, dispatch?.ThreadsY ?? 1, dispatch?.ThreadsZ ?? 1) : null);
                definitions.Add(definition);
                keys.Add(key, definition);
            }
            var arguments = node.Resources.Select(binding =>
                new VmArgument(binding.Port, physical[binding.Resource], binding.View?.ByteOffset ?? 0)).ToArray();
            VmInstruction instruction;
            if (target == VmTarget.Cpu)
                instruction = new VmCall(definition.Id, arguments);
            else if (dispatch is not null)
                instruction = new VmDispatch(definition.Id, arguments, new(dispatch.GroupsX, dispatch.GroupsY, dispatch.GroupsZ));
            else
            {
                var output = parameters.Single(parameter => parameter.Access == VmAccess.ReadWrite);
                if (cooperative)
                {
                    var groups = GpuMatVecExecution.Groups(output.Tensor.ElementCount);
                    instruction = new VmDispatch(definition.Id, arguments, new(groups.X, groups.Y));
                }
                else
                {
                    var groups = checked((uint)((output.Tensor.ElementCount + threads - 1) / threads));
                    if (groups > 65535)
                        throw new NotSupportedException($"Node '{node.Id}' exceeds the configured one-dimensional GPU dispatch grid.");
                    instruction = new VmDispatch(definition.Id, arguments, new(groups));
                }
            }
            var id = node.Id.Value;
            sequence.Add(new VmNode(id, instruction, previous is null ? [] : [previous]));
            previous = id;
            if (target == VmTarget.Direct3D12)
            {
                var writes = node.Resources.Where(binding => binding.Access != GraphResourceAccess.Read)
                    .Select(binding => physical[binding.Resource]).Distinct(StringComparer.Ordinal).ToArray();
                var barrierId = $"{id}.barrier";
                sequence.Add(new VmNode(barrierId, new VmBarrier(writes), [id]));
                previous = barrierId;
            }
        }
        definitions.Add(new VmDefinition("forward", VmDefinitionKind.Orchestration, rootParameters, sequence));
        var entries = new List<VmEntry>
        {
            new("forward", "forward", slots.Select(slot => new VmArgument(slot.Id, slot.Id))),
        };
        if (logical.Model.Attributes.GetValueOrDefault("execution.token-prefill") == "true" &&
            logical.Inputs.Count == 1 && descriptors[logical.Inputs[0]].Tensor.ElementType == GraphElementType.Int32 &&
            descriptors[logical.Inputs[0]].Tensor.Dimensions.SequenceEqual([1]))
        {
            var tokenSlot = physical[logical.Inputs[0]];
            var index = slots.FindIndex(slot => slot.Id == tokenSlot);
            slots[index] = slots[index] with { Tensor = new VmTensor(VmElementType.Int32, [options.PrefillCapacity]) };
            for (var count = 1; count <= options.PrefillCapacity; count *= 2)
            {
                var prefillParameters = slots.Select(slot => new VmParameter(slot.Id, slot.Access,
                    slot.Id == tokenSlot ? new VmTensor(VmElementType.Int32, [count]) : slot.Tensor)).ToArray();
                var child = count == 1 ? "forward" : $"prefill.{count / 2}";
                var calls = Enumerable.Range(0, count == 1 ? 1 : 2).Select(position => new VmNode($"token.{position}",
                    new VmCall(child, slots.Select(slot => new VmArgument(slot.Id, slot.Id,
                        slot.Id == tokenSlot ? checked((ulong)position * (ulong)(count / 2) * sizeof(int)) : 0))),
                    position == 0 ? [] : [$"token.{position - 1}"])).ToArray();
                var id = $"prefill.{count}";
                definitions.Add(new VmDefinition(id, VmDefinitionKind.Orchestration, prefillParameters, calls));
                entries.Add(new VmEntry(id, id, slots.Select(slot => new VmArgument(slot.Id, slot.Id))));
            }
        }
        var program = new VmProgram(logical.Identity.Name, $"vm:{logical.Model.StateAbiId}", target,
            slots, definitions, entries, new VmState(logical.GraphState.Schema.Name, 1,
                logical.GraphState.Entries.Select(entry => new VmStateEntry(entry.Name, physical[entry.Resource]))));
        return new(program, selection.Report, new(matVecCalls, cooperativeCalls,
            Array.AsReadOnly(matVecDiagnostics.Order(StringComparer.Ordinal).ToArray())));
    }

    private static (IReadOnlyList<ExecutionNode> Nodes, Dictionary<ResourceId, ResourceId> Aliases) LowerWeightViews(
        ExecutionGraph graph, Dictionary<ResourceId, GraphResource> resources, bool nativeHalf)
    {
        var aliases = new Dictionary<ResourceId, ResourceId>();
        var removed = new Dictionary<ExecutionNodeId, ExecutionNode>();
        var consumers = graph.Nodes.SelectMany(node => node.Resources.Select(binding => (node, binding)))
            .GroupBy(pair => pair.binding.Resource).ToDictionary(group => group.Key, group => group.ToArray());
        bool ExternallyVisible(ResourceId id) => graph.Inputs.Contains(id) || graph.Outputs.Contains(id) ||
            graph.GraphState.Entries.Any(entry => entry.Resource == id);
        bool ReadOnlyUses(ResourceId id, ExecutionNodeId producer) => consumers[id]
            .Where(pair => pair.node.Id != producer).All(pair => pair.binding.Access == GraphResourceAccess.Read);
        foreach (var node in graph.Nodes.Where(node => nativeHalf &&
            node.Operation == PortableTensorOperationContracts.CastFp16ToFp32))
        {
            var input = node.Resources.Single(binding => binding.Port == "input").Resource;
            var output = node.Resources.Single(binding => binding.Port == "output").Resource;
            if (resources[input].Kind != GraphResourceKind.Weight || resources[input].Tensor.ElementType != GraphElementType.Float16 ||
                ExternallyVisible(output))
                continue;
            var path = new Dictionary<ExecutionNodeId, ExecutionNode> { [node.Id] = node };
            var pending = new Queue<ResourceId>();
            var visited = new HashSet<ResourceId>();
            var safe = true;
            pending.Enqueue(output);
            while (pending.TryDequeue(out var value) && safe)
            {
                if (!visited.Add(value)) continue;
                if (ExternallyVisible(value)) { safe = false; break; }
                foreach (var (consumer, binding) in consumers[value].Where(pair => !path.ContainsKey(pair.node.Id)))
                {
                    if (binding.Access != GraphResourceAccess.Read) { safe = false; break; }
                    if (consumer.Operation == PortableTensorOperationContracts.Reshape && binding.Port == "input")
                    {
                        path.TryAdd(consumer.Id, consumer);
                        pending.Enqueue(consumer.Resources.Single(binding => binding.Port == "output").Resource);
                    }
                    else if (!(consumer.Operation == PrimitiveGraphOperations.MatVec && binding.Port is "matrix" or "weight" ||
                        consumer.Operation == PrimitiveGraphOperations.GatherRow && binding.Port == "table" ||
                        consumer.Operation == PortableTensorOperationContracts.BatchedMatVec && binding.Port == "matrix"))
                    { safe = false; break; }
                }
            }
            if (!safe) continue;
            foreach (var item in path.Values)
            {
                var id = item.Resources.Single(binding => binding.Port == "output").Resource;
                aliases[id] = input;
                resources[id] = View(resources[id], GraphElementType.Float16);
                removed[item.Id] = item;
            }
        }
        var immutable = graph.Resources.Where(resource => resource.Kind == GraphResourceKind.Weight)
            .Select(resource => resource.Id).Concat(aliases.Keys).ToHashSet();
        foreach (var cast in graph.Nodes.Where(node => node.Operation == PortableTensorOperationContracts.CastFp16ToFp32 &&
            !removed.ContainsKey(node.Id)))
        {
            var input = cast.Resources.Single(binding => binding.Port == "input").Resource;
            var output = cast.Resources.Single(binding => binding.Port == "output").Resource;
            if (immutable.Contains(input) && ReadOnlyUses(output, cast.Id)) immutable.Add(output);
        }
        var views = new Queue<ResourceId>(immutable);
        while (views.TryDequeue(out var input))
        {
            if (!consumers.TryGetValue(input, out var uses)) continue;
            foreach (var (node, binding) in uses.Where(pair => pair.node.Operation == PortableTensorOperationContracts.Reshape &&
                pair.binding.Port == "input" && !removed.ContainsKey(pair.node.Id)))
            {
                var output = node.Resources.Single(binding => binding.Port == "output").Resource;
                if (ExternallyVisible(output) || !ReadOnlyUses(output, node.Id)) continue;
                aliases[output] = aliases.TryGetValue(input, out var source) ? source : input;
                resources[output] = View(resources[output], resources[input].Tensor.ElementType);
                removed[node.Id] = node;
                views.Enqueue(output);
            }
        }
        if (removed.Count == 0) return (graph.Nodes, aliases);
        var nodes = graph.Nodes.Where(node => !removed.ContainsKey(node.Id)).Select(node => node with
        {
            Dependencies = Expand(node.Dependencies),
        }).ToArray();
        return (nodes, aliases);

        static GraphResource View(GraphResource resource, GraphElementType type) => resource with
        {
            Tensor = new TensorDescriptor(type, resource.Tensor.Dimensions, resource.Tensor.Layout),
        };

        ExecutionNodeId[] Expand(IReadOnlyList<ExecutionNodeId> dependencies)
        {
            var pending = new Stack<ExecutionNodeId>(dependencies.Reverse());
            var result = new List<ExecutionNodeId>();
            var seen = new HashSet<ExecutionNodeId>();
            while (pending.TryPop(out var id))
            {
                if (!seen.Add(id)) continue;
                if (removed.TryGetValue(id, out var cast))
                    foreach (var dependency in cast.Dependencies.Reverse()) pending.Push(dependency);
                else result.Add(id);
            }
            return result.ToArray();
        }
    }

    private static VmSlotScope Scope(GraphResourceScope scope) => scope switch
    {
        GraphResourceScope.Global => VmSlotScope.Global,
        GraphResourceScope.Session => VmSlotScope.Session,
        GraphResourceScope.Local => VmSlotScope.Local,
        _ => throw new InvalidDataException("Unknown graph resource scope."),
    };

    private static VmTensor Tensor(TensorDescriptor tensor) => new(tensor.ElementType switch
    {
        GraphElementType.Byte => VmElementType.Byte,
        GraphElementType.Int32 => VmElementType.Int32,
        GraphElementType.UInt32 => VmElementType.UInt32,
        GraphElementType.Float16 => VmElementType.Float16,
        GraphElementType.Float32 => VmElementType.Float32,
        _ => throw new NotSupportedException($"Unsupported tensor type '{tensor.ElementType}'."),
    }, tensor.Dimensions);
}
