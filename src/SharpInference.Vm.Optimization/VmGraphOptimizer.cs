using System.Xml.Linq;
using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

/// <summary>Configures resource reuse, tensor views, GPU launch sizes, and optional profile-driven selection.</summary>
/// <param name="ReuseLocalStorage">Whether compatible non-overlapping temporary lifetimes share storage.</param>
/// <param name="ThreadsPerGroup">The number of GPU threads in each dispatch group.</param>
/// <param name="PrefillCapacity">The token capacity used for prefill lowering.</param>
/// <param name="NativeHalfWeights">Whether native half-precision weight storage is retained during lowering.</param>
/// <param name="WeightViews">Whether eligible weight conversion and slicing operations are lowered to views.</param>
/// <param name="TierOne">The optional trusted-profile settings for pointwise fusion selection.</param>
/// <param name="GpuMatVec">The optional GPU matrix-vector implementation-selection settings.</param>
/// <param name="TensorViews">Whether eligible general tensor operations are lowered to storage views.</param>
public sealed record VmOptimizationOptions(
    bool ReuseLocalStorage = true, uint ThreadsPerGroup = 64, int PrefillCapacity = 64,
    bool NativeHalfWeights = true, bool WeightViews = true, TierOneOptimizationSettings? TierOne = null,
    GpuMatVecOptimizationSettings? GpuMatVec = null, bool TensorViews = true);

/// <summary>Contains a target execution plan and its optimization-selection reports.</summary>
/// <param name="Plan">The optimized target execution plan.</param>
/// <param name="Report">The tier-one fusion selection report.</param>
/// <param name="MatVecReport">The GPU matrix-vector selection report.</param>
public sealed record ExecutionPlanOptimizationResult(TargetExecutionPlan Plan,
    TierOneOptimizationReport Report, GpuMatVecOptimizationReport MatVecReport);

/// <summary>Lowers logical or IL graphs into resource-allocated target execution plans and VM programs.</summary>
public static class VmGraphOptimizer
{
    /// <summary>Lowers and optimizes a logical graph into a VM program for the specified target.</summary>
    /// <param name="logical">The logical graph to lower and optimize.</param>
    /// <param name="target">The CPU or Direct3D12 VM execution backend.</param>
    /// <param name="options">The lowering, resource reuse, launch, and optional offline-selection settings; null uses defaults.</param>
    public static VmProgram Optimize(LogicalGraph logical, VmTarget target, VmOptimizationOptions? options = null)
        => OptimizeWithReport(logical, target, options).Program;

    /// <summary>Builds an optimized VM program together with fusion and matrix-vector selection reports.</summary>
    /// <param name="logical">The logical graph to lower and optimize.</param>
    /// <param name="target">The CPU or Direct3D12 VM execution backend.</param>
    /// <param name="options">The lowering, resource reuse, launch, and optional offline-selection settings; null uses defaults.</param>
    /// <param name="capabilities">The registered target-specific instruction optimization capabilities.</param>
    /// <param name="operationBindings">The optional model-owned graph-operation instruction mappings.</param>
    public static VmOptimizationResult OptimizeWithReport(LogicalGraph logical, VmTarget target,
        VmOptimizationOptions? options = null, IReadOnlyList<InstructionOptimizationCapability>? capabilities = null,
        IReadOnlyList<GraphInstructionBinding>? operationBindings = null)
    {
        var result = BuildPlan(logical, target, options, capabilities, operationBindings);
        return new(ExecutionPlanAssembler.Assemble(result.Plan), result.Report, result.MatVecReport);
    }

    /// <summary>Lowers a graph into a target execution plan with resource allocation and optional instruction selection.</summary>
    /// <param name="logical">The logical graph to lower and optimize.</param>
    /// <param name="target">The CPU or Direct3D12 VM execution backend.</param>
    /// <param name="options">The lowering, resource reuse, launch, and optional offline-selection settings; null uses defaults.</param>
    /// <param name="capabilities">The registered target-specific instruction optimization capabilities.</param>
    /// <param name="operationBindings">The optional model-owned graph-operation instruction mappings.</param>
    public static ExecutionPlanOptimizationResult BuildPlan(LogicalGraph logical, VmTarget target,
        VmOptimizationOptions? options = null, IReadOnlyList<InstructionOptimizationCapability>? capabilities = null,
        IReadOnlyList<GraphInstructionBinding>? operationBindings = null)
        => BuildPlan(new IlExecutionPlan(logical), target, options, capabilities, operationBindings);

    /// <summary>Lowers a graph into a target execution plan with resource allocation and optional instruction selection.</summary>
    /// <param name="il">The intermediate execution graph to lower and optimize.</param>
    /// <param name="target">The CPU or Direct3D12 VM execution backend.</param>
    /// <param name="options">The lowering, resource reuse, launch, and optional offline-selection settings; null uses defaults.</param>
    /// <param name="capabilities">The registered target-specific instruction optimization capabilities.</param>
    /// <param name="operationBindings">The optional model-owned graph-operation instruction mappings.</param>
    public static ExecutionPlanOptimizationResult BuildPlan(IlExecutionPlan il, VmTarget target,
        VmOptimizationOptions? options = null, IReadOnlyList<InstructionOptimizationCapability>? capabilities = null,
        IReadOnlyList<GraphInstructionBinding>? operationBindings = null)
    {
        ArgumentNullException.ThrowIfNull(il);
        options ??= new VmOptimizationOptions();
        if (options.ThreadsPerGroup is 0 or > 1024)
            throw new ArgumentOutOfRangeException(nameof(options), "Thread group size must be between 1 and 1024.");
        if (options.PrefillCapacity is < 1 or > 1024)
            throw new ArgumentOutOfRangeException(nameof(options), "Prefill capacity must be between 1 and 1024.");
        if (target != VmTarget.Direct3D12 && options.GpuMatVec is not null)
            throw new NotSupportedException("GPU MatVec settings require a Direct3D12 target.");
        var hasViews = il.Nodes.Any(node => node.Resources.Any(binding => binding.View is not null));
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
        var resourceDescriptions = il.Resources.ToDictionary(resource => resource.Id);
        foreach (var node in il.Nodes)
            if (!bindings.ContainsKey(node.Operation)) TierZeroOperationContracts.ValidateNode(
                new LogicalNode(new(node.Id.Value), node.Operation, node.Region, node.Resources, [], node.Attributes, node.Requirements),
                resourceDescriptions);
        var execution = il;
        var descriptors = il.Resources.ToDictionary(resource => resource.Id);
        var aliases = new Dictionary<ResourceId, ResourceId>();
        IReadOnlyList<ExecutionNode> nodes = execution.Nodes;
        if ((options.WeightViews || options.TensorViews) && !hasViews)
            (nodes, aliases) = LowerViews(execution, descriptors,
                options.WeightViews && options.NativeHalfWeights, options.WeightViews, options.TensorViews);
        execution = execution.WithNodes(descriptors.Values, nodes);
        var selection = TierOneGraphSelection.Apply(execution, target, options, capabilities ?? [], aliases);
        execution = selection.Graph;
        nodes = execution.Nodes;
        var retainedResources = execution.Resources.Select(resource => resource.Id).ToHashSet();
        var physical = new Dictionary<ResourceId, string>();
        var slots = new List<ExecutionPlanSlot>();
        if (options.ReuseLocalStorage)
        {
            var allocationGraph = aliases.Count == 0 ? execution : execution.WithNodes(
                execution.Resources.Where(resource => !aliases.ContainsKey(resource.Id)),
                nodes.Select(node => node with
                {
                    Resources = node.Resources.Select(binding => aliases.TryGetValue(binding.Resource, out var source)
                        ? binding with { Resource = source } : binding).ToArray(),
                }));
            var allocation = GraphResourceAllocator.PlanLocal(allocationGraph);
            foreach (var slot in allocation.Slots)
                slots.Add(new ExecutionPlanSlot($"local.{slot.Id}", GraphResourceScope.Local,
                    GraphResourceAccess.ReadWrite, slot.Tensor));
            foreach (var (resource, slot) in allocation.SlotByResource)
                physical.Add(resource, $"local.{slot}");
        }
        foreach (var resource in il.Resources)
        {
            if (!retainedResources.Contains(resource.Id)) continue;
            if (aliases.ContainsKey(resource.Id)) continue;
            if (physical.ContainsKey(resource.Id)) continue;
            physical.Add(resource.Id, resource.Id.Value);
            if (resource.Tensor.Layout != "dense")
                throw new NotSupportedException($"Resource '{resource.Id}' requires unsupported layout '{resource.Tensor.Layout}'.");
            slots.Add(new ExecutionPlanSlot(resource.Id.Value, resource.Scope,
                resource.Kind is GraphResourceKind.Weight or GraphResourceKind.Constant or GraphResourceKind.Input
                    ? GraphResourceAccess.Read : GraphResourceAccess.ReadWrite,
                resource.Tensor, resource.BindingKey));
        }
        foreach (var (alias, source) in aliases) physical.Add(alias, physical[source]);
        var definitions = new List<ExecutionPlanDefinition>();
        var keys = new Dictionary<string, ExecutionPlanDefinition>(StringComparer.Ordinal);
        var sequence = new List<ExecutionPlanNode>();
        var nodeRegions = new Dictionary<string, RegionId>(StringComparer.Ordinal);
        var rootParameters = slots.Select(slot => new ExecutionPlanParameter(slot.Id, slot.Access, slot.Tensor)).ToArray();
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
            var parameters = node.Resources.Select(binding => new ExecutionPlanParameter(binding.Port,
                binding.Access == GraphResourceAccess.Read ? GraphResourceAccess.Read : GraphResourceAccess.ReadWrite,
                binding.View?.Tensor ?? descriptors[binding.Resource].Tensor)).ToArray();
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
                var choice = options.GpuMatVec is { } settings ? settings.Select(parameters.Select(ExecutionPlanAssembler.Parameter).ToArray(), options) :
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
                definition = new ExecutionPlanDefinition($"op.{definitions.Count:D4}",
                    target == VmTarget.Cpu ? ExecutionPlanDefinitionKind.Function : ExecutionPlanDefinitionKind.Kernel,
                    parameters, [new ExecutionPlanNode("body", new ExecutionPlanOperator(node.Requirements,
                        instructionBinding?.CollectionId ?? implementation?.CollectionId ?? (parameters.Single(parameter => parameter.Name == "output").Tensor.ElementType == GraphElementType.Float16
                            ? InstructionCollectionIds.TierZeroFloat16 : InstructionCollectionIds.TierZeroFloat32),
                        instructionBinding?.InstructionName ?? implementation?.Name ?? node.Operation.Name,
                        parameters.Select(parameter => new ExecutionPlanArgument(parameter.Name, parameter.Name)),
                        node.Attributes, implementation: configuration?.Implementation.Value,
                        cooperativeThreads: target == VmTarget.Direct3D12 && node.Operation.Name == "core.mat-vec"
                            ? cooperative ? 64u : 1u : null))],
                    target == VmTarget.Direct3D12 ? new ExecutionPlanGrid(threads, dispatch?.ThreadsY ?? 1, dispatch?.ThreadsZ ?? 1) : null);
                definitions.Add(definition);
                keys.Add(key, definition);
            }
            var arguments = node.Resources.Select(binding =>
                new ExecutionPlanArgument(binding.Port, physical[binding.Resource], binding.View?.ByteOffset ?? 0)).ToArray();
            ExecutionPlanInstruction instruction;
            if (target == VmTarget.Cpu)
                instruction = new ExecutionPlanCall(definition.Id, arguments);
            else if (dispatch is not null)
                instruction = new ExecutionPlanDispatch(definition.Id, arguments, new(dispatch.GroupsX, dispatch.GroupsY, dispatch.GroupsZ));
            else
            {
                var output = parameters.Single(parameter => parameter.Access == GraphResourceAccess.ReadWrite);
                if (cooperative)
                {
                    var groups = GpuMatVecExecution.Groups(ElementCount(output.Tensor));
                    instruction = new ExecutionPlanDispatch(definition.Id, arguments, new(groups.X, groups.Y));
                }
                else
                {
                    var groups = checked((uint)((ElementCount(output.Tensor) + threads - 1) / threads));
                    if (groups > 65535)
                        throw new NotSupportedException($"Node '{node.Id}' exceeds the configured one-dimensional GPU dispatch grid.");
                    instruction = new ExecutionPlanDispatch(definition.Id, arguments, new(groups));
                }
            }
            var id = node.Id.Value;
            sequence.Add(new ExecutionPlanNode(id, instruction, previous is null ? [] : [previous],
                node.Source.LogicalNodes.Select(source => source.Value)));
            nodeRegions.Add(id, node.Region);
            previous = id;
            if (target == VmTarget.Direct3D12)
            {
                var writes = node.Resources.Where(binding => binding.Access != GraphResourceAccess.Read)
                    .Select(binding => physical[binding.Resource]).Distinct(StringComparer.Ordinal).ToArray();
                var barrierId = $"{id}.barrier";
                sequence.Add(new ExecutionPlanNode(barrierId, new ExecutionPlanBarrier(writes), [id]));
                nodeRegions.Add(barrierId, node.Region);
                previous = barrierId;
            }
        }
        var forwardNodes = il.Structure is { LayerDefinitions.Count: > 0 }
            ? GroupRegions(execution.Regions, nodeRegions, sequence, rootParameters, definitions) : sequence;
        definitions.Add(new ExecutionPlanDefinition("forward", ExecutionPlanDefinitionKind.Orchestration, rootParameters, forwardNodes));
        var entries = new List<ExecutionPlanEntry>
        {
            new("forward", "forward", slots.Select(slot => new ExecutionPlanArgument(slot.Id, slot.Id))),
        };
        if (il.Model.Attributes.GetValueOrDefault("execution.token-prefill") == "true" &&
            il.Inputs.Count == 1 && descriptors[il.Inputs[0]].Tensor.ElementType == GraphElementType.Int32 &&
            descriptors[il.Inputs[0]].Tensor.Dimensions.SequenceEqual([1]))
        {
            var tokenSlot = physical[il.Inputs[0]];
            var index = slots.FindIndex(slot => slot.Id == tokenSlot);
            slots[index] = slots[index] with { Tensor = new TensorDescriptor(GraphElementType.Int32, [options.PrefillCapacity]) };
            for (var count = 1; count <= options.PrefillCapacity; count *= 2)
            {
                var prefillParameters = slots.Select(slot => new ExecutionPlanParameter(slot.Id, slot.Access,
                    slot.Id == tokenSlot ? new TensorDescriptor(GraphElementType.Int32, [count]) : slot.Tensor)).ToArray();
                var child = count == 1 ? "forward" : $"prefill.{count / 2}";
                var calls = Enumerable.Range(0, count == 1 ? 1 : 2).Select(position => new ExecutionPlanNode($"token.{position}",
                    new ExecutionPlanCall(child, slots.Select(slot => new ExecutionPlanArgument(slot.Id, slot.Id,
                        slot.Id == tokenSlot ? checked((ulong)position * (ulong)(count / 2) * sizeof(int)) : 0))),
                    position == 0 ? [] : [$"token.{position - 1}"])).ToArray();
                var id = $"prefill.{count}";
                definitions.Add(new ExecutionPlanDefinition(id, ExecutionPlanDefinitionKind.Orchestration, prefillParameters, calls));
                entries.Add(new ExecutionPlanEntry(id, id, slots.Select(slot => new ExecutionPlanArgument(slot.Id, slot.Id))));
            }
        }
        var plan = new TargetExecutionPlan(il.Identity.Name, $"vm:{il.Model.StateAbiId}",
            target == VmTarget.Cpu ? ExecutionPlanTarget.Cpu : ExecutionPlanTarget.Direct3D12,
            slots, definitions, entries, il.GraphState.Schema.Name, 1,
            il.GraphState.Entries.Select(entry => new ExecutionPlanStateEntry(entry.Name, physical[entry.Resource])),
            TargetExecutionPlan.CreateBindings(il),
            physical.Select(pair => new ExecutionPlanResourceBinding(pair.Key.Value, pair.Value,
                descriptors[pair.Key].Tensor, IsView: aliases.ContainsKey(pair.Key))));
        return new(plan, selection.Report, new(matVecCalls, cooperativeCalls,
            Array.AsReadOnly(matVecDiagnostics.Order(StringComparer.Ordinal).ToArray())));
    }

    private static (IReadOnlyList<ExecutionNode> Nodes, Dictionary<ResourceId, ResourceId> Aliases) LowerViews(
        IExecutionPlanningGraph graph, Dictionary<ResourceId, GraphResource> resources, bool nativeHalf, bool weightViews, bool tensorViews)
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
        ResourceId Base(ResourceId id) => aliases.TryGetValue(id, out var baseId) ? baseId : id;
        var positions = graph.Nodes.Select((node, index) => (node.Id, index)).ToDictionary(pair => pair.Id, pair => pair.index);
        foreach (var node in graph.Nodes.Where(node => node.Operation == PortableTensorOperationContracts.Reshape &&
                     !removed.ContainsKey(node.Id)))
        {
            var input = node.Resources.Single(binding => binding.Port == "input").Resource;
            var output = node.Resources.Single(binding => binding.Port == "output").Resource;
            var baseId = Base(input);
            if (!(resources[baseId].Kind == GraphResourceKind.Weight ? weightViews : tensorViews) ||
                ExternallyVisible(output) || !ReadOnlyUses(output, node.Id) ||
                resources[input].Tensor.Layout != "dense" || resources[output].Tensor.Layout != "dense")
                continue;
            var uses = consumers[output].Where(pair => pair.node.Id != node.Id).ToArray();
            if (uses.Length == 0 || uses.Any(pair => positions[pair.node.Id] <= positions[node.Id]))
                continue;
            var last = uses.Max(pair => positions[pair.node.Id]);
            if (graph.Nodes.Skip(positions[node.Id] + 1).Take(last - positions[node.Id])
                .Where(other => !removed.ContainsKey(other.Id))
                .Any(other => other.Resources.Any(binding =>
                    binding.Access != GraphResourceAccess.Read && Base(binding.Resource) == baseId)))
                continue;
            aliases[output] = baseId;
            resources[output] = View(resources[output], resources[input].Tensor.ElementType);
            removed[node.Id] = node;
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

    private static ulong ElementCount(TensorDescriptor tensor) =>
        tensor.Dimensions.Aggregate(1UL, (count, size) => checked(count * (ulong)size));

    private static List<ExecutionPlanNode> GroupRegions(IReadOnlyList<GraphRegion> regions,
        IReadOnlyDictionary<string, RegionId> nodeRegions, IReadOnlyList<ExecutionPlanNode> nodes,
        IReadOnlyList<ExecutionPlanParameter> rootParameters, List<ExecutionPlanDefinition> definitions)
    {
        var byId = regions.ToDictionary(region => region.Id);
        string Group(ExecutionPlanNode node)
        {
            var region = byId[nodeRegions[node.Id]];
            var outer = region;
            while (region.ParentId is { } parent)
            {
                if (region.Type == GraphRegionTypes.Layer) return region.Id.Value;
                outer = region;
                region = byId[parent];
            }
            return outer.Id.Value;
        }
        var result = new List<ExecutionPlanNode>();
        var templates = new Dictionary<string, int>(StringComparer.Ordinal);
        var rootByName = rootParameters.ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);
        var offset = 0;
        while (offset < nodes.Count)
        {
            var group = Group(nodes[offset]);
            var end = offset + 1;
            while (end < nodes.Count && Group(nodes[end]) == group) end++;
            var body = nodes.Skip(offset).Take(end - offset).ToArray();
            var bodyIds = body.Select((node, index) => (node.Id, Name: $"step.{index:D4}"))
                .ToDictionary(pair => pair.Id, pair => pair.Name, StringComparer.Ordinal);
            var sources = body.SelectMany(node => node.Instruction switch
            {
                ExecutionPlanCall call => call.Arguments.Select(argument => argument.Source),
                ExecutionPlanDispatch dispatch => dispatch.Arguments.Select(argument => argument.Source),
                ExecutionPlanBarrier barrier => barrier.Resources,
                _ => throw new InvalidDataException("Region grouping requires an already lowered target sequence."),
            }).Distinct(StringComparer.Ordinal).ToArray();
            var formals = sources.Select((source, index) => (source, Name: $"slot.{index:D4}"))
                .ToDictionary(pair => pair.source, pair => pair.Name, StringComparer.Ordinal);
            var parameters = sources.Select(source =>
                rootByName[source] with { Name = formals[source] }).ToArray();
            ExecutionPlanArgument Argument(ExecutionPlanArgument argument) =>
                argument with { Source = formals[argument.Source] };
            var normalized = body.Select(node => new ExecutionPlanNode(bodyIds[node.Id], node.Instruction switch
            {
                ExecutionPlanCall call => new ExecutionPlanCall(call.Definition, call.Arguments.Select(Argument)),
                ExecutionPlanDispatch dispatch => new ExecutionPlanDispatch(dispatch.Definition,
                    dispatch.Arguments.Select(Argument), dispatch.Groups),
                ExecutionPlanBarrier barrier => new ExecutionPlanBarrier(barrier.Resources.Select(source => formals[source])),
                _ => throw new InvalidDataException("Unexpected unlowered region instruction."),
            }, node.Dependencies.Where(bodyIds.ContainsKey).Select(dependency => bodyIds[dependency]), node.Sources)).ToArray();
            XElement Arguments(IEnumerable<ExecutionPlanArgument> arguments) => new("Arguments", arguments.Select(argument =>
                new XElement("Argument", new XAttribute("parameter", argument.Parameter),
                    new XAttribute("source", argument.Source), new XAttribute("offset", argument.ByteOffset))));
            var key = new XElement("Region",
                parameters.Select(parameter => new XElement("Parameter", new XAttribute("access", parameter.Access),
                    new XAttribute("type", parameter.Tensor.ElementType),
                    new XAttribute("shape", string.Join(",", parameter.Tensor.Dimensions)))),
                normalized.Select(node => new XElement("Node",
                    new XAttribute("dependencies", string.Join(",", node.Dependencies)), node.Instruction switch
                    {
                        ExecutionPlanCall call => new XElement("Call", new XAttribute("definition", call.Definition), Arguments(call.Arguments)),
                        ExecutionPlanDispatch dispatch => new XElement("Dispatch", new XAttribute("definition", dispatch.Definition),
                            new XAttribute("groups", $"{dispatch.Groups.X},{dispatch.Groups.Y},{dispatch.Groups.Z}"), Arguments(dispatch.Arguments)),
                        ExecutionPlanBarrier barrier => new XElement("Barrier", new XAttribute("resources", string.Join(",", barrier.Resources))),
                        _ => throw new InvalidDataException("Unexpected region instruction."),
                    }))).ToString(SaveOptions.DisableFormatting);
            string id;
            if (templates.TryGetValue(key, out var template))
            {
                var existing = definitions[template];
                id = existing.Id;
                definitions[template] = new(existing.Id, existing.Kind, existing.Parameters,
                    existing.Nodes.Select((node, index) => new ExecutionPlanNode(node.Id, node.Instruction,
                        node.Dependencies, node.Sources.Concat(normalized[index].Sources).Distinct(StringComparer.Ordinal))));
            }
            else
            {
                id = $"region.{templates.Count:D4}";
                templates.Add(key, definitions.Count);
                definitions.Add(new(id, ExecutionPlanDefinitionKind.Orchestration, parameters, normalized));
            }
            result.Add(new($"region-call.{result.Count:D4}", new ExecutionPlanCall(id,
                sources.Select(source => new ExecutionPlanArgument(formals[source], source))),
                result.Count == 0 ? [] : [result[^1].Id], [group]));
            offset = end;
        }
        return result;
    }
}
