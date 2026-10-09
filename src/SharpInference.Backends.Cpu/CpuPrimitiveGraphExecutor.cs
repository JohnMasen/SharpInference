using System.Buffers;
using System.Diagnostics;
using System.Numerics.Tensors;
using SharpInference.Graphs;

namespace SharpInference.Backends.Cpu;

public sealed record CpuPrimitiveGraphResult(
    IReadOnlyDictionary<ResourceId, Array> Outputs,
    IReadOnlyList<GraphStateValue> State);

/// <summary>Executes unfused, dense primitive graphs with explicit inputs and state snapshots.</summary>
public sealed class CpuPrimitiveGraphExecutor
{
    private static readonly IReadOnlySet<GraphOperationId> SupportedPrimitives =
        new HashSet<GraphOperationId>
        {
            PrimitiveGraphOperations.Copy, PrimitiveGraphOperations.Add,
            PrimitiveGraphOperations.Subtract, PrimitiveGraphOperations.Multiply,
            PrimitiveGraphOperations.Divide, PrimitiveGraphOperations.Maximum,
            PrimitiveGraphOperations.Exp, PrimitiveGraphOperations.Tanh,
            PrimitiveGraphOperations.Sigmoid, PrimitiveGraphOperations.ReciprocalSquareRoot,
            PrimitiveGraphOperations.Square, PrimitiveGraphOperations.Relu,
            PrimitiveGraphOperations.ReduceSum, PrimitiveGraphOperations.ReduceMean,
            PrimitiveGraphOperations.MatVec, PrimitiveGraphOperations.GatherRow,
        };

    private readonly ExecutionGraph graph;
    private readonly IPrimitiveOperatorBackend backend;
    private readonly Dictionary<ResourceId, GraphResource> resources;
    private readonly Dictionary<ResourceId, Array> weights = [];
    private readonly Dictionary<ResourceId, IModelTensor> retainedWeights = [];
    private readonly ExecutionNode[] nodes;
    private readonly (ExecutionNode Node, IReadOnlyDictionary<string, int> Ports,
        TensorParameters Parameters, int AliasSource, bool CachedCast)[] dispatchPlan;
    private readonly Dictionary<ResourceId, int> slots;
    private readonly GraphResource[] descriptors;
    private readonly Array?[] workspace;
    private readonly IModelTensor?[] borrowed;
    private readonly bool[] constantCastReady;
    private readonly bool[] nativeHalf;
    private readonly object workspaceGate = new();
    private bool workspaceReady;

    private enum ExpressionOp : byte
    {
        Copy, Add, Subtract, Multiply, Divide, Maximum, Exp, Tanh, Sigmoid, Rsqrt, Square, Relu,
    }

    private readonly record struct FusedStep(ExpressionOp Operation, int Left, int Right);

    private readonly record struct TensorParameters(
        float FillValue, int SliceAxis, int SliceStart, int SliceLength, int SliceOuter, int SliceInner,
        FusedStep[]? Steps = null, bool StraightLine = false);

    public ExecutionGraph Graph => graph;

    /// <summary>
    /// Borrows weight tensors from the model catalog without recopying them.
    /// The caller must keep that catalog alive for the executor's lifetime.
    /// </summary>
    public CpuPrimitiveGraphExecutor(PortableGraphModel model,
        IPrimitiveOperatorBackend? backend = null)
        : this((model ?? throw new ArgumentNullException(nameof(model))).Graph, model.Tensors, backend,
            retainCatalogWeights: true)
    {
    }

    public CpuPrimitiveGraphExecutor(LogicalGraph logical, IModelTensorCatalog catalog,
        IPrimitiveOperatorBackend? backend = null)
        : this(new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded)), catalog, backend)
    {
    }

    public CpuPrimitiveGraphExecutor(ExecutionGraph graph, IModelTensorCatalog catalog,
        IPrimitiveOperatorBackend? backend = null)
        : this(graph, catalog ?? throw new ArgumentNullException(nameof(catalog)), backend,
            retainCatalogWeights: false)
    {
    }

    internal static void Validate(ExecutionGraph graph, IPrimitiveOperatorBackend backend) =>
        _ = new CpuPrimitiveGraphExecutor(graph, null, backend, retainCatalogWeights: false);

    private CpuPrimitiveGraphExecutor(ExecutionGraph graph, IModelTensorCatalog? catalog,
        IPrimitiveOperatorBackend? backend, bool retainCatalogWeights)
    {
        this.graph = graph ?? throw new ArgumentNullException(nameof(graph));
        if (graph.Nodes.Any(node => node.Resources.Any(binding => binding.View is not null)))
            throw new NotSupportedException("The primitive reference executor does not support tensor-view bindings; use a VM adapter.");
        this.backend = backend ?? CpuPrimitiveOperatorBackend.Instance;
        resources = graph.Resources.ToDictionary(resource => resource.Id);
        if (graph.Inputs.Any(id => resources[id].Kind != GraphResourceKind.Input) ||
            graph.Outputs.Any(id => resources[id].Kind != GraphResourceKind.Output))
            throw new InvalidDataException("Graph endpoints must reference input and output resources respectively.");
        descriptors = graph.Resources.ToArray();
        slots = descriptors.Select((resource, index) => (resource.Id, index))
            .ToDictionary(item => item.Id, item => item.index);
        workspace = new Array?[descriptors.Length];
        borrowed = new IModelTensor?[descriptors.Length];
        constantCastReady = new bool[descriptors.Length];
        nativeHalf = new bool[descriptors.Length];
        if (graph.Nodes.Any(node => node.InternalResources.Count != 0))
            throw new NotSupportedException("Primitive execution does not support fused internal resources.");
        if (graph.Resources.Any(resource => resource.Tensor.Layout != "dense" ||
            resource.Tensor.ElementType is not (GraphElementType.Float32 or GraphElementType.Float16 or GraphElementType.Int32) ||
            resource.Kind == GraphResourceKind.Constant))
            throw new NotSupportedException("Primitive execution requires dense float tensors and int32 indices; constants are not supported.");
        foreach (var resource in graph.Resources)
            if (resource.Tensor.Dimensions.Aggregate(1L,
                    (count, dimension) => checked(count * dimension)) > int.MaxValue)
                throw new InvalidDataException($"Resource '{resource.Id}' exceeds the CPU array length limit.");
        if (graph.Resources.Count(resource => resource.Kind == GraphResourceKind.SessionState) !=
            graph.GraphState.Slots.Count ||
            graph.GraphState.Slots.Any(slot =>
                resources[slot.Resource].Tensor.ElementType != GraphElementType.Float32))
            throw new NotSupportedException("Primitive execution requires GraphState to cover all float32 session resources.");

        foreach (var resource in graph.Resources.Where(resource => resource.Kind == GraphResourceKind.Weight))
        {
            if (catalog is null) continue;
            var tensor = catalog.GetRequired(resource.BindingKey!);
            var type = tensor.DataType switch
            {
                TensorDataType.Float32 => GraphElementType.Float32,
                TensorDataType.Float16 => GraphElementType.Float16,
                _ => throw new NotSupportedException($"Weight '{resource.Id}' has unsupported storage type."),
            };
            if (type != resource.Tensor.ElementType ||
                !tensor.Dimensions.SequenceEqual(resource.Tensor.Dimensions))
                throw new InvalidDataException($"Weight '{resource.Id}' does not match its graph descriptor.");
            var length = resource.Tensor.Dimensions.Aggregate(1L,
                (count, dimension) => checked(count * dimension));
            if ((type == GraphElementType.Float32 ? tensor.FloatValues.Length : tensor.HalfValues.Length) != length)
                throw new InvalidDataException($"Weight '{resource.Id}' does not match its graph descriptor.");
            if (retainCatalogWeights)
                retainedWeights.Add(resource.Id, tensor);
            else
                weights.Add(resource.Id, type == GraphElementType.Float32
                    ? tensor.FloatValues.ToArray() : tensor.HalfValues.ToArray());
        }

        var remaining = graph.Nodes.ToDictionary(node => node.Id);
        var scheduled = new List<ExecutionNode>();
        var initialized = graph.Inputs.Concat(graph.Resources
                .Where(resource => resource.Kind == GraphResourceKind.Weight).Select(resource => resource.Id))
            .Concat(graph.GraphState.Slots.Select(slot => slot.Resource)).ToHashSet();
        while (remaining.Count != 0)
        {
            var ready = graph.Nodes.FirstOrDefault(node => remaining.ContainsKey(node.Id) &&
                node.Dependencies.All(dependency => !remaining.ContainsKey(dependency)));
            if (ready is null) throw new InvalidDataException("Graph dependencies cannot be scheduled.");
            ValidateNode(ready, initialized);
            scheduled.Add(ready);
            remaining.Remove(ready.Id);
        }
        if (graph.Outputs.Any(output => !initialized.Contains(output)))
            throw new InvalidDataException("Graph output is not initialized.");
        nodes = scheduled.ToArray();
        var writeCounts = nodes.SelectMany(node => node.Resources
                .Where(binding => binding.Access != GraphResourceAccess.Read)
                .Select(binding => binding.Resource))
            .GroupBy(id => id).ToDictionary(group => group.Key, group => group.Count());
        var constants = graph.Resources.Where(resource => resource.Kind == GraphResourceKind.Weight)
            .Select(resource => resource.Id).ToHashSet();
        var castBySource = new Dictionary<ResourceId, ResourceId>();
        var readersByResource = nodes.SelectMany(node => node.Resources
                .Where(binding => binding.Access != GraphResourceAccess.Write)
                .Select(binding => (Node: node, Binding: binding)))
            .ToLookup(reader => reader.Binding.Resource);
        bool CanConsumeNativeHalf(ResourceId id)
        {
            if (writeCounts.GetValueOrDefault(id) != 1 ||
                resources[id].Scope != GraphResourceScope.Local ||
                resources[id].Kind is not (GraphResourceKind.Temporary or GraphResourceKind.TokenTransient) ||
                graph.Outputs.Contains(id))
                return false;
            var readers = readersByResource[id];
            return readers.Any() && readers.All(reader =>
                reader.Binding.Access == GraphResourceAccess.Read &&
                ((reader.Node.Operation == PrimitiveGraphOperations.MatVec &&
                  reader.Binding.Port == "matrix") ||
                 (reader.Node.Operation == PrimitiveGraphOperations.GatherRow &&
                  reader.Binding.Port == "table") ||
                 (reader.Node.Operation == PortableTensorOperationContracts.Reshape &&
                  reader.Binding.Port == "input" &&
                  reader.Node.Resources.Single(binding => binding.Port == "output") is { } reshapeOutput &&
                  CanConsumeNativeHalf(reshapeOutput.Resource))));
        }
        var planned = new List<(ExecutionNode Node, IReadOnlyDictionary<string, int> Ports,
            TensorParameters Parameters, int AliasSource, bool CachedCast)>();
        foreach (var node in nodes)
        {
            var bindings = (IReadOnlyDictionary<string, int>)node.Resources.ToDictionary(
                binding => binding.Port, binding => slots[binding.Resource], StringComparer.Ordinal);
            var output = node.Resources.SingleOrDefault(binding => binding.Port == "output");
            var source = node.Resources.SingleOrDefault(binding => binding.Port == "input");
            var safeConstant = output is not null && source is not null &&
                constants.Contains(source.Resource) &&
                writeCounts.GetValueOrDefault(output.Resource) == 1 &&
                resources[output.Resource].Scope == GraphResourceScope.Local &&
                resources[output.Resource].Kind is GraphResourceKind.Temporary or GraphResourceKind.TokenTransient;
            var alias = -1;
            var cachedCast = false;
            if (safeConstant && node.Operation == PortableTensorOperationContracts.Reshape)
            {
                alias = slots[source!.Resource];
                nativeHalf[slots[output!.Resource]] = nativeHalf[alias];
                constants.Add(output!.Resource);
            }
            else if (safeConstant && node.Operation == PortableTensorOperationContracts.CastFp16ToFp32)
            {
                if (this.backend is CpuPrimitiveOperatorBackend &&
                    CanConsumeNativeHalf(output!.Resource))
                {
                    alias = slots[source!.Resource];
                    nativeHalf[slots[output.Resource]] = true;
                }
                else if (castBySource.TryGetValue(source!.Resource, out var first))
                    alias = slots[first];
                else
                {
                    castBySource.Add(source.Resource, output!.Resource);
                    cachedCast = true;
                }
                constants.Add(output!.Resource);
            }
            planned.Add((node, bindings, GetParameters(node), alias, cachedCast));
        }
        dispatchPlan = planned.ToArray();
        if (catalog is null) return;
        foreach (var (id, buffer) in weights)
            workspace[slots[id]] = buffer;
        foreach (var (id, tensor) in retainedWeights)
            borrowed[slots[id]] = tensor;
    }

    public CpuPrimitiveGraphResult Execute(IReadOnlyDictionary<ResourceId, Array> inputs,
        IReadOnlyList<GraphStateValue>? state = null)
    {
        ValidateInputs(inputs);
        state ??= [];
        if (state.Count != graph.GraphState.Slots.Count)
            throw new InvalidDataException("State does not match GraphState.");

        foreach (var (slot, index) in graph.GraphState.Slots.Select((slot, index) => (slot, index)))
        {
            var entry = state[index];
            var resource = resources[slot.Resource];
            if (resource.Tensor.ElementType != GraphElementType.Float32 ||
                entry is null || entry.Name != slot.Name ||
                !entry.Dimensions.SequenceEqual(resource.Tensor.Dimensions))
                throw new InvalidDataException($"State slot '{slot.Name}' does not match GraphState.");
            CheckArray(resource, entry.Values);
        }
        lock (workspaceGate)
        {
            EnsureWorkspace();
            BindInputs(inputs);
            for (var index = 0; index < graph.GraphState.Slots.Count; index++)
            {
                var id = graph.GraphState.Slots[index].Resource;
                state[index].Values.CopyTo((float[])workspace[slots[id]]!, 0);
            }
            return RunBoundGraph();
        }
    }

    /// <summary>Executes a graph and commits its state only after successful execution.</summary>
    public CpuPrimitiveGraphResult Execute(IReadOnlyDictionary<ResourceId, Array> inputs,
        PortableGraphState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateInputs(inputs);
        var views = state.Views;
        if (views.Count != graph.GraphState.Slots.Count ||
            views.Where((view, index) => view.Name != graph.GraphState.Slots[index].Name).Any())
        {
            var snapshot = GraphSessionStateAccess.Read(graph.GraphState, graph.Resources, state);
            var fallback = Execute(inputs, snapshot);
            GraphSessionStateAccess.Write(graph.GraphState, graph.Resources, state, fallback.State);
            return fallback;
        }
        ValidateViews(views);
        lock (workspaceGate)
        {
            EnsureWorkspace();
            BindInputs(inputs);
            for (var index = 0; index < views.Count; index++)
                views[index].Values.CopyTo((float[])workspace[slots[graph.GraphState.Slots[index].Resource]]!, 0);
            var result = RunBoundGraph();
            for (var index = 0; index < views.Count; index++)
                ((float[])workspace[slots[graph.GraphState.Slots[index].Resource]]!).CopyTo(views[index].Values, 0);
            state.CommitViews();
            return result;
        }
    }

    /// <summary>Runs one invocation into a caller-owned FP32 output without allocating state snapshots.</summary>
    public void ExecuteInto(IReadOnlyDictionary<ResourceId, Array> inputs,
        PortableGraphState state, ResourceId output, Span<float> destination)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (!graph.Outputs.Contains(output) || resources[output].Tensor.ElementType != GraphElementType.Float32 ||
            destination.Length != Product(resources[output].Tensor.Dimensions))
            throw new ArgumentException($"Output '{output}' does not match the destination.", nameof(destination));
        ValidateInputs(inputs);
        var views = state.Views;
        if (views.Count != graph.GraphState.Slots.Count ||
            views.Where((view, index) => view.Name != graph.GraphState.Slots[index].Name).Any())
        {
            var result = Execute(inputs, state);
            ((float[])result.Outputs[output]).CopyTo(destination);
            return;
        }
        ValidateViews(views);
        lock (workspaceGate)
        {
            EnsureWorkspace();
            BindInputs(inputs);
            for (var index = 0; index < views.Count; index++)
                views[index].Values.CopyTo((float[])workspace[slots[graph.GraphState.Slots[index].Resource]]!, 0);
            DispatchNodes();
            ((float[])workspace[slots[output]]!).CopyTo(destination);
            for (var index = 0; index < views.Count; index++)
                ((float[])workspace[slots[graph.GraphState.Slots[index].Resource]]!).CopyTo(views[index].Values, 0);
            state.CommitViews();
        }
    }

    private void ValidateInputs(IReadOnlyDictionary<ResourceId, Array> inputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        if (inputs.Count != graph.Inputs.Count || inputs.Keys.Any(id => !graph.Inputs.Contains(id)))
            throw new InvalidDataException("Inputs do not match the graph input resources.");
        foreach (var id in graph.Inputs)
        {
            if (!inputs.TryGetValue(id, out var input))
                throw new InvalidDataException($"Missing input '{id}'.");
            CheckArray(resources[id], input);
        }
    }

    private void ValidateViews(IReadOnlyList<Float32StateView> views)
    {
        for (var index = 0; index < views.Count; index++)
        {
            var id = graph.GraphState.Slots[index].Resource;
            var view = views[index];
            if (!view.Dimensions.SequenceEqual(resources[id].Tensor.Dimensions) ||
                view.Values.Length != Product(resources[id].Tensor.Dimensions))
                throw new InvalidDataException($"State tensor '{view.Name}' does not match GraphState.");
        }
    }

    private void EnsureWorkspace()
    {
        if (workspaceReady) return;
        var deferred = dispatchPlan.Where(step => step.AliasSource >= 0 || step.CachedCast)
            .Select(step => step.Ports["output"]).ToHashSet();
        var required = nodes.SelectMany(node => node.Resources
                .Where(binding => binding.Access != GraphResourceAccess.Read)
                .Select(binding => binding.Resource))
            .Concat(graph.GraphState.Slots.Select(slot => slot.Resource)).ToHashSet();
        foreach (var resource in descriptors.Where(resource =>
                     resource.Kind != GraphResourceKind.Weight &&
                     resource.Kind != GraphResourceKind.Input && required.Contains(resource.Id)))
            if (!deferred.Contains(slots[resource.Id]))
                workspace[slots[resource.Id]] = NewArray(resource);
        workspaceReady = true;
    }

    private void BindInputs(IReadOnlyDictionary<ResourceId, Array> inputs)
    {
        foreach (var id in graph.Inputs)
            workspace[slots[id]] = inputs[id];
    }

    private CpuPrimitiveGraphResult RunBoundGraph()
    {
        DispatchNodes();
        return new CpuPrimitiveGraphResult(
            graph.Outputs.ToDictionary(id => id, id => (Array)workspace[slots[id]]!.Clone()),
            graph.GraphState.Slots.Select(slot =>
                new GraphStateValue(slot.Name, resources[slot.Resource].Tensor.Dimensions.ToArray(),
                    (float[])workspace[slots[slot.Resource]]!.Clone())).ToArray());
    }

    private void DispatchNodes()
    {
        foreach (var (node, portMap, parameters, aliasSource, cachedCast) in dispatchPlan)
        {
            if (aliasSource >= 0)
            {
                var output = portMap["output"];
                workspace[output] = workspace[aliasSource];
                borrowed[output] = borrowed[aliasSource];
                continue;
            }
            if (cachedCast)
            {
                var output = portMap["output"];
                if (constantCastReady[output]) continue;
                workspace[output] ??= NewArray(descriptors[output]);
                Dispatch(node, portMap, parameters, workspace);
                constantCastReady[output] = true;
                continue;
            }
            Dispatch(node, portMap, parameters, workspace);
        }
    }

    private void ValidateNode(ExecutionNode node, HashSet<ResourceId> initialized)
    {
        var operation = node.Operation;
        if (operation == FusedElementwiseExpressionContract.Operation)
        {
            FusedElementwiseExpressionContract.Read(node, resources);
            foreach (var binding in node.Resources.Take(node.Resources.Count - 1))
                if (!initialized.Contains(binding.Resource))
                    throw new InvalidDataException(
                        $"Node '{node.Id}' reads uninitialized resource '{binding.Resource}'.");
            initialized.Add(node.Resources[^1].Resource);
            return;
        }
        if (operation.Name.StartsWith("core.tensor.", StringComparison.Ordinal))
        {
            PortableTensorOperationContracts.ValidateNode(
                new LogicalNode(new LogicalNodeId(node.Id.Value), operation, node.Region,
                    node.Resources, [], node.Attributes, node.Requirements), resources);
            foreach (var binding in node.Resources.Where(binding => binding.Access == GraphResourceAccess.Read))
                if (!initialized.Contains(binding.Resource))
                    throw new InvalidDataException(
                        $"Node '{node.Id}' reads uninitialized resource '{binding.Resource}'.");
            initialized.Add(node.Resources.Single(binding => binding.Port == "output").Resource);
            return;
        }
        var inputs = operation == PrimitiveGraphOperations.GatherRow
            ? new[] { "table", "index" }
            : operation == PrimitiveGraphOperations.MatVec ? new[] { "matrix", "input" }
            : operation is var binary && (binary == PrimitiveGraphOperations.Add ||
                binary == PrimitiveGraphOperations.Subtract || binary == PrimitiveGraphOperations.Multiply ||
                binary == PrimitiveGraphOperations.Divide || binary == PrimitiveGraphOperations.Maximum)
                ? new[] { "left", "right" } : new[] { "input" };
        if (!SupportedPrimitives.Contains(operation) ||
            !backend.PrimitiveOperators.Any(desc => desc.Operation == operation))
            throw new NotSupportedException($"Unsupported primitive '{operation}'.");
        if (node.Attributes.Count != 0)
            throw new InvalidDataException(
                $"Node '{node.Id}' has unsupported attributes for primitive '{operation}'.");
        var expected = inputs.Append("output").ToHashSet(StringComparer.Ordinal);
        if (node.Resources.Count != expected.Count ||
            node.Resources.Any(binding => !expected.Contains(binding.Port)) ||
            inputs.Any(port => node.Resources.SingleOrDefault(binding => binding.Port == port) is not
                { Access: GraphResourceAccess.Read }) ||
            node.Resources.SingleOrDefault(binding => binding.Port == "output") is not
                { Access: GraphResourceAccess.Write } output)
            throw new InvalidDataException($"Node '{node.Id}' has invalid primitive ports or access.");

        var source = inputs.Select(port => resources[node.Resources.Single(binding => binding.Port == port).Resource]).ToArray();
        var target = resources[output.Resource];
        var type = source[0].Tensor.ElementType;
        if (type is not (GraphElementType.Float16 or GraphElementType.Float32) ||
            target.Tensor.ElementType != type ||
            source.Skip(1).Any(resource => resource.Tensor.ElementType !=
                (operation == PrimitiveGraphOperations.GatherRow ? GraphElementType.Int32 : type)) ||
            !backend.OperatorImplementations.Any(implementation =>
                implementation.Operation == operation &&
                implementation.Signature.Matches(new OperatorSignature(
                    Enumerable.Repeat(type, inputs.Length == 2 && operation != PrimitiveGraphOperations.GatherRow ? 2 : 1),
                    [type])) &&
                NumericTypeCompatibility.Satisfies(implementation.Precision.ArithmeticType,
                    node.Requirements.MinimumArithmeticType) &&
                NumericTypeCompatibility.Satisfies(implementation.Precision.AccumulatorType,
                    node.Requirements.MinimumAccumulatorType)))
            throw new NotSupportedException($"Node '{node.Id}' has unsupported tensor types or precision.");

        int Size(GraphResource resource) => checked((int)resource.Tensor.Dimensions.Aggregate(
            1L, (size, dimension) => checked(size * dimension)));
        static string Shape(GraphResource resource) =>
            $"[{string.Join(",", resource.Tensor.Dimensions)}]";
        var nodeName = node.Source.LogicalNodes.Count == 0
            ? $"Node '{node.Id}'"
            : $"Node '{node.Id}' (logical: {string.Join(",", node.Source.LogicalNodes)})";
        var targetSize = Size(target);
        if (operation == PrimitiveGraphOperations.MatVec)
        {
            var dims = source[0].Tensor.Dimensions;
            if (dims.Count != 2 || source[1].Tensor.Dimensions.Count != 1 ||
                Size(source[1]) != dims[1] || target.Tensor.Dimensions.Count != 1 ||
                targetSize != dims[0])
                throw new InvalidDataException(
                    $"{nodeName} has invalid matvec dimensions: matrix {Shape(source[0])}, " +
                    $"input {Shape(source[1])}, output {Shape(target)}; expected [rows,columns] x [columns] -> [rows].");
        }
        else if (operation == PrimitiveGraphOperations.GatherRow)
        {
            var dims = source[0].Tensor.Dimensions;
            if (dims.Count != 2 || !source[1].Tensor.Dimensions.SequenceEqual([1]) ||
                target.Tensor.Dimensions.Count != 1 || targetSize != dims[1])
                throw new InvalidDataException(
                    $"{nodeName} has invalid gather dimensions: table {Shape(source[0])}, " +
                    $"index {Shape(source[1])}, output {Shape(target)}; expected [rows,columns], [1] -> [columns].");
        }
        else if (operation == PrimitiveGraphOperations.ReduceSum ||
                 operation == PrimitiveGraphOperations.ReduceMean)
        {
            if (!target.Tensor.Dimensions.SequenceEqual([1]))
                throw new InvalidDataException($"{nodeName} needs a scalar output [1], got {Shape(target)}.");
        }
        else if (source.Any(resource => !resource.Tensor.Dimensions.SequenceEqual(target.Tensor.Dimensions)))
            throw new InvalidDataException(
                $"{nodeName} has incompatible elementwise dimensions: " +
                $"inputs [{string.Join(", ", source.Select(Shape))}], output {Shape(target)}; " +
                "all shapes must be equal (broadcasting is not supported).");

        foreach (var binding in node.Resources.Where(binding => binding.Access == GraphResourceAccess.Read))
            if (!initialized.Contains(binding.Resource))
                throw new InvalidDataException($"Node '{node.Id}' reads uninitialized resource '{binding.Resource}'.");
        if (target.Kind is GraphResourceKind.Input or GraphResourceKind.Weight)
            throw new InvalidDataException($"Node '{node.Id}' writes a read-only external resource.");
        initialized.Add(output.Resource);
    }

    private void Dispatch(ExecutionNode node, IReadOnlyDictionary<string, int> bindings,
        TensorParameters parameters, Array?[] values)
    {
        var profiler = CpuPrimitiveGraphProfiler.Active;
        if (profiler is null)
        {
            DispatchCore(node, bindings, parameters, values);
            return;
        }
        var operation = node.Operation == PrimitiveGraphOperations.MatVec &&
            nativeHalf[bindings["matrix"]]
                ? "core.mat-vec/native-half"
                : node.Operation == PrimitiveGraphOperations.GatherRow &&
                  nativeHalf[bindings["table"]]
                    ? "core.gather-row/native-half" : node.Operation.Name;
        var started = Stopwatch.GetTimestamp();
        try
        {
            DispatchCore(node, bindings, parameters, values);
        }
        finally
        {
            profiler.Record(operation, started);
        }
    }

    private void DispatchCore(ExecutionNode node, IReadOnlyDictionary<string, int> bindings,
        TensorParameters parameters, Array?[] values)
    {
        if (node.Operation == FusedElementwiseExpressionContract.Operation)
        {
            DispatchExpression((float[])values[bindings["output"]]!, parameters, values);
            return;
        }
        if (node.Operation.Name.StartsWith("core.tensor.", StringComparison.Ordinal))
        {
            DispatchTensor(node, bindings, parameters, values);
            return;
        }
        int Id(string port) => bindings[port];
        Array Get(string port) => values[Id(port)]!;
        ReadOnlySpan<float> Float(string port)
        {
            var id = Id(port);
            return borrowed[id] is { } tensor
                ? tensor.FloatValues : (float[])values[id]!;
        }
        ReadOnlySpan<Half> HalfValue(string port)
        {
            var id = Id(port);
            return borrowed[id] is { } tensor
                ? tensor.HalfValues : (Half[])values[id]!;
        }
        var output = Get("output");
        if (output is float[] floats)
        {
            if (node.Operation == PrimitiveGraphOperations.GatherRow)
            {
                var index = ((int[])Get("index"))[0];
                if (nativeHalf[Id("table")])
                    CpuHalfMatrixOperators.GatherRow(HalfValue("table"), index, floats);
                else
                    backend.GatherRow(Float("table"), index, floats.Length, floats);
            }
            else if (node.Operation == PrimitiveGraphOperations.MatVec)
            {
                var input = Float("input");
                if (nativeHalf[Id("matrix")])
                    CpuHalfMatrixOperators.MatVec(HalfValue("matrix"), input, floats);
                else
                    backend.MatVec(Float("matrix"), input, floats, floats.Length, input.Length);
            }
            else if (node.Operation == PrimitiveGraphOperations.ReduceSum)
                floats[0] = backend.ReduceSum(Float("input"));
            else if (node.Operation == PrimitiveGraphOperations.ReduceMean)
                floats[0] = backend.ReduceMean(Float("input"));
            else if (node.Operation == PrimitiveGraphOperations.Copy) backend.Copy(Float("input"), floats);
            else if (node.Operation == PrimitiveGraphOperations.Add) backend.Add(Float("left"), Float("right"), floats);
            else if (node.Operation == PrimitiveGraphOperations.Subtract) backend.Subtract(Float("left"), Float("right"), floats);
            else if (node.Operation == PrimitiveGraphOperations.Multiply) backend.Multiply(Float("left"), Float("right"), floats);
            else if (node.Operation == PrimitiveGraphOperations.Divide) backend.Divide(Float("left"), Float("right"), floats);
            else if (node.Operation == PrimitiveGraphOperations.Maximum) backend.Maximum(Float("left"), Float("right"), floats);
            else if (node.Operation == PrimitiveGraphOperations.Exp) backend.Exp(Float("input"), floats);
            else if (node.Operation == PrimitiveGraphOperations.Tanh) backend.Tanh(Float("input"), floats);
            else if (node.Operation == PrimitiveGraphOperations.Sigmoid) backend.Sigmoid(Float("input"), floats);
            else if (node.Operation == PrimitiveGraphOperations.ReciprocalSquareRoot) backend.ReciprocalSquareRoot(Float("input"), floats);
            else if (node.Operation == PrimitiveGraphOperations.Square) backend.Square(Float("input"), floats);
            else if (node.Operation == PrimitiveGraphOperations.Relu) backend.Relu(Float("input"), floats);
        }
        else
        {
            var halves = (Half[])output;
            if (node.Operation == PrimitiveGraphOperations.GatherRow)
                backend.GatherRow(HalfValue("table"), ((int[])Get("index"))[0], halves.Length, halves);
            else if (node.Operation == PrimitiveGraphOperations.MatVec)
                backend.MatVec(HalfValue("matrix"), HalfValue("input"), halves,
                    halves.Length, HalfValue("input").Length);
            else if (node.Operation == PrimitiveGraphOperations.ReduceSum)
                halves[0] = backend.ReduceSum(HalfValue("input"));
            else if (node.Operation == PrimitiveGraphOperations.ReduceMean)
                halves[0] = backend.ReduceMean(HalfValue("input"));
            else if (node.Operation == PrimitiveGraphOperations.Copy) backend.Copy(HalfValue("input"), halves);
            else if (node.Operation == PrimitiveGraphOperations.Add) backend.Add(HalfValue("left"), HalfValue("right"), halves);
            else if (node.Operation == PrimitiveGraphOperations.Subtract) backend.Subtract(HalfValue("left"), HalfValue("right"), halves);
            else if (node.Operation == PrimitiveGraphOperations.Multiply) backend.Multiply(HalfValue("left"), HalfValue("right"), halves);
            else if (node.Operation == PrimitiveGraphOperations.Divide) backend.Divide(HalfValue("left"), HalfValue("right"), halves);
            else if (node.Operation == PrimitiveGraphOperations.Maximum) backend.Maximum(HalfValue("left"), HalfValue("right"), halves);
            else if (node.Operation == PrimitiveGraphOperations.Exp) backend.Exp(HalfValue("input"), halves);
            else if (node.Operation == PrimitiveGraphOperations.Tanh) backend.Tanh(HalfValue("input"), halves);
            else if (node.Operation == PrimitiveGraphOperations.Sigmoid) backend.Sigmoid(HalfValue("input"), halves);
            else if (node.Operation == PrimitiveGraphOperations.ReciprocalSquareRoot) backend.ReciprocalSquareRoot(HalfValue("input"), halves);
            else if (node.Operation == PrimitiveGraphOperations.Square) backend.Square(HalfValue("input"), halves);
            else if (node.Operation == PrimitiveGraphOperations.Relu) backend.Relu(HalfValue("input"), halves);
        }
    }

    private void DispatchExpression(float[] output, TensorParameters parameters, Array?[] values)
    {
        var instructions = parameters.Steps!;
        if (parameters.StraightLine)
        {
            ReadOnlySpan<float> Source(int slot) => slot < 0
                ? output
                : borrowed[slot] is { } tensor ? tensor.FloatValues : (float[])values[slot]!;
            foreach (var instruction in instructions)
            {
                var left = Source(instruction.Left);
                var right = IsBinary(instruction.Operation)
                    ? Source(instruction.Right) : ReadOnlySpan<float>.Empty;
                DispatchExpressionStep(instruction.Operation, left, right, output);
            }
            return;
        }

        const int blockLength = 256;
        var stride = Math.Min(output.Length, blockLength);
        var scratchLength = checked(instructions.Length * stride);
        var rented = scratchLength > 1024 ? ArrayPool<float>.Shared.Rent(scratchLength) : null;
        Span<float> scratch = rented is null
            ? stackalloc float[scratchLength] : rented.AsSpan(0, scratchLength);
        try
        {
            for (var offset = 0; offset < output.Length; offset += stride)
            {
                var length = Math.Min(stride, output.Length - offset);
                for (var stepIndex = 0; stepIndex < instructions.Length; stepIndex++)
                {
                    var instruction = instructions[stepIndex];
                    var left = ExpressionSource(instruction.Left, scratch, stride, offset, length, values);
                    var right = IsBinary(instruction.Operation)
                        ? ExpressionSource(instruction.Right, scratch, stride, offset, length, values)
                        : ReadOnlySpan<float>.Empty;
                    DispatchExpressionStep(instruction.Operation, left, right,
                        scratch.Slice(stepIndex * stride, length));
                }
                scratch.Slice((instructions.Length - 1) * stride, length)
                    .CopyTo(output.AsSpan(offset, length));
            }
        }
        finally
        {
            if (rented is not null) ArrayPool<float>.Shared.Return(rented);
        }
    }

    private ReadOnlySpan<float> ExpressionSource(int slot, Span<float> scratch,
        int stride, int offset, int length, Array?[] values) =>
        slot < 0 ? scratch.Slice(~slot * stride, length)
            : borrowed[slot] is { } tensor ? tensor.FloatValues.Slice(offset, length)
            : ((float[])values[slot]!).AsSpan(offset, length);

    private static bool IsBinary(ExpressionOp operation) =>
        operation is ExpressionOp.Add or ExpressionOp.Subtract or ExpressionOp.Multiply
            or ExpressionOp.Divide or ExpressionOp.Maximum;

    private void DispatchExpressionStep(ExpressionOp operation, ReadOnlySpan<float> left,
        ReadOnlySpan<float> right, Span<float> output)
    {
        switch (operation)
        {
            case ExpressionOp.Copy: backend.Copy(left, output); break;
            case ExpressionOp.Add: backend.Add(left, right, output); break;
            case ExpressionOp.Subtract: backend.Subtract(left, right, output); break;
            case ExpressionOp.Multiply: backend.Multiply(left, right, output); break;
            case ExpressionOp.Divide: backend.Divide(left, right, output); break;
            case ExpressionOp.Maximum: backend.Maximum(left, right, output); break;
            case ExpressionOp.Exp: backend.Exp(left, output); break;
            case ExpressionOp.Tanh: backend.Tanh(left, output); break;
            case ExpressionOp.Sigmoid: backend.Sigmoid(left, output); break;
            case ExpressionOp.Rsqrt: backend.ReciprocalSquareRoot(left, output); break;
            case ExpressionOp.Square: backend.Square(left, output); break;
            case ExpressionOp.Relu: backend.Relu(left, output); break;
            default: throw new NotSupportedException($"Unsupported fused step '{operation}'.");
        }
    }

    private void DispatchTensor(ExecutionNode node, IReadOnlyDictionary<string, int> bindings,
        TensorParameters parameters, Array?[] values)
    {
        int Id(string port) => bindings[port];
        ReadOnlySpan<float> Float(string port)
        {
            var id = Id(port);
            return borrowed[id] is { } tensor
                ? tensor.FloatValues : (float[])values[id]!;
        }
        ReadOnlySpan<Half> HalfValue(string port)
        {
            var id = Id(port);
            return borrowed[id] is { } tensor
                ? tensor.HalfValues : (Half[])values[id]!;
        }
        var output = (float[])values[Id("output")]!;
        var operation = node.Operation;
        if (operation == PortableTensorOperationContracts.Fill)
        {
            output.AsSpan().Fill(parameters.FillValue);
        }
        else if (operation == PortableTensorOperationContracts.CastFp16ToFp32)
        {
            TensorPrimitives.ConvertToSingle(HalfValue("input"), output);
        }
        else if (operation == PortableTensorOperationContracts.Reshape)
        {
            backend.Copy(Float("input"), output);
        }
        else if (operation == PortableTensorOperationContracts.Slice)
        {
            var shape = descriptors[Id("input")].Tensor.Dimensions;
            var axis = parameters.SliceAxis;
            var start = parameters.SliceStart;
            var length = parameters.SliceLength;
            var outer = parameters.SliceOuter;
            var inner = parameters.SliceInner;
            var source = Float("input");
            for (var index = 0; index < outer; index++)
                backend.Copy(
                    source.Slice(checked((index * shape[axis] + start) * inner), checked(length * inner)),
                    output.AsSpan(checked(index * length * inner), checked(length * inner)));
        }
        else if (operation == PortableTensorOperationContracts.Broadcast)
        {
            var source = Float("input");
            var sourceShape = descriptors[Id("input")].Tensor.Dimensions;
            var outputShape = descriptors[Id("output")].Tensor.Dimensions;
            Span<nint> sourceLengths = sourceShape.Count <= 16
                ? stackalloc nint[sourceShape.Count] : new nint[sourceShape.Count];
            Span<nint> outputLengths = outputShape.Count <= 16
                ? stackalloc nint[outputShape.Count] : new nint[outputShape.Count];
            for (var axis = 0; axis < sourceShape.Count; axis++) sourceLengths[axis] = sourceShape[axis];
            for (var axis = 0; axis < outputShape.Count; axis++) outputLengths[axis] = outputShape[axis];
            var sourceTensor = new ReadOnlyTensorSpan<float>(source, sourceLengths);
            var outputTensor = new TensorSpan<float>(output, outputLengths);
            Tensor.BroadcastTo(sourceTensor, outputTensor);
        }
        else if (operation == PortableTensorOperationContracts.BatchedMatVec)
        {
            var shape = descriptors[Id("matrix")].Tensor.Dimensions;
            var matrix = Float("matrix");
            var vector = Float("vector");
            var rows = shape[1];
            var columns = shape[2];
            for (var batch = 0; batch < shape[0]; batch++)
                backend.MatVec(matrix.Slice(checked(batch * rows * columns), checked(rows * columns)),
                    vector.Slice(checked(batch * columns), columns),
                    output.AsSpan(checked(batch * rows), rows), rows, columns);
        }
        else if (operation == PortableTensorOperationContracts.ReduceLastSum ||
                 operation == PortableTensorOperationContracts.ReduceLastMean)
        {
            var shape = descriptors[Id("input")].Tensor.Dimensions;
            var width = shape[^1];
            var input = Float("input");
            for (var i = 0; i < output.Length; i++)
            {
                var row = input.Slice(checked(i * width), width);
                output[i] = operation == PortableTensorOperationContracts.ReduceLastSum
                    ? backend.ReduceSum(row) : backend.ReduceMean(row);
            }
        }
        else if (operation == PortableTensorOperationContracts.HeadOuter)
        {
            var left = Float("left");
            var right = Float("right");
            var shape = descriptors[Id("output")].Tensor.Dimensions;
            var leftWidth = shape[1];
            var rightWidth = shape[2];
            for (var head = 0; head < shape[0]; head++)
                for (var row = 0; row < leftWidth; row++)
                    TensorPrimitives.Multiply(right.Slice(head * rightWidth, rightWidth),
                        left[head * leftWidth + row],
                        output.AsSpan(checked((head * leftWidth + row) * rightWidth), rightWidth));
        }
    }

    private static int Product(IEnumerable<int> dimensions) =>
        dimensions.Aggregate(1, (size, dimension) => checked(size * dimension));

    private TensorParameters GetParameters(ExecutionNode node)
    {
        if (node.Operation == FusedElementwiseExpressionContract.Operation)
        {
            var expression = FusedElementwiseExpressionContract.Read(node, resources);
            int Source(ElementwiseOperand operand) => operand.StepIndex is int prior
                ? ~prior : slots[node.Resources[operand.InputIndex!.Value].Resource];
            var steps = expression.Steps.Select(instruction => new FusedStep(
                instruction.Operation == PrimitiveGraphOperations.Copy ? ExpressionOp.Copy
                    : instruction.Operation == PrimitiveGraphOperations.Add ? ExpressionOp.Add
                    : instruction.Operation == PrimitiveGraphOperations.Subtract ? ExpressionOp.Subtract
                    : instruction.Operation == PrimitiveGraphOperations.Multiply ? ExpressionOp.Multiply
                    : instruction.Operation == PrimitiveGraphOperations.Divide ? ExpressionOp.Divide
                    : instruction.Operation == PrimitiveGraphOperations.Maximum ? ExpressionOp.Maximum
                    : instruction.Operation == PrimitiveGraphOperations.Exp ? ExpressionOp.Exp
                    : instruction.Operation == PrimitiveGraphOperations.Tanh ? ExpressionOp.Tanh
                    : instruction.Operation == PrimitiveGraphOperations.Sigmoid ? ExpressionOp.Sigmoid
                    : instruction.Operation == PrimitiveGraphOperations.ReciprocalSquareRoot ? ExpressionOp.Rsqrt
                    : instruction.Operation == PrimitiveGraphOperations.Square ? ExpressionOp.Square
                    : instruction.Operation == PrimitiveGraphOperations.Relu ? ExpressionOp.Relu
                    : throw new NotSupportedException($"Unsupported fused step '{instruction.Operation}'."),
                Source(instruction.Operands[0]),
                instruction.Operands.Count == 2 ? Source(instruction.Operands[1]) : 0)).ToArray();
            var straightLine = steps.Select((step, index) => (step, index)).All(item =>
                (item.step.Left >= 0 || ~item.step.Left == item.index - 1) &&
                (item.step.Right >= 0 || ~item.step.Right == item.index - 1));
            return new TensorParameters(0, 0, 0, 0, 0, 0, steps, straightLine);
        }
        if (node.Operation == PortableTensorOperationContracts.Fill)
            return new TensorParameters(
                float.Parse(node.Attributes["value"], System.Globalization.CultureInfo.InvariantCulture),
                0, 0, 0, 0, 0);
        if (node.Operation != PortableTensorOperationContracts.Slice) return default;
        var inputId = node.Resources.Single(binding => binding.Port == "input").Resource;
        var shape = resources[inputId].Tensor.Dimensions;
        var axis = int.Parse(node.Attributes["axis"], System.Globalization.CultureInfo.InvariantCulture);
        return new TensorParameters(0, axis,
            int.Parse(node.Attributes["start"], System.Globalization.CultureInfo.InvariantCulture),
            int.Parse(node.Attributes["length"], System.Globalization.CultureInfo.InvariantCulture),
            Product(shape.Take(axis)), Product(shape.Skip(axis + 1)));
    }

    private static Array NewArray(GraphResource resource)
    {
        var length = checked((int)resource.Tensor.Dimensions.Aggregate(1L,
            (size, dimension) => checked(size * dimension)));
        return resource.Tensor.ElementType == GraphElementType.Float32
            ? new float[length] : new Half[length];
    }

    private static void CheckArray(GraphResource resource, Array array)
    {
        var expected = resource.Tensor.ElementType switch
        {
            GraphElementType.Float32 => typeof(float[]),
            GraphElementType.Float16 => typeof(Half[]),
            GraphElementType.Int32 => typeof(int[]),
            _ => throw new NotSupportedException($"Unsupported resource '{resource.Id}'."),
        };
        if (array is null || array.GetType() != expected ||
            array.Length != resource.Tensor.Dimensions.Aggregate(1L,
                (size, dimension) => checked(size * dimension)))
            throw new InvalidDataException($"Resource '{resource.Id}' does not match its graph descriptor.");
    }
}
