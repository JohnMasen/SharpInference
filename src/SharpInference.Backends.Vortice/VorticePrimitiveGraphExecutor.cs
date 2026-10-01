using System.Diagnostics;
using SharpInference.Graphs;
using Vortice.Direct3D12;
using Vortice.Dxc;
using SharpInference;

namespace SharpInference.Backends.Vortice;

/// <summary>Owns GPU model buffers and serializes command recording across independent sessions.</summary>
public sealed class VorticePrimitiveGraphExecutor : IModelWeightOwnershipPolicy, IDisposable
{
    private readonly ID3D12Device device;
    private readonly ID3D12CommandQueue queue;
    private readonly ID3D12RootSignature signature;
    private readonly ID3D12CommandAllocator allocator;
    private readonly ID3D12GraphicsCommandList commands;
    private readonly ID3D12Fence fence;
    private readonly EventWaitHandle completion = new(false, EventResetMode.AutoReset);
    private readonly Dictionary<string, ID3D12PipelineState> pipelines = new(StringComparer.Ordinal);
    private readonly Dictionary<ResourceId, ID3D12Resource> modelBuffers = [];
    private readonly VorticePrimitiveBufferPlanner localBuffers;
    private readonly ulong logicalCastOutputBytes;
    private readonly HashSet<ResourceId> uploadedWeights = [];
    private readonly object gate = new();
    private VorticePrimitiveGraphMetrics metrics = new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    private ulong fenceValue;
    private ulong modelBufferBytes;
    private ulong activeSessionBufferBytes;
    private ulong peakBufferBytes;
    private VorticePrimitiveGraphExecutionProfile? lastExecutionProfile;
    private bool enableCommandReplay;
    private bool disposed;

    public VorticePrimitiveGraphExecutor(ID3D12Device device, ExecutionGraph graph)
        : this(device, VorticePrimitiveGraphPlan.Compile(graph)) { }

    public VorticePrimitiveGraphExecutor(ID3D12Device device, VorticePrimitiveGraphPlan plan)
    {
        this.device = device ?? throw new ArgumentNullException(nameof(device));
        Plan = plan ?? throw new ArgumentNullException(nameof(plan));
        localBuffers = VorticePrimitiveBufferPlanner.Create(plan);
        var graphResources = plan.Graph.Resources.ToDictionary(item => item.Id);
        logicalCastOutputBytes = plan.Steps.Where(step => step.Kernel == "CastFp16ToFp32")
            .Aggregate(0UL, (sum, step) => checked(sum + Bytes(graphResources[step.Output])));
        queue = device.CreateCommandQueue(CommandListType.Compute);
        fence = device.CreateFence();
        signature = device.CreateRootSignature(new RootSignatureDescription(
            RootSignatureFlags.None,
            [
                .. Enumerable.Range(0, VorticeFusedExpressionKernel.MaximumInputs)
                    .Select(index => new RootParameter(RootParameterType.ShaderResourceView,
                        new RootDescriptor(checked((uint)index), 0), ShaderVisibility.All)),
                new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new RootParameter(new RootConstants(0, 0, 12), ShaderVisibility.All),
            ]), RootSignatureVersion.Version10);
        allocator = device.CreateCommandAllocator(CommandListType.Compute);
        commands = device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, allocator);
        commands.Close();
        foreach (var resource in plan.Graph.Resources.Where(item => item.Scope == GraphResourceScope.Global))
            modelBuffers.Add(resource.Id, CreateBuffer(resource));
        modelBufferBytes = modelBuffers.Values.Aggregate(0UL,
            (sum, buffer) => checked(sum + buffer.Description.Width));
        peakBufferBytes = modelBufferBytes;
    }

    public VorticePrimitiveGraphPlan Plan { get; }
    public int DispatchesPerToken => Plan.Steps.Count - localBuffers.ElidedSteps.Count;
    public bool ProfileCommandRecording { get; set; }
    public bool EnableCommandReplay
    {
        get => enableCommandReplay;
        set
        {
            lock (gate)
            {
                ThrowIfDisposed();
                if (value) ValidateReplay(Plan);
                enableCommandReplay = value;
            }
        }
    }
    public VorticePrimitiveGraphExecutionProfile? LastExecutionProfile
    {
        get { lock (gate) return lastExecutionProfile; }
    }
    public VorticePrimitiveGraphMetrics Metrics
    {
        get { lock (gate) return metrics; }
    }
    public VorticePrimitiveGraphMemoryMetrics Memory
    {
        get
        {
            lock (gate)
                return new VorticePrimitiveGraphMemoryMetrics(
                    modelBufferBytes, activeSessionBufferBytes, peakBufferBytes,
                    localBuffers.Capacities.Aggregate(0UL,
                        (sum, bytes) => checked(sum + ((bytes + 3UL) & ~3UL))),
                    logicalCastOutputBytes);
        }
    }

    /// <summary>
    /// The executor retains default-heap GPU copies only. Call <see cref="PrepareModelWeights"/>
    /// during model binding, before the source catalog or reader is released.
    /// </summary>
    public bool RequiresCpuWeightCopy => false;

    public static void ValidateReplay(VorticePrimitiveGraphPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        var graph = plan.Graph;
        if (graph.Inputs.Count != 1 ||
            graph.Resources.SingleOrDefault(resource => resource.Id == graph.Inputs[0]) is not { } input ||
            input.Tensor.ElementType != GraphElementType.Int32 ||
            !input.Tensor.Dimensions.SequenceEqual([1]) ||
            plan.Steps.All(step => step.Kernel != "GatherRow") ||
            plan.Steps.Any(step => step.Index is ResourceId index && index != input.Id))
            throw new NotSupportedException(
                "GPU command replay requires one Int32[1] token input used by GatherRow; " +
                "FP32 activation inputs and other dynamic scalar bindings require normal recording.");
    }

    /// <summary>
    /// Resolves graph weight binding keys against the model catalog and uploads FP16/FP32 weights.
    /// Does not retain the model, catalog, or source tensor arrays.
    /// </summary>
    public void PrepareModelWeights(PortableGraphModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (!ReferenceEquals(model.Graph, Plan.Graph))
            throw new ArgumentException("The model belongs to a different execution graph.", nameof(model));
        lock (gate)
        {
            ThrowIfDisposed();
            var weights = Plan.Graph.Resources
                .Where(resource => resource.Kind == GraphResourceKind.Weight)
                .Select(resource =>
                {
                    var tensor = model.Tensors.GetRequired(resource.BindingKey!);
                    if (resource.Tensor.ElementType != (tensor.DataType switch
                        {
                            RwkvTensorDataType.Float16 => GraphElementType.Float16,
                            RwkvTensorDataType.Float32 => GraphElementType.Float32,
                            _ => throw new InvalidDataException($"Unsupported tensor type for '{resource.Id}'."),
                        }) ||
                        !resource.Tensor.Dimensions.SequenceEqual(tensor.Dimensions))
                        throw new InvalidDataException(
                            $"Graph weight '{resource.Id}' (binding '{resource.BindingKey}') requires matching type and dimensions.");
                    return (resource.Id, Tensor: tensor);
                }).ToArray();
            foreach (var (id, tensor) in weights)
            {
                if (tensor.DataType == RwkvTensorDataType.Float16)
                    UploadWeight(id, tensor.HalfValues);
                else
                    UploadWeight(id, tensor.FloatValues);
            }
        }
    }

    public void UploadWeight(ResourceId id, ReadOnlySpan<float> values)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            var resource = Describe(id);
            if (!modelBuffers.TryGetValue(id, out var target) ||
                resource.Kind is not (GraphResourceKind.Weight or GraphResourceKind.Constant))
                throw new ArgumentException($"Resource '{id}' is not a model weight or constant.", nameof(id));
            if (resource.Tensor.ElementType != GraphElementType.Float32)
                throw new ArgumentException($"Resource '{id}' requires FP16 weight data.", nameof(id));
            ValidateLength(resource, values.Length);
            using var upload = CreateUpload(values);
            Begin();
            RecordUpload(target, upload, Bytes(resource));
            Submit();
            metrics = metrics with
            {
                WeightCommandSubmissions = checked(metrics.WeightCommandSubmissions + 1),
                WeightUploadBytes = checked(metrics.WeightUploadBytes + Bytes(resource)),
            };
            uploadedWeights.Add(id);
        }
    }

    public void UploadWeight(ResourceId id, ReadOnlySpan<Half> values)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            var resource = Describe(id);
            if (!modelBuffers.TryGetValue(id, out var target) ||
                resource.Kind is not (GraphResourceKind.Weight or GraphResourceKind.Constant) ||
                resource.Tensor.ElementType != GraphElementType.Float16)
                throw new ArgumentException($"Resource '{id}' is not an FP16 model weight or constant.", nameof(id));
            if ((ulong)values.Length * sizeof(ushort) != Bytes(resource))
                throw new ArgumentException($"Resource '{id}' requires {Bytes(resource) / sizeof(ushort)} FP16 elements.", nameof(values));
            using var upload = CreateUpload(values);
            Begin();
            RecordUpload(target, upload, AllocatedBytes(resource));
            Submit();
            metrics = metrics with
            {
                WeightCommandSubmissions = checked(metrics.WeightCommandSubmissions + 1),
                WeightUploadBytes = checked(metrics.WeightUploadBytes + Bytes(resource)),
            };
            uploadedWeights.Add(id);
        }
    }

    public VorticePrimitiveGraphSession CreateSession()
    {
        lock (gate)
        {
            ThrowIfDisposed();
            var buffers = new Dictionary<ResourceId, ID3D12Resource>();
            var allocated = new List<ID3D12Resource>();
            try
            {
                foreach (var capacity in localBuffers.Capacities)
                    allocated.Add(CreateBuffer(capacity));
                foreach (var resource in Plan.Graph.Resources.Where(item =>
                    item.Scope != GraphResourceScope.Global &&
                    item.Tensor.ElementType == GraphElementType.Float32))
                    buffers.Add(resource.Id,
                        localBuffers.Slots.TryGetValue(resource.Id, out var slot)
                            ? allocated[slot] : CreateAndTrack(resource));
                var session = new VorticePrimitiveGraphSession(this, buffers);
                activeSessionBufferBytes = checked(activeSessionBufferBytes + session.AllocatedBufferBytes);
                peakBufferBytes = Math.Max(peakBufferBytes,
                    checked(modelBufferBytes + activeSessionBufferBytes));
                return session;
            }
            catch
            {
                foreach (var buffer in allocated) buffer.Dispose();
                throw;
            }

            ID3D12Resource CreateAndTrack(GraphResource descriptor)
            {
                var buffer = CreateBuffer(descriptor);
                allocated.Add(buffer);
                return buffer;
            }
        }
    }

    internal void UploadState(VorticePrimitiveGraphSession session, ResourceId id, ReadOnlySpan<float> values)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            ObjectDisposedException.ThrowIf(session.IsDisposed, session);
            var resource = Describe(id);
            if (resource.Kind != GraphResourceKind.SessionState || !session.Buffers.TryGetValue(id, out var target))
                throw new ArgumentException($"Resource '{id}' is not session state.", nameof(id));
            ValidateLength(resource, values.Length);
            using var upload = CreateUpload(values);
            Begin();
            RecordUpload(target, upload, Bytes(resource));
            Submit();
            metrics = metrics with
            {
                StateUploadCommandSubmissions = checked(metrics.StateUploadCommandSubmissions + 1),
                StateUploadBytes = checked(metrics.StateUploadBytes + Bytes(resource)),
            };
            session.InitializedState.Add(id);
        }
    }

    internal IReadOnlyDictionary<ResourceId, float[]> ExecuteToken(
        VorticePrimitiveGraphSession session,
        IReadOnlyDictionary<ResourceId, float[]> inputs,
        IReadOnlyDictionary<ResourceId, int>? scalars)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            ObjectDisposedException.ThrowIf(session.IsDisposed, session);
            ArgumentNullException.ThrowIfNull(inputs);
            scalars ??= new Dictionary<ResourceId, int>();
            var graph = Plan.Graph;
            foreach (var resource in graph.Resources.Where(item =>
                item.Kind is GraphResourceKind.Weight or GraphResourceKind.Constant))
                if (!uploadedWeights.Contains(resource.Id))
                    throw new InvalidOperationException($"Model resource '{resource.Id}' has not been uploaded.");
            foreach (var resource in graph.Resources.Where(item => item.Kind == GraphResourceKind.SessionState))
                if (!session.InitializedState.Contains(resource.Id))
                    throw new InvalidOperationException($"Session state '{resource.Id}' has not been initialized.");
            foreach (var id in inputs.Keys)
                if (!graph.Inputs.Contains(id) || Describe(id).Tensor.ElementType != GraphElementType.Float32)
                    throw new ArgumentException($"'{id}' is not an FP32 graph input.", nameof(inputs));
            foreach (var id in scalars.Keys)
                if (!graph.Inputs.Contains(id) || Describe(id).Tensor.ElementType != GraphElementType.Int32)
                    throw new ArgumentException($"'{id}' is not an Int32 graph input.", nameof(scalars));
            foreach (var id in graph.Inputs)
            {
                var descriptor = Describe(id);
                if (descriptor.Tensor.ElementType == GraphElementType.Int32)
                {
                    if (!scalars.ContainsKey(id))
                        throw new ArgumentException($"Missing scalar input '{id}'.", nameof(scalars));
                }
                else if (!inputs.TryGetValue(id, out var values))
                    throw new ArgumentException($"Missing FP32 input '{id}'.", nameof(inputs));
                else
                    ValidateLength(descriptor, values.Length);
            }
            foreach (var step in Plan.Steps.Where(item => item.Index is ResourceId))
            {
                var index = scalars[step.Index!.Value];
                if (index < 0 || (uint)index >= step.Rows)
                    throw new ArgumentOutOfRangeException(nameof(scalars),
                        $"Gather node '{step.Node}' index {index} is outside [0, {step.Rows}).");
            }
            if (enableCommandReplay)
                return ExecuteReplay(session, scalars[graph.Inputs[0]]);

            var uploads = new List<ID3D12Resource>();
            var readbacks = new Dictionary<ResourceId, ID3D12Resource>();
            try
            {
                foreach (var (id, values) in inputs)
                    uploads.Add(CreateUpload(values));
                foreach (var id in graph.Outputs)
                    readbacks.Add(id, device.CreateCommittedResource(
                        HeapProperties.ReadbackHeapProperties, HeapFlags.None,
                        ResourceDescription.Buffer(Bytes(Describe(id))), ResourceStates.CopyDest));
                Begin();
                var uploadIndex = 0;
                foreach (var (id, _) in inputs)
                    RecordUpload(session.Buffers[id], uploads[uploadIndex++], Bytes(Describe(id)));
                var kernelRecordings = ProfileCommandRecording
                    ? new Dictionary<string, (int Dispatches, long Ticks)>(StringComparer.Ordinal)
                    : null;
                var recordingStarted = kernelRecordings is null ? 0 : Stopwatch.GetTimestamp();
                RecordGraph(commands, session, scalars, null, readbacks, kernelRecordings);
                var recordingFinished = kernelRecordings is null ? 0 : Stopwatch.GetTimestamp();
                Submit();
                if (kernelRecordings is not null)
                {
                    var submitted = Stopwatch.GetTimestamp();
                    lastExecutionProfile = new VorticePrimitiveGraphExecutionProfile(
                        Stopwatch.GetElapsedTime(recordingStarted, recordingFinished).TotalMilliseconds,
                        Stopwatch.GetElapsedTime(recordingFinished, submitted).TotalMilliseconds,
                        kernelRecordings.Select(pair => new VorticePrimitiveGraphKernelRecording(
                            pair.Key, pair.Value.Dispatches,
                            pair.Value.Ticks * 1000d / Stopwatch.Frequency))
                            .OrderByDescending(item => item.CpuRecordingMilliseconds).ToArray());
                }
                else
                    lastExecutionProfile = null;
                var uploadedBytes = inputs.Keys.Aggregate(0UL,
                    (total, id) => checked(total + Bytes(Describe(id))));
                var outputBytes = graph.Outputs.Aggregate(0UL,
                    (total, id) => checked(total + Bytes(Describe(id))));
                metrics = metrics with
                {
                    TokenCount = checked(metrics.TokenCount + 1),
                    TokenCommandSubmissions = checked(metrics.TokenCommandSubmissions + 1),
                    ActivationUploadBytes = checked(metrics.ActivationUploadBytes + uploadedBytes),
                    OutputReadbackBytes = checked(metrics.OutputReadbackBytes + outputBytes),
                };
                session.TokenMetrics = new VorticePrimitiveGraphTokenMetrics(
                    1, DispatchesPerToken, uploadedBytes, outputBytes);
                return readbacks.ToDictionary(pair => pair.Key, pair =>
                {
                    var result = new float[checked((int)(Bytes(Describe(pair.Key)) / sizeof(float)))];
                    pair.Value.GetData(result.AsSpan());
                    return result;
                });
            }
            finally
            {
                foreach (var upload in uploads) upload.Dispose();
                foreach (var readback in readbacks.Values) readback.Dispose();
            }
        }
    }

    private void RecordGraph(
        ID3D12GraphicsCommandList list,
        VorticePrimitiveGraphSession session,
        IReadOnlyDictionary<ResourceId, int> scalars,
        ID3D12Resource? tokenUpload,
        IReadOnlyDictionary<ResourceId, ID3D12Resource> readbacks,
        Dictionary<string, (int Dispatches, long Ticks)>? kernelRecordings)
    {
        list.SetComputeRootSignature(signature);
        foreach (var step in Plan.Steps)
        {
            if (localBuffers.ElidedSteps.Contains(step.Node))
                continue;
            var replayGather = tokenUpload is not null && step.Kernel == "GatherRow";
            var kernel = replayGather ? "ReplayGatherRow" : step.Kernel;
            var stepStarted = kernelRecordings is null ? 0 : Stopwatch.GetTimestamp();
            var inputIds = step.FusedInputs ??
                (step.Kernel == "Fill" ? [] : step.Input1 is ResourceId secondId
                    ? [step.Input0, secondId] : [step.Input0]);
            var sources = inputIds.Select(id => Buffer(session, id)).ToArray();
            foreach (var source in sources.Distinct<ID3D12Resource>(ReferenceEqualityComparer.Instance))
                list.ResourceBarrierTransition(source, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            list.SetPipelineState(GetPipeline(replayGather ? step with { Kernel = kernel } : step));
            for (var inputSlot = 0; inputSlot < sources.Length; inputSlot++)
                list.SetComputeRootShaderResourceView((uint)inputSlot, sources[inputSlot].GPUVirtualAddress);
            if (replayGather)
                list.SetComputeRootShaderResourceView(2, tokenUpload!.GPUVirtualAddress);
            list.SetComputeRootUnorderedAccessView(VorticeFusedExpressionKernel.MaximumInputs,
                Buffer(session, step.Output).GPUVirtualAddress);
            list.SetComputeRoot32BitConstant(VorticeFusedExpressionKernel.MaximumInputs + 1, step.ElementCount, 0);
            list.SetComputeRoot32BitConstant(VorticeFusedExpressionKernel.MaximumInputs + 1, step.Rows, 1);
            list.SetComputeRoot32BitConstant(VorticeFusedExpressionKernel.MaximumInputs + 1, step.Columns, 2);
            list.SetComputeRoot32BitConstant(VorticeFusedExpressionKernel.MaximumInputs + 1,
                step.Index is ResourceId index && tokenUpload is null
                    ? checked((uint)scalars[index]) : step.Parameter, 3);
            for (var axis = 0; axis < 4; axis++)
            {
                list.SetComputeRoot32BitConstant(VorticeFusedExpressionKernel.MaximumInputs + 1,
                    step.SourceDimensions is { } source && axis < source.Count ? source[axis] : 1u,
                    checked((uint)(4 + axis)));
                list.SetComputeRoot32BitConstant(VorticeFusedExpressionKernel.MaximumInputs + 1,
                    step.OutputDimensions is { } destination && axis < destination.Count ? destination[axis] : 1u,
                    checked((uint)(8 + axis)));
            }
            var twoDimensional = step.Kernel is "CastFp16ToFp32" or "PortableCopy" or
                "Slice" or "Broadcast" or "Fill" or "BatchedMatVec" or
                "ReduceLastSum" or "ReduceLastMean" or "HeadOuter" or "MatVecLarge" or
                "FusedExpression";
            list.Dispatch(
                Math.Min(step.DispatchX, 65535u),
                twoDimensional ? checked((step.DispatchX + 65534u) / 65535u) : 1u, 1);
            list.ResourceBarrierUnorderedAccessView(null!);
            foreach (var source in sources.Distinct<ID3D12Resource>(ReferenceEqualityComparer.Instance))
                list.ResourceBarrierTransition(source, ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess);
            if (kernelRecordings is not null)
            {
                kernelRecordings.TryGetValue(kernel, out var elapsed);
                kernelRecordings[kernel] = (
                    elapsed.Dispatches + 1,
                    elapsed.Ticks + Stopwatch.GetTimestamp() - stepStarted);
            }
        }
        foreach (var (id, readback) in readbacks)
        {
            var source = Buffer(session, id);
            list.ResourceBarrierTransition(source, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            list.CopyBufferRegion(readback, 0, source, 0, Bytes(Describe(id)));
            list.ResourceBarrierTransition(source, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        }
    }

    private IReadOnlyDictionary<ResourceId, float[]> ExecuteReplay(
        VorticePrimitiveGraphSession session, int token)
    {
        var replay = session.Replay;
        var recordingMilliseconds = 0d;
        IReadOnlyList<VorticePrimitiveGraphKernelRecording> kernelProfiles = [];
        if (replay is null)
        {
            var started = Stopwatch.GetTimestamp();
            var recordings = ProfileCommandRecording
                ? new Dictionary<string, (int Dispatches, long Ticks)>(StringComparer.Ordinal)
                : null;
            replay = CreateReplay(session, recordings);
            session.Replay = replay;
            recordingMilliseconds = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
            if (recordings is not null)
                kernelProfiles = recordings.Select(pair => new VorticePrimitiveGraphKernelRecording(
                    pair.Key, pair.Value.Dispatches,
                    pair.Value.Ticks * 1000d / Stopwatch.Frequency))
                    .OrderByDescending(item => item.CpuRecordingMilliseconds).ToArray();
            metrics = metrics with
            {
                ReplayListRecordings = checked(metrics.ReplayListRecordings + 1),
            };
        }

        Span<uint> tokenValue = stackalloc uint[1];
        tokenValue[0] = checked((uint)token);
        replay.TokenIndex.SetData(tokenValue);
        var submittedAt = Stopwatch.GetTimestamp();
        SubmitClosed(replay.Commands);
        if (ProfileCommandRecording)
            lastExecutionProfile = new VorticePrimitiveGraphExecutionProfile(
                recordingMilliseconds, Stopwatch.GetElapsedTime(submittedAt).TotalMilliseconds,
                kernelProfiles);
        else
            lastExecutionProfile = null;
        var outputBytes = Plan.Graph.Outputs.Aggregate(0UL,
            (total, id) => checked(total + Bytes(Describe(id))));
        metrics = metrics with
        {
            TokenCount = checked(metrics.TokenCount + 1),
            TokenCommandSubmissions = checked(metrics.TokenCommandSubmissions + 1),
            ReplayedTokens = checked(metrics.ReplayedTokens + 1),
            ScalarUploadBytes = checked(metrics.ScalarUploadBytes + sizeof(uint)),
            OutputReadbackBytes = checked(metrics.OutputReadbackBytes + outputBytes),
        };
        session.TokenMetrics = new VorticePrimitiveGraphTokenMetrics(
            1, DispatchesPerToken, 0, outputBytes);
        return replay.Readbacks.ToDictionary(pair => pair.Key, pair =>
        {
            var result = new float[checked((int)(Bytes(Describe(pair.Key)) / sizeof(float)))];
            pair.Value.GetData(result.AsSpan());
            return result;
        });
    }

    private VorticePrimitiveGraphReplay CreateReplay(
        VorticePrimitiveGraphSession session,
        Dictionary<string, (int Dispatches, long Ticks)>? kernelRecordings)
    {
        ID3D12Resource? tokenIndex = null;
        ID3D12CommandAllocator? replayAllocator = null;
        ID3D12GraphicsCommandList? replayCommands = null;
        var readbacks = new Dictionary<ResourceId, ID3D12Resource>();
        try
        {
            tokenIndex = device.CreateCommittedResource(
                HeapProperties.UploadHeapProperties, HeapFlags.None,
                ResourceDescription.Buffer(sizeof(uint)), ResourceStates.GenericRead);
            foreach (var id in Plan.Graph.Outputs)
                readbacks.Add(id, device.CreateCommittedResource(
                    HeapProperties.ReadbackHeapProperties, HeapFlags.None,
                    ResourceDescription.Buffer(Bytes(Describe(id))), ResourceStates.CopyDest));
            replayAllocator = device.CreateCommandAllocator(CommandListType.Compute);
            replayCommands = device.CreateCommandList<ID3D12GraphicsCommandList>(
                0, CommandListType.Compute, replayAllocator);
            RecordGraph(replayCommands, session, new Dictionary<ResourceId, int>(),
                tokenIndex, readbacks, kernelRecordings);
            replayCommands.Close();
            return new VorticePrimitiveGraphReplay(
                tokenIndex, replayAllocator, replayCommands, readbacks);
        }
        catch
        {
            replayCommands?.Dispose();
            replayAllocator?.Dispose();
            tokenIndex?.Dispose();
            foreach (var readback in readbacks.Values) readback.Dispose();
            throw;
        }
    }

    internal float[] ReadState(VorticePrimitiveGraphSession session, ResourceId id)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            ObjectDisposedException.ThrowIf(session.IsDisposed, session);
            var resource = Describe(id);
            if (resource.Kind != GraphResourceKind.SessionState ||
                !session.InitializedState.Contains(id))
                throw new ArgumentException($"Session state '{id}' is not initialized.", nameof(id));
            using var readback = device.CreateCommittedResource(
                HeapProperties.ReadbackHeapProperties, HeapFlags.None,
                ResourceDescription.Buffer(Bytes(resource)), ResourceStates.CopyDest);
            Begin();
            var source = session.Buffers[id];
            commands.ResourceBarrierTransition(source, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            commands.CopyBufferRegion(readback, 0, source, 0, Bytes(resource));
            commands.ResourceBarrierTransition(source, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            Submit();
            metrics = metrics with
            {
                StateReadbackCommandSubmissions = checked(metrics.StateReadbackCommandSubmissions + 1),
                StateReadbackBytes = checked(metrics.StateReadbackBytes + Bytes(resource)),
            };
            var values = new float[checked((int)(Bytes(resource) / sizeof(float)))];
            readback.GetData(values.AsSpan());
            return values;
        }
    }

    internal void ReleaseSession(VorticePrimitiveGraphSession session)
    {
        lock (gate)
        {
            var allocatedBytes = session.AllocatedBufferBytes;
            session.Replay?.Dispose();
            foreach (var buffer in session.Buffers.Values.Distinct<ID3D12Resource>(ReferenceEqualityComparer.Instance))
                buffer.Dispose();
            activeSessionBufferBytes -= allocatedBytes;
        }
    }

    internal VorticePrimitiveGraphTokenMetrics? GetLastTokenMetrics(VorticePrimitiveGraphSession session)
    {
        lock (gate) return session.TokenMetrics;
    }

    private ID3D12Resource Buffer(VorticePrimitiveGraphSession session, ResourceId id) =>
        modelBuffers.TryGetValue(id, out var model) ? model : session.Buffers[id];

    private GraphResource Describe(ResourceId id) =>
        Plan.Graph.Resources.FirstOrDefault(resource => resource.Id == id) ??
        throw new ArgumentException($"Unknown graph resource '{id}'.", nameof(id));

    private static ulong Bytes(GraphResource resource) =>
        checked((ulong)resource.Tensor.Dimensions.Aggregate(1L, (a, b) => checked(a * b)) *
            (uint)(resource.Tensor.ElementType == GraphElementType.Float16 ? sizeof(ushort) : sizeof(float)));

    private static ulong AllocatedBytes(GraphResource resource) =>
        checked((Bytes(resource) + 3UL) & ~3UL);

    private static void ValidateLength(GraphResource resource, int length)
    {
        if (resource.Tensor.ElementType != GraphElementType.Float32)
            throw new ArgumentException($"Resource '{resource.Id}' is not an FP32 tensor.");
        if ((ulong)length * sizeof(float) != Bytes(resource))
            throw new ArgumentException($"Resource '{resource.Id}' expects {Bytes(resource) / sizeof(float)} FP32 elements, received {length}.");
    }

    private ID3D12Resource CreateBuffer(GraphResource descriptor) => CreateBuffer(AllocatedBytes(descriptor));

    private ID3D12Resource CreateBuffer(ulong bytes) =>
        device.CreateCommittedResource(HeapProperties.DefaultHeapProperties, HeapFlags.None,
            ResourceDescription.Buffer(checked((bytes + 3UL) & ~3UL), ResourceFlags.AllowUnorderedAccess),
            ResourceStates.UnorderedAccess);

    private ID3D12Resource CreateUpload(ReadOnlySpan<Half> values)
    {
        var bytes = checked((ulong)values.Length * sizeof(ushort));
        var upload = device.CreateCommittedResource(HeapProperties.UploadHeapProperties, HeapFlags.None,
            ResourceDescription.Buffer(checked((bytes + 3UL) & ~3UL)), ResourceStates.GenericRead);
        upload.SetData(values);
        return upload;
    }

    private ID3D12Resource CreateUpload(ReadOnlySpan<float> values)
    {
        var upload = device.CreateCommittedResource(HeapProperties.UploadHeapProperties, HeapFlags.None,
            ResourceDescription.Buffer(checked((ulong)values.Length * sizeof(float))), ResourceStates.GenericRead);
        upload.SetData(values);
        return upload;
    }

    private void RecordUpload(ID3D12Resource destination, ID3D12Resource source, ulong bytes)
    {
        commands.ResourceBarrierTransition(destination, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        commands.CopyBufferRegion(destination, 0, source, 0, bytes);
        commands.ResourceBarrierTransition(destination, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
    }

    private ID3D12PipelineState GetPipeline(VorticePrimitiveGraphStep step)
    {
        var kernel = step.Kernel;
        var key = step.Expression is { } expression
            ? VorticeFusedExpressionKernel.Key(expression) : kernel;
        if (pipelines.TryGetValue(key, out var pipeline)) return pipeline;
        if (step.Expression is not null)
        {
            using var compilation = DxcCompiler.Compile(
                DxcShaderStage.Compute, VorticeFusedExpressionKernel.Source(step.Expression),
                "FusedExpression", new DxcCompilerOptions { ShaderModel = DxcShaderModel.Model6_0 });
            pipeline = device.CreateComputePipelineState(new ComputePipelineStateDescription
            {
                RootSignature = signature,
                ComputeShader = compilation.GetObjectBytecodeArray(),
            });
            pipelines.Add(key, pipeline);
            return pipeline;
        }
        var portable = kernel is "Fill" or "Slice" or "Broadcast" or "CastFp16ToFp32" or
            "PortableCopy" or "MatVecLarge" or "BatchedMatVec" or "ReduceLastSum" or
            "ReduceLastMean" or "HeadOuter" or "ReplayGatherRow";
        var name = kernel == "CastFp16ToFp32"
            ? "SharpInference.Backends.Vortice.Shaders.PortableCastFp16.hlsl"
            : kernel == "ReplayGatherRow"
            ? "SharpInference.Backends.Vortice.Shaders.PortableReplayGatherFp32.hlsl"
            : portable
            ? "SharpInference.Backends.Vortice.Shaders.PortableTensorFp32.hlsl"
            : "SharpInference.Backends.Vortice.Shaders.PrimitiveOperatorsFp32.hlsl";
        using var stream = typeof(VorticePrimitiveGraphExecutor).Assembly.GetManifestResourceStream(name) ??
            throw new InvalidOperationException($"Embedded shader '{name}' is missing.");
        var bytecode = new byte[checked((int)stream.Length)];
        stream.ReadExactly(bytecode);
        using (var compilation = DxcCompiler.Compile(
                   DxcShaderStage.Compute, System.Text.Encoding.UTF8.GetString(bytecode), kernel,
                   new DxcCompilerOptions { ShaderModel = DxcShaderModel.Model6_0 }))
            bytecode = compilation.GetObjectBytecodeArray();
        pipeline = device.CreateComputePipelineState(new ComputePipelineStateDescription
        {
            RootSignature = signature,
            ComputeShader = bytecode,
        });
        pipelines.Add(key, pipeline);
        return pipeline;
    }

    private void Begin()
    {
        allocator.Reset();
        commands.Reset(allocator);
    }

    private void Submit()
    {
        commands.Close();
        SubmitClosed(commands);
    }

    private void SubmitClosed(ID3D12GraphicsCommandList list)
    {
        queue.ExecuteCommandList(list);
        var value = checked(++fenceValue);
        queue.Signal(fence, value).CheckError();
        if (fence.CompletedValue < value)
        {
            fence.SetEventOnCompletion(value, completion).CheckError();
            completion.WaitOne();
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            foreach (var pipeline in pipelines.Values) pipeline.Dispose();
            foreach (var buffer in modelBuffers.Values) buffer.Dispose();
            commands.Dispose();
            allocator.Dispose();
            signature.Dispose();
            fence.Dispose();
            queue.Dispose();
            completion.Dispose();
        }
    }
}

/// <summary>Owns per-session GPU state and token-local buffers; dispose after the last token.</summary>
public sealed class VorticePrimitiveGraphSession : IDisposable
{
    private readonly VorticePrimitiveGraphExecutor executor;
    private bool disposed;

    internal VorticePrimitiveGraphSession(
        VorticePrimitiveGraphExecutor executor,
        Dictionary<ResourceId, ID3D12Resource> buffers)
    {
        this.executor = executor;
        Buffers = buffers;
    }

    internal Dictionary<ResourceId, ID3D12Resource> Buffers { get; }
    internal VorticePrimitiveGraphReplay? Replay { get; set; }
    internal HashSet<ResourceId> InitializedState { get; } = [];
    internal bool IsDisposed => disposed;
    internal VorticePrimitiveGraphTokenMetrics? TokenMetrics { get; set; }
    public ExecutionGraph Graph => executor.Plan.Graph;
    public ulong AllocatedBufferBytes => Buffers.Values
        .Distinct<ID3D12Resource>(ReferenceEqualityComparer.Instance)
        .Aggregate(0UL, (sum, buffer) => checked(sum + buffer.Description.Width));
    public VorticePrimitiveGraphTokenMetrics? LastTokenMetrics => executor.GetLastTokenMetrics(this);

    public void UploadState(ResourceId id, ReadOnlySpan<float> values)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        executor.UploadState(this, id, values);
    }

    public float[] ReadState(ResourceId id)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return executor.ReadState(this, id);
    }

    public IReadOnlyDictionary<ResourceId, float[]> ExecuteToken(
        IReadOnlyDictionary<ResourceId, float[]> inputs,
        IReadOnlyDictionary<ResourceId, int>? scalarInputs = null)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return executor.ExecuteToken(this, inputs, scalarInputs);
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        executor.ReleaseSession(this);
    }
}

internal sealed class VorticePrimitiveGraphReplay : IDisposable
{
    private readonly ID3D12CommandAllocator allocator;

    public VorticePrimitiveGraphReplay(ID3D12Resource tokenIndex,
        ID3D12CommandAllocator allocator,
        ID3D12GraphicsCommandList commands,
        IReadOnlyDictionary<ResourceId, ID3D12Resource> readbacks)
    {
        TokenIndex = tokenIndex;
        this.allocator = allocator;
        Commands = commands;
        Readbacks = readbacks;
    }

    public ID3D12Resource TokenIndex { get; }
    public ID3D12GraphicsCommandList Commands { get; }
    public IReadOnlyDictionary<ResourceId, ID3D12Resource> Readbacks { get; }

    public void Dispose()
    {
        Commands.Dispose();
        allocator.Dispose();
        TokenIndex.Dispose();
        foreach (var readback in Readbacks.Values) readback.Dispose();
    }
}
