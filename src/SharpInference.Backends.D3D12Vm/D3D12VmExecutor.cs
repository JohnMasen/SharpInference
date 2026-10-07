using System.Runtime.CompilerServices;
using SharpInference.Vm;
using Vortice.Direct3D12;

namespace SharpInference.Backends.D3D12Vm;

/// <summary>
/// Standalone execution owns one persistent GPU allocation per slot. Pooled execution binds the
/// manager-owned CPU shadows to their shared pool resources, without allocating duplicate GPU slots.
/// Direct construction borrows the device; the artifact's parameterless factory owns its device.
/// Upload/readback are control-plane operations; tensor operators execute only on the GPU.
/// Managed execution uses explicit CPU-authoritative mutable-state copies.
/// Session migration is safe because every managed entry uploads session backing bytes and copies
/// writable results back before returning. CPU array identity never suppresses these state transfers.
/// Idle pooled command caches retain only allocation IDs, never storage, CPU arrays or resource wrappers.
/// </summary>
public sealed class D3D12VmExecutor : IVmExecutable
{
    private readonly ID3D12Device device;
    private readonly ID3D12Resource[] buffers;
    private readonly D3D12VmResourcePool? pool;
    private readonly D3D12VmPooledStorage?[] pooledBindings;
    private readonly D3D12VmPooledStorage?[] candidateBindings;
    private readonly ulong[] recordedAllocationIds;
    private int pooledUseDepth;
    private readonly IReadOnlyDictionary<string, D3D12VmCommand[]> schedules;
    private bool poolAttached;
    private bool entriesRecorded;
    private ulong bindingRecordCount;
    private readonly D3D12VmGatherIndex[] gatherIndices;
    private readonly Dictionary<string, D3D12VmGatherIndex[]> entryGatherIndices;
    private readonly WeakReference<byte[]>?[] globalBindings;
    private readonly ID3D12CommandQueue queue = null!;
    private readonly ID3D12Fence fence = null!;
    private readonly EventWaitHandle completion = new(false, EventResetMode.AutoReset);
    private readonly ID3D12CommandAllocator transferAllocator = null!;
    private readonly ID3D12GraphicsCommandList transfer = null!;
    private readonly Dictionary<string, ID3D12RootSignature> signatures = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ID3D12PipelineState> pipelines = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (ID3D12CommandAllocator Allocator, ID3D12GraphicsCommandList Commands)> entries = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CombinedTransferEntry> combinedTransferEntries =
        new(StringComparer.Ordinal);
    private readonly object gate = new();
    private readonly ConditionalWeakTable<VmProgram, string> leasePrograms = new();
    private string? programXml;
    private bool ownsDevice;
    private ulong fenceValue;
    private ulong submissionCount;
    private bool disposed;

    public D3D12VmExecutor(ID3D12Device device, D3D12VmArtifact artifact)
        : this(device, artifact, null, false) { }

    public D3D12VmExecutor(D3D12VmResourcePool pool, D3D12VmArtifact artifact)
        : this((pool ?? throw new ArgumentNullException(nameof(pool))).Device, artifact, pool, false) { }

    private D3D12VmExecutor(
        ID3D12Device device,
        D3D12VmArtifact artifact,
        D3D12VmResourcePool? pool,
        bool skipReadOnlyGlobalInitialization,
        bool skipAllInitialization = false)
    {
        this.device = device ?? throw new ArgumentNullException(nameof(device));
        Artifact = artifact ?? throw new ArgumentNullException(nameof(artifact));
        this.pool = pool;
        schedules = D3D12VmSchedule.Create(artifact.Program);
        gatherIndices = D3D12VmSchedule.GatherIndices(artifact.Contracts);
        entryGatherIndices = schedules.Keys.ToDictionary(name => name,
            name => D3D12VmSchedule.GatherIndices(artifact.Contracts, name), StringComparer.Ordinal);
        buffers = new ID3D12Resource[artifact.Program.Slots.Count];
        pooledBindings = new D3D12VmPooledStorage?[buffers.Length];
        candidateBindings = new D3D12VmPooledStorage?[buffers.Length];
        recordedAllocationIds = new ulong[buffers.Length];
        globalBindings = new WeakReference<byte[]>?[buffers.Length];
        try
        {
            if (pool is not null)
            {
                pool.AttachExecutor();
                poolAttached = true;
            }
            queue = device.CreateCommandQueue(CommandListType.Compute);
            fence = device.CreateFence();
            transferAllocator = device.CreateCommandAllocator(CommandListType.Compute);
            transfer = device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, transferAllocator);
            transfer.Close();
            if (pool is null)
                for (var i = 0; i < buffers.Length; i++)
                    buffers[i] = device.CreateCommittedResource(HeapProperties.DefaultHeapProperties, HeapFlags.None,
                        ResourceDescription.Buffer(Align(artifact.Program.Slots[i].Tensor.ByteLength), ResourceFlags.AllowUnorderedAccess),
                        ResourceStates.UnorderedAccess);
            foreach (var kernel in artifact.Kernels)
            {
                var definition = artifact.Program.Definitions.Single(d => d.Id == kernel.Definition);
                var parameters = definition.Parameters.Select((p, i) => new RootParameter(
                    RootParameterType.UnorderedAccessView, new RootDescriptor((uint)i, 0), ShaderVisibility.All)).ToList();
                parameters.Add(new RootParameter(new RootConstants(0, 0, (uint)(2 + definition.Parameters.Count)), ShaderVisibility.All));
                var signature = device.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, parameters.ToArray()),
                    RootSignatureVersion.Version10);
                signatures.Add(kernel.Definition, signature);
                try
                {
                    pipelines.Add(kernel.Definition, device.CreateComputePipelineState(new ComputePipelineStateDescription
                    {
                        RootSignature = signature,
                        ComputeShader = kernel.Bytecode,
                    }));
                }
                catch (Exception error)
                {
                    throw new InvalidOperationException(
                        $"Failed to create the D3D12 pipeline for kernel '{kernel.Definition}'.", error);
                }
            }
            if (pool is null)
            {
                RecordEntries();
                // Pooled resources are initialized from their authoritative shadows at first use.
                for (var i = 0; i < buffers.Length; i++)
                {
                    if (skipAllInitialization ||
                        skipReadOnlyGlobalInitialization && IsFixedGlobal(i))
                        continue;
                    Upload(i, new byte[checked((int)artifact.Program.Slots[i].Tensor.ByteLength)]);
                }
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public D3D12VmArtifact Artifact { get; }
    public VmProgram Program => Artifact.Program;
    public D3D12VmResourcePool? ResourcePool => pool;
    public bool OwnsDevice => ownsDevice;
    /// <summary>Executor-owned slot bytes. Pooled allocations are counted only by ResourcePool.</summary>
    public ulong AllocatedGpuBytes => pool is null ?
        Artifact.Program.Slots.Aggregate(0UL, (sum, s) => checked(sum + Align(s.Tensor.ByteLength))) : 0;
    public ulong BindingRecordCount { get { lock (gate) return bindingRecordCount; } }
    public ulong SubmissionCount { get { lock (gate) return submissionCount; } }
    public int DispatchCount(string entry) => D3D12VmSchedule.Create(Artifact.Program)[entry].Count(c => c is D3D12VmDispatch);

    public ulong GetGpuVirtualAddress(string slot)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            using var use = BeginPoolUse();
            var index = SlotIndex(slot);
            RequirePooledBinding(index);
            return buffers[index].GPUVirtualAddress;
        }
    }

    private void RecordEntries()
    {
        ClearEntries();
        foreach (var value in combinedTransferEntries.Values) value.Dispose();
        combinedTransferEntries.Clear();
        try
        {
            foreach (var (name, schedule) in schedules)
            {
                var allocator = device.CreateCommandAllocator(CommandListType.Compute);
                ID3D12GraphicsCommandList? list = null;
                try
                {
                    list = device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, allocator);
                    RecordCommands(list, schedule);
                    list.Close();
                    entries.Add(name, (allocator, list));
                }
                catch
                {
                    list?.Dispose();
                    allocator.Dispose();
                    throw;
                }
            }
            entriesRecorded = true;
            bindingRecordCount = checked(bindingRecordCount + 1);
        }
        catch
        {
            ClearEntries();
            throw;
        }
    }

    private void ClearEntries()
    {
        foreach (var value in entries.Values) { value.Commands.Dispose(); value.Allocator.Dispose(); }
        entries.Clear();
        entriesRecorded = false;
    }

    internal static D3D12VmExecutor CreateOwned(
        ID3D12Device device,
        D3D12VmArtifact artifact,
        bool skipReadOnlyGlobalInitialization = false,
        bool skipAllInitialization = false)
    {
        try
        {
            var executor = new D3D12VmExecutor(
                device, artifact, null, skipReadOnlyGlobalInitialization, skipAllInitialization);
            executor.ownsDevice = true;
            return executor;
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-only execution: replays explicit dispatches/barriers and waits for completion.
    /// No CPU mirror is synchronized. Call Upload/UploadSlots for CPU changes before execution and
    /// Readback/ReadbackSlots before consuming GPU changes through CPU backing arrays.
    /// Managed workers should instead use the byte[][] or VmExecutionLease overload.
    /// Pooled bindings must first be
    /// initialized by managed execution or UploadSlots. Read writable slots back before handing
    /// state to another managed worker; managed execution treats the CPU shadows as authoritative.
    /// </summary>
    public void Execute(string entry)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            using var use = BeginPoolUse();
            PrepareGpuEntry(entry);
            Submit(entries[entry].Commands);
        }
    }

    /// <summary>
    /// Uploads one control slot, executes an entry, and reads one result slot with a single queue
    /// submission and fence wait. Intended for token-boundary inference transfers.
    /// </summary>
    public void ExecuteWithTransfers(
            string entry,
            string uploadSlot,
            byte[] uploadBytes,
            string readbackSlot,
            byte[] readbackBytes)
        {
            ArgumentNullException.ThrowIfNull(uploadBytes);
            ArgumentNullException.ThrowIfNull(readbackBytes);
            lock (gate)
            {
                ThrowIfDisposed();
                using var use = BeginPoolUse();
                if (!schedules.TryGetValue(entry, out var schedule))
                    throw new ArgumentException($"Unknown VM entry '{entry}'.", nameof(entry));
                PrepareGpuEntry(entry);
                var uploadIndex = SlotIndex(uploadSlot);
                var readbackIndex = SlotIndex(readbackSlot);
                ValidateBuffer(uploadIndex, uploadBytes);
                ValidateBuffer(readbackIndex, readbackBytes);
                var key = $"{entry}\0{uploadIndex}\0{readbackIndex}";
                if (!combinedTransferEntries.TryGetValue(key, out var combined))
                {
                    var upload = device.CreateCommittedResource(
                        HeapProperties.UploadHeapProperties,
                        HeapFlags.None,
                        ResourceDescription.Buffer(Align((ulong)uploadBytes.Length)),
                        ResourceStates.GenericRead);
                    var readback = device.CreateCommittedResource(
                        HeapProperties.ReadbackHeapProperties,
                        HeapFlags.None,
                        ResourceDescription.Buffer(Align((ulong)readbackBytes.Length)),
                        ResourceStates.CopyDest);
                    var allocator = device.CreateCommandAllocator(CommandListType.Compute);
                    var commands = device.CreateCommandList<ID3D12GraphicsCommandList>(
                        0, CommandListType.Compute, allocator);
                    try
                    {
                        commands.ResourceBarrierTransition(
                            buffers[uploadIndex], ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                        commands.CopyBufferRegion(
                            buffers[uploadIndex], 0, upload, 0, (ulong)uploadBytes.Length);
                        commands.ResourceBarrierTransition(
                            buffers[uploadIndex], ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                        RecordCommands(commands, schedule);
                        commands.ResourceBarrierTransition(
                            buffers[readbackIndex], ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                        commands.CopyBufferRegion(
                            readback, 0, buffers[readbackIndex], 0, (ulong)readbackBytes.Length);
                        commands.ResourceBarrierTransition(
                            buffers[readbackIndex], ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                        commands.Close();
                        combined = new(upload, readback, allocator, commands);
                        combinedTransferEntries.Add(key, combined);
                    }
                    catch
                    {
                        commands.Dispose();
                        allocator.Dispose();
                        readback.Dispose();
                        upload.Dispose();
                        throw;
                    }
                }
                combined.Upload.SetData<byte>(uploadBytes);
                Submit(combined.Commands);
                combined.Readback.GetData<byte>(readbackBytes);
        }
    }

    private sealed record CombinedTransferEntry(
        ID3D12Resource Upload,
        ID3D12Resource Readback,
        ID3D12CommandAllocator Allocator,
        ID3D12GraphicsCommandList Commands) : IDisposable
    {
        public void Dispose()
        {
            Commands.Dispose();
            Allocator.Dispose();
            Readback.Dispose();
            Upload.Dispose();
        }
    }

    private void PrepareGpuEntry(string entry)
    {
        if (!schedules.ContainsKey(entry))
            throw new ArgumentException($"Unknown VM entry '{entry}'.", nameof(entry));
        for (var i = 0; pool is not null && i < buffers.Length; i++)
        {
            var binding = RequirePooledBinding(i)!;
            if (!binding.GpuInitialized || IsFixedGlobal(i) && !binding.GlobalUploaded)
                throw new InvalidOperationException($"Slot '{Program.Slots[i].Id}' requires a managed execution or explicit upload before GPU-only execution.");
        }
        if (!entriesRecorded) RecordEntries();
    }

    private void RecordCommands(ID3D12GraphicsCommandList list, IReadOnlyList<D3D12VmCommand> schedule)
    {
        foreach (var command in schedule)
            switch (command)
            {
                case D3D12VmBarrier barrier:
                    foreach (var slot in barrier.Slots)
                        list.ResourceBarrierUnorderedAccessView(buffers[slot]);
                    break;
                case D3D12VmDispatch dispatch:
                    var definition = Program.Definitions.Single(d => d.Id == dispatch.Kernel);
                    list.SetComputeRootSignature(signatures[dispatch.Kernel]);
                    list.SetPipelineState(pipelines[dispatch.Kernel]);
                    var constants = new uint[2 + dispatch.Bindings.Length];
                    constants[0] = checked(dispatch.Groups.X * definition.Threads!.X);
                    constants[1] = checked(dispatch.Groups.Y * definition.Threads.Y);
                    for (var index = 0; index < dispatch.Bindings.Length; index++)
                    {
                        var binding = dispatch.Bindings[index];
                        list.SetComputeRootUnorderedAccessView((uint)index, buffers[binding.Slot].GPUVirtualAddress);
                        constants[index + 2] = checked((uint)binding.Offset);
                    }
                    list.SetComputeRoot32BitConstants((uint)dispatch.Bindings.Length, constants);
                    list.Dispatch(dispatch.Groups.X, dispatch.Groups.Y, dispatch.Groups.Z);
                    break;
            }
    }

    /// <summary>
    /// Measures GPU-resident entry execution using queue timestamps, including scheduled barriers.
    /// Uploads and result readback are excluded. The entry executes exactly repetitions times.
    /// This explicit profiling API is never invoked during normal inference or plan preparation.
    /// </summary>
    public double MeasureGpuMicroseconds(string entry, int repetitions = 8)
    {
        if (repetitions is < 1 or > 1024) throw new ArgumentOutOfRangeException(nameof(repetitions));
        lock (gate)
        {
            ThrowIfDisposed();
            using var use = BeginPoolUse();
            PrepareGpuEntry(entry);
            queue.GetTimestampFrequency(out var frequency).CheckError();
            if (frequency == 0) throw new InvalidOperationException("GPU timestamp frequency is zero.");
            using var queries = device.CreateQueryHeap<ID3D12QueryHeap>(new QueryHeapDescription(QueryHeapType.Timestamp, 2));
            using var readback = device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties, HeapFlags.None,
                ResourceDescription.Buffer(16), ResourceStates.CopyDest);
            using var allocator = device.CreateCommandAllocator(CommandListType.Compute);
            using var commands = device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, allocator);
            commands.EndQuery(queries, QueryType.Timestamp, 0);
            for (var index = 0; index < repetitions; index++) RecordCommands(commands, schedules[entry]);
            commands.EndQuery(queries, QueryType.Timestamp, 1);
            commands.ResolveQueryData(queries, QueryType.Timestamp, 0, 2, readback, 0);
            commands.Close();
            Submit(commands);
            var ticks = new ulong[2];
            readback.GetData<ulong>(ticks);
            if (ticks[1] <= ticks[0]) throw new InvalidOperationException("GPU timestamps did not advance.");
            return (ticks[1] - ticks[0]) * (1_000_000d / frequency) / repetitions;
        }
    }

    /// <summary>
    /// Executes against an active managed lease, validating its complete program contract.
    /// Equivalent deserialized programs are accepted; only the control-plane pointer table is obtained.
    /// The caller retains ownership of the lease, including across a complete prefill request.
    /// </summary>
    public void Execute(string entryName, VmExecutionLease lease)
    {
        ArgumentNullException.ThrowIfNull(lease);
        lock (gate)
        {
            ThrowIfDisposed();
            var leasedProgram = lease.Program;
            if (!ReferenceEquals(leasedProgram, Program))
            {
                programXml ??= VmProgramXml.Serialize(Program);
                if (leasePrograms.GetValue(leasedProgram, static p => VmProgramXml.Serialize(p)) != programXml)
                    throw new ArgumentException("The execution lease belongs to a different VM program.", nameof(lease));
            }
            Execute(entryName, lease.GetBuffers());
        }
    }

    /// <summary>
    /// Managed execution bridge for VmExecutionLease.GetBuffers(), in Program.Slots order.
    /// Standalone read-only globals are immutable bindings; replacement requires a new executor.
    /// Pooled bindings are resolved from the allocator-owned arrays; shared globals upload once
    /// per storage, or again after Storage.Write/ResourcePool.MarkCpuModified. Other slots upload at entry boundaries and writable
    /// slots are read back into the existing arrays. No replacement tensor arrays are allocated.
    /// In particular, mutable session/local bytes are uploaded on EVERY call, including in-place CPU
    /// edits and session migrations to another worker; they are not cached by backing-array identity.
    /// Readback completes before return, making CPU state authoritative for the next worker.
    /// Read-only globals must remain immutable unless explicitly refreshed using Upload/UploadSlots.
    /// Use selective UploadSlots/ReadbackSlots plus Execute(entry) when retaining GPU-only scratch.
    /// </summary>
    public void Execute(string entryName, byte[][] slots)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            using var use = BeginPoolUse();
            if (!schedules.ContainsKey(entryName))
                throw new ArgumentException($"Unknown VM entry '{entryName}'.", nameof(entryName));
            ValidateContext(slots);
            for (var i = 0; i < slots.Length; i++)
            {
                ValidateUpload(i, slots[i], entryGatherIndices[entryName]);
                if (pool is null && IsFixedGlobal(i) && globalBindings[i] is { } binding &&
                    (!binding.TryGetTarget(out var existing) || !ReferenceEquals(existing, slots[i])))
                    throw new ArgumentException($"Read-only global slot '{Program.Slots[i].Id}' belongs to a different managed binding. Create a new executor.", nameof(slots));
            }
            BindPooledContext(slots);
            for (var i = 0; i < slots.Length; i++)
                if (!IsFixedGlobal(i) || (pool is null ? globalBindings[i] is null : !pooledBindings[i]!.GlobalUploaded))
                {
                    UploadCore(i, slots[i]);
                    if (pool is null && IsFixedGlobal(i)) globalBindings[i] = new(slots[i]);
                }
            Execute(entryName);
            for (var i = 0; i < slots.Length; i++)
                if (Program.Slots[i].Access == VmAccess.ReadWrite)
                    Readback(i, slots[i]);
        }
    }

    /// <summary>Slot-ordered byte[][] bridge. A null element leaves the persistent GPU resource unchanged.</summary>
    public void UploadSlots(byte[]?[] context)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (context.Length != buffers.Length) throw new ArgumentException("Context must match Program.Slots order.", nameof(context));
        lock (gate)
        {
            ThrowIfDisposed();
            using var use = BeginPoolUse();
            if (pool is not null)
                throw new NotSupportedException("Pooled selective uploads require a complete context and explicit slot indices.");
            for (var i = 0; i < context.Length; i++)
                if (context[i] is { } bytes) ValidateUpload(i, bytes);
            for (var i = 0; i < context.Length; i++)
                if (context[i] is { } bytes) Upload(i, bytes);
        }
    }

    /// <summary>Uploads selected slot indices directly from existing backing arrays, including explicit weights.</summary>
    public void UploadSlots(byte[][] context, IReadOnlyList<int> slots)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            using var use = BeginPoolUse();
            ValidateContext(context);
            ArgumentNullException.ThrowIfNull(slots);
            foreach (var slot in slots) { ValidateSlot(slot); ValidateUpload(slot, context[slot]); }
            BindPooledContext(context);
            foreach (var slot in slots) UploadCore(slot, context[slot]);
        }
    }

    /// <summary>Reads selected GPU slots into existing backing arrays, without tensor-array allocation.</summary>
    public void ReadbackSlots(byte[][] context, IReadOnlyList<int> slots)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            using var use = BeginPoolUse();
            ValidateContext(context);
            ArgumentNullException.ThrowIfNull(slots);
            foreach (var slot in slots) ValidateSlot(slot);
            BindPooledContext(context);
            foreach (var slot in slots) Readback(slot, context[slot]);
        }
    }

    /// <summary>Uploads an exact slot-sized CPU payload. CPU gather indices are range-checked before transfer.</summary>
    public void Upload(string slot, byte[] bytes) => Upload(SlotIndex(slot), bytes);
    public void Upload(string slot, ReadOnlySpan<byte> bytes) => Upload(SlotIndex(slot), bytes);
    public void Upload(int slot, byte[] bytes)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            using var use = BeginPoolUse();
            ValidateUpload(slot, bytes);
            UploadCore(slot, bytes);
        }
    }

    public void Upload(int slot, ReadOnlySpan<byte> bytes)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            using var use = BeginPoolUse();
            ValidateSlot(slot);
            if ((ulong)bytes.Length != Program.Slots[slot].Tensor.ByteLength)
                throw new ArgumentException(
                    $"Expected {Program.Slots[slot].Tensor.ByteLength} bytes for slot '{Program.Slots[slot].Id}', received {bytes.Length}.",
                    nameof(bytes));
            if (RequirePooledBinding(slot) is not null)
                throw new NotSupportedException("Span uploads do not support pooled resource bindings.");
            UploadCore(slot, bytes);
        }
    }

    public void InitializeSlots(IReadOnlyList<byte[]> buffers, IEnumerable<int> slots)
    {
        ArgumentNullException.ThrowIfNull(buffers);
        ArgumentNullException.ThrowIfNull(slots);
        lock (gate)
        {
            ThrowIfDisposed();
            using var use = BeginPoolUse();
            foreach (var slot in slots.Distinct())
            {
                ValidateSlot(slot);
                var binding = RequirePooledBinding(slot);
                if (binding is not null && binding.GpuInitialized)
                    continue;
                UploadCore(slot, buffers[slot]);
            }
        }
    }

    private void UploadCore(int slot, byte[] bytes)
    {
        var binding = RequirePooledBinding(slot);
        if (binding is not null && !ReferenceEquals(binding.Buffer, bytes))
        {
            binding.Write(0, bytes);
            bytes = binding.Buffer;
        }
        using var upload = device.CreateCommittedResource(HeapProperties.UploadHeapProperties, HeapFlags.None,
            ResourceDescription.Buffer(Align((ulong)bytes.Length)), ResourceStates.GenericRead);
        upload.SetData<byte>(bytes);
        BeginTransfer();
        transfer.ResourceBarrierTransition(buffers[slot], ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        transfer.CopyBufferRegion(buffers[slot], 0, upload, 0, (ulong)bytes.Length);
        transfer.ResourceBarrierTransition(buffers[slot], ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
        transfer.Close();
        Submit(transfer);
        if (binding is not null)
        {
            binding.GpuInitialized = true;
            if (IsFixedGlobal(slot)) pool!.UploadedGlobal(binding);
        }
    }

    private void UploadCore(int slot, ReadOnlySpan<byte> bytes)
    {
        using var upload = device.CreateCommittedResource(HeapProperties.UploadHeapProperties, HeapFlags.None,
            ResourceDescription.Buffer(Align((ulong)bytes.Length)), ResourceStates.GenericRead);
        upload.SetData(bytes);
        BeginTransfer();
        transfer.ResourceBarrierTransition(buffers[slot], ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        transfer.CopyBufferRegion(buffers[slot], 0, upload, 0, (ulong)bytes.Length);
        transfer.ResourceBarrierTransition(buffers[slot], ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
        transfer.Close();
        Submit(transfer);
    }

    /// <summary>Copies one persistent GPU slot back to an owned CPU byte array.</summary>
    public byte[] Readback(string slot) => Readback(SlotIndex(slot));
    public byte[] Readback(int slot)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            ValidateSlot(slot);
            var result = new byte[checked((int)Program.Slots[slot].Tensor.ByteLength)];
            Readback(slot, result);
            return result;
        }
    }

    public void Readback(string slot, byte[] destination) => Readback(SlotIndex(slot), destination);
    public void Readback(int slot, byte[] destination)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            using var use = BeginPoolUse();
            ValidateBuffer(slot, destination);
            var binding = RequirePooledBinding(slot);
            if (binding is not null && (!binding.GpuInitialized || IsFixedGlobal(slot) && !binding.GlobalUploaded))
                throw new InvalidOperationException($"Slot '{Program.Slots[slot].Id}' requires an upload before GPU readback.");
            var bytes = Program.Slots[slot].Tensor.ByteLength;
            using var readback = device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties, HeapFlags.None,
                ResourceDescription.Buffer(Align(bytes)), ResourceStates.CopyDest);
            BeginTransfer();
            transfer.ResourceBarrierTransition(buffers[slot], ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            transfer.CopyBufferRegion(readback, 0, buffers[slot], 0, bytes);
            transfer.ResourceBarrierTransition(buffers[slot], ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            transfer.Close();
            Submit(transfer);
            if (binding is null)
                readback.GetData<byte>(destination);
            else
            {
                readback.GetData<byte>(binding.Buffer);
                if (!ReferenceEquals(binding.Buffer, destination)) binding.Buffer.CopyTo(destination, 0);
            }
        }
    }

    private bool IsFixedGlobal(int slot) =>
        Program.Slots[slot] is { Scope: VmSlotScope.Global, Access: VmAccess.ReadOnly };

    private D3D12VmPooledStorage? RequirePooledBinding(int slot)
    {
        if (pool is null) return null;
        if (recordedAllocationIds[slot] == 0)
            throw new InvalidOperationException("Bind pooled storage with a managed execution or UploadSlots(context, indices) first.");
        var binding = pooledBindings[slot] ?? pool.Resolve(recordedAllocationIds[slot]);
        pool.ValidateLive(binding);
        pooledBindings[slot] = binding;
        buffers[slot] = binding.Resource;
        return binding;
    }

    private IDisposable? BeginPoolUse()
    {
        if (pool is null) return null;
        var use = pool.BeginUse();
        var borrowed = new BorrowedBindings(this, use);
        pooledUseDepth++;
        return borrowed;
    }

    private sealed class BorrowedBindings(D3D12VmExecutor executor, IDisposable use) : IDisposable
    {
        public void Dispose()
        {
            try
            {
                if (--executor.pooledUseDepth == 0)
                {
                    Array.Clear(executor.pooledBindings);
                    Array.Clear(executor.candidateBindings);
                    Array.Clear(executor.buffers);
                }
            }
            finally { use.Dispose(); }
        }
    }

    private void BindPooledContext(byte[][] context)
    {
        if (pool is null) return;
        var changed = false;
        try
        {
            for (var i = 0; i < context.Length; i++)
            {
                var allocation = pool.Resolve(Program.Slots[i], context[i]);
                candidateBindings[i] = allocation;
                changed |= allocation.Id != recordedAllocationIds[i];
            }
            if (changed) ValidatePooledAliases();
            for (var i = 0; i < buffers.Length; i++)
            {
                pooledBindings[i] = candidateBindings[i];
                buffers[i] = candidateBindings[i]!.Resource;
            }
            if (changed || !entriesRecorded)
            {
                RecordEntries();
                for (var i = 0; i < buffers.Length; i++)
                    recordedAllocationIds[i] = pooledBindings[i]!.Id;
            }
        }
        finally { Array.Clear(candidateBindings); }
    }

    private void ValidatePooledAliases()
    {
        foreach (var schedule in schedules.Values)
        {
            var pending = new HashSet<D3D12VmPooledStorage>();
            foreach (var command in schedule)
                switch (command)
                {
                    case D3D12VmBarrier barrier:
                        foreach (var slot in barrier.Slots) pending.Remove(candidateBindings[slot]!);
                        break;
                    case D3D12VmDispatch dispatch:
                        var parameters = Program.Definitions.Single(d => d.Id == dispatch.Kernel).Parameters;
                        for (var i = 0; i < dispatch.Bindings.Length; i++)
                        {
                            var first = dispatch.Bindings[i];
                            if (pending.Contains(candidateBindings[first.Slot]!))
                                throw new InvalidDataException("Aliased pooled UAV writes require an explicit VmBarrier.");
                            for (var j = i + 1; j < dispatch.Bindings.Length; j++)
                            {
                                var second = dispatch.Bindings[j];
                                if (ReferenceEquals(candidateBindings[first.Slot], candidateBindings[second.Slot]) &&
                                    (parameters[i].Access == VmAccess.ReadWrite || parameters[j].Access == VmAccess.ReadWrite) &&
                                    first.Offset < second.Offset + parameters[j].Tensor.ByteLength &&
                                    second.Offset < first.Offset + parameters[i].Tensor.ByteLength)
                                    throw new NotSupportedException("Writable kernel parameters alias the same pooled GPU storage.");
                            }
                        }
                        for (var i = 0; i < dispatch.Bindings.Length; i++)
                            if (parameters[i].Access == VmAccess.ReadWrite)
                                pending.Add(candidateBindings[dispatch.Bindings[i].Slot]!);
                        break;
                }
        }
    }
    private void ValidateContext(byte[][] context)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (context.Length != buffers.Length)
            throw new ArgumentException("Context must match Program.Slots order.", nameof(context));
        for (var i = 0; i < context.Length; i++) ValidateBuffer(i, context[i]);
    }
    private void ValidateBuffer(int slot, byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ValidateSlot(slot);
        if ((ulong)bytes.Length != Artifact.Program.Slots[slot].Tensor.ByteLength)
            throw new ArgumentException($"Slot '{Artifact.Program.Slots[slot].Id}' requires {Artifact.Program.Slots[slot].Tensor.ByteLength} bytes.", nameof(bytes));
    }
    private void ValidateUpload(int slot, byte[] bytes, IReadOnlyList<D3D12VmGatherIndex>? indices = null)
    {
        ValidateBuffer(slot, bytes);
        foreach (var index in (indices ?? gatherIndices).Where(i => i.Slot == slot))
        {
            var row = BitConverter.ToInt32(bytes.AsSpan(checked((int)index.ByteOffset), sizeof(int)));
            if ((uint)row >= (uint)index.Rows)
                throw new ArgumentOutOfRangeException(nameof(bytes), $"Gather index in slot '{Artifact.Program.Slots[slot].Id}' must be in [0, {index.Rows}).");
        }
    }
    private void ValidateSlot(int slot)
    {
        if ((uint)slot >= (uint)buffers.Length) throw new ArgumentOutOfRangeException(nameof(slot));
    }
    private int SlotIndex(string slot)
    {
        for (var i = 0; i < Artifact.Program.Slots.Count; i++)
            if (Artifact.Program.Slots[i].Id == slot) return i;
        throw new ArgumentException($"Unknown VM slot '{slot}'.", nameof(slot));
    }
    private static ulong Align(ulong bytes) => checked((bytes + 3UL) & ~3UL);
    private void BeginTransfer()
    {
        transferAllocator.Reset();
        transfer.Reset(transferAllocator);
    }
    private void Submit(ID3D12GraphicsCommandList list)
    {
        queue.ExecuteCommandList(list);
        submissionCount = checked(submissionCount + 1);
        WaitForCompletion();
    }
    private void WaitForCompletion()
    {
        var value = checked(++fenceValue);
        queue.Signal(fence, value).CheckError();
        if (fence.CompletedValue < value)
        {
            fence.SetEventOnCompletion(value, completion).CheckError();
            while (!completion.WaitOne(TimeSpan.FromSeconds(1)))
                device.DeviceRemovedReason.CheckError();
        }
        device.DeviceRemovedReason.CheckError();
    }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            foreach (var entry in combinedTransferEntries.Values) entry.Dispose();
            combinedTransferEntries.Clear();
            ClearEntries();
            foreach (var pipeline in pipelines.Values) pipeline.Dispose();
            foreach (var signature in signatures.Values) signature.Dispose();
            if (pool is null)
                foreach (var buffer in buffers) buffer?.Dispose();
            Array.Clear(buffers);
            Array.Clear(pooledBindings);
            Array.Clear(candidateBindings);
            Array.Clear(recordedAllocationIds);
            transfer?.Dispose();
            transferAllocator?.Dispose();
            fence?.Dispose();
            queue?.Dispose();
            completion.Dispose();
            if (poolAttached)
            {
                pool!.DetachExecutor();
                poolAttached = false;
            }
            if (ownsDevice) device.Dispose();
        }
    }
}
