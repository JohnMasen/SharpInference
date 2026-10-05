using SharpInference.Vm;
using Vortice.Direct3D12;

namespace SharpInference.Backends.D3D12Vm;

/// <summary>
/// Caller-device-owned GPU allocator for VmResourceManager. Each allocation owns a CPU shadow and
/// one default-heap resource. Binding the same managed storage to different workers shares its GPU
/// resource; separate local allocations remain private. Storage disposal unregisters and releases
/// the resource immediately. No weak cache or backing-array copy is retained after disposal.
/// Resource use is serialized across workers. Dispose executors and core resource owners before
/// disposing this pool, then dispose the caller-owned device.
/// </summary>
public sealed class D3D12VmResourcePool : IDisposable
{
    private readonly Dictionary<byte[], D3D12VmPooledStorage> storage = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<ulong, D3D12VmPooledStorage> allocations = new();
    internal object Gate { get; } = new();
    internal ID3D12Device Device { get; }
    private ulong allocatedGpuBytes;
    private ulong globalUploadBytes;
    private ulong globalUploadCount;
    private ulong nextAllocationId;
    private int executors;
    private bool disposed;
    private ID3D12CommandQueue? readQueue;
    private ID3D12CommandAllocator? readAllocator;
    private ID3D12GraphicsCommandList? readCommands;
    private ID3D12Fence? readFence;
    private EventWaitHandle? readCompletion;
    private ID3D12Resource? readStaging;
    private ulong readCapacity;
    private ulong readFenceValue;
    private ulong completedTasks;
    private ulong computeRecordings;
    private ulong inputUploadBytes;
    private ulong initialStateUploadBytes;
    private ulong initialLocalUploadBytes;
    private ulong stateLoadBytes;
    private ulong stateCommitBytes;
    private ulong cpuReadbackBytes;
    private double preparationMilliseconds;
    private double computeMilliseconds;
    private double commitMilliseconds;

    public D3D12VmResourcePool(ID3D12Device device) =>
        Device = device ?? throw new ArgumentNullException(nameof(device));

    /// <summary>Live, aligned buffer lengths; excludes committed-heap alignment and driver overhead.</summary>
    public ulong AllocatedGpuBytes { get { lock (Gate) return allocatedGpuBytes; } }
    public int AllocationCount { get { lock (Gate) return storage.Count; } }
    public int ExecutorCount { get { lock (Gate) return executors; } }
    public ulong GlobalUploadCount { get { lock (Gate) return globalUploadCount; } }
    public ulong GlobalUploadBytes { get { lock (Gate) return globalUploadBytes; } }
    public D3D12VmTaskStatistics TaskStatistics
    {
        get
        {
            lock (Gate) return new(completedTasks, computeRecordings, inputUploadBytes,
                initialStateUploadBytes, initialLocalUploadBytes, stateLoadBytes, stateCommitBytes, cpuReadbackBytes,
                preparationMilliseconds, computeMilliseconds, commitMilliseconds);
        }
    }

    internal void RecordedTaskEntry() => computeRecordings = checked(computeRecordings + 1);

    internal void CompletedTask(ulong inputBytes, ulong initialStateBytes, ulong localBytes, ulong loadBytes, ulong commitBytes,
        double preparation, double compute, double commit)
    {
        completedTasks = checked(completedTasks + 1);
        inputUploadBytes = checked(inputUploadBytes + inputBytes);
        initialStateUploadBytes = checked(initialStateUploadBytes + initialStateBytes);
        initialLocalUploadBytes = checked(initialLocalUploadBytes + localBytes);
        stateLoadBytes = checked(stateLoadBytes + loadBytes);
        stateCommitBytes = checked(stateCommitBytes + commitBytes);
        preparationMilliseconds += preparation;
        computeMilliseconds += compute;
        commitMilliseconds += commit;
    }

    /// <summary>
    /// Use as VmResourceManager's or VmInferenceEngine's allocate delegate.
    /// The returned IVmStorage implements IVmManagedStorage; its Buffer is the stable identity.
    /// </summary>
    public IVmStorage Allocate(VmSlot slot)
    {
        ArgumentNullException.ThrowIfNull(slot);
        if (slot.Tensor is null || !Enum.IsDefined(slot.Scope) || !Enum.IsDefined(slot.Access))
            throw new ArgumentException("Invalid VM slot descriptor.", nameof(slot));
        if (slot.Tensor.ByteLength > int.MaxValue)
            throw new NotSupportedException("Pooled storage exceeds the managed byte[] limit.");
        lock (Gate)
        {
            ThrowIfDisposed();
            var id = checked(++nextAllocationId);
            var bytes = new byte[checked((int)slot.Tensor.ByteLength)];
            var resource = Device.CreateCommittedResource(HeapProperties.DefaultHeapProperties, HeapFlags.None,
                ResourceDescription.Buffer(Align(slot.Tensor.ByteLength), ResourceFlags.AllowUnorderedAccess),
                ResourceStates.UnorderedAccess);
            try
            {
                var allocation = new D3D12VmPooledStorage(this, id, slot, bytes, resource);
                var totalBytes = checked(allocatedGpuBytes + resource.Description.Width);
                storage.Add(bytes, allocation);
                allocations.Add(id, allocation);
                allocatedGpuBytes = totalBytes;
                return allocation;
            }
            catch
            {
                storage.Remove(bytes);
                allocations.Remove(id);
                resource.Dispose();
                throw;
            }
        }
    }

    public ulong GetGpuVirtualAddress(byte[] buffer)
    {
        lock (Gate) return Find(buffer).Resource.GPUVirtualAddress;
    }

    public bool Contains(byte[] buffer)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        lock (Gate)
        {
            ThrowIfDisposed();
            return storage.ContainsKey(buffer);
        }
    }

    /// <summary>
    /// Notify in-place CPU edits to read-only globals made through Buffer. Storage.Write does this
    /// automatically. The next managed execution uploads the global once for all workers.
    /// Mutable session/local shadows are always uploaded and do not need this notification.
    /// </summary>
    public void MarkCpuModified(byte[] buffer)
    {
        lock (Gate)
        {
            var allocation = Find(buffer);
            allocation.CpuModified();
        }
    }

    internal D3D12VmPooledStorage Resolve(VmSlot slot, byte[] buffer)
    {
        var allocation = Find(buffer);
        var descriptor = allocation.Slot;
        if (descriptor.Scope != slot.Scope || descriptor.Access != slot.Access ||
            descriptor.Tensor.ElementType != slot.Tensor.ElementType ||
            !descriptor.Tensor.Dimensions.SequenceEqual(slot.Tensor.Dimensions))
            throw new ArgumentException($"Pooled backing for slot '{slot.Id}' has a different scope, access, type or shape.", nameof(buffer));
        return allocation;
    }

    internal void ValidateLive(D3D12VmPooledStorage allocation)
    {
        ThrowIfDisposed();
        if (!storage.TryGetValue(allocation.RawBuffer, out var current) || !ReferenceEquals(allocation, current))
            throw new ObjectDisposedException(nameof(IVmStorage), "The pooled allocation is no longer registered.");
    }

    internal D3D12VmPooledStorage Resolve(ulong allocationId)
    {
        ThrowIfDisposed();
        return allocations.TryGetValue(allocationId, out var allocation) ? allocation :
            throw new ObjectDisposedException(nameof(IVmStorage), "The cached GPU allocation has been released.");
    }

    internal void UploadedGlobal(D3D12VmPooledStorage allocation)
    {
        allocation.GlobalUploaded = true;
        globalUploadCount = checked(globalUploadCount + 1);
        globalUploadBytes = checked(globalUploadBytes + allocation.ByteLength);
    }

    internal void EnsureCpuCurrent(D3D12VmPooledStorage allocation)
    {
        ValidateLive(allocation);
        if (allocation.CpuCurrent) return;
        if (!allocation.GpuInitialized)
            throw new InvalidOperationException("The GPU state has no valid CPU or device representation.");
        if (readQueue is null)
        {
            readQueue = Device.CreateCommandQueue(CommandListType.Compute);
            readAllocator = Device.CreateCommandAllocator(CommandListType.Compute);
            readCommands = Device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, readAllocator);
            readCommands.Close();
            readFence = Device.CreateFence();
            readCompletion = new EventWaitHandle(false, EventResetMode.AutoReset);
        }
        var bytes = Align(allocation.ByteLength);
        if (readCapacity < bytes)
        {
            readStaging?.Dispose();
            readStaging = Device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties, HeapFlags.None,
                ResourceDescription.Buffer(bytes), ResourceStates.CopyDest);
            readCapacity = bytes;
        }
        readAllocator!.Reset();
        readCommands!.Reset(readAllocator);
        readCommands.ResourceBarrierTransition(allocation.Resource, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        readCommands.CopyBufferRegion(readStaging!, 0, allocation.Resource, 0, allocation.ByteLength);
        readCommands.ResourceBarrierTransition(allocation.Resource, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        readCommands.Close();
        readQueue.ExecuteCommandList(readCommands);
        var value = checked(++readFenceValue);
        readQueue.Signal(readFence!, value).CheckError();
        if (readFence!.CompletedValue < value)
        {
            readFence.SetEventOnCompletion(value, readCompletion!).CheckError();
            while (!readCompletion!.WaitOne(TimeSpan.FromSeconds(1))) Device.DeviceRemovedReason.CheckError();
        }
        Device.DeviceRemovedReason.CheckError();
        readStaging!.GetData<byte>(allocation.RawBuffer);
        cpuReadbackBytes = checked(cpuReadbackBytes + allocation.ByteLength);
        allocation.CpuCurrent = true;
    }

    internal void AttachExecutor()
    {
        lock (Gate)
        {
            ThrowIfDisposed();
            executors = checked(executors + 1);
        }
    }

    internal void DetachExecutor()
    {
        lock (Gate)
        {
            if (executors <= 0) throw new InvalidOperationException("Unbalanced pooled executor lifetime.");
            executors--;
        }
    }

    // A pool serializes resource use, transfers and disposal across compute queues. CPU mirrors stay
    // coherent and shared UAV resources are never transitioned or released while another worker uses them.
    internal IDisposable BeginUse()
    {
        Monitor.Enter(Gate);
        try
        {
            ThrowIfDisposed();
            return new Use(Gate);
        }
        catch
        {
            Monitor.Exit(Gate);
            throw;
        }
    }

    internal void Release(D3D12VmPooledStorage allocation)
    {
        lock (Gate)
        {
            if (allocation.Disposed) return;
            if (!storage.Remove(allocation.RawBuffer))
                throw new InvalidOperationException("Pooled allocation registration is missing.");
            allocations.Remove(allocation.Id);
            allocatedGpuBytes -= allocation.Resource.Description.Width;
            allocation.Release();
        }
    }

    private D3D12VmPooledStorage Find(byte[] buffer)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        ThrowIfDisposed();
        return storage.TryGetValue(buffer, out var allocation) ? allocation :
            throw new ArgumentException("Buffer is not a live allocation of this GPU pool.", nameof(buffer));
    }

    private static ulong Align(ulong bytes) => checked((bytes + 3UL) & ~3UL);
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);

    public void Dispose()
    {
        lock (Gate)
        {
            if (disposed) return;
            if (executors != 0)
                throw new InvalidOperationException("Dispose all pooled executors before disposing their resource pool.");
            foreach (var allocation in storage.Values) allocation.Release();
            readStaging?.Dispose();
            readCommands?.Dispose();
            readAllocator?.Dispose();
            readFence?.Dispose();
            readQueue?.Dispose();
            readCompletion?.Dispose();
            storage.Clear();
            allocations.Clear();
            allocatedGpuBytes = 0;
            disposed = true;
        }
    }

    private sealed class Use(object gate) : IDisposable
    {
        private bool released;
        public void Dispose()
        {
            if (released) return;
            released = true;
            Monitor.Exit(gate);
        }
    }
}

internal sealed class D3D12VmPooledStorage(
    D3D12VmResourcePool pool, ulong id, VmSlot slot, byte[] buffer, ID3D12Resource resource) : IVmManagedStorage
{
    private byte[]? data = buffer;
    private ID3D12Resource? resource = resource;
    internal VmSlot Slot { get; } = slot;
    internal ulong Id { get; } = id;
    public ulong ByteLength { get; } = slot.Tensor.ByteLength;
    public byte[] Buffer
    {
        get
        {
            lock (pool.Gate)
            {
                pool.EnsureCpuCurrent(this);
                return RawBuffer;
            }
        }
    }
    internal byte[] RawBuffer => data ?? throw new ObjectDisposedException(nameof(IVmStorage));
    internal D3D12VmResourcePool Owner => pool;
    internal ID3D12Resource Resource => resource ?? throw new ObjectDisposedException(nameof(IVmStorage));
    internal bool GlobalUploaded { get; set; }
    internal bool GpuInitialized { get; set; }
    internal bool CpuCurrent { get; set; } = true;
    internal bool Disposed => data is null;

    public void Read(ulong offset, Span<byte> destination)
    {
        lock (pool.Gate)
        {
            pool.EnsureCpuCurrent(this);
            Data(offset, destination.Length).CopyTo(destination);
        }
    }
    public void Write(ulong offset, ReadOnlySpan<byte> source)
    {
        lock (pool.Gate)
        {
            if (offset != 0 || (ulong)source.Length != ByteLength) pool.EnsureCpuCurrent(this);
            source.CopyTo(Data(offset, source.Length));
            CpuModified();
        }
    }
    internal void CpuModified()
    {
        GlobalUploaded = false;
        GpuInitialized = false;
        CpuCurrent = true;
    }
    internal void GpuModified()
    {
        GpuInitialized = true;
        CpuCurrent = false;
    }
    private Span<byte> Data(ulong offset, int length)
    {
        var bytes = data ?? throw new ObjectDisposedException(nameof(IVmStorage));
        if (offset > ByteLength || (ulong)length > ByteLength - offset)
            throw new ArgumentOutOfRangeException(nameof(offset), "Storage access exceeds capacity.");
        return bytes.AsSpan(checked((int)offset), length);
    }
    internal void Release()
    {
        resource?.Dispose();
        resource = null;
        data = null;
        GlobalUploaded = false;
    }
    public void Dispose() => pool.Release(this);
}
