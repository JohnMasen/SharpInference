using System.Diagnostics;
using SharpInference.Vm;
using Vortice.Direct3D12;

namespace SharpInference.Backends.D3D12Vm;

internal sealed class D3D12VmTaskWindow(D3D12VmExecutor executor, D3D12VmResourcePool pool) : IDisposable
{
    private readonly List<D3D12VmPooledStorage> owned = [];
    private D3D12VmPooledStorage[]? window;
    private byte[][]? context;
    private ID3D12Resource? upload;
    private ulong uploadCapacity;
    private bool localInitialized;

    public void Execute(string entry, VmExecutionLease lease, CancellationToken cancellation)
    {
        if (!ReferenceEquals(lease.Program, executor.Program) &&
            VmProgramXml.Serialize(lease.Program) != VmProgramXml.Serialize(executor.Program))
            throw new ArgumentException("The task lease belongs to a different GPU program.", nameof(lease));
        cancellation.ThrowIfCancellationRequested();
        using var use = pool.BeginUse();
        executor.ValidateTaskEntry(entry, lease);
        var watch = Stopwatch.StartNew();
        var sources = executor.Program.Slots.Select(slot =>
        {
            if (lease.GetStorage(slot.Id) is not D3D12VmPooledStorage storage || !ReferenceEquals(storage.Owner, pool))
                throw new ArgumentException($"Slot '{slot.Id}' is not owned by this GPU pool.", nameof(lease));
            pool.ValidateLive(storage);
            return storage;
        }).ToArray();
        Initialize(sources);
        var active = window!;
        var globals = Enumerable.Range(0, sources.Length).Where(index =>
            FixedGlobal(index) && !sources[index].GlobalUploaded).ToArray();
        if (globals.Length != 0) executor.UploadTaskGlobals(context!, globals);
        var uploads = sources.Where((storage, index) => !FixedGlobal(index) &&
            (executor.Program.Slots[index].Scope != VmSlotScope.Local ||
                executor.Program.Slots[index].Access == VmAccess.ReadOnly) && !storage.GpuInitialized).ToList();
        var inputBytes = uploads.Where(storage => !executor.Program.State.Entries.Any(entry =>
            entry.Slot == storage.Slot.Id)).Aggregate(0UL, (sum, storage) => checked(sum + storage.ByteLength));
        var initialStateBytes = uploads.Where(storage => executor.Program.State.Entries.Any(entry =>
            entry.Slot == storage.Slot.Id)).Aggregate(0UL, (sum, storage) => checked(sum + storage.ByteLength));
        var localBytes = localInitialized ? 0UL : active.Where((_, index) =>
            executor.Program.Slots[index] is { Scope: VmSlotScope.Local, Access: VmAccess.ReadWrite })
            .Aggregate(0UL, (sum, storage) => checked(sum + storage.ByteLength));
        if (!localInitialized)
            uploads.AddRange(active.Where((_, index) =>
                executor.Program.Slots[index] is { Scope: VmSlotScope.Local, Access: VmAccess.ReadWrite }));
        Upload(uploads);
        localInitialized = true;
        executor.ExecuteTransfer(list =>
        {
            for (var index = 0; index < sources.Length; index++)
                if (!FixedGlobal(index) && executor.Program.Slots[index].Scope != VmSlotScope.Local)
                    Copy(list, sources[index].Resource, active[index].Resource, sources[index].ByteLength);
        });
        foreach (var storage in active) storage.GpuInitialized = true;
        cancellation.ThrowIfCancellationRequested();
        var preparation = watch.Elapsed.TotalMilliseconds;
        watch.Restart();
        executor.Execute(entry);
        var compute = watch.Elapsed.TotalMilliseconds;
        watch.Restart();
        executor.ExecuteTransfer(list =>
        {
            for (var index = 0; index < sources.Length; index++)
                if (executor.Program.Slots[index] is { Access: VmAccess.ReadWrite, Scope: not VmSlotScope.Local })
                    Copy(list, active[index].Resource, sources[index].Resource, sources[index].ByteLength);
        });
        for (var index = 0; index < sources.Length; index++)
            if (executor.Program.Slots[index].Access == VmAccess.ReadWrite)
                sources[index].GpuModified();
        cancellation.ThrowIfCancellationRequested();
        var stateBytes = sources.Where(storage => executor.Program.State.Entries.Any(entry =>
            entry.Slot == storage.Slot.Id)).Aggregate(0UL, (sum, storage) => checked(sum + storage.ByteLength));
        pool.CompletedTask(inputBytes, initialStateBytes, localBytes, stateBytes, stateBytes,
            preparation, compute, watch.Elapsed.TotalMilliseconds);
    }

    private bool FixedGlobal(int index) =>
        executor.Program.Slots[index] is { Scope: VmSlotScope.Global, Access: VmAccess.ReadOnly };

    private void Initialize(D3D12VmPooledStorage[] sources)
    {
        if (window is null)
        {
            try
            {
                window = executor.Program.Slots.Select((slot, index) =>
                {
                    if (FixedGlobal(index) || slot.Scope == VmSlotScope.Local) return sources[index];
                    if (pool.Allocate(slot) is not D3D12VmPooledStorage storage)
                        throw new InvalidOperationException("The GPU allocator did not provide pooled storage.");
                    owned.Add(storage);
                    return storage;
                }).ToArray();
                context = window.Select(storage => storage.RawBuffer).ToArray();
            }
            catch
            {
                Dispose();
                throw;
            }
        }
        for (var index = 0; index < sources.Length; index++)
            if ((FixedGlobal(index) || executor.Program.Slots[index].Scope == VmSlotScope.Local) &&
                !ReferenceEquals(sources[index], window[index]))
                throw new InvalidOperationException("A task instance cannot replace its fixed globals or local workspace.");
        executor.BindTaskWindow(context!);
    }

    private void Upload(IReadOnlyList<D3D12VmPooledStorage> storages)
    {
        if (storages.Count == 0) return;
        var bytes = storages.Aggregate(0UL, (sum, storage) => checked(sum + Align(storage.ByteLength)));
        if (bytes > int.MaxValue) throw new NotSupportedException("A task upload exceeds the managed staging capacity.");
        if (uploadCapacity < bytes)
        {
            upload?.Dispose();
            upload = pool.Device.CreateCommittedResource(HeapProperties.UploadHeapProperties, HeapFlags.None,
                ResourceDescription.Buffer(bytes), ResourceStates.GenericRead);
            uploadCapacity = bytes;
        }
        var payload = new byte[checked((int)bytes)];
        var offset = 0;
        foreach (var storage in storages)
        {
            storage.RawBuffer.CopyTo(payload, offset);
            offset += checked((int)Align(storage.ByteLength));
        }
        upload!.SetData<byte>(payload);
        executor.ExecuteTransfer(list =>
        {
            ulong position = 0;
            foreach (var storage in storages)
            {
                list.ResourceBarrierTransition(storage.Resource, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                list.CopyBufferRegion(storage.Resource, 0, upload, position, storage.ByteLength);
                list.ResourceBarrierTransition(storage.Resource, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                position += Align(storage.ByteLength);
            }
        });
        foreach (var storage in storages) storage.GpuInitialized = true;
    }

    private static void Copy(ID3D12GraphicsCommandList list, ID3D12Resource source, ID3D12Resource destination, ulong bytes)
    {
        list.ResourceBarrierTransition(source, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        list.ResourceBarrierTransition(destination, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        list.CopyBufferRegion(destination, 0, source, 0, bytes);
        list.ResourceBarrierTransition(destination, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
        list.ResourceBarrierTransition(source, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
    }

    private static ulong Align(ulong bytes) => checked((bytes + 3UL) & ~3UL);

    public void Dispose()
    {
        upload?.Dispose();
        upload = null;
        foreach (var storage in owned) storage.Dispose();
        owned.Clear();
        window = null;
        context = null;
    }
}
