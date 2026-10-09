namespace SharpInference.Vm;

/// <summary>Provides byte-addressable storage for VM resources.</summary>
public interface IVmStorage : IDisposable
{
    ulong ByteLength { get; }
    void Read(ulong offset, Span<byte> destination);
    void Write(ulong offset, ReadOnlySpan<byte> source);
}

/// <summary>Provides VM storage backed by a managed byte array.</summary>
public interface IVmManagedStorage : IVmStorage
{
    byte[] Buffer { get; }
}

/// <summary>Stores VM resource bytes in a managed array.</summary>
public sealed class VmMemoryStorage : IVmManagedStorage
{
    private byte[]? data;

    /// <summary>Allocates storage with the specified byte capacity.</summary>
    /// <param name="byteLength">The positive storage capacity in bytes.</param>
    public VmMemoryStorage(int byteLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
        data = new byte[byteLength];
        ByteLength = (ulong)byteLength;
    }

    public ulong ByteLength { get; }
    public byte[] Buffer => data ?? throw new ObjectDisposedException(nameof(VmMemoryStorage));

    /// <summary>Reads bytes from storage into the destination span.</summary>
    public void Read(ulong offset, Span<byte> destination) =>
        Data(offset, destination.Length).CopyTo(destination);

    /// <summary>Writes source bytes into storage.</summary>
    public void Write(ulong offset, ReadOnlySpan<byte> source) =>
        source.CopyTo(Data(offset, source.Length));

    private Span<byte> Data(ulong offset, int length)
    {
        var buffer = data ?? throw new ObjectDisposedException(nameof(VmMemoryStorage));
        if (offset > ByteLength || (ulong)length > ByteLength - offset)
            throw new ArgumentOutOfRangeException(nameof(offset), "Storage access exceeds capacity.");
        return buffer.AsSpan(checked((int)offset), length);
    }

    /// <summary>Releases the managed byte array.</summary>
    public void Dispose() => data = null;
}

/// <summary>Owns storage and coordinates VM resource references and access leases.</summary>
public sealed class VmResource : IDisposable
{
    private readonly object gate = new();
    private readonly IVmStorage storage;
    private int references = 1;
    private bool ownerDisposed;
    private int localBindingReaders;
    private bool localBindingWriter;
    private bool valid = true;
    private int readers;
    private bool writer;

    /// <summary>Creates a resource whose storage capacity matches a tensor descriptor.</summary>
    /// <param name="tensor">The tensor descriptor represented by the storage.</param>
    /// <param name="scope">The VM slot scope of the resource.</param>
    /// <param name="access">The permitted resource access.</param>
    /// <param name="storage">The storage backing the resource.</param>
    public VmResource(VmTensor tensor, VmSlotScope scope, VmAccess access, IVmStorage storage)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentNullException.ThrowIfNull(storage);
        if (!Enum.IsDefined(scope))
            throw new ArgumentOutOfRangeException(nameof(scope));
        if (!Enum.IsDefined(access))
            throw new ArgumentOutOfRangeException(nameof(access));
        if (storage.ByteLength != tensor.ByteLength)
            throw new ArgumentException("Storage capacity must match the resource tensor.", nameof(storage));
        Tensor = tensor;
        Scope = scope;
        Access = access;
        this.storage = storage;
    }

    public VmTensor Tensor { get; }
    public VmSlotScope Scope { get; }
    public VmAccess Access { get; }
    internal bool Valid { get { lock (gate) return valid; } }
    internal void SetValidity(bool value) { lock (gate) valid = value; }

    /// <summary>Acquires an owning reference lease for the resource.</summary>
    public VmResourceLease Acquire()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(ownerDisposed, this);
            references = checked(references + 1);
            return new VmResourceLease(this);
        }
    }

    internal VmResourceLease Retain()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(references == 0, this);
            references = checked(references + 1);
            return new VmResourceLease(this);
        }
    }

    internal VmResourceLease AcquireBinding(VmAccess access)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(references == 0, this);
            if (Scope == VmSlotScope.Local &&
                (localBindingWriter || access == VmAccess.ReadWrite && localBindingReaders != 0))
                throw new InvalidOperationException(
                    "Local storage cannot have overlapping writable bindings.");
            references = checked(references + 1);
            if (Scope == VmSlotScope.Local)
            {
                if (access == VmAccess.ReadWrite) localBindingWriter = true;
                else localBindingReaders++;
            }
            return new VmResourceLease(this, access);
        }
    }

    internal VmResourceUse AcquireUse(VmAccess access)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(references == 0, this);
            if (writer || access == VmAccess.ReadWrite && readers != 0)
                throw new InvalidOperationException("The physical resource is already in use.");
            references = checked(references + 1);
            if (access == VmAccess.ReadWrite) writer = true;
            else readers++;
            return new VmResourceUse(this, access);
        }
    }

    internal void ReleaseUse(VmAccess access)
    {
        lock (gate)
        {
            if (access == VmAccess.ReadWrite) writer = false;
            else readers--;
        }
        Release();
    }

    internal void Read(ulong offset, Span<byte> destination) => storage.Read(offset, destination);
    internal byte[] GetManagedBuffer() => storage is IVmManagedStorage memory ? memory.Buffer :
        throw new NotSupportedException("This resource does not expose managed backing storage.");
    internal IVmStorage Storage => storage;

    internal void Write(ulong offset, ReadOnlySpan<byte> source)
    {
        if (Access != VmAccess.ReadWrite)
            throw new InvalidOperationException("The resource is read-only.");
        storage.Write(offset, source);
    }

    internal void Release(VmAccess? bindingAccess = null)
    {
        var dispose = false;
        lock (gate)
        {
            if (references <= 0)
                throw new InvalidOperationException("Unbalanced resource release.");
            if (bindingAccess is not null && Scope == VmSlotScope.Local)
            {
                if (bindingAccess == VmAccess.ReadWrite) localBindingWriter = false;
                else localBindingReaders--;
            }
            dispose = --references == 0;
        }
        if (dispose) storage.Dispose();
    }

    /// <summary>Releases the owner's reference to the resource.</summary>
    public void Dispose()
    {
        lock (gate)
        {
            if (ownerDisposed) return;
            ownerDisposed = true;
        }
        Release();
    }
}

/// <summary>Represents a disposable reference to a VM resource.</summary>
public sealed class VmResourceLease : IDisposable
{
    private readonly object gate = new();
    private VmResource? resource;
    private readonly VmAccess? bindingAccess;

    internal VmResourceLease(VmResource resource, VmAccess? bindingAccess = null)
    {
        this.resource = resource;
        this.bindingAccess = bindingAccess;
    }
    /// <summary>Gets the tensor descriptor of the leased resource.</summary>
    public VmTensor Tensor { get { lock (gate) return Resource.Tensor; } }

    /// <summary>Gets the slot scope of the leased resource.</summary>
    public VmSlotScope Scope { get { lock (gate) return Resource.Scope; } }

    /// <summary>Gets the access mode of the leased resource.</summary>
    public VmAccess Access { get { lock (gate) return Resource.Access; } }
    internal bool Valid { get { lock (gate) return Resource.Valid; } }
    internal void SetValidity(bool value) { lock (gate) Resource.SetValidity(value); }
    internal VmResourceLease AcquireBinding(VmAccess access) { lock (gate) return Resource.AcquireBinding(access); }
    internal VmResource Identity { get { lock (gate) return Resource; } }
    internal VmResourceUse AcquireUse(VmAccess access) { lock (gate) return Resource.AcquireUse(access); }
    internal byte[] GetManagedBuffer() { lock (gate) return Resource.GetManagedBuffer(); }
    internal IVmStorage Storage { get { lock (gate) return Resource.Storage; } }

    public VmResourceLease Retain()
    {
        lock (gate) return Resource.Retain();
    }

    public void Read(ulong offset, Span<byte> destination)
    {
        lock (gate)
        {
            using var use = Resource.AcquireUse(VmAccess.ReadOnly);
            Resource.Read(offset, destination);
        }
    }

    public void Write(ulong offset, ReadOnlySpan<byte> source)
    {
        lock (gate)
        {
            using var use = Resource.AcquireUse(VmAccess.ReadWrite);
            Resource.Write(offset, source);
        }
    }

    internal void ReadBound(ulong offset, Span<byte> destination)
    {
        lock (gate) Resource.Read(offset, destination);
    }

    internal void WriteBound(ulong offset, ReadOnlySpan<byte> source)
    {
        lock (gate) Resource.Write(offset, source);
    }

    private VmResource Resource => resource ?? throw new ObjectDisposedException(nameof(VmResourceLease));

    public void Dispose()
    {
        VmResource? released;
        lock (gate)
        {
            released = resource;
            resource = null;
        }
        released?.Release(bindingAccess);
    }
}

internal sealed class VmResourceUse(VmResource resource, VmAccess access) : IDisposable
{
    private VmResource? resource = resource;
    public void Dispose() => Interlocked.Exchange(ref resource, null)?.ReleaseUse(access);
}

public sealed class VmBindings : IDisposable
{
    private readonly object gate = new();
    private readonly VmProgram program;
    private readonly Dictionary<string, VmResourceLease> resources = new(StringComparer.Ordinal);
    private bool executing;
    private bool disposed;
    private readonly List<VmResourceUse> activeUses = [];

    public VmBindings(VmProgram program) => this.program =
        program ?? throw new ArgumentNullException(nameof(program));
    public VmProgram Program => program;
    internal bool StateValid
    {
        get
        {
            lock (gate)
                return program.State.Entries.All(entry => resources[entry.Slot].Valid);
        }
    }

    public void Bind(string slotId, VmResourceLease resource)
    {
        ArgumentNullException.ThrowIfNull(resource);
        lock (gate)
        {
            EnsureIdle();
            var slot = program.Slots.SingleOrDefault(slot => slot.Id == slotId) ??
                throw new ArgumentException($"Unknown slot '{slotId}'.", nameof(slotId));
            using var validationLease = resource.Retain();
            var tensor = validationLease.Tensor;
            if (slot.Tensor.ElementType != tensor.ElementType ||
                !slot.Tensor.Dimensions.SequenceEqual(tensor.Dimensions) ||
                slot.Scope != validationLease.Scope ||
                slot.Access == VmAccess.ReadWrite && validationLease.Access != VmAccess.ReadWrite)
                throw new ArgumentException($"Resource is incompatible with slot '{slotId}'.", nameof(resource));
            var owned = validationLease.AcquireBinding(slot.Access);
            resources.Remove(slotId, out var previous);
            resources.Add(slotId, owned);
            previous?.Dispose();
        }
    }

    public VmExecutionLease BeginExecution() => BeginAccess(allowInvalidState: false);

    public VmExecutionLease BeginStateAccess() => BeginAccess(allowInvalidState: true);

    private VmExecutionLease BeginAccess(bool allowInvalidState)
    {
        lock (gate)
        {
            EnsureIdle();
            if (program.Slots.Any(slot => !resources.ContainsKey(slot.Id)))
                throw new InvalidOperationException("All program slots must be bound before execution.");
            try
            {
                foreach (var group in program.Slots.GroupBy(slot => resources[slot.Id].Identity))
                {
                    var access = group.Any(slot => slot.Access == VmAccess.ReadWrite ||
                        allowInvalidState && resources[slot.Id].Access == VmAccess.ReadWrite &&
                            program.State.Entries.Any(entry => entry.Slot == slot.Id))
                        ? VmAccess.ReadWrite : VmAccess.ReadOnly;
                    activeUses.Add(resources[group.First().Id].AcquireUse(access));
                }
                if (!allowInvalidState && !StateValid)
                    throw new InvalidOperationException("State must be restored before this VM can execute.");
            }
            catch
            {
                foreach (var use in activeUses) use.Dispose();
                activeUses.Clear();
                throw;
            }
            executing = true;
            return new VmExecutionLease(this, stateAccess: allowInvalidState);
        }
    }

    internal void EndExecution()
    {
        lock (gate)
        {
            foreach (var use in activeUses) use.Dispose();
            activeUses.Clear();
            executing = false;
        }
    }

    internal byte[][] GetBuffers()
    {
        lock (gate)
        {
            if (!executing)
                throw new InvalidOperationException("CPU buffer views require an active execution lease.");
            return program.Slots.Select(slot => resources[slot.Id].GetManagedBuffer()).ToArray();
        }

    }

    internal IVmStorage GetStorage(string slotId)
    {
        lock (gate)
        {
            if (!executing)
                throw new InvalidOperationException("Storage access requires an active execution lease.");
            return resources.TryGetValue(slotId, out var resource) ? resource.Storage :
                throw new ArgumentException($"Unknown slot '{slotId}'.", nameof(slotId));
        }
    }

    internal void SetStateValidity(bool valid)
    {
        lock (gate)
            foreach (var entry in program.State.Entries)
                resources[entry.Slot].SetValidity(valid);
    }

    internal void Read(string slotId, ulong offset, Span<byte> destination)
    {
        lock (gate) resources[slotId].ReadBound(offset, destination);
    }

    internal void Write(string slotId, ulong offset, ReadOnlySpan<byte> source)
    {
        lock (gate)
        {
            if (program.Slots.Single(slot => slot.Id == slotId).Access != VmAccess.ReadWrite)
                throw new InvalidOperationException($"Slot '{slotId}' is read-only.");
            resources[slotId].WriteBound(offset, source);
        }
    }

    internal bool CanRestoreState(string slotId)
    {
        lock (gate) return resources[slotId].Access == VmAccess.ReadWrite;
    }

    internal void WriteState(string slotId, ReadOnlySpan<byte> source)
    {
        lock (gate)
        {
            if (!program.State.Entries.Any(entry => entry.Slot == slotId))
                throw new InvalidOperationException("State transfer cannot write an unregistered slot.");
            resources[slotId].WriteBound(0, source);
        }
    }

    public void UnbindSession()
    {
        lock (gate)
        {
            EnsureIdle();
            foreach (var slot in program.Slots.Where(slot => slot.Scope == VmSlotScope.Session))
                if (resources.Remove(slot.Id, out var resource)) resource.Dispose();
        }
    }

    private void EnsureIdle()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (executing)
            throw new InvalidOperationException("An executing VM cannot be rebound, disposed or executed again.");
    }

    public void Dispose()
    {
        List<Exception>? errors = null;
        lock (gate)
        {
            if (disposed) return;
            EnsureIdle();
            disposed = true;
            foreach (var resource in resources.Values)
            {
                try { resource.Dispose(); }
                catch (Exception error) { (errors ??= []).Add(error); }
            }
            resources.Clear();
        }
        if (errors is not null)
            throw new AggregateException("Failed to release VM resources.", errors);
    }
}

public sealed class VmExecutionLease : IDisposable
{
    private readonly object gate = new();
    private VmBindings? bindings;
    internal VmExecutionLease(VmBindings bindings, bool stateAccess)
    {
        this.bindings = bindings;
        IsStateAccess = stateAccess;
    }
    internal bool IsStateAccess { get; }
    public VmProgram Program { get { lock (gate) return Bindings.Program; } }
    internal bool StateValid { get { lock (gate) return Bindings.StateValid; } }
    internal bool CanRestoreState(string slotId) { lock (gate) return Bindings.CanRestoreState(slotId); }

    internal void WriteState(string slotId, ReadOnlySpan<byte> source)
    {
        lock (gate)
        {
            if (!IsStateAccess)
                throw new InvalidOperationException("State restoration requires a state-access lease.");
            Bindings.WriteState(slotId, source);
        }
    }

    public void Read(string slotId, ulong offset, Span<byte> destination)
    {
        lock (gate) Bindings.Read(slotId, offset, destination);
    }

    public void Write(string slotId, ulong offset, ReadOnlySpan<byte> source)
    {
        lock (gate) Bindings.Write(slotId, offset, source);
    }

    public byte[][] GetBuffers()
    {
        lock (gate) return Bindings.GetBuffers();
    }

    public IVmStorage GetStorage(string slotId)
    {
        lock (gate) return Bindings.GetStorage(slotId);
    }

    public void InvalidateState()
    {
        lock (gate) Bindings.SetStateValidity(false);
    }

    public void RestoreState(IReadOnlyDictionary<string, ReadOnlyMemory<byte>> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        lock (gate)
        {
            if (!IsStateAccess)
                throw new InvalidOperationException("State restoration requires a state-access lease.");
            var program = Bindings.Program;
            if (values.Count != program.State.Entries.Count)
                throw new InvalidDataException("State buffers must exactly cover the program state entries.");
            var staged = program.State.Entries.Select(entry =>
            {
                var slot = program.Slots.Single(candidate => candidate.Id == entry.Slot);
                if (!values.TryGetValue(entry.Name, out var bytes) || (ulong)bytes.Length != slot.Tensor.ByteLength ||
                    !Bindings.CanRestoreState(entry.Slot))
                    throw new InvalidDataException($"State entry '{entry.Name}' has incompatible storage.");
                return (entry.Slot, Bytes: bytes.ToArray());
            }).ToArray();
            try
            {
                foreach (var item in staged) Bindings.WriteState(item.Slot, item.Bytes);
                Bindings.SetStateValidity(true);
            }
            catch
            {
                Bindings.SetStateValidity(false);
                throw;
            }
        }
    }

    internal void ValidateState()
    {
        lock (gate) Bindings.SetStateValidity(true);
    }

    private VmBindings Bindings => bindings ?? throw new ObjectDisposedException(nameof(VmExecutionLease));

    public void Dispose()
    {
        lock (gate)
        {
            bindings?.EndExecution();
            bindings = null;
        }
    }
}
