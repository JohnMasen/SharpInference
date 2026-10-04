namespace SharpInference.Vm;

public sealed class VmResourceManager : IDisposable
{
    private readonly object gate = new();
    private readonly Dictionary<string, VmResource> globals = new(StringComparer.Ordinal);
    private readonly Action<VmSlot, IVmStorage> initializeGlobal;
    private readonly Func<VmSlot, IVmStorage> allocate;
    private bool disposed;

    public VmResourceManager(Action<VmSlot, IVmStorage> initializeGlobal,
        Func<VmSlot, IVmStorage>? allocate = null)
    {
        this.initializeGlobal = initializeGlobal ?? throw new ArgumentNullException(nameof(initializeGlobal));
        this.allocate = allocate ?? (slot => new VmMemoryStorage(checked((int)slot.Tensor.ByteLength)));
    }

    public VmBindings CreateBindings(VmProgram program)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var bindings = new VmBindings(program);
            try
            {
                foreach (var slot in program.Slots.Where(slot => slot.Scope != VmSlotScope.Session))
                {
                    if (slot.Scope == VmSlotScope.Global)
                    {
                        var key = slot.BindingKey ?? slot.Id;
                        if (!globals.TryGetValue(key, out var resource))
                        {
                            resource = CreateResource(slot, slot.Access, initialize: true);
                            globals.Add(key, resource);
                        }
                        using var lease = resource.Acquire();
                        bindings.Bind(slot.Id, lease);
                    }
                    else
                    {
                        using var resource = CreateResource(slot, VmAccess.ReadWrite, initialize: false);
                        using var lease = resource.Acquire();
                        bindings.Bind(slot.Id, lease);
                    }
                }
                return bindings;
            }
            catch (Exception error)
            {
                try { bindings.Dispose(); }
                catch (Exception cleanup) { throw new AggregateException("Binding construction and cleanup failed.", error, cleanup); }
                throw;
            }
        }
    }

    public VmSessionResources CreateSession(params VmProgram[] programs)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return new VmSessionResources(programs, allocate);
        }
    }

    public void BindSession(VmBindings bindings, VmSessionResources session)
    {
        ArgumentNullException.ThrowIfNull(bindings);
        ArgumentNullException.ThrowIfNull(session);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            foreach (var slot in bindings.Program.Slots.Where(slot => slot.Scope == VmSlotScope.Session))
            {
                using var lease = session.Acquire(bindings.Program, slot);
                bindings.Bind(slot.Id, lease);
            }
        }
    }

    private VmResource CreateResource(VmSlot slot, VmAccess access, bool initialize)
    {
        var storage = allocate(slot);
        try
        {
            if (initialize) initializeGlobal(slot, storage);
            return new VmResource(slot.Tensor, slot.Scope, access, storage);
        }
        catch (Exception error)
        {
            try { storage.Dispose(); }
            catch (Exception cleanup) { throw new AggregateException("Resource construction and cleanup failed.", error, cleanup); }
            throw;
        }
    }

    public void Dispose()
    {
        List<Exception>? errors = null;
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            foreach (var resource in globals.Values)
            {
                try { resource.Dispose(); }
                catch (Exception error) { (errors ??= []).Add(error); }
            }
            globals.Clear();
        }
        if (errors is not null) throw new AggregateException("Global resource cleanup failed.", errors);
    }
}

public sealed class VmSessionResources : IDisposable
{
    private readonly object gate = new();
    private readonly Dictionary<string, VmResource> resources = new(StringComparer.Ordinal);
    private bool disposed;

    internal VmSessionResources(IReadOnlyList<VmProgram> programs, Func<VmSlot, IVmStorage> allocate)
    {
        ValidateContracts(programs);
        var first = programs[0];
        var stateSlots = first.Slots.ToDictionary(slot => slot.Id, StringComparer.Ordinal);
        StateProgram = new VmProgram("session-state", first.Abi, VmTarget.Cpu,
            first.State.Entries.Select(entry => stateSlots[entry.Slot] with { Id = entry.Name, Access = VmAccess.ReadWrite }),
            [new VmDefinition("state", VmDefinitionKind.Function, [], [])], [new VmEntry("state", "state", [])],
            new VmState(first.State.Schema, first.State.Version,
                first.State.Entries.Select(entry => new VmStateEntry(entry.Name, entry.Name))));
        try
        {
            foreach (var program in programs)
            {
                foreach (var slot in program.Slots.Where(slot => slot.Scope == VmSlotScope.Session))
                {
                    var key = Key(program, slot);
                    if (resources.TryGetValue(key, out var existing))
                    {
                        if (existing.Tensor.ElementType != slot.Tensor.ElementType ||
                            !existing.Tensor.Dimensions.SequenceEqual(slot.Tensor.Dimensions))
                            throw new InvalidDataException($"Session slot '{slot.Id}' has incompatible descriptors.");
                        continue;
                    }
                    var storage = allocate(slot);
                    VmResource resource;
                    try { resource = new VmResource(slot.Tensor, VmSlotScope.Session, VmAccess.ReadWrite, storage); }
                    catch (Exception error)
                    {
                        try { storage.Dispose(); }
                        catch (Exception cleanup) { throw new AggregateException("Session storage construction and cleanup failed.", error, cleanup); }
                        throw;
                    }
                    resources.Add(key, resource);
                }
            }
        }
        catch (Exception error)
        {
            try { Dispose(); }
            catch (Exception cleanup) { throw new AggregateException("Session construction and cleanup failed.", error, cleanup); }
            throw;
        }
    }

    public VmProgram StateProgram { get; }

    internal static void ValidateContracts(IReadOnlyList<VmProgram> programs)
    {
        ArgumentNullException.ThrowIfNull(programs);
        if (programs.Count == 0) throw new ArgumentException("At least one program is required.", nameof(programs));
        var first = programs[0];
        var descriptors = new Dictionary<string, VmTensor>(StringComparer.Ordinal);
        foreach (var program in programs)
        {
            if (program.State.Schema != first.State.Schema || program.State.Version != first.State.Version ||
                !program.State.Entries.Select(entry => entry.Name).SequenceEqual(first.State.Entries.Select(entry => entry.Name)))
                throw new InvalidDataException("Session programs require compatible State contracts.");
            foreach (var slot in program.Slots.Where(slot => slot.Scope == VmSlotScope.Session))
            {
                var key = Key(program, slot);
                if (descriptors.TryGetValue(key, out var descriptor))
                {
                    if (descriptor.ElementType != slot.Tensor.ElementType ||
                        !descriptor.Dimensions.SequenceEqual(slot.Tensor.Dimensions))
                        throw new InvalidDataException($"Session slot '{slot.Id}' has incompatible descriptors.");
                }
                else descriptors.Add(key, slot.Tensor);
            }
        }
    }

    internal VmResourceLease Acquire(VmProgram program, VmSlot slot)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return resources[Key(program, slot)].Acquire();
        }
    }

    public VmBindings CreateStateBindings()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var bindings = new VmBindings(StateProgram);
            try
            {
                foreach (var slot in StateProgram.Slots)
                {
                    using var lease = resources[$"state:{slot.Id}"].Acquire();
                    bindings.Bind(slot.Id, lease);
                }
                return bindings;
            }
            catch (Exception error)
            {
                try { bindings.Dispose(); }
                catch (Exception cleanup) { throw new AggregateException("State binding construction and cleanup failed.", error, cleanup); }
                throw;
            }
        }
    }

    private static string Key(VmProgram program, VmSlot slot)
    {
        var entry = program.State.Entries.SingleOrDefault(entry => entry.Slot == slot.Id);
        return entry is not null ? $"state:{entry.Name}" : $"private:{program.Name}:{slot.Id}";
    }

    public void Dispose()
    {
        List<Exception>? errors = null;
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            foreach (var resource in resources.Values)
            {
                try { resource.Dispose(); }
                catch (Exception error) { (errors ??= []).Add(error); }
            }
            resources.Clear();
        }
        if (errors is not null) throw new AggregateException("Session resource cleanup failed.", errors);
    }
}
