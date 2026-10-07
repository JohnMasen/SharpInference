using SharpInference.Graphs;
using SharpInference.Vm;

namespace SharpInference.Runtime;

public enum VmPortDirection
{
    Input,
    Output,
}

public readonly record struct VmPortId
{
    public VmPortId(string component, string port)
    {
        Component = Required(component, nameof(component));
        Port = Required(port, nameof(port));
    }

    public string Component { get; }
    public string Port { get; }
    public override string ToString() => $"{Component}.{Port}";

    private static string Required(string value, string parameter) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException("A port identifier value is required.", parameter)
            : value;
}

public sealed record VmProgramPort
{
    public VmProgramPort(
        VmPortId id,
        string slotId,
        VmPortDirection direction,
        ComponentPortDescriptor descriptor)
    {
        Id = id;
        SlotId = string.IsNullOrWhiteSpace(slotId)
            ? throw new ArgumentException("A VM slot identifier is required.", nameof(slotId))
            : slotId;
        if (!Enum.IsDefined(direction)) throw new ArgumentOutOfRangeException(nameof(direction));
        Descriptor = descriptor ?? throw new ArgumentNullException(nameof(descriptor));
        if (direction == VmPortDirection.Input && descriptor.Access != GraphResourceAccess.Read)
            throw new ArgumentException("VM input ports must be read-only.", nameof(descriptor));
        if (direction == VmPortDirection.Output && descriptor.Access == GraphResourceAccess.Read)
            throw new ArgumentException("VM output ports must be writable.", nameof(descriptor));
        Direction = direction;
    }

    public VmPortId Id { get; }
    public string SlotId { get; }
    public VmPortDirection Direction { get; }
    public ComponentPortDescriptor Descriptor { get; }
}

public sealed class VmComponentProgram
{
    public VmComponentProgram(
        string name,
        VmProgram program,
        StorageDomain storageDomain,
        IEnumerable<VmProgramPort> ports)
    {
        Name = string.IsNullOrWhiteSpace(name)
            ? throw new ArgumentException("A component name is required.", nameof(name))
            : name;
        Program = program ?? throw new ArgumentNullException(nameof(program));
        StorageDomain = storageDomain ?? throw new ArgumentNullException(nameof(storageDomain));
        ArgumentNullException.ThrowIfNull(ports);
        var values = ports.ToArray();
        if (values.Any(port => !string.Equals(port.Id.Component, Name, StringComparison.Ordinal)))
            throw new ArgumentException("Every VM port must belong to its component.", nameof(ports));
        if (values.GroupBy(port => port.Id).Any(group => group.Count() != 1))
            throw new ArgumentException("VM port identifiers must be unique.", nameof(ports));
        var slots = program.Slots.ToDictionary(slot => slot.Id, StringComparer.Ordinal);
        foreach (var port in values)
        {
            if (!slots.TryGetValue(port.SlotId, out var slot))
                throw new ArgumentException(
                    $"Port '{port.Id}' references unknown slot '{port.SlotId}'.", nameof(ports));
            if (!TensorEquals(slot.Tensor, port.Descriptor.Tensor))
                throw new ArgumentException(
                    $"Port '{port.Id}' does not match slot '{port.SlotId}'.", nameof(ports));
        }
        Ports = Array.AsReadOnly(values);
    }

    public string Name { get; }
    public VmProgram Program { get; }
    public StorageDomain StorageDomain { get; }
    public IReadOnlyList<VmProgramPort> Ports { get; }
    public IReadOnlyList<VmProgramPort> Inputs =>
        Ports.Where(port => port.Direction == VmPortDirection.Input).ToArray();
    public IReadOnlyList<VmProgramPort> Outputs =>
        Ports.Where(port => port.Direction == VmPortDirection.Output).ToArray();

    private static bool TensorEquals(VmTensor slot, TensorDescriptor port) =>
        slot.ElementType == (port.ElementType switch
        {
            GraphElementType.Byte => VmElementType.Byte,
            GraphElementType.Float16 => VmElementType.Float16,
            GraphElementType.Float32 => VmElementType.Float32,
            GraphElementType.Int32 => VmElementType.Int32,
            GraphElementType.UInt32 => VmElementType.UInt32,
            _ => (VmElementType)(-1),
        }) && slot.Dimensions.SequenceEqual(port.Dimensions);
}

public sealed class VmConnectionRegistry
{
    private readonly IReadOnlyDictionary<string, VmComponentProgram> components;
    private readonly Dictionary<VmPortId, VmPortId> producers = [];
    private readonly Dictionary<VmPortId, List<VmPortId>> consumers = [];

    public VmConnectionRegistry(IEnumerable<VmComponentProgram> components)
    {
        ArgumentNullException.ThrowIfNull(components);
        this.components = components.ToDictionary(component => component.Name, StringComparer.Ordinal);
    }

    public void Connect(VmPortId source, VmPortId target)
    {
        var sourcePort = GetPort(source, VmPortDirection.Output);
        var targetPort = GetPort(target, VmPortDirection.Input);
        if (source.Component == target.Component)
            throw new InvalidDataException($"Connection '{source}' to '{target}' creates a self-dependency.");
        ComponentPortDescriptor.ValidateConnection(sourcePort.Descriptor, targetPort.Descriptor);
        if (producers.ContainsKey(target))
            throw new InvalidOperationException($"Input port '{target}' already has a producer.");

        producers.Add(target, source);
        if (!consumers.TryGetValue(source, out var targets))
        {
            targets = [];
            consumers.Add(source, targets);
        }
        targets.Add(target);
        if (CreatesCycle(source.Component, target.Component))
        {
            targets.RemoveAt(targets.Count - 1);
            if (targets.Count == 0) consumers.Remove(source);
            producers.Remove(target);
            throw new InvalidDataException(
                $"Connection '{source}' to '{target}' creates a component cycle.");
        }
    }

    public VmPortId GetProducer(VmPortId input) =>
        producers.TryGetValue(input, out var producer)
            ? producer
            : throw new KeyNotFoundException($"Input port '{input}' is not connected.");

    public IReadOnlyList<VmPortId> GetConsumers(VmPortId output) =>
        consumers.TryGetValue(output, out var values)
            ? values.AsReadOnly()
            : [];

    private VmProgramPort GetPort(VmPortId id, VmPortDirection direction)
    {
        if (!components.TryGetValue(id.Component, out var component))
            throw new KeyNotFoundException($"Unknown VM component '{id.Component}'.");
        var port = component.Ports.SingleOrDefault(port => port.Id == id) ??
            throw new KeyNotFoundException($"Unknown VM port '{id}'.");
        if (port.Direction != direction)
            throw new InvalidDataException(
                $"Port '{id}' is a {port.Direction.ToString().ToLowerInvariant()} port, not a " +
                $"{direction.ToString().ToLowerInvariant()} port.");
        return port;
    }

    private bool CreatesCycle(string source, string target)
    {
        var visited = new HashSet<string>(StringComparer.Ordinal);
        return Visit(target);

        bool Visit(string component)
        {
            if (string.Equals(component, source, StringComparison.Ordinal)) return true;
            if (!visited.Add(component)) return false;
            return consumers
                .Where(pair => string.Equals(pair.Key.Component, component, StringComparison.Ordinal))
                .SelectMany(pair => pair.Value)
                .Any(port => Visit(port.Component));
        }
    }
}

public sealed record TensorView
{
    public TensorView(IEnumerable<int> dimensions, IEnumerable<int>? origin = null)
    {
        Dimensions = Array.AsReadOnly(dimensions?.ToArray() ??
            throw new ArgumentNullException(nameof(dimensions)));
        Origin = Array.AsReadOnly(origin?.ToArray() ?? new int[Dimensions.Count]);
        if (Dimensions.Count == 0 || Dimensions.Any(dimension => dimension <= 0))
            throw new ArgumentException("Tensor view dimensions must be positive.", nameof(dimensions));
        if (Origin.Count != Dimensions.Count || Origin.Any(value => value < 0))
            throw new ArgumentException("Tensor view origin must match the non-negative dimensions.", nameof(origin));
    }

    public IReadOnlyList<int> Dimensions { get; }
    public IReadOnlyList<int> Origin { get; }
}

public readonly record struct VmResourceId(Guid Value)
{
    public static VmResourceId Create() => new(Guid.NewGuid());
    public override string ToString() => Value.ToString("N");
}

public sealed record VmDataReference(
    VmResourceId ResourceId,
    long Version,
    ComponentPortDescriptor Port,
    TensorView View);

public sealed class LogicalVmResourceManager : IAsyncDisposable
{
    private sealed class ResourceState(
        VmDataReference reference,
        StorageDomain sourceDomain,
        IStorageLease source)
    {
        public VmDataReference Reference { get; } = reference;
        public Dictionary<StorageDomain, IStorageLease> Replicas { get; } =
            new() { [sourceDomain] = source };
        public Dictionary<StorageDomain, Task<IStorageLease>> Transfers { get; } = [];
    }

    private readonly object gate = new();
    private readonly IStorageTransferService transfers;
    private readonly Dictionary<VmResourceId, ResourceState> resources = [];
    private readonly CancellationTokenSource lifetime = new();
    private bool disposed;

    public LogicalVmResourceManager(IStorageTransferService transfers) =>
        this.transfers = transfers ?? throw new ArgumentNullException(nameof(transfers));

    public VmDataReference Publish(
        ComponentPortDescriptor outputPort,
        IStorageLease storage,
        TensorView? view = null)
    {
        ArgumentNullException.ThrowIfNull(outputPort);
        ArgumentNullException.ThrowIfNull(storage);
        if (outputPort.Access == GraphResourceAccess.Read)
            throw new ArgumentException("Published ports must be writable outputs.", nameof(outputPort));
        var handle = storage.Handle;
        if (!ComponentPortDescriptor.TensorEquals(outputPort.Tensor, handle.Descriptor))
            throw new InvalidDataException("Published storage does not match the output port.");
        if (outputPort.StorageDomains.Count != 0 &&
            !outputPort.StorageDomains.Contains(handle.Domain))
            throw new InvalidDataException("Published storage belongs to an unsupported domain.");
        var actualView = view ?? new TensorView(outputPort.Tensor.Dimensions);
        ValidateView(actualView, outputPort.Tensor);
        var reference = new VmDataReference(VmResourceId.Create(), 1, outputPort, actualView);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            resources.Add(reference.ResourceId,
                new ResourceState(reference, handle.Domain, storage.Retain()));
        }
        return reference;
    }

    public async ValueTask<IStorageLease> AcquireAsync(
        VmDataReference reference,
        StorageDomain targetDomain,
        ComponentPortDescriptor targetPort,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(reference);
        ArgumentNullException.ThrowIfNull(targetDomain);
        ArgumentNullException.ThrowIfNull(targetPort);
        ComponentPortDescriptor.ValidateConnection(reference.Port, targetPort);
        Task<IStorageLease>? transfer;
        TaskCompletionSource<IStorageLease>? pending = null;
        IStorageLease? source = null;
        ResourceState? transferState = null;
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var state = GetState(reference);
            if (state.Replicas.TryGetValue(targetDomain, out var existing))
                return existing.Retain();
            if (!state.Transfers.TryGetValue(targetDomain, out transfer))
            {
                pending = new(TaskCreationOptions.RunContinuationsAsynchronously);
                transfer = pending.Task;
                state.Transfers.Add(targetDomain, transfer);
                source = state.Replicas.Values.First().Retain();
                transferState = state;
            }
        }
        if (pending is not null)
            _ = TransferAsync(transferState!, source!, targetDomain, targetPort, pending);
        var replica = await transfer!.WaitAsync(cancellationToken).ConfigureAwait(false);
        return replica.Retain();
    }

    public async ValueTask PrepareBindingsAsync(
        VmComponentProgram component,
        VmConnectionRegistry connections,
        IReadOnlyDictionary<VmPortId, VmDataReference> publishedOutputs,
        VmBindings bindings,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(component);
        ArgumentNullException.ThrowIfNull(connections);
        ArgumentNullException.ThrowIfNull(publishedOutputs);
        ArgumentNullException.ThrowIfNull(bindings);
        if (!ReferenceEquals(component.Program, bindings.Program))
            throw new ArgumentException("Bindings belong to a different VM program.", nameof(bindings));
        foreach (var input in component.Inputs)
        {
            var producer = connections.GetProducer(input.Id);
            if (!publishedOutputs.TryGetValue(producer, out var value))
                throw new KeyNotFoundException($"Output port '{producer}' has not published a value.");
            await BindAsync(
                value,
                component.StorageDomain,
                input.Descriptor,
                bindings,
                input.SlotId,
                cancellationToken).ConfigureAwait(false);
        }
    }

    public async ValueTask BindAsync(
        VmDataReference reference,
        StorageDomain targetDomain,
        ComponentPortDescriptor targetPort,
        VmBindings bindings,
        string slotId,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(bindings);
        ArgumentException.ThrowIfNullOrWhiteSpace(slotId);
        await using var lease = await AcquireAsync(
            reference, targetDomain, targetPort, cancellationToken).ConfigureAwait(false);
        if (lease.Handle is not IVmBindableStorageHandle bindable)
            throw new NotSupportedException(
                $"Storage handle '{lease.Handle.GetType().Name}' cannot bind VM slots.");
        bindable.Bind(bindings, slotId);
    }

    public async ValueTask DisposeAsync()
    {
        Task<IStorageLease>[] pending;
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            lifetime.Cancel();
            pending = resources.Values.SelectMany(state => state.Transfers.Values).ToArray();
        }
        try { await Task.WhenAll(pending).ConfigureAwait(false); }
        catch (OperationCanceledException) when (lifetime.IsCancellationRequested) { }
        finally
        {
            IStorageLease[] leases;
            lock (gate)
            {
                leases = resources.Values.SelectMany(state => state.Replicas.Values).ToArray();
                resources.Clear();
            }
            foreach (var lease in leases)
                await lease.DisposeAsync().ConfigureAwait(false);
            lifetime.Dispose();
        }
    }

    private async Task TransferAsync(
        ResourceState state,
        IStorageLease source,
        StorageDomain targetDomain,
        ComponentPortDescriptor targetPort,
        TaskCompletionSource<IStorageLease> completion)
    {
        try
        {
            var replica = await transfers.TransferAsync(
                source, targetDomain, targetPort, lifetime.Token).ConfigureAwait(false);
            lock (gate)
            {
                if (disposed)
                {
                    replica.Dispose();
                    throw new ObjectDisposedException(nameof(LogicalVmResourceManager));
                }
                state.Replicas.Add(targetDomain, replica);
                state.Transfers.Remove(targetDomain);
            }
            completion.TrySetResult(replica);
        }
        catch (Exception error)
        {
            lock (gate)
                state.Transfers.Remove(targetDomain);
            completion.TrySetException(error);
        }
        finally
        {
            source.Dispose();
        }
    }

    private ResourceState GetState(VmDataReference reference)
    {
        if (!resources.TryGetValue(reference.ResourceId, out var state) ||
            state.Reference.Version != reference.Version)
            throw new KeyNotFoundException(
                $"Logical resource '{reference.ResourceId}' version {reference.Version} is not available.");
        return state;
    }

    private static void ValidateView(TensorView view, TensorDescriptor descriptor)
    {
        if (view.Dimensions.Count != descriptor.Dimensions.Count)
            throw new InvalidDataException("Tensor view rank does not match its physical storage.");
        for (var index = 0; index < view.Dimensions.Count; index++)
            if (view.Origin[index] + view.Dimensions[index] > descriptor.Dimensions[index])
                throw new InvalidDataException("Tensor view exceeds its physical storage.");
    }
}
