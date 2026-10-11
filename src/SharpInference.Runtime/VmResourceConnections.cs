using SharpInference.Graphs;
using SharpInference.Vm;

namespace SharpInference.Runtime;

/// <summary>Distinguishes component input ports from output ports.</summary>
public enum VmPortDirection
{
    /// <summary>Identifies a read-only port supplied by another component.</summary>
    Input,
    /// <summary>Identifies a writable port that publishes a component&apos;s result.</summary>
    Output,
}

/// <summary>Identifies a named port within a VM component.</summary>
public readonly record struct VmPortId
{
    /// <summary>Creates a port identifier with non-empty component and port names.</summary>
    /// <param name="component">The non-empty name of the owning VM component.</param>
    /// <param name="port">The non-empty port name within its component.</param>
    public VmPortId(string component, string port)
    {
        Component = Required(component, nameof(component));
        Port = Required(port, nameof(port));
    }

    /// <summary>Gets the name of the owning VM component.</summary>
    public string Component { get; }
    /// <summary>Gets the port name within the component.</summary>
    public string Port { get; }
    /// <summary>Returns the component and port names separated by a period.</summary>
    public override string ToString() => $"{Component}.{Port}";

    private static string Required(string value, string parameter) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException("A port identifier value is required.", parameter)
            : value;
}

/// <summary>Maps a component port&apos;s ABI and direction to a VM program slot.</summary>
public sealed record VmProgramPort
{
    /// <summary>Creates and validates a VM slot mapping with direction-compatible resource access.</summary>
    /// <param name="id">The component-qualified identifier of the port.</param>
    /// <param name="slotId">The identifier of the VM slot to bind.</param>
    /// <param name="direction">Whether the port consumes input or publishes output.</param>
    /// <param name="descriptor">The semantic ABI and tensor access requirements of the port.</param>
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

    /// <summary>Gets the component-qualified port identifier.</summary>
    public VmPortId Id { get; }
    /// <summary>Gets the VM program slot bound to this port.</summary>
    public string SlotId { get; }
    /// <summary>Gets whether this port consumes input or publishes output.</summary>
    public VmPortDirection Direction { get; }
    /// <summary>Gets the semantic ABI and tensor requirements of this port.</summary>
    public ComponentPortDescriptor Descriptor { get; }
}

/// <summary>Groups a VM program with its storage domain and validated component port mappings.</summary>
public sealed class VmComponentProgram
{
    /// <summary>Creates a component and validates that its unique ports match existing program slots and tensor shapes.</summary>
    /// <param name="name">The unique component name used to qualify all of its ports.</param>
    /// <param name="program">The VM program executed by the component.</param>
    /// <param name="storageDomain">The backend and device storage domain used by the component.</param>
    /// <param name="ports">The unique component port mappings to validate against program slots.</param>
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

    /// <summary>Gets the component name used by its port identifiers.</summary>
    public string Name { get; }
    /// <summary>Gets the VM program executed by this component.</summary>
    public VmProgram Program { get; }
    /// <summary>Gets the storage domain used to bind the component&apos;s inputs.</summary>
    public StorageDomain StorageDomain { get; }
    /// <summary>Gets all validated component ports.</summary>
    public IReadOnlyList<VmProgramPort> Ports { get; }
    /// <summary>Gets the component&apos;s read-only input ports.</summary>
    public IReadOnlyList<VmProgramPort> Inputs =>
        Ports.Where(port => port.Direction == VmPortDirection.Input).ToArray();
    /// <summary>Gets the component&apos;s writable output ports.</summary>
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

/// <summary>Registers ABI-compatible, acyclic producer-to-consumer connections between VM components.</summary>
public sealed class VmConnectionRegistry
{
    private readonly IReadOnlyDictionary<string, VmComponentProgram> components;
    private readonly Dictionary<VmPortId, VmPortId> producers = [];
    private readonly Dictionary<VmPortId, List<VmPortId>> consumers = [];

    /// <summary>Registers the named VM components available for port connections.</summary>
    /// <param name="components">The named VM components available for connections.</param>
    public VmConnectionRegistry(IEnumerable<VmComponentProgram> components)
    {
        ArgumentNullException.ThrowIfNull(components);
        this.components = components.ToDictionary(component => component.Name, StringComparer.Ordinal);
    }

    /// <summary>Connects an output to an unconnected input after checking ABI compatibility and component cycles.</summary>
    /// <param name="source">The component output port publishing the resource.</param>
    /// <param name="target">The unconnected component input port consuming the output.</param>
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

    /// <summary>Returns the output port connected to the specified input, or throws if it is unconnected.</summary>
    /// <param name="input">The connected component input port whose producer is requested.</param>
    public VmPortId GetProducer(VmPortId input) =>
        producers.TryGetValue(input, out var producer)
            ? producer
            : throw new KeyNotFoundException($"Input port '{input}' is not connected.");

    /// <summary>Returns the input ports connected to the specified output.</summary>
    /// <param name="output">The output port whose connected consumer ports are requested.</param>
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

/// <summary>Describes a positive tensor extent and non-negative origin in each dimension.</summary>
public sealed record TensorView
{
    /// <summary>Creates a tensor view with positive dimensions and a matching origin, defaulting to zero.</summary>
    /// <param name="dimensions">The positive extent of the tensor view in each dimension.</param>
    /// <param name="origin">The optional non-negative origin coordinates; omitted coordinates default to zero.</param>
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

    /// <summary>Gets the ordered extents of the tensor view.</summary>
    public IReadOnlyList<int> Dimensions { get; }
    /// <summary>Gets the non-negative starting coordinate of the view along each dimension.</summary>
    public IReadOnlyList<int> Origin { get; }
}

/// <summary>Identifies a logical VM resource independently of its physical storage replicas.</summary>
/// <param name="Value">The GUID identifying the logical resource.</param>
public readonly record struct VmResourceId(Guid Value)
{
    /// <summary>Creates a new logical resource identifier using a random GUID.</summary>
    public static VmResourceId Create() => new(Guid.NewGuid());
    /// <summary>Returns the resource GUID as hexadecimal digits without separators.</summary>
    public override string ToString() => Value.ToString("N");
}

/// <summary>References a versioned logical tensor resource with its output-port ABI and tensor view.</summary>
/// <param name="ResourceId">The identity of the logical resource.</param>
/// <param name="Version">The version of the logical resource referenced by this value.</param>
/// <param name="Port">The output-port ABI describing the logical resource.</param>
/// <param name="View">The full-tensor view represented by this reference.</param>
public sealed record VmDataReference(
    VmResourceId ResourceId,
    long Version,
    ComponentPortDescriptor Port,
    TensorView View);

/// <summary>Owns logical tensor resources, coalesces cross-domain transfers, and binds compatible replicas to VM inputs.</summary>
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

    /// <summary>Creates a logical resource manager using the supplied storage transfer service.</summary>
    /// <param name="transfers">The service used to transfer tensor storage between domains.</param>
    public LogicalVmResourceManager(IStorageTransferService transfers) =>
        this.transfers = transfers ?? throw new ArgumentNullException(nameof(transfers));

    /// <summary>Validates and retains output storage, then returns a versioned logical reference to its full tensor view.</summary>
    /// <param name="outputPort">The writable port whose ABI describes the published tensor.</param>
    /// <param name="storage">The storage lease to retain for the published output.</param>
    /// <param name="view">The optional full-tensor view; omitted dimensions are taken from the output port.</param>
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

    /// <summary>Acquires a retained replica in the target domain, sharing any in-progress transfer for the resource.</summary>
    /// <param name="reference">The versioned logical resource reference to acquire and bind.</param>
    /// <param name="targetDomain">The storage domain in which a compatible replica is required.</param>
    /// <param name="targetPort">The readable consumer port whose ABI and domain requirements must be satisfied.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
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

    /// <summary>Resolves connected producers and binds their published resources to every component input slot.</summary>
    /// <param name="component">The component whose connected input resources are being prepared.</param>
    /// <param name="connections">The registry mapping component inputs to producer outputs.</param>
    /// <param name="publishedOutputs">Producer output ports mapped to their published logical resources.</param>
    /// <param name="bindings">The VM binding set receiving the storage handle.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
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

    /// <summary>Acquires compatible target-domain storage and binds its handle to a VM slot.</summary>
    /// <param name="reference">The versioned logical resource reference to acquire and bind.</param>
    /// <param name="targetDomain">The storage domain in which a compatible replica is required.</param>
    /// <param name="targetPort">The readable consumer port whose ABI and domain requirements must be satisfied.</param>
    /// <param name="bindings">The VM binding set receiving the storage handle.</param>
    /// <param name="slotId">The identifier of the VM slot to bind.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
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

    /// <summary>Cancels outstanding transfers, waits for them to finish, and releases all retained storage replicas.</summary>
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
