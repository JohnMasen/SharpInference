using SharpInference.Graphs;
using SharpInference.Vm;

namespace SharpInference.Runtime;

/// <summary>Identifies a storage namespace by backend, device, and domain name.</summary>
public sealed record StorageDomain
{
    /// <summary>Creates a storage domain with non-empty backend, device, and namespace identifiers.</summary>
    /// <param name="backend">The non-empty backend identifier that owns the storage.</param>
    /// <param name="device">The non-empty identifier of the device within the backend.</param>
    /// <param name="name">The non-empty storage namespace within the device.</param>
    public StorageDomain(string backend, string device, string name)
    {
        Backend = Required(backend, nameof(backend));
        Device = Required(device, nameof(device));
        Name = Required(name, nameof(name));
    }

    /// <summary>Gets the backend identifier that owns the storage.</summary>
    public string Backend { get; }
    /// <summary>Gets the device identifier within the backend.</summary>
    public string Device { get; }
    /// <summary>Gets the storage namespace on the selected device.</summary>
    public string Name { get; }
    /// <summary>Returns the backend, device, and domain name separated by slashes.</summary>
    public override string ToString() => $"{Backend}/{Device}/{Name}";

    private static string Required(string value, string parameter) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException("A storage domain value is required.", parameter)
            : value;
}

/// <summary>Describes a component port&apos;s semantic ABI, tensor shape, access, lifetime, and supported storage domains.</summary>
public sealed class ComponentPortDescriptor
{
    /// <summary>Creates and validates the ABI and storage requirements of a component port.</summary>
    /// <param name="semanticName">The non-empty semantic name used to match compatible ports.</param>
    /// <param name="abiVersion">The positive version of the semantic port ABI.</param>
    /// <param name="tensor">The tensor element type, dimensions, and layout required by the port.</param>
    /// <param name="access">The required read or write access to the resource.</param>
    /// <param name="lifetime">The lifetime required for the port&apos;s resource.</param>
    /// <param name="storageDomains">The supported storage domains, or null to impose no domain restriction.</param>
    public ComponentPortDescriptor(
        string semanticName,
        int abiVersion,
        TensorDescriptor tensor,
        GraphResourceAccess access,
        GraphResourceLifetime lifetime,
        IEnumerable<StorageDomain>? storageDomains = null)
    {
        if (string.IsNullOrWhiteSpace(semanticName))
            throw new ArgumentException("A semantic port name is required.", nameof(semanticName));
        if (abiVersion <= 0) throw new ArgumentOutOfRangeException(nameof(abiVersion));
        if (!Enum.IsDefined(access)) throw new ArgumentOutOfRangeException(nameof(access));
        if (!Enum.IsDefined(lifetime)) throw new ArgumentOutOfRangeException(nameof(lifetime));
        SemanticName = semanticName;
        AbiVersion = abiVersion;
        Tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
        Access = access;
        Lifetime = lifetime;
        StorageDomains = Array.AsReadOnly(storageDomains?.Distinct().ToArray() ?? []);
    }

    /// <summary>Gets the semantic name used to match producer and consumer ports.</summary>
    public string SemanticName { get; }
    /// <summary>Gets the positive version of the port&apos;s semantic ABI.</summary>
    public int AbiVersion { get; }
    /// <summary>Gets the tensor element type, dimensions, and layout required by the port.</summary>
    public TensorDescriptor Tensor { get; }
    /// <summary>Gets whether the port reads, writes, or reads and writes its resource.</summary>
    public GraphResourceAccess Access { get; }
    /// <summary>Gets the lifetime required for the port&apos;s resource.</summary>
    public GraphResourceLifetime Lifetime { get; }
    /// <summary>Gets the supported storage domains; an empty list imposes no domain restriction.</summary>
    public IReadOnlyList<StorageDomain> StorageDomains { get; }

    /// <summary>Checks that a writable producer and readable consumer have matching semantic names, ABI versions, and tensor descriptors.</summary>
    /// <param name="producer">The writable port publishing the tensor.</param>
    /// <param name="consumer">The readable port consuming the producer&apos;s tensor.</param>
    public static void ValidateConnection(ComponentPortDescriptor producer, ComponentPortDescriptor consumer)
    {
        ArgumentNullException.ThrowIfNull(producer);
        ArgumentNullException.ThrowIfNull(consumer);
        if (producer.Access == GraphResourceAccess.Read ||
            consumer.Access == GraphResourceAccess.Write)
            throw new InvalidDataException("A component connection requires a writable producer and readable consumer.");
        if (!string.Equals(producer.SemanticName, consumer.SemanticName, StringComparison.Ordinal) ||
            producer.AbiVersion != consumer.AbiVersion ||
            !TensorEquals(producer.Tensor, consumer.Tensor))
        {
            throw new InvalidDataException(
                $"Component ports '{producer.SemanticName}@{producer.AbiVersion}' and " +
                $"'{consumer.SemanticName}@{consumer.AbiVersion}' are ABI-incompatible.");
        }
    }

    /// <summary>Compares tensor element types, layouts, and ordered dimensions for ABI compatibility.</summary>
    /// <param name="left">The first tensor descriptor to compare.</param>
    /// <param name="right">The second tensor descriptor to compare.</param>
    public static bool TensorEquals(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType &&
        string.Equals(left.Layout, right.Layout, StringComparison.Ordinal) &&
        left.Dimensions.SequenceEqual(right.Dimensions);
}

/// <summary>Describes storage identity, ownership domain, tensor shape, and byte length without controlling its lifetime.</summary>
public interface IStorageHandle
{
    /// <summary>Gets the unique identity of the backing storage.</summary>
    Guid Id { get; }
    /// <summary>Gets the backend and device domain that owns the storage.</summary>
    StorageDomain Domain { get; }
    /// <summary>Gets the tensor descriptor represented by the storage.</summary>
    TensorDescriptor Descriptor { get; }
    /// <summary>Gets the number of bytes in the backing storage.</summary>
    long ByteLength { get; }
}

/// <summary>Exposes a storage handle that can be bound directly to a VM program slot.</summary>
public interface IVmBindableStorageHandle : IStorageHandle
{
    /// <summary>Binds this storage to the specified slot in a VM binding set.</summary>
    /// <param name="bindings">The VM binding set receiving the storage handle.</param>
    /// <param name="slotId">The identifier of the VM slot to bind.</param>
    void Bind(VmBindings bindings, string slotId);
}

/// <summary>Controls a reference to storage that remains alive until every retained lease is released.</summary>
public interface IStorageLease : IDisposable, IAsyncDisposable
{
    /// <summary>Gets the backing storage handle while this lease remains valid.</summary>
    IStorageHandle Handle { get; }
    /// <summary>Creates an independently disposable lease that retains the same backing storage.</summary>
    IStorageLease Retain();
}

/// <summary>Implements thread-safe reference-counted storage ownership with a final-release callback.</summary>
public sealed class StorageLease : IStorageLease
{
    private sealed class SharedState(IStorageHandle handle, Action<IStorageHandle> release)
    {
        public readonly IStorageHandle Handle = handle;
        public readonly Action<IStorageHandle> Release = release;
        public int References = 1;
    }

    private readonly SharedState state;
    private int disposed;

    private StorageLease(SharedState state)
    {
        this.state = state;
    }

    /// <inheritdoc/>
    public IStorageHandle Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref disposed) != 0, this);
            return state.Handle;
        }
    }

    /// <summary>Creates the initial lease for a storage handle and its final-release callback.</summary>
    /// <param name="handle">The backing storage whose lifetime is controlled by the lease.</param>
    /// <param name="release">The callback invoked once when the final retained lease is disposed.</param>
    public static IStorageLease Create(IStorageHandle handle, Action<IStorageHandle> release)
    {
        ArgumentNullException.ThrowIfNull(handle);
        ArgumentNullException.ThrowIfNull(release);
        return new StorageLease(new SharedState(handle, release));
    }

    /// <inheritdoc/>
    public IStorageLease Retain()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref disposed) != 0, this);
        while (true)
        {
            var current = Volatile.Read(ref state.References);
            if (current == 0) throw new ObjectDisposedException(nameof(StorageLease));
            if (current == int.MaxValue)
                throw new InvalidOperationException("The storage lease reference count is exhausted.");
            if (Interlocked.CompareExchange(ref state.References, current + 1, current) == current)
                return new StorageLease(state);
        }
    }

    /// <summary>Releases this lease and invokes the release callback when the final reference is gone.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0) return;
        if (Interlocked.Decrement(ref state.References) == 0)
            state.Release(state.Handle);
    }

    /// <summary>Releases this lease synchronously and returns a completed asynchronous operation.</summary>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }
}

/// <summary>Provides temporary host memory for transfers between storage domains.</summary>
public interface IHostStagingBuffer : IAsyncDisposable
{
    /// <summary>Gets the host memory available for staging until the buffer is disposed.</summary>
    Memory<byte> Memory { get; }
}

/// <summary>Allocates temporary host buffers for cross-domain storage transfers.</summary>
public interface IHostStagingAllocator
{
    /// <summary>Obtains a disposable host staging buffer of the requested byte length.</summary>
    /// <param name="byteLength">The positive number of bytes required for staging.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    ValueTask<IHostStagingBuffer> RentAsync(int byteLength, CancellationToken cancellationToken);
}

/// <summary>Allocates host staging buffers backed by managed byte arrays.</summary>
public sealed class ArrayHostStagingAllocator : IHostStagingAllocator
{
    private sealed class Buffer(int byteLength) : IHostStagingBuffer
    {
        private byte[]? bytes = new byte[byteLength];
        public Memory<byte> Memory =>
            bytes ?? throw new ObjectDisposedException(nameof(Buffer));
        public ValueTask DisposeAsync()
        {
            bytes = null;
            return ValueTask.CompletedTask;
        }
    }

    /// <inheritdoc/>
    public ValueTask<IHostStagingBuffer> RentAsync(
        int byteLength,
        CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult<IHostStagingBuffer>(new Buffer(byteLength));
    }
}

/// <summary>Transfers tensor bytes between one storage domain and host memory.</summary>
public interface IStorageDomainAdapter
{
    /// <summary>Gets the storage domain served by this adapter.</summary>
    StorageDomain Domain { get; }
    /// <summary>Copies all bytes from a storage handle in this domain into host memory.</summary>
    /// <param name="source">The storage handle owned by this adapter&apos;s domain.</param>
    /// <param name="destination">The host memory receiving all bytes from the source storage.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    ValueTask DownloadAsync(
        IStorageHandle source,
        Memory<byte> destination,
        CancellationToken cancellationToken);
    /// <summary>Creates a storage lease in this domain from host bytes and a tensor descriptor.</summary>
    /// <param name="descriptor">The tensor descriptor for the transferred bytes.</param>
    /// <param name="source">The host tensor bytes to copy into the target domain.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    ValueTask<IStorageLease> UploadAsync(
        TensorDescriptor descriptor,
        ReadOnlyMemory<byte> source,
        CancellationToken cancellationToken);
}

/// <summary>Stores tensors in managed byte arrays and transfers their contents to and from host memory.</summary>
public sealed class HostMemoryStorageAdapter : IStorageDomainAdapter
{
    private sealed record HostMemoryHandle(
        Guid Id,
        StorageDomain Domain,
        TensorDescriptor Descriptor,
        byte[] Bytes) : IStorageHandle
    {
        public long ByteLength => Bytes.LongLength;
    }

    /// <summary>Creates a managed-memory storage adapter for the specified domain.</summary>
    /// <param name="domain">The storage domain served by the adapter.</param>
    public HostMemoryStorageAdapter(StorageDomain domain) =>
        Domain = domain ?? throw new ArgumentNullException(nameof(domain));

    /// <inheritdoc/>
    public StorageDomain Domain { get; }

    /// <summary>Copies source bytes into a new managed-memory storage lease with the specified tensor descriptor.</summary>
    /// <param name="descriptor">The tensor descriptor for the transferred bytes.</param>
    /// <param name="source">The tensor bytes copied into new managed storage.</param>
    public IStorageLease CreateLease(TensorDescriptor descriptor, ReadOnlySpan<byte> source)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        var handle = new HostMemoryHandle(Guid.NewGuid(), Domain, descriptor, source.ToArray());
        return StorageLease.Create(handle, static _ => { });
    }

    /// <inheritdoc/>
    public ValueTask DownloadAsync(
        IStorageHandle source,
        Memory<byte> destination,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (source is not HostMemoryHandle handle || handle.Domain != Domain)
            throw new ArgumentException("The source is not owned by this host storage adapter.", nameof(source));
        if (destination.Length != handle.Bytes.Length)
            throw new ArgumentException("The destination size does not match the host storage.", nameof(destination));
        handle.Bytes.CopyTo(destination);
        return ValueTask.CompletedTask;
    }

    /// <inheritdoc/>
    public ValueTask<IStorageLease> UploadAsync(
        TensorDescriptor descriptor,
        ReadOnlyMemory<byte> source,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(CreateLease(descriptor, source.Span));
    }
}

/// <summary>Moves leased tensor storage to a compatible component port in another storage domain.</summary>
public interface IStorageTransferService
{
    /// <summary>Retains storage already in the target domain or transfers it to satisfy the target port.</summary>
    /// <param name="source">The leased tensor storage to retain or transfer.</param>
    /// <param name="targetDomain">The storage domain in which a compatible replica is required.</param>
    /// <param name="targetPort">The readable consumer port whose ABI and domain requirements must be satisfied.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    ValueTask<IStorageLease> TransferAsync(
        IStorageLease source,
        StorageDomain targetDomain,
        ComponentPortDescriptor targetPort,
        CancellationToken cancellationToken = default);
}

/// <summary>Transfers tensor storage between registered domains through a disposable host staging buffer.</summary>
public sealed class HostStagingStorageTransferService : IStorageTransferService
{
    private readonly IReadOnlyDictionary<StorageDomain, IStorageDomainAdapter> adapters;
    private readonly IHostStagingAllocator staging;

    /// <summary>Registers storage-domain adapters and the allocator used to stage their transfers.</summary>
    /// <param name="adapters">The storage adapters indexed by their distinct domains.</param>
    /// <param name="staging">The allocator providing temporary host memory for cross-domain transfers.</param>
    public HostStagingStorageTransferService(
        IEnumerable<IStorageDomainAdapter> adapters,
        IHostStagingAllocator staging)
    {
        ArgumentNullException.ThrowIfNull(adapters);
        this.adapters = adapters.ToDictionary(adapter => adapter.Domain);
        this.staging = staging ?? throw new ArgumentNullException(nameof(staging));
    }

    /// <inheritdoc/>
    public async ValueTask<IStorageLease> TransferAsync(
        IStorageLease source,
        StorageDomain targetDomain,
        ComponentPortDescriptor targetPort,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(targetDomain);
        ArgumentNullException.ThrowIfNull(targetPort);
        if (targetPort.Access == GraphResourceAccess.Write)
            throw new InvalidDataException("A transfer target port must be readable.");
        var handle = source.Handle;
        if (!ComponentPortDescriptor.TensorEquals(handle.Descriptor, targetPort.Tensor))
            throw new InvalidDataException("The source storage descriptor does not match the target port.");
        if (targetPort.StorageDomains.Count != 0 && !targetPort.StorageDomains.Contains(targetDomain))
            throw new InvalidDataException($"Target port '{targetPort.SemanticName}' does not support '{targetDomain}'.");
        if (handle.Domain == targetDomain)
            return source.Retain();
        if (!adapters.TryGetValue(handle.Domain, out var sourceAdapter))
            throw new NotSupportedException($"No storage adapter is registered for '{handle.Domain}'.");
        if (!adapters.TryGetValue(targetDomain, out var targetAdapter))
            throw new NotSupportedException($"No storage adapter is registered for '{targetDomain}'.");
        var byteLength = GetByteLength(targetPort.Tensor);
        if (handle.ByteLength != byteLength)
            throw new InvalidDataException("The source storage size does not match its tensor descriptor.");

        var buffer = await staging.RentAsync(byteLength, cancellationToken).ConfigureAwait(false);
        IStorageLease? result = null;
        Exception? transferError = null;
        try
        {
            await sourceAdapter.DownloadAsync(handle, buffer.Memory[..byteLength], cancellationToken)
                .ConfigureAwait(false);
            result = await targetAdapter.UploadAsync(
                targetPort.Tensor, buffer.Memory[..byteLength], cancellationToken).ConfigureAwait(false);
            var uploaded = result?.Handle ??
                throw new InvalidOperationException("The target adapter returned no storage lease.");
            if (uploaded.Domain != targetDomain ||
                uploaded.ByteLength != byteLength ||
                !ComponentPortDescriptor.TensorEquals(uploaded.Descriptor, targetPort.Tensor))
                throw new InvalidDataException("The uploaded storage does not match the target domain and descriptor.");
        }
        catch (Exception error)
        {
            transferError = error;
        }

        Exception? cleanupError = null;
        try { await buffer.DisposeAsync().ConfigureAwait(false); }
        catch (Exception error) { cleanupError = error; }

        if (transferError is not null || cleanupError is not null)
        {
            if (result is not null)
            {
                try { await result.DisposeAsync().ConfigureAwait(false); }
                catch (Exception error)
                {
                    cleanupError = cleanupError is null
                        ? error
                        : new AggregateException(cleanupError, error);
                }
            }
            if (transferError is not null && cleanupError is not null)
                throw new AggregateException("Storage transfer and cleanup failed.", transferError, cleanupError);
            if (transferError is not null)
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(transferError).Throw();
            throw new AggregateException("Storage transfer cleanup failed.", cleanupError!);
        }

        return result ?? throw new InvalidOperationException("The target adapter returned no storage lease.");
    }

    private static int GetByteLength(TensorDescriptor descriptor)
    {
        var elementSize = descriptor.ElementType switch
        {
            GraphElementType.Byte => 1,
            GraphElementType.Float16 => 2,
            GraphElementType.Int32 or GraphElementType.UInt32 or GraphElementType.Float32 => 4,
            _ => throw new NotSupportedException($"Element type '{descriptor.ElementType}' has no storage size."),
        };
        return descriptor.Dimensions.Aggregate(elementSize,
            static (length, dimension) => checked(length * dimension));
    }
}
