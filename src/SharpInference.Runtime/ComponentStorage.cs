using SharpInference.Graphs;

namespace SharpInference.Runtime;

public sealed record StorageDomain
{
    public StorageDomain(string backend, string device, string name)
    {
        Backend = Required(backend, nameof(backend));
        Device = Required(device, nameof(device));
        Name = Required(name, nameof(name));
    }

    public string Backend { get; }
    public string Device { get; }
    public string Name { get; }
    public override string ToString() => $"{Backend}/{Device}/{Name}";

    private static string Required(string value, string parameter) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException("A storage domain value is required.", parameter)
            : value;
}

public sealed class ComponentPortDescriptor
{
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

    public string SemanticName { get; }
    public int AbiVersion { get; }
    public TensorDescriptor Tensor { get; }
    public GraphResourceAccess Access { get; }
    public GraphResourceLifetime Lifetime { get; }
    public IReadOnlyList<StorageDomain> StorageDomains { get; }

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

    public static bool TensorEquals(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType &&
        string.Equals(left.Layout, right.Layout, StringComparison.Ordinal) &&
        left.Dimensions.SequenceEqual(right.Dimensions);
}

public interface IStorageHandle
{
    Guid Id { get; }
    StorageDomain Domain { get; }
    TensorDescriptor Descriptor { get; }
    long ByteLength { get; }
}

public interface IStorageLease : IDisposable, IAsyncDisposable
{
    IStorageHandle Handle { get; }
    IStorageLease Retain();
}

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

    public IStorageHandle Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref disposed) != 0, this);
            return state.Handle;
        }
    }

    public static IStorageLease Create(IStorageHandle handle, Action<IStorageHandle> release)
    {
        ArgumentNullException.ThrowIfNull(handle);
        ArgumentNullException.ThrowIfNull(release);
        return new StorageLease(new SharedState(handle, release));
    }

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

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0) return;
        if (Interlocked.Decrement(ref state.References) == 0)
            state.Release(state.Handle);
    }

    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }
}

public interface IHostStagingBuffer : IAsyncDisposable
{
    Memory<byte> Memory { get; }
}

public interface IHostStagingAllocator
{
    ValueTask<IHostStagingBuffer> RentAsync(int byteLength, CancellationToken cancellationToken);
}

public interface IStorageDomainAdapter
{
    StorageDomain Domain { get; }
    ValueTask DownloadAsync(
        IStorageHandle source,
        Memory<byte> destination,
        CancellationToken cancellationToken);
    ValueTask<IStorageLease> UploadAsync(
        TensorDescriptor descriptor,
        ReadOnlyMemory<byte> source,
        CancellationToken cancellationToken);
}

public interface IStorageTransferService
{
    ValueTask<IStorageLease> TransferAsync(
        IStorageLease source,
        StorageDomain targetDomain,
        ComponentPortDescriptor targetPort,
        CancellationToken cancellationToken = default);
}

public sealed class HostStagingStorageTransferService : IStorageTransferService
{
    private readonly IReadOnlyDictionary<StorageDomain, IStorageDomainAdapter> adapters;
    private readonly IHostStagingAllocator staging;

    public HostStagingStorageTransferService(
        IEnumerable<IStorageDomainAdapter> adapters,
        IHostStagingAllocator staging)
    {
        ArgumentNullException.ThrowIfNull(adapters);
        this.adapters = adapters.ToDictionary(adapter => adapter.Domain);
        this.staging = staging ?? throw new ArgumentNullException(nameof(staging));
    }

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
