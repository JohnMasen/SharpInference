using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Runtime;
using SharpInference.Vm;

namespace SharpInference.Architectures.Phi4.D3D12;

internal sealed class Phi4D3D12StorageAdapter(
    D3D12VmResourcePool pool,
    StorageDomain domain) : IStorageDomainAdapter
{
    public StorageDomain Domain { get; } =
        domain ?? throw new ArgumentNullException(nameof(domain));

    public ValueTask DownloadAsync(
        IStorageHandle source,
        Memory<byte> destination,
        CancellationToken cancellationToken) =>
        throw new NotSupportedException(
            "Generic D3D12 readback requires an executor that owns the producing program.");

    public ValueTask<IStorageLease> UploadAsync(
        TensorDescriptor descriptor,
        ReadOnlyMemory<byte> source,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        cancellationToken.ThrowIfCancellationRequested();
        var tensor = ToVmTensor(descriptor);
        if ((ulong)source.Length != tensor.ByteLength)
            throw new ArgumentException("Upload data size does not match its tensor descriptor.", nameof(source));
        var slot = new VmSlot(
            $"transfer_{Guid.NewGuid():N}",
            VmSlotScope.Local,
            VmAccess.ReadWrite,
            tensor);
        IVmStorage? storage = null;
        VmResource? resource = null;
        VmResourceLease? lease = null;
        try
        {
            storage = pool.Allocate(slot);
            resource = new VmResource(tensor, slot.Scope, slot.Access, storage);
            storage = null;
            lease = resource.Acquire();
            resource.Dispose();
            resource = null;
            lease.Write(0, source.Span);
            var handle = new UploadedHandle(Domain, descriptor, lease);
            lease = null;
            return ValueTask.FromResult<IStorageLease>(
                StorageLease.Create(handle, static released => ((UploadedHandle)released).Dispose()));
        }
        catch
        {
            lease?.Dispose();
            resource?.Dispose();
            storage?.Dispose();
            throw;
        }
    }

    private static VmTensor ToVmTensor(TensorDescriptor descriptor) =>
        new(
            descriptor.ElementType switch
            {
                GraphElementType.Byte => VmElementType.Byte,
                GraphElementType.Float16 => VmElementType.Float16,
                GraphElementType.Float32 => VmElementType.Float32,
                GraphElementType.Int32 => VmElementType.Int32,
                GraphElementType.UInt32 => VmElementType.UInt32,
                _ => throw new NotSupportedException(
                    $"D3D12 transfer does not support '{descriptor.ElementType}'."),
            },
            descriptor.Dimensions);

    private sealed class UploadedHandle(
        StorageDomain domain,
        TensorDescriptor descriptor,
        VmResourceLease resource) : IVmBindableStorageHandle, IDisposable
    {
        private VmResourceLease? resource = resource;
        public Guid Id { get; } = Guid.NewGuid();
        public StorageDomain Domain { get; } = domain;
        public TensorDescriptor Descriptor { get; } = descriptor;
        public long ByteLength { get; } = checked((long)resource.Tensor.ByteLength);

        public void Bind(VmBindings bindings, string slotId)
        {
            using var retained = Resource.Retain();
            bindings.Bind(slotId, retained);
        }

        private VmResourceLease Resource =>
            resource ?? throw new ObjectDisposedException(nameof(UploadedHandle));

        public void Dispose() => Interlocked.Exchange(ref resource, null)?.Dispose();
    }
}
