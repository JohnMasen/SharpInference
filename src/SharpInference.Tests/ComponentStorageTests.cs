using SharpInference.Graphs;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class ComponentStorageTests
{
    private static readonly TensorDescriptor Descriptor =
        new(GraphElementType.Float32, [2], "dense");

    [Fact]
    public async Task SameDomainTransfer_RetainsZeroCopyHandleUntilLastLease()
    {
        var domain = new StorageDomain("cpu", "0", "main");
        var handle = new TestHandle(Guid.NewGuid(), domain, Descriptor, 8);
        var releases = 0;
        var source = StorageLease.Create(handle, _ => Interlocked.Increment(ref releases));
        var service = new HostStagingStorageTransferService([], new TestStagingAllocator([]));
        var port = Port(domain);

        var transferred = await service.TransferAsync(source, domain, port);

        Assert.Same(handle, transferred.Handle);
        await source.DisposeAsync();
        Assert.Equal(0, releases);
        await transferred.DisposeAsync();
        Assert.Equal(1, releases);
    }

    [Fact]
    public async Task CrossDomainTransfer_StagesThroughHostAndWaitsBeforeReleasingBuffer()
    {
        var events = new List<string>();
        var sourceDomain = new StorageDomain("gpu", "0", "device");
        var targetDomain = new StorageDomain("cpu", "0", "main");
        var sourceAdapter = new TestAdapter(sourceDomain, events);
        var targetAdapter = new TestAdapter(targetDomain, events);
        var sourceHandle = sourceAdapter.Add([1, 2, 3, 4, 5, 6, 7, 8], Descriptor);
        await using var source = StorageLease.Create(sourceHandle, _ => { });
        var service = new HostStagingStorageTransferService(
            [sourceAdapter, targetAdapter], new TestStagingAllocator(events));

        await using var transferred = await service.TransferAsync(source, targetDomain, Port(targetDomain));

        Assert.Equal(["download", "upload", "staging.dispose"], events);
        Assert.Equal(sourceAdapter.Read(sourceHandle), targetAdapter.Read(transferred.Handle));
        Assert.Equal(targetDomain, transferred.Handle.Domain);
    }

    [Fact]
    public async Task Transfer_RejectsDescriptorMismatchWithoutStaging()
    {
        var events = new List<string>();
        var domain = new StorageDomain("cpu", "0", "main");
        var handle = new TestHandle(Guid.NewGuid(), domain, Descriptor, 8);
        await using var source = StorageLease.Create(handle, _ => { });
        var service = new HostStagingStorageTransferService([], new TestStagingAllocator(events));
        var mismatched = new ComponentPortDescriptor(
            "embedding", 1, new TensorDescriptor(GraphElementType.Float16, [2]),
            GraphResourceAccess.Read, GraphResourceLifetime.Invocation, [domain]);

        await Assert.ThrowsAsync<InvalidDataException>(
            () => service.TransferAsync(source, domain, mismatched).AsTask());
        Assert.Empty(events);
    }

    private static ComponentPortDescriptor Port(StorageDomain domain) =>
        new("embedding", 1, Descriptor, GraphResourceAccess.Read,
            GraphResourceLifetime.Invocation, [domain]);

    private sealed record TestHandle(
        Guid Id,
        StorageDomain Domain,
        TensorDescriptor Descriptor,
        long ByteLength) : IStorageHandle;

    private sealed class TestStagingAllocator(List<string> events) : IHostStagingAllocator
    {
        public ValueTask<IHostStagingBuffer> RentAsync(int byteLength, CancellationToken cancellationToken) =>
            ValueTask.FromResult<IHostStagingBuffer>(new TestStagingBuffer(byteLength, events));
    }

    private sealed class TestStagingBuffer(int length, List<string> events) : IHostStagingBuffer
    {
        private readonly byte[] bytes = new byte[length];
        public Memory<byte> Memory => bytes;
        public ValueTask DisposeAsync()
        {
            events.Add("staging.dispose");
            return ValueTask.CompletedTask;
        }
    }

    private sealed class TestAdapter(StorageDomain domain, List<string> events) : IStorageDomainAdapter
    {
        private readonly Dictionary<Guid, byte[]> storage = [];
        public StorageDomain Domain => domain;

        public IStorageHandle Add(byte[] data, TensorDescriptor descriptor)
        {
            var handle = new TestHandle(Guid.NewGuid(), domain, descriptor, data.Length);
            storage.Add(handle.Id, data.ToArray());
            return handle;
        }

        public byte[] Read(IStorageHandle handle) => storage[handle.Id];

        public ValueTask DownloadAsync(
            IStorageHandle source,
            Memory<byte> destination,
            CancellationToken cancellationToken)
        {
            events.Add("download");
            storage[source.Id].CopyTo(destination);
            return ValueTask.CompletedTask;
        }

        public ValueTask<IStorageLease> UploadAsync(
            TensorDescriptor descriptor,
            ReadOnlyMemory<byte> source,
            CancellationToken cancellationToken)
        {
            events.Add("upload");
            var handle = Add(source.ToArray(), descriptor);
            return ValueTask.FromResult(StorageLease.Create(handle, released => storage.Remove(released.Id)));
        }
    }
}
