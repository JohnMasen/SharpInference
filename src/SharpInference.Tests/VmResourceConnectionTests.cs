using SharpInference.Graphs;
using SharpInference.Runtime;
using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class VmResourceConnectionTests
{
    private static readonly TensorDescriptor Descriptor =
        new(GraphElementType.Float32, [2], "dense");

    [Fact]
    public void Connections_AllowMulticastButRejectInvalidSourcesAndCycles()
    {
        var domain = new StorageDomain("cpu", "0", "main");
        var first = Component("first", domain, input: true, output: true);
        var second = Component("second", domain, input: true, output: true);
        var third = Component("third", domain, input: true, output: true);
        var registry = new VmConnectionRegistry([first, second, third]);

        registry.Connect(Id("first", "output"), Id("second", "input"));
        registry.Connect(Id("first", "output"), Id("third", "input"));

        Assert.Equal(
            [Id("second", "input"), Id("third", "input")],
            registry.GetConsumers(Id("first", "output")));
        Assert.Throws<InvalidOperationException>(() =>
            registry.Connect(Id("second", "output"), Id("third", "input")));
        Assert.Throws<InvalidDataException>(() =>
            registry.Connect(Id("second", "input"), Id("first", "input")));

        var cycle = new VmConnectionRegistry([first, second, third]);
        cycle.Connect(Id("first", "output"), Id("second", "input"));
        cycle.Connect(Id("second", "output"), Id("third", "input"));
        Assert.Throws<InvalidDataException>(() =>
            cycle.Connect(Id("third", "output"), Id("first", "input")));
    }

    [Fact]
    public async Task PrepareBindings_MulticastsOneLocalResourceToReadOnlyConsumers()
    {
        var domain = new StorageDomain("cpu", "0", "main");
        var producer = Component("producer", domain, input: false, output: true);
        var first = Component("first", domain, input: true, output: false);
        var second = Component("second", domain, input: true, output: false);
        var registry = new VmConnectionRegistry([producer, first, second]);
        registry.Connect(Id("producer", "output"), Id("first", "input"));
        registry.Connect(Id("producer", "output"), Id("second", "input"));
        var source = CreateStorage(domain, [1, 2, 3, 4, 5, 6, 7, 8]);
        await using var manager = new LogicalVmResourceManager(
            new HostStagingStorageTransferService([], new ThrowingStagingAllocator()));
        var value = manager.Publish(OutputPort(domain), source);
        var published = new Dictionary<VmPortId, VmDataReference>
        {
            [Id("producer", "output")] = value,
        };
        using var firstBindings = new VmBindings(first.Program);
        using var secondBindings = new VmBindings(second.Program);

        await manager.PrepareBindingsAsync(first, registry, published, firstBindings);
        await manager.PrepareBindingsAsync(second, registry, published, secondBindings);
        using var firstExecution = firstBindings.BeginExecution();
        using var secondExecution = secondBindings.BeginExecution();

        source.Dispose();
    }

    [Fact]
    public async Task Acquire_CachesOneCrossDomainReplicaForConcurrentConsumers()
    {
        var sourceDomain = new StorageDomain("gpu", "0", "pool");
        var targetDomain = new StorageDomain("cpu", "0", "main");
        var sourceAdapter = new TestAdapter(sourceDomain);
        var targetAdapter = new TestAdapter(targetDomain);
        await using var source = sourceAdapter.Add([1, 2, 3, 4, 5, 6, 7, 8]);
        await using var manager = new LogicalVmResourceManager(
            new HostStagingStorageTransferService(
                [sourceAdapter, targetAdapter],
                new TestStagingAllocator()));
        var value = manager.Publish(OutputPort(sourceDomain), source);
        var target = InputPort(targetDomain);

        var acquisitions = Enumerable.Range(0, 8)
            .Select(_ => manager.AcquireAsync(value, targetDomain, target).AsTask())
            .ToArray();
        var leases = await Task.WhenAll(acquisitions);
        try
        {
            Assert.Equal(1, sourceAdapter.Downloads);
            Assert.Equal(1, targetAdapter.Uploads);
            Assert.All(leases, lease => Assert.Equal(targetDomain, lease.Handle.Domain));
            Assert.All(leases, lease =>
                Assert.Equal(targetAdapter.Read(lease.Handle), sourceAdapter.Read(source.Handle)));
        }
        finally
        {
            foreach (var lease in leases)
                await lease.DisposeAsync();
        }
    }

    private static VmComponentProgram Component(
        string name,
        StorageDomain domain,
        bool input,
        bool output)
    {
        var slots = new List<VmSlot>();
        var parameters = new List<VmParameter>();
        var ports = new List<VmProgramPort>();
        if (input)
        {
            var tensor = new VmTensor(VmElementType.Float32, [2]);
            slots.Add(new("input", VmSlotScope.Local, VmAccess.ReadOnly, tensor));
            parameters.Add(new("input", VmAccess.ReadOnly, tensor));
            ports.Add(new(Id(name, "input"), "input", VmPortDirection.Input, InputPort(domain)));
        }
        if (output)
        {
            var tensor = new VmTensor(VmElementType.Float32, [2]);
            slots.Add(new("output", VmSlotScope.Local, VmAccess.ReadWrite, tensor));
            parameters.Add(new("output", VmAccess.ReadWrite, tensor));
            ports.Add(new(Id(name, "output"), "output", VmPortDirection.Output, OutputPort(domain)));
        }
        var definition = new VmDefinition(
            "run", VmDefinitionKind.Orchestration, parameters, []);
        var program = new VmProgram(
            name,
            "test",
            VmTarget.Cpu,
            slots,
            [definition],
            [new VmEntry("run", "run", parameters.Select(parameter =>
                new VmArgument(parameter.Name, parameter.Name)))],
            new VmState("none", 1, []));
        return new(name, program, domain, ports);
    }

    private static VmPortId Id(string component, string port) => new(component, port);

    private static ComponentPortDescriptor InputPort(StorageDomain domain) =>
        new("embedding", 1, Descriptor, GraphResourceAccess.Read,
            GraphResourceLifetime.Invocation, [domain]);

    private static ComponentPortDescriptor OutputPort(StorageDomain domain) =>
        new("embedding", 1, Descriptor, GraphResourceAccess.Write,
            GraphResourceLifetime.Invocation, [domain]);

    private static IStorageLease CreateStorage(StorageDomain domain, byte[] bytes)
    {
        var storage = new VmMemoryStorage(bytes.Length);
        storage.Write(0, bytes);
        var resource = new VmResource(
            new VmTensor(VmElementType.Float32, [2]),
            VmSlotScope.Local,
            VmAccess.ReadWrite,
            storage);
        var lease = resource.Acquire();
        resource.Dispose();
        return StorageLease.Create(
            new TestHandle(domain, Descriptor, lease),
            handle => ((TestHandle)handle).Dispose());
    }

    private sealed class TestHandle(
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

        public VmResourceLease Resource =>
            resource ?? throw new ObjectDisposedException(nameof(TestHandle));

        public void Dispose() => Interlocked.Exchange(ref resource, null)?.Dispose();
    }

    private sealed class ThrowingStagingAllocator : IHostStagingAllocator
    {
        public ValueTask<IHostStagingBuffer> RentAsync(
            int byteLength,
            CancellationToken cancellationToken) =>
            throw new InvalidOperationException("Same-domain multicast must not stage data.");
    }

    private sealed class TestStagingAllocator : IHostStagingAllocator
    {
        public ValueTask<IHostStagingBuffer> RentAsync(
            int byteLength,
            CancellationToken cancellationToken) =>
            ValueTask.FromResult<IHostStagingBuffer>(new TestStagingBuffer(byteLength));
    }

    private sealed class TestStagingBuffer(int byteLength) : IHostStagingBuffer
    {
        private readonly byte[] buffer = new byte[byteLength];
        public Memory<byte> Memory => buffer;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class TestAdapter(StorageDomain domain) : IStorageDomainAdapter
    {
        private readonly Dictionary<Guid, byte[]> storage = [];
        public StorageDomain Domain { get; } = domain;
        public int Downloads { get; private set; }
        public int Uploads { get; private set; }

        public IStorageLease Add(byte[] bytes)
        {
            var lease = CreateStorage(Domain, bytes);
            storage.Add(lease.Handle.Id, bytes.ToArray());
            return StorageLease.Create(
                lease.Handle,
                handle =>
                {
                    storage.Remove(handle.Id);
                    lease.Dispose();
                });
        }

        public byte[] Read(IStorageHandle handle) => storage[handle.Id];

        public ValueTask DownloadAsync(
            IStorageHandle source,
            Memory<byte> destination,
            CancellationToken cancellationToken)
        {
            Downloads++;
            storage[source.Id].CopyTo(destination);
            return ValueTask.CompletedTask;
        }

        public ValueTask<IStorageLease> UploadAsync(
            TensorDescriptor descriptor,
            ReadOnlyMemory<byte> source,
            CancellationToken cancellationToken)
        {
            Uploads++;
            return ValueTask.FromResult(Add(source.ToArray()));
        }
    }
}
