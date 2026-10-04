using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class VmResourceTests
{
    private static readonly VmTensor Bytes = new(VmElementType.Byte, [4]);

    [Fact]
    public void SharedStateSurvivesVmAndOwnerDisposal()
    {
        var storage = new CountingStorage(4);
        using var owner = new VmResource(Bytes, VmSlotScope.Session, VmAccess.ReadWrite, storage);
        using var handle = owner.Acquire();
        using var first = new VmBindings(Program());
        using var second = new VmBindings(Program());
        first.Bind("state", handle);
        second.Bind("state", handle);
        owner.Dispose();
        handle.Dispose();
        using (var execution = first.BeginExecution())
            execution.Write("state", 0, [1, 2, 3, 4]);
        first.Dispose();
        var result = new byte[4];
        using (var execution = second.BeginExecution())
            execution.Read("state", 0, result);
        Assert.Equal(new byte[] { 1, 2, 3, 4 }, result);
        Assert.Equal(0, storage.Disposals);
        second.Dispose();
        Assert.Equal(1, storage.Disposals);
    }

    [Fact]
    public void RebindingAndDisposalAreRejectedWhileExecuting()
    {
        var program = Program();
        using var bindings = new VmBindings(program);
        using var owner = new VmResource(Bytes, VmSlotScope.Session, VmAccess.ReadWrite,
            new VmMemoryStorage(4));
        using var handle = owner.Acquire();
        bindings.Bind("state", handle);
        using (var execution = bindings.BeginExecution())
        {
            Assert.Throws<InvalidOperationException>(() => bindings.Bind("state", handle));
            Assert.Throws<InvalidOperationException>(() => bindings.BeginExecution());
            Assert.Throws<InvalidOperationException>(() => bindings.Dispose());
        }

        bindings.Bind("state", handle);
        using var next = bindings.BeginExecution();
    }

    [Fact]
    public void SharedWritableStateCannotExecuteConcurrentlyAcrossVms()
    {
        using var first = new VmBindings(Program());
        using var second = new VmBindings(Program());
        using var owner = new VmResource(Bytes, VmSlotScope.Session, VmAccess.ReadWrite,
            new VmMemoryStorage(4));
        using var resource = owner.Acquire();
        first.Bind("state", resource);
        second.Bind("state", resource);
        using (var execution = first.BeginExecution())
        {
            Assert.Throws<InvalidOperationException>(() => second.BeginExecution());
            Assert.Throws<InvalidOperationException>(() => resource.Write(0, [1]));
            Assert.Throws<InvalidOperationException>(() => resource.Read(0, new byte[4]));
        }
        using var next = second.BeginExecution();
    }

    [Fact]
    public void BindingRejectsScopeShapeAndPermissions()
    {
        using var bindings = new VmBindings(Program());
        using var wrongScope = new VmResource(Bytes, VmSlotScope.Global, VmAccess.ReadWrite,
            new VmMemoryStorage(4));
        using var readOnly = new VmResource(Bytes, VmSlotScope.Session, VmAccess.ReadOnly,
            new VmMemoryStorage(4));
        using var wrongShape = new VmResource(new VmTensor(VmElementType.Byte, [2, 2]),
            VmSlotScope.Session, VmAccess.ReadWrite, new VmMemoryStorage(4));
        using var scope = wrongScope.Acquire();
        using var access = readOnly.Acquire();
        using var shape = wrongShape.Acquire();
        Assert.Throws<ArgumentException>(() => bindings.Bind("state", scope));
        Assert.Throws<ArgumentException>(() => bindings.Bind("state", access));
        Assert.Throws<ArgumentException>(() => bindings.Bind("state", shape));
        Assert.Throws<InvalidOperationException>(() => bindings.BeginExecution());
    }

    [Fact]
    public void LocalPhysicalStorageCannotBeSharedByTwoVms()
    {
        var program = Program([new VmSlot("scratch", VmSlotScope.Local, VmAccess.ReadWrite, Bytes)]);
        using var owner = new VmResource(Bytes, VmSlotScope.Local, VmAccess.ReadWrite, new VmMemoryStorage(4));
        using var handle = owner.Acquire();
        using var first = new VmBindings(program);
        using var second = new VmBindings(program);
        first.Bind("scratch", handle);
        Assert.Throws<InvalidOperationException>(() => second.Bind("scratch", handle));
        first.Dispose();
        second.Bind("scratch", handle);
    }

    [Fact]
    public void SlotPermissionsCannotBeWidenedByWritablePhysicalStorage()
    {
        var program = Program([new VmSlot("weight", VmSlotScope.Global, VmAccess.ReadOnly, Bytes)]);
        using var bindings = new VmBindings(program);
        using var owner = new VmResource(Bytes, VmSlotScope.Global, VmAccess.ReadWrite, new VmMemoryStorage(4));
        using var handle = owner.Acquire();
        bindings.Bind("weight", handle);
        using var execution = bindings.BeginExecution();
        Assert.Throws<InvalidOperationException>(() => execution.Write("weight", 0, [1]));
    }

    [Fact]
    public void StateStreamTransfersBetweenIndependentVmInstances()
    {
        var program = Program();
        using var first = Bind(program, [1, 2, 3, 4]);
        using var second = Bind(program, [0, 0, 0, 0]);
        using var stream = new MemoryStream();
        using (var access = first.BeginStateAccess())
            VmStateStream.Export(stream, program, access, "model@1");
        Assert.True(stream.CanWrite);
        stream.Position = 0;
        using (var source = new NonSeekableReadStream(stream))
        using (var access = second.BeginStateAccess())
            VmStateStream.Import(source, program, access, "model@1", 4);
        var result = new byte[4];
        using (var access = second.BeginExecution()) access.Read("state", 0, result);
        Assert.Equal(new byte[] { 1, 2, 3, 4 }, result);
        Assert.True(stream.CanRead);
    }

    [Theory]
    [InlineData("truncated")]
    [InlineData("corrupt")]
    [InlineData("context")]
    [InlineData("budget")]
    public void InvalidImportLeavesExistingStateUnchanged(string failure)
    {
        var program = Program();
        using var bindings = Bind(program, [1, 2, 3, 4]);
        using var stream = new MemoryStream();
        using (var access = bindings.BeginStateAccess())
            VmStateStream.Export(stream, program, access, "model@1");
        var bytes = stream.ToArray();
        if (failure == "truncated") bytes = bytes[..^1];
        if (failure == "corrupt") bytes[^1] ^= 1;
        using (var source = new MemoryStream(bytes))
        using (var access = bindings.BeginStateAccess())
            Assert.Throws<InvalidDataException>(() => VmStateStream.Import(source, program, access,
                failure == "context" ? "other-model" : "model@1", failure == "budget" ? 3UL : 4UL));
        var result = new byte[4];
        using (var access = bindings.BeginExecution()) access.Read("state", 0, result);
        Assert.Equal(new byte[] { 1, 2, 3, 4 }, result);
    }

    [Fact]
    public void SessionPrivateStorageIsNotPartOfStateStream()
    {
        var slots = new[]
        {
            new VmSlot("state", VmSlotScope.Session, VmAccess.ReadWrite, Bytes),
            new VmSlot("private", VmSlotScope.Session, VmAccess.ReadWrite, Bytes),
        };
        var program = Program(slots);
        using var bindings = Bind(program, [1, 2, 3, 4]);
        using var stream = new MemoryStream();
        using (var access = bindings.BeginStateAccess())
        {
            VmStateStream.Export(stream, program, access, "model@1");
            access.Write("private", 0, [9, 9, 9, 9]);
            stream.Position = 0;
            VmStateStream.Import(stream, program, access, "model@1", 4);
            var result = new byte[4];
            access.Read("private", 0, result);
            Assert.Equal(new byte[] { 9, 9, 9, 9 }, result);
        }
    }

    [Fact]
    public void FailedCommitInvalidatesSharedStateUntilSuccessfulRestore()
    {
        var program = Program();
        var storage = new CountingStorage(4);
        using var owner = new VmResource(Bytes, VmSlotScope.Session, VmAccess.ReadWrite, storage);
        using var resource = owner.Acquire();
        using var first = new VmBindings(program);
        using var second = new VmBindings(program);
        first.Bind("state", resource);
        second.Bind("state", resource);
        using var stream = new MemoryStream();
        using (var access = first.BeginStateAccess())
            VmStateStream.Export(stream, program, access, "model@1");
        stream.Position = 0;
        storage.FailWrites = true;
        using (var access = first.BeginStateAccess())
            Assert.Throws<IOException>(() => VmStateStream.Import(stream, program, access, "model@1", 4));
        Assert.Throws<InvalidOperationException>(() => first.BeginExecution());
        Assert.Throws<InvalidOperationException>(() => second.BeginExecution());
        storage.FailWrites = false;
        stream.Position = 0;
        using (var access = first.BeginStateAccess())
            VmStateStream.Import(stream, program, access, "model@1", 4);
        using var execution = second.BeginExecution();
    }

    [Fact]
    public void HostCanRestoreStateWithoutGrantingGeneratedCodeWriteAccess()
    {
        var program = Program([new VmSlot("state", VmSlotScope.Session, VmAccess.ReadOnly, Bytes)]);
        using var owner = new VmResource(Bytes, VmSlotScope.Session, VmAccess.ReadWrite, new VmMemoryStorage(4));
        using var resource = owner.Acquire();
        using var bindings = new VmBindings(program);
        bindings.Bind("state", resource);
        using var stream = new MemoryStream();
        resource.Write(0, [1, 2, 3, 4]);
        using (var access = bindings.BeginStateAccess())
            VmStateStream.Export(stream, program, access, "model@1");
        resource.Write(0, [0, 0, 0, 0]);
        stream.Position = 0;
        using (var access = bindings.BeginStateAccess())
            VmStateStream.Import(stream, program, access, "model@1", 4);
        var result = new byte[4];
        using (var execution = bindings.BeginExecution())
        {
            execution.Read("state", 0, result);
            Assert.Throws<InvalidOperationException>(() => execution.Write("state", 0, [9]));
        }
        Assert.Equal(new byte[] { 1, 2, 3, 4 }, result);
    }

    private static VmProgram Program(VmSlot[]? slots = null)
    {
        slots ??= [new VmSlot("state", VmSlotScope.Session, VmAccess.ReadWrite, Bytes)];
        return new VmProgram("resource-test", "test@1", VmTarget.Cpu, slots,
            [new VmDefinition("empty", VmDefinitionKind.Function, [], [])],
            [new VmEntry("run", "empty", [])], new VmState("state@1", 1,
                slots.Any(slot => slot.Id == "state") ? [new VmStateEntry("recurrent", "state")] : []));
    }

    private static VmBindings Bind(VmProgram program, byte[] initial)
    {
        var bindings = new VmBindings(program);
        foreach (var slot in program.Slots)
        {
            var storage = new VmMemoryStorage((int)slot.Tensor.ByteLength);
            storage.Write(0, initial);
            using var owner = new VmResource(slot.Tensor, slot.Scope, slot.Access, storage);
            using var lease = owner.Acquire();
            bindings.Bind(slot.Id, lease);
        }
        return bindings;
    }

    private sealed class CountingStorage(int size) : IVmStorage
    {
        private readonly VmMemoryStorage inner = new(size);
        public ulong ByteLength => inner.ByteLength;
        public int Disposals { get; private set; }
        public bool FailWrites { get; set; }
        public void Read(ulong offset, Span<byte> destination) => inner.Read(offset, destination);
        public void Write(ulong offset, ReadOnlySpan<byte> source)
        {
            if (FailWrites) throw new IOException("Simulated upload failure.");
            inner.Write(offset, source);
        }
        public void Dispose()
        {
            Disposals++;
            inner.Dispose();
        }
    }

    private sealed class NonSeekableReadStream(Stream inner) : Stream
    {
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
        public override int Read(Span<byte> buffer) => inner.Read(buffer);
        public override void Flush() => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
