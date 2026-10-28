using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class VmStateTests
{
    private static readonly VmTensor Tensor = new(VmElementType.Byte, [1]);

    [Theory]
    [InlineData(VmSlotScope.Global, VmAccess.ReadOnly)]
    [InlineData(VmSlotScope.Global, VmAccess.ReadWrite)]
    [InlineData(VmSlotScope.Session, VmAccess.ReadOnly)]
    [InlineData(VmSlotScope.Session, VmAccess.ReadWrite)]
    [InlineData(VmSlotScope.Local, VmAccess.ReadOnly)]
    [InlineData(VmSlotScope.Local, VmAccess.ReadWrite)]
    public void StateRegistrationIsIndependentOfScopeAndAccess(VmSlotScope scope, VmAccess access)
    {
        var program = Program("test", [new("value", scope, access, Tensor)], [new("saved", "value")]);

        Assert.Equal(scope, Assert.Single(program.Slots).Scope);
        Assert.Equal(access, Assert.Single(program.Slots).Access);
        Assert.Equal("value", Assert.Single(program.State.Entries).Slot);
    }

    [Fact]
    public void StateStillRequiresDistinctExistingSlots()
    {
        VmSlot[] slots = [new("value", VmSlotScope.Global, VmAccess.ReadWrite, Tensor)];
        Assert.Throws<InvalidDataException>(() => Program("missing", slots, [new("saved", "missing")]));
        Assert.Throws<InvalidDataException>(() =>
            Program("duplicate", slots, [new("first", "value"), new("second", "value")]));
    }

    [Fact]
    public void SessionViewExcludesWorkspaceAndNonSessionState()
    {
        var first = Program("first",
            [new("state", VmSlotScope.Session, VmAccess.ReadWrite, Tensor),
             new("workspace", VmSlotScope.Session, VmAccess.ReadWrite, Tensor),
             new("global", VmSlotScope.Global, VmAccess.ReadWrite, Tensor),
             new("local", VmSlotScope.Local, VmAccess.ReadWrite, Tensor)],
            [new("global-saved", "global"), new("saved", "state"), new("local-saved", "local")]);
        var second = Program("second",
            [new("renamed", VmSlotScope.Session, VmAccess.ReadWrite, Tensor),
             new("other-global", VmSlotScope.Global, VmAccess.ReadOnly, Tensor)],
            [new("saved", "renamed"), new("unrelated", "other-global")]);
        using var manager = Manager();
        using var session = manager.CreateSession(first, second);
        Assert.Equal("saved", Assert.Single(session.StateProgram.Slots).Id);
        Assert.Equal("saved", Assert.Single(session.StateProgram.State.Entries).Name);
        using var firstBindings = manager.CreateBindings(first);
        using var secondBindings = manager.CreateBindings(second);
        manager.BindSession(firstBindings, session);
        manager.BindSession(secondBindings, session);
        using (var access = firstBindings.BeginExecution())
        {
            access.Write("workspace", 0, new byte[] { 9 });
        }
        using (var state = session.CreateStateBindings())
        using (var access = state.BeginStateAccess())
        {
            access.RestoreState(Snapshot(("saved", 7)));
        }
        using (var access = firstBindings.BeginExecution())
        {
            Assert.Equal(7, Read(access, "state"));
            Assert.Equal(9, Read(access, "workspace"));
        }
        using (var access = secondBindings.BeginExecution())
        {
            Assert.Equal(7, Read(access, "renamed"));
        }
        using var otherSession = manager.CreateSession(first);
        using var otherState = otherSession.CreateStateBindings();
        using var otherAccess = otherState.BeginStateAccess();
        Assert.Equal(0, Read(otherAccess, "saved"));
    }

    [Fact]
    public void EmptySessionStateDoesNotCompareUnrelatedProgramSchemas()
    {
        var first = Program("first", [new("global", VmSlotScope.Global, VmAccess.ReadWrite, Tensor)],
            [new("global-saved", "global")], "first-schema");
        var second = Program("second", [new("local", VmSlotScope.Local, VmAccess.ReadWrite, Tensor)],
            [new("local-saved", "local")], "second-schema");
        using var manager = Manager();
        using var session = manager.CreateSession(first, second);
        Assert.Empty(session.StateProgram.Slots);
        Assert.Empty(session.StateProgram.State.Entries);
    }

    [Theory]
    [InlineData("name")]
    [InlineData("schema")]
    [InlineData("tensor")]
    [InlineData("access")]
    public void ActualSessionStateContractsMustRemainCompatible(string difference)
    {
        var first = Program("first", [new("state", VmSlotScope.Session, VmAccess.ReadWrite, Tensor)],
            [new("saved", "state")]);
        var second = Program("second",
            [new("state", VmSlotScope.Session,
                difference == "access" ? VmAccess.ReadOnly : VmAccess.ReadWrite,
                difference == "tensor" ? new(VmElementType.Byte, [2]) : Tensor)],
            [new(difference == "name" ? "other" : "saved", "state")],
            difference == "schema" ? "other" : "schema");
        using var manager = Manager();
        Assert.Throws<InvalidDataException>(() => manager.CreateSession(first, second));
    }

    [Fact]
    public void GlobalStateUsesSharedPhysicalBindingAndLocksAcrossPrograms()
    {
        var first = Program("first", [new("global", VmSlotScope.Global, VmAccess.ReadWrite, Tensor, "shared")],
            [new("saved", "global")]);
        var second = Program("second", [new("alias", VmSlotScope.Global, VmAccess.ReadWrite, Tensor, "shared")],
            [new("saved", "alias")]);
        using var manager = Manager();
        using var left = manager.CreateBindings(first);
        using var right = manager.CreateBindings(second);
        using var session = manager.CreateSession(first, second);
        Assert.Empty(session.StateProgram.Slots);
        using var stream = new MemoryStream();
        using (var access = left.BeginStateAccess())
        {
            access.RestoreState(Snapshot(("saved", 4)));
            VmStateStream.Export(stream, first, access, "context");
            Assert.Throws<InvalidOperationException>(() => right.BeginStateAccess());
            Assert.Throws<InvalidOperationException>(() => right.BeginExecution());
        }
        using (var access = right.BeginExecution())
        {
            Assert.Equal(4, Read(access, "alias"));
            access.Write("alias", 0, new byte[] { 8 });
        }
        using (var access = left.BeginStateAccess())
        {
            stream.Position = 0;
            VmStateStream.Import(stream, first, access, "context", 1);
        }
        using var restored = right.BeginExecution();
        Assert.Equal(4, Read(restored, "alias"));
    }

    [Fact]
    public void CallMappedLocalStateRetainsOnlyItsBindingLifetime()
    {
        var program = new VmProgram("local-call", "abi", VmTarget.Cpu,
            [new("local", VmSlotScope.Local, VmAccess.ReadWrite, Tensor)],
            [new("root", VmDefinitionKind.Function, [new("value", VmAccess.ReadWrite, Tensor)],
                [new("call", new VmCall("child", [new("value", "value")]))]),
             new("child", VmDefinitionKind.Function, [new("value", VmAccess.ReadWrite, Tensor)], [])],
            [new("run", "root", [new("value", "local")])],
            new("schema", 1, [new("saved", "local")]));
        List<TestStorage> allocations = [];
        using var manager = new VmResourceManager((_, _) => { }, slot =>
        {
            var storage = new TestStorage((int)slot.Tensor.ByteLength);
            allocations.Add(storage);
            return storage;
        });
        using var session = manager.CreateSession(program);
        Assert.Empty(session.StateProgram.Slots);
        var bindings = manager.CreateBindings(program);
        using (var access = bindings.BeginStateAccess())
        {
            access.RestoreState(Snapshot(("saved", 6)));
            Assert.Equal(6, Read(access, "local"));
            Assert.Throws<InvalidOperationException>(() => bindings.Dispose());
        }
        bindings.Dispose();
        Assert.True(Assert.Single(allocations).Disposed);
        using var replacement = manager.CreateBindings(program);
        using var fresh = replacement.BeginExecution();
        Assert.Equal(0, Read(fresh, "local"));
        Assert.Equal(VmSlotScope.Local, Assert.Single(program.Slots).Scope);
    }

    [Theory]
    [InlineData(VmSlotScope.Global, VmAccess.ReadOnly)]
    [InlineData(VmSlotScope.Global, VmAccess.ReadWrite)]
    [InlineData(VmSlotScope.Session, VmAccess.ReadOnly)]
    [InlineData(VmSlotScope.Session, VmAccess.ReadWrite)]
    [InlineData(VmSlotScope.Local, VmAccess.ReadOnly)]
    [InlineData(VmSlotScope.Local, VmAccess.ReadWrite)]
    public void RestoreCannotElevateReadOnlySlotPermissions(VmSlotScope scope, VmAccess physicalAccess)
    {
        var program = Program("test",
            [new("writable", scope, VmAccess.ReadWrite, Tensor),
             new("readonly", scope, VmAccess.ReadOnly, Tensor)],
            [new("first", "writable"), new("second", "readonly")]);
        using var writable = new VmResource(Tensor, scope, VmAccess.ReadWrite, new VmMemoryStorage(1));
        var readOnlyStorage = new VmMemoryStorage(1);
        readOnlyStorage.Write(0, new byte[] { 5 });
        using var readOnly = new VmResource(Tensor, scope, physicalAccess, readOnlyStorage);
        using var writableLease = writable.Acquire();
        using var readOnlyLease = readOnly.Acquire();
        using var bindings = new VmBindings(program);
        bindings.Bind("writable", writableLease);
        bindings.Bind("readonly", readOnlyLease);
        using var access = bindings.BeginStateAccess();
        using var stream = new MemoryStream();
        VmStateStream.Export(stream, program, access, "context");
        Assert.Throws<InvalidOperationException>(() => access.RestoreState(Snapshot(("first", 9), ("second", 9))));
        stream.Position = 0;
        Assert.Throws<InvalidOperationException>(() => VmStateStream.Import(stream, program, access, "context", 2));
        Assert.Throws<InvalidOperationException>(() => access.Write("readonly", 0, new byte[] { 9 }));
        Assert.Equal(0, Read(access, "writable"));
        Assert.Equal(5, Read(access, "readonly"));
        using var unchanged = new MemoryStream();
        VmStateStream.Export(unchanged, program, access, "context");
        Assert.Equal(stream.ToArray(), unchanged.ToArray());
    }

    [Fact]
    public void SessionStateViewDoesNotTurnReadOnlyStateIntoWritableState()
    {
        var program = Program("test", [new("state", VmSlotScope.Session, VmAccess.ReadOnly, Tensor)],
            [new("saved", "state")]);
        using var manager = Manager();
        using var session = manager.CreateSession(program);
        Assert.Equal(VmAccess.ReadOnly, Assert.Single(session.StateProgram.Slots).Access);
        using var bindings = session.CreateStateBindings();
        using var access = bindings.BeginStateAccess();
        Assert.Throws<InvalidOperationException>(() => access.RestoreState(Snapshot(("saved", 1))));
        using var stream = new MemoryStream();
        VmStateStream.Export(stream, session.StateProgram, access, "context");
    }

    [Fact]
    public void WritableStateCannotBindReadOnlyPhysicalStorage()
    {
        var program = Program("test", [new("state", VmSlotScope.Global, VmAccess.ReadWrite, Tensor)],
            [new("saved", "state")]);
        using var resource = new VmResource(Tensor, VmSlotScope.Global, VmAccess.ReadOnly, new VmMemoryStorage(1));
        using var lease = resource.Acquire();
        using var bindings = new VmBindings(program);
        Assert.Throws<ArgumentException>(() => bindings.Bind("state", lease));
    }

    [Theory]
    [InlineData("budget")]
    [InlineData("integrity")]
    [InlineData("context")]
    [InlineData("truncated")]
    public void FailedImportLeavesAllScopesUnchangedAndValid(string failure)
    {
        var program = MixedProgram();
        using var manager = Manager();
        using var session = manager.CreateSession(program);
        using var bindings = manager.CreateBindings(program);
        manager.BindSession(bindings, session);
        using (var access = bindings.BeginStateAccess())
        {
            access.RestoreState(Snapshot(("global", 1), ("session", 2), ("local", 3)));
        }
        using var stream = Export(bindings);
        var bytes = stream.ToArray();
        if (failure == "integrity")
        {
            bytes[^1] ^= 1;
        }

        if (failure == "truncated")
        {
            bytes = bytes[..^1];
        }

        using var source = new MemoryStream(bytes);
        using (var access = bindings.BeginStateAccess())
        {
            Assert.Throws<InvalidDataException>(() => VmStateStream.Import(source, program, access,
                failure == "context" ? "other" : "context", failure == "budget" ? 2UL : 3UL));
        }
        using var unchanged = Export(bindings);
        Assert.Equal(stream.ToArray(), unchanged.ToArray());
        using var execution = bindings.BeginExecution();
    }

    [Fact]
    public void StorageFailureInvalidatesSharedStateUntilRestore()
    {
        TestStorage? storage = null;
        using var manager = new VmResourceManager((_, _) => { }, _ => storage = new TestStorage(1));
        var program = Program("test", [new("state", VmSlotScope.Global, VmAccess.ReadWrite, Tensor)],
            [new("saved", "state")]);
        using var first = manager.CreateBindings(program);
        using var second = manager.CreateBindings(program);
        storage!.FailWrites = true;
        using (var access = first.BeginStateAccess())
        {
            Assert.Throws<IOException>(() => access.RestoreState(Snapshot(("saved", 9))));
        }
        Assert.Throws<InvalidOperationException>(() => second.BeginExecution());
        storage.FailWrites = false;
        using (var access = second.BeginStateAccess())
        {
            access.RestoreState(Snapshot(("saved", 7)));
        }
        using var repaired = first.BeginExecution();
        Assert.Equal(7, Read(repaired, "state"));
    }

    private static VmProgram MixedProgram() => Program("mixed",
        [new("global", VmSlotScope.Global, VmAccess.ReadWrite, Tensor),
         new("session", VmSlotScope.Session, VmAccess.ReadWrite, Tensor),
         new("local", VmSlotScope.Local, VmAccess.ReadWrite, Tensor)],
        [new("global", "global"), new("session", "session"), new("local", "local")]);

    private static VmProgram Program(string name, VmSlot[] slots, VmStateEntry[] state, string schema = "schema") =>
        new(name, "abi", VmTarget.Cpu, slots,
            [new("run", VmDefinitionKind.Function, [], [])], [new("run", "run", [])], new(schema, 1, state));

    private static VmResourceManager Manager() => new((_, _) => { });

    private static Dictionary<string, ReadOnlyMemory<byte>> Snapshot(params (string Name, byte Value)[] values) =>
        values.ToDictionary(value => value.Name, value => (ReadOnlyMemory<byte>)new byte[] { value.Value });

    private static byte Read(VmExecutionLease access, string slot)
    {
        byte[] bytes = new byte[1];
        access.Read(slot, 0, bytes);
        return bytes[0];
    }

    private static MemoryStream Export(VmBindings bindings)
    {
        var stream = new MemoryStream();
        using var access = bindings.BeginStateAccess();
        VmStateStream.Export(stream, bindings.Program, access, "context");
        stream.Position = 0;
        return stream;
    }

    private sealed class TestStorage(int size) : IVmStorage
    {
        private readonly VmMemoryStorage storage = new(size);
        public bool Disposed
        {
            get; private set;
        }
        public bool FailWrites
        {
            get; set;
        }
        public ulong ByteLength => storage.ByteLength;
        public void Read(ulong offset, Span<byte> destination)
        {
            storage.Read(offset, destination);
        }
        public void Write(ulong offset, ReadOnlySpan<byte> source)
        {
            if (FailWrites)
            {
                throw new IOException("Injected write failure.");
            }

            storage.Write(offset, source);
        }
        public void Dispose()
        {
            Disposed = true;
            storage.Dispose();
        }
    }
}
