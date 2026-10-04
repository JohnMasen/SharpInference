using SharpInference.Backends.D3D12Vm;
using SharpInference.Vm;
using Xunit.Abstractions;

namespace SharpInference.Tests;

public sealed class D3D12VmResourcePoolTests(ITestOutputHelper output)
{
    [Fact]
    public void ImportedArtifactFourWorkersPhysicallyShareWeightsAndSessionButNotLocals()
    {
        var program = D3D12VmTests.AccumulatingProgram();
        using var archive = new MemoryStream();
        new D3D12VmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program).Export(archive);
        archive.Position = 0;
        var artifact = D3D12VmArtifact.Import(archive);
        var (device, name) = D3D12VmDeviceFactory.Create();
        output.WriteLine($"Physical GPU sharing on {name}");
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        using (var manager = new VmResourceManager(Initialize, pool.Allocate))
        using (var session = manager.CreateSession(program))
        {
            var workers = new List<D3D12VmExecutor>();
            var bindings = new List<VmBindings>();
            try
            {
                for (var i = 0; i < 4; i++)
                {
                    workers.Add(artifact.CreateExecutor(pool));
                    bindings.Add(manager.CreateBindings(program));
                    manager.BindSession(bindings[i], session);
                    using var lease = bindings[i].BeginExecution();
                    BitConverter.GetBytes(i).CopyTo(lease.GetBuffers()[1], 0);
                    workers[i].Execute("forward", lease);
                    Assert.Equal((float)(Enumerable.Range(1, i + 1).Sum() + i * 64),
                        BitConverter.ToSingle(lease.GetBuffers()[2]));
                    Assert.Equal(0UL, workers[i].AllocatedGpuBytes);
                    Assert.Same(pool, workers[i].ResourcePool);
                    Assert.Equal(workers[0].GetGpuVirtualAddress("table"), workers[i].GetGpuVirtualAddress("table"));
                    Assert.Equal(workers[0].GetGpuVirtualAddress("state"), workers[i].GetGpuVirtualAddress("state"));
                    if (i > 0)
                        foreach (var local in program.Slots.Where(s => s.Scope == VmSlotScope.Local))
                            Assert.NotEqual(workers[0].GetGpuVirtualAddress(local.Id), workers[i].GetGpuVirtualAddress(local.Id));
                    var records = workers[i].BindingRecordCount;
                    for (var token = 0; token < 64; token++)
                        BitConverter.GetBytes(0).CopyTo(lease.GetBuffers()[1], token * 4);
                    workers[i].Execute("prefill.64", lease);
                    Assert.Equal((float)(Enumerable.Range(1, i + 1).Sum() + (i + 1) * 64),
                        BitConverter.ToSingle(lease.GetBuffers()[2]));
                    Assert.Equal(records, workers[i].BindingRecordCount);
                }
                Assert.Equal(18, pool.AllocationCount);
                Assert.Equal(1084UL, pool.AllocatedGpuBytes);
                Assert.Equal(1UL, pool.GlobalUploadCount);
                Assert.Equal(8UL, pool.GlobalUploadBytes);
                Assert.Equal(4, pool.ExecutorCount);
                Assert.Throws<InvalidOperationException>(pool.Dispose);
            }
            finally
            {
                foreach (var worker in workers) worker.Dispose();
                foreach (var binding in bindings) binding.Dispose();
            }
            Assert.Equal(0, pool.ExecutorCount);
            Assert.Equal(2, pool.AllocationCount);
            session.Dispose();
            manager.Dispose();
            Assert.Equal(0, pool.AllocationCount);
            Assert.Equal(0UL, pool.AllocatedGpuBytes);
        }
    }

    [Fact]
    public async Task ConcurrentWorkersUploadSharedGlobalsOnceAndKeepSessionsIndependent()
    {
        var program = D3D12VmTests.AccumulatingProgram();
        var artifact = new D3D12VmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
        var (device, _) = D3D12VmDeviceFactory.Create();
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        using (var manager = new VmResourceManager(Initialize, pool.Allocate))
        {
            var workers = new List<D3D12VmExecutor>();
            var bindings = new List<VmBindings>();
            var sessions = new List<VmSessionResources>();
            try
            {
                for (var i = 0; i < 4; i++)
                {
                    workers.Add(artifact.CreateExecutor(pool));
                    bindings.Add(manager.CreateBindings(program));
                    sessions.Add(manager.CreateSession(program));
                    manager.BindSession(bindings[i], sessions[i]);
                }
                await Task.WhenAll(Enumerable.Range(0, 4).Select(i => Task.Run(() =>
                {
                    for (var step = 1; step <= 5; step++)
                        Assert.Equal((float)((i + 1) * step), Run(workers[i], bindings[i], i));
                })));
                Assert.Equal(1UL, pool.GlobalUploadCount);
                Assert.Single(workers.Select(worker => worker.GetGpuVirtualAddress("table")).Distinct());
                Assert.Equal(4, workers.Select(worker => worker.GetGpuVirtualAddress("state")).Distinct().Count());
            }
            finally
            {
                foreach (var worker in workers) worker.Dispose();
                foreach (var binding in bindings) binding.Dispose();
                foreach (var session in sessions) session.Dispose();
            }
            manager.Dispose();
            Assert.Equal(0, pool.AllocationCount);
        }
    }

    [Fact]
    public void StateEditsImportsAndSessionRebindingRemainCoherentAcrossWorkers()
    {
        var program = D3D12VmTests.AccumulatingProgram();
        var artifact = new D3D12VmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
        var (device, _) = D3D12VmDeviceFactory.Create();
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        using (var manager = new VmResourceManager(Initialize, pool.Allocate))
        using (var firstSession = manager.CreateSession(program))
        using (var secondSession = manager.CreateSession(program))
        using (var firstBindings = manager.CreateBindings(program))
        using (var secondBindings = manager.CreateBindings(program))
        using (var first = artifact.CreateExecutor(pool))
        using (var second = artifact.CreateExecutor(pool))
        {
            manager.BindSession(firstBindings, firstSession);
            manager.BindSession(secondBindings, firstSession);
            Assert.Equal(1f, Run(first, firstBindings, 0));
            Assert.Equal(3f, Run(second, secondBindings, 1));
            using var snapshot = new MemoryStream();
            using (var access = firstBindings.BeginStateAccess())
                VmStateStream.Export(snapshot, program, access, "pool");
            using (var lease = secondBindings.BeginExecution())
                BitConverter.GetBytes(20f).CopyTo(lease.GetBuffers()[2], 0);
            Assert.Equal(23f, Run(first, firstBindings, 2));
            snapshot.Position = 0;
            using (var access = secondBindings.BeginStateAccess())
                VmStateStream.Import(snapshot, program, access, "pool", 4);
            Assert.Equal(7f, Run(first, firstBindings, 3));
            Assert.Equal(first.GetGpuVirtualAddress("state"), second.GetGpuVirtualAddress("state"));
            var previousAddress = first.GetGpuVirtualAddress("state");
            var records = first.BindingRecordCount;
            firstBindings.UnbindSession();
            manager.BindSession(firstBindings, secondSession);
            Assert.Equal(2f, Run(first, firstBindings, 1));
            Assert.NotEqual(previousAddress, first.GetGpuVirtualAddress("state"));
            Assert.Equal(records + 1, first.BindingRecordCount);
            Assert.Equal(8f, Run(second, secondBindings, 0));
            firstBindings.UnbindSession();
            manager.BindSession(firstBindings, firstSession);
            Assert.Equal(9f, Run(first, firstBindings, 0));
            Assert.Equal(previousAddress, first.GetGpuVirtualAddress("state"));
            Assert.Equal(1UL, pool.GlobalUploadCount);
        }
    }

    [Fact]
    public void GlobalWriteAndExplicitDirtyNotificationUploadOnceAcrossWorkers()
    {
        var program = D3D12VmTests.AccumulatingProgram();
        var artifact = new D3D12VmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
        var (device, _) = D3D12VmDeviceFactory.Create();
        IVmStorage? global = null;
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        using (var manager = new VmResourceManager((slot, storage) =>
        {
            Initialize(slot, storage);
            global = storage;
        }, pool.Allocate))
        using (var session = manager.CreateSession(program))
        using (var firstBindings = manager.CreateBindings(program))
        using (var secondBindings = manager.CreateBindings(program))
        using (var first = artifact.CreateExecutor(pool))
        using (var second = artifact.CreateExecutor(pool))
        {
            manager.BindSession(firstBindings, session);
            manager.BindSession(secondBindings, session);
            Assert.Equal(1f, Run(first, firstBindings, 0));
            Assert.Equal(2f, Run(second, secondBindings, 0));
            global!.Write(0, BitConverter.GetBytes((Half)10));
            Assert.Throws<InvalidOperationException>(() => first.Execute("forward"));
            Assert.Equal(12f, Run(second, secondBindings, 0));
            Assert.Equal(22f, Run(first, firstBindings, 0));
            Assert.Equal(2UL, pool.GlobalUploadCount);
            var buffer = ((IVmManagedStorage)global).Buffer;
            BitConverter.GetBytes((Half)20).CopyTo(buffer, 0);
            pool.MarkCpuModified(buffer);
            Assert.Throws<InvalidOperationException>(() => second.Execute("forward"));
            Assert.Equal(42f, Run(first, firstBindings, 0));
            Assert.Equal(62f, Run(second, secondBindings, 0));
            Assert.Equal(3UL, pool.GlobalUploadCount);
        }
    }

    [Fact]
    public void StorageDisposalUnregistersImmediatelyAndRejectsStaleOrForeignBindings()
    {
        var program = D3D12VmTests.AccumulatingProgram();
        var artifact = new D3D12VmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
        var (device, _) = D3D12VmDeviceFactory.Create();
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        using (var worker = artifact.CreateExecutor(pool))
        {
            var storage = program.Slots.Select(pool.Allocate).ToArray();
            try
            {
                Initialize(program.Slots[0], storage[0]);
                var context = storage.Cast<IVmManagedStorage>().Select(s => s.Buffer).ToArray();
                worker.Execute("forward", context);
                Assert.Equal(pool.GetGpuVirtualAddress(context[0]), worker.GetGpuVirtualAddress("table"));
                var original = context[0];
                context[0] = (byte[])original.Clone();
                Assert.Throws<ArgumentException>(() => worker.Execute("forward", context));
                context[0] = original;
                context[4] = context[3];
                Assert.Throws<NotSupportedException>(() => worker.Execute("forward", context));
                context[4] = ((IVmManagedStorage)storage[4]).Buffer;
                storage[2].Dispose();
                Assert.False(pool.Contains(context[2]));
                Assert.Equal(5, pool.AllocationCount);
                Assert.Throws<ObjectDisposedException>(() => worker.Execute("forward"));
                Assert.Throws<ArgumentException>(() => worker.Execute("forward", context));
                Assert.Throws<ObjectDisposedException>(() => storage[2].Read(0, new byte[4]));
                Assert.Throws<ObjectDisposedException>(() => ((IVmManagedStorage)storage[2]).Buffer);
                storage[2].Dispose();
                using var replacement = pool.Allocate(program.Slots[2]);
                context[2] = ((IVmManagedStorage)replacement).Buffer;
                worker.Execute("forward", context);
                Assert.Equal(1f, BitConverter.ToSingle(context[2]));
                worker.Dispose();
                Assert.Equal(6, pool.AllocationCount);
                Assert.Equal(0, pool.ExecutorCount);
            }
            finally { foreach (var allocation in storage) allocation.Dispose(); }
            Assert.Equal(0, pool.AllocationCount);
            Assert.Equal(0UL, pool.AllocatedGpuBytes);
        }
    }

    [Fact]
    public void IdleCommandCacheBorrowsNoStorageAndRejectsClosedSessionBeforeSubmission()
    {
        var program = D3D12VmTests.AccumulatingProgram();
        var artifact = new D3D12VmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
        var (device, _) = D3D12VmDeviceFactory.Create();
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        using (var manager = new VmResourceManager(Initialize, pool.Allocate))
        using (var bindings = manager.CreateBindings(program))
        using (var session = manager.CreateSession(program))
        using (var worker = artifact.CreateExecutor(pool))
        {
            manager.BindSession(bindings, session);
            Assert.Equal(1f, Run(worker, bindings, 0));
            AssertNoBorrowedReferences(worker);
            var records = worker.BindingRecordCount;
            Assert.NotEqual(0UL, worker.GetGpuVirtualAddress("state"));
            AssertNoBorrowedReferences(worker);
            session.Dispose();
            bindings.UnbindSession();
            Assert.Equal(5, pool.AllocationCount);
            Assert.Throws<ObjectDisposedException>(() => worker.Execute("forward"));
            AssertNoBorrowedReferences(worker);
            Assert.Throws<ObjectDisposedException>(() => worker.GetGpuVirtualAddress("state"));
            AssertNoBorrowedReferences(worker);
            using var replacement = manager.CreateSession(program);
            manager.BindSession(bindings, replacement);
            Assert.Equal(2f, Run(worker, bindings, 1));
            Assert.Equal(records + 1, worker.BindingRecordCount);
            AssertNoBorrowedReferences(worker);
            using var lease = bindings.BeginExecution();
            var context = lease.GetBuffers();
            context[2] = (byte[])context[2].Clone();
            Assert.Throws<ArgumentException>(() => worker.Execute("forward", context));
            AssertNoBorrowedReferences(worker);
            worker.Execute("forward");
            AssertNoBorrowedReferences(worker);
            worker.Readback("state");
            AssertNoBorrowedReferences(worker);
        }
    }

    [Fact]
    public void CoreOwnersReleaseAllocationsOnlyAfterBindingsAndLeasesReleaseThem()
    {
        var program = D3D12VmTests.AccumulatingProgram();
        var (device, _) = D3D12VmDeviceFactory.Create();
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        using (var manager = new VmResourceManager(Initialize, pool.Allocate))
        using (var session = manager.CreateSession(program))
        using (var bindings = manager.CreateBindings(program))
        {
            manager.BindSession(bindings, session);
            using var lease = bindings.BeginExecution();
            var context = lease.GetBuffers();
            manager.Dispose();
            session.Dispose();
            Assert.Equal(6, pool.AllocationCount);
            Assert.True(pool.Contains(context[0]));
            Assert.True(pool.Contains(context[2]));
            lease.Dispose();
            bindings.UnbindSession();
            Assert.False(pool.Contains(context[2]));
            Assert.Equal(5, pool.AllocationCount);
            bindings.Dispose();
            Assert.Equal(0, pool.AllocationCount);
        }
    }

    [Fact]
    public void ExplicitTransfersRefreshOwnerShadowAndPoolDisposalReleasesRemainingStorage()
    {
        var program = D3D12VmTests.AccumulatingProgram();
        var artifact = new D3D12VmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
        var (device, _) = D3D12VmDeviceFactory.Create();
        using (device)
        using (var pool = new D3D12VmResourcePool(device))
        {
            var storage = program.Slots.Select(pool.Allocate).ToArray();
            try
            {
                Initialize(program.Slots[0], storage[0]);
                var context = storage.Cast<IVmManagedStorage>().Select(s => s.Buffer).ToArray();
                using (var worker = artifact.CreateExecutor(pool))
                {
                    Assert.Throws<InvalidOperationException>(() => worker.Execute("forward"));
                    worker.UploadSlots(context, Enumerable.Range(0, context.Length).ToArray());
                    worker.Upload("state", BitConverter.GetBytes(10f));
                    Assert.Equal(10f, BitConverter.ToSingle(context[2]));
                    worker.Execute("forward");
                    var result = worker.Readback("state");
                    Assert.Equal(11f, BitConverter.ToSingle(result));
                    Assert.Equal(11f, BitConverter.ToSingle(context[2]));
                    worker.ReadbackSlots(context, [5]);
                    Assert.Equal(11f, BitConverter.ToSingle(context[5]));
                    Assert.Throws<ArgumentOutOfRangeException>(() => worker.UploadSlots(context, [context.Length]));
                    Assert.Throws<ArgumentException>(() => worker.Execute("missing", context));
                }
                Assert.Equal(6, pool.AllocationCount);
                pool.Dispose();
                Assert.Equal(0UL, pool.AllocatedGpuBytes);
                Assert.Throws<ObjectDisposedException>(() => ((IVmManagedStorage)storage[0]).Buffer);
                Assert.Throws<ObjectDisposedException>(() => pool.Allocate(program.Slots[0]));
            }
            finally { foreach (var allocation in storage) allocation.Dispose(); }
        }
    }

    private static float Run(D3D12VmExecutor worker, VmBindings bindings, int token)
    {
        using var lease = bindings.BeginExecution();
        BitConverter.GetBytes(token).CopyTo(lease.GetBuffers()[1], 0);
        worker.Execute("forward", lease);
        return BitConverter.ToSingle(lease.GetBuffers()[2]);
    }

    private static void AssertNoBorrowedReferences(D3D12VmExecutor worker)
    {
        foreach (var field in new[] { "buffers", "pooledBindings", "candidateBindings" })
        {
            var references = (Array)typeof(D3D12VmExecutor)
                .GetField(field, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
                .GetValue(worker)!;
            Assert.All(references.Cast<object?>(), reference => Assert.Null(reference));
        }
    }

    private static void Initialize(VmSlot _, IVmStorage storage)
    {
        var bytes = new byte[8];
        for (var i = 0; i < 4; i++) BitConverter.GetBytes((Half)(i + 1)).CopyTo(bytes, i * 2);
        storage.Write(0, bytes);
    }
}
