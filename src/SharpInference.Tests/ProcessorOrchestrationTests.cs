using SharpInference.Graphs;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class ProcessorOrchestrationTests
{
    [Fact]
    public async Task Scheduler_RunsIndependentBranchesInParallelAndJoinAfterBoth()
    {
        var firstEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var secondEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var completed = 0;
        var joinedAfter = -1;
        var graph = new ProcessorOrchestrationGraph(
        [
            Node("vision", [], async (_, token) =>
            {
                firstEntered.SetResult();
                await release.Task.WaitAsync(token);
                Interlocked.Increment(ref completed);
            }),
            Node("audio", [], async (_, token) =>
            {
                secondEntered.SetResult();
                await release.Task.WaitAsync(token);
                Interlocked.Increment(ref completed);
            }),
            new ProcessorOrchestrationNode(
                "join", ["vision", "audio"], [],
                (_, _) =>
                {
                    joinedAfter = Volatile.Read(ref completed);
                    return ValueTask.CompletedTask;
                }),
        ]);

        var execution = new ProcessorOrchestrator().ExecuteAsync(graph).AsTask();
        await Task.WhenAll(firstEntered.Task, secondEntered.Task);
        Assert.False(execution.IsCompleted);
        release.SetResult();
        await using var result = await execution;

        Assert.Equal(2, joinedAfter);
    }

    [Fact]
    public void Graph_RejectsUnorderedReadWriteAndWriteWriteConflicts()
    {
        var read = new ProcessorResourceAccess("shared", GraphResourceAccess.Read);
        var write = new ProcessorResourceAccess("shared", GraphResourceAccess.Write);

        Assert.Throws<InvalidDataException>(() => new ProcessorOrchestrationGraph(
        [
            Node("reader", [], static (_, _) => ValueTask.CompletedTask, read),
            Node("writer", [], static (_, _) => ValueTask.CompletedTask, write),
        ]));
        Assert.Throws<InvalidDataException>(() => new ProcessorOrchestrationGraph(
        [
            Node("first", [], static (_, _) => ValueTask.CompletedTask, write),
            Node("second", [], static (_, _) => ValueTask.CompletedTask, write),
        ]));

        _ = new ProcessorOrchestrationGraph(
        [
            Node("writer", [], static (_, _) => ValueTask.CompletedTask, write),
            Node("reader", ["writer"], static (_, _) => ValueTask.CompletedTask, read),
        ]);
    }

    [Fact]
    public async Task Failure_CancelsParallelWorkAndReleasesPublishedStorage()
    {
        var branchStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var branchCancelled = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var released = 0;
        var domain = new StorageDomain("cpu", "0", "main");
        var descriptor = new TensorDescriptor(GraphElementType.Byte, [1]);
        var graph = new ProcessorOrchestrationGraph(
        [
            Node("branch", [], async (context, token) =>
            {
                context.Publish("temporary", StorageLease.Create(
                    new TestHandle(Guid.NewGuid(), domain, descriptor, 1),
                    _ => Interlocked.Increment(ref released)));
                branchStarted.SetResult();
                try { await Task.Delay(Timeout.InfiniteTimeSpan, token); }
                catch (OperationCanceledException)
                {
                    branchCancelled.SetResult();
                    throw;
                }
            }, new ProcessorResourceAccess("temporary", GraphResourceAccess.Write)),
            Node("failure", [], async (_, _) =>
            {
                await branchStarted.Task;
                throw new InvalidOperationException("component failed");
            }),
        ]);

        var error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => new ProcessorOrchestrator().ExecuteAsync(graph).AsTask());

        Assert.Equal("component failed", error.Message);
        await branchCancelled.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(1, released);
    }

    private static ProcessorOrchestrationNode Node(
        string name,
        IEnumerable<string> dependencies,
        Func<ProcessorOrchestrationContext, CancellationToken, ValueTask> execute,
        params ProcessorResourceAccess[] resources) =>
        new(name, dependencies, resources, execute);

    private sealed record TestHandle(
        Guid Id,
        StorageDomain Domain,
        TensorDescriptor Descriptor,
        long ByteLength) : IStorageHandle;
}
