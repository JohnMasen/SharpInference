using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class SymbolicShapeTests
{
    [Fact]
    public void Binding_SelectsBucketAndSpecializesFixedDescriptor()
    {
        var binding = ShapeBinding.Create(
            [new ShapeConstraint("S", minimum: 1, maximum: 16, buckets: [4, 8, 16])],
            new Dictionary<string, int> { ["S"] = 5 });
        var symbolic = new SymbolicTensorDescriptor(
            GraphElementType.Float16,
            [new FixedTensorDimension(2), new SymbolicTensorDimension("S")],
            "sequence-major");

        var descriptor = symbolic.Specialize(binding);

        Assert.Equal([2, 8], descriptor.Dimensions);
        Assert.Equal(5, binding.GetActual("S"));
        Assert.Equal(3, binding.GetPadding("S"));
        Assert.Equal("sequence-major", descriptor.Layout);
    }

    [Fact]
    public void Binding_RejectsValuesOutsideConstraintsAndUnknownBindings()
    {
        var constraint = new ShapeConstraint("T", minimum: 2, maximum: 8, multipleOf: 2);
        Assert.Throws<ArgumentException>(() => ShapeBinding.Create(
            [constraint], new Dictionary<string, int> { ["T"] = 3 }));
        Assert.Throws<ArgumentException>(() => ShapeBinding.Create(
            [constraint], new Dictionary<string, int> { ["other"] = 2 }));
    }

    [Fact]
    public async Task BucketCache_CompilesConcurrentKeyOnceAndEvictsLeastRecentlyUsed()
    {
        var cache = new ShapeBucketCache<string, string>(2);
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var compilations = 0;
        ValueTask<string> Compile(string key, CancellationToken _)
        {
            return new ValueTask<string>(CompileCore(key));
        }
        async Task<string> CompileCore(string key)
        {
            Interlocked.Increment(ref compilations);
            entered.TrySetResult();
            await release.Task;
            return $"compiled:{key}";
        }

        var first = cache.GetOrAddAsync("a", Compile).AsTask();
        await entered.Task;
        var second = cache.GetOrAddAsync("a", Compile).AsTask();
        release.SetResult();

        Assert.Equal(["compiled:a", "compiled:a"], await Task.WhenAll(first, second));
        Assert.Equal(1, compilations);
        Assert.Equal("compiled:b", await cache.GetOrAddAsync(
            "b", static (key, _) => ValueTask.FromResult($"compiled:{key}")));
        Assert.Equal("compiled:a", await cache.GetOrAddAsync(
            "a", static (key, _) => ValueTask.FromResult($"unexpected:{key}")));
        Assert.Equal("compiled:c", await cache.GetOrAddAsync(
            "c", static (key, _) => ValueTask.FromResult($"compiled:{key}")));
        Assert.Equal(2, cache.Count);
        Assert.Equal("recompiled:b", await cache.GetOrAddAsync(
            "b", static (key, _) => ValueTask.FromResult($"recompiled:{key}")));
    }
}
