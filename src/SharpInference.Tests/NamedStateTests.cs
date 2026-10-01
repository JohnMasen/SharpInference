using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class NamedStateTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PortableState_ViewsAreMutableAndMaintainGraphSlotOrder(bool rwkv7)
    {
        var model = rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6;
        using var catalog = TestModelLoader.OpenCatalog(model);
        ILogicalGraphProvider provider = rwkv7
            ? new PortableRwkv7GraphProvider() : new PortableRwkv6GraphProvider();
        var graph = new GraphOptimizer().Optimize(provider.Build(catalog));
        var state = new PortableGraphState(graph);
        var slots = graph.GraphState.Slots;
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var views = state.Views;
        Assert.Equal(slots.Count, views.Count);
        for (var index = 0; index < slots.Count; index++)
        {
            Assert.Equal(slots[index].Name, views[index].Name);
            Assert.Equal(resources[slots[index].Resource].Tensor.Dimensions, views[index].Dimensions);
            Assert.Equal(views[index].Dimensions.Aggregate(1, (product, dimension) => product * dimension),
                views[index].Values.Length);
            views[index].Values[0] = index + 1;
        }
        state.CommitViews();
        var snapshot = new float[state.ElementCount];
        state.CopyTo(snapshot);
        var offset = 0;
        for (var index = 0; index < views.Count; index++)
        {
            Assert.Equal(index + 1, snapshot[offset]);
            offset += views[index].Values.Length;
        }

        var cloned = Assert.IsAssignableFrom<INamedRwkvState>(state.Clone());
        Assert.Equal(views.Select(view => view.Name), cloned.Views.Select(view => view.Name));
        Assert.NotSame(views[0].Values, cloned.Views[0].Values);
        (cloned as IDisposable)?.Dispose();
    }
}
