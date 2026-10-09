namespace SharpInference.Graphs;

public sealed class PortableGraphModel(
    ModelMetadata metadata,
    IModelTensorCatalog tensors,
    ExecutionGraph graph) : IModel
{
    public ModelMetadata ModelMetadata { get; } = metadata ?? throw new ArgumentNullException(nameof(metadata));
    public IModelTensorCatalog Tensors { get; } = tensors ?? throw new ArgumentNullException(nameof(tensors));
    public ExecutionGraph Graph { get; } = graph ?? throw new ArgumentNullException(nameof(graph));
}

public sealed class PortableGraphState : INamedFloat32ModelState
{
    private readonly Float32StateView[] views;
    private Action<IReadOnlyList<Float32StateView>>? synchronizeFromDevice;
    private bool deviceModified;

    public PortableGraphState(ExecutionGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArchitectureId = graph.Identity.ArchitectureId;
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var stateResources = graph.Resources.Where(resource =>
            resource.Kind == GraphResourceKind.SessionState).Select(resource => resource.Id).ToHashSet();
        if (stateResources.Count != graph.GraphState.Slots.Count ||
            graph.GraphState.Slots.Any(slot => !stateResources.Contains(slot.Resource)))
            throw new InvalidDataException("The graph state schema must cover every session-state resource.");
        views = graph.GraphState.Slots.Select(slot =>
        {
            var resource = resources[slot.Resource];
            if (resource.Kind != GraphResourceKind.SessionState ||
                resource.Tensor.ElementType != GraphElementType.Float32 ||
                resource.Tensor.Layout != "dense")
                throw new InvalidDataException($"State resource '{slot.Resource}' must be dense FP32.");
            var length = resource.Tensor.Dimensions.Aggregate(1, (count, dimension) => checked(count * dimension));
            return new Float32StateView(slot.Name, resource.Tensor.Dimensions.ToArray(), new float[length]);
        }).ToArray();
        ElementCount = views.Sum(view => view.Values.Length);
    }

    private PortableGraphState(string architectureId, Float32StateView[] views)
    {
        ArchitectureId = architectureId;
        this.views = views;
        ElementCount = views.Sum(view => view.Values.Length);
    }

    public string ArchitectureId { get; }
    public int ElementCount { get; }
    public IReadOnlyList<Float32StateView> Views
    {
        get
        {
            Synchronize();
            return views;
        }
    }
    public long Revision { get; private set; }

    IModelState IModelState.Clone() => Clone();

    public PortableGraphState Clone()
    {
        Synchronize();
        return new PortableGraphState(ArchitectureId,
            views.Select(view => new Float32StateView(view.Name, view.Dimensions.ToArray(),
                (float[])view.Values.Clone())).ToArray());
    }

    public void AttachDeviceSynchronizer(Action<IReadOnlyList<Float32StateView>> synchronize)
    {
        ArgumentNullException.ThrowIfNull(synchronize);
        if (synchronizeFromDevice is not null)
            throw new InvalidOperationException("The graph state is already attached to a device session.");
        synchronizeFromDevice = synchronize;
    }

    public void DetachDeviceSynchronizer(bool synchronize = true)
    {
        if (deviceModified && synchronize)
            Synchronize();
        deviceModified = false;
        synchronizeFromDevice = null;
    }

    public void MarkDeviceModified()
    {
        if (synchronizeFromDevice is null)
            throw new InvalidOperationException("The graph state has no attached device session.");
        deviceModified = true;
    }

    public void Reset()
    {
        foreach (var view in views)
            Array.Clear(view.Values);
        CommitViews();
    }

    public void CopyTo(Span<float> destination)
    {
        if (destination.Length != ElementCount)
            throw new ArgumentException("The destination does not match graph state size.", nameof(destination));
        Synchronize();
        foreach (var view in views)
        {
            view.Values.CopyTo(destination);
            destination = destination[view.Values.Length..];
        }
    }

    public void Restore(ReadOnlySpan<float> source)
    {
        if (source.Length != ElementCount)
            throw new ArgumentException("The source does not match graph state size.", nameof(source));
        foreach (var view in views)
        {
            source[..view.Values.Length].CopyTo(view.Values);
            source = source[view.Values.Length..];
        }
        CommitViews();
    }

    public void CommitViews()
    {
        deviceModified = false;
        Revision++;
    }

    private void Synchronize()
    {
        if (!deviceModified)
            return;
        var synchronize = synchronizeFromDevice ??
            throw new InvalidOperationException("The modified graph state has no attached device session.");
        synchronize(views);
        deviceModified = false;
    }
}
