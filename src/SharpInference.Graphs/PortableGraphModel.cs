namespace SharpInference.Graphs;

/// <summary>Combines model metadata, a tensor catalog, and graph binding metadata into a portable model.</summary>
/// <param name="metadata">The model metadata.</param>
/// <param name="tensors">The model tensor catalog.</param>
/// <param name="graph">The graph supplying operations and resource binding metadata.</param>
public sealed class PortableGraphModel(
    ModelMetadata metadata,
    IModelTensorCatalog tensors,
    IGraphBindingMetadata graph) : IModel
{
    /// <summary>Gets the model metadata.</summary>
    public ModelMetadata ModelMetadata { get; } = metadata ?? throw new ArgumentNullException(nameof(metadata));
    /// <summary>Gets the model tensor catalog.</summary>
    public IModelTensorCatalog Tensors { get; } = tensors ?? throw new ArgumentNullException(nameof(tensors));
    /// <summary>Gets the graph represented by this model or plan.</summary>
    public IGraphBindingMetadata Graph { get; } = graph ?? throw new ArgumentNullException(nameof(graph));
}

/// <summary>Stores named dense FP32 session state and synchronizes device modifications on demand.</summary>
public sealed class PortableGraphState : INamedFloat32ModelState
{
    private readonly Float32StateView[] views;
    private Action<IReadOnlyList<Float32StateView>>? synchronizeFromDevice;
    private bool deviceModified;

    /// <summary>Allocates zero-initialized dense FP32 state views covering every session-state resource.</summary>
    /// <param name="graph">The graph supplying operations and resource binding metadata.</param>
    public PortableGraphState(IGraphBindingMetadata graph)
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

    /// <summary>Gets the model architecture identifier.</summary>
    public string ArchitectureId { get; }
    /// <summary>Gets the total number of FP32 state elements.</summary>
    public int ElementCount { get; }
    /// <summary>Gets named FP32 host state views after synchronizing pending device changes.</summary>
    public IReadOnlyList<Float32StateView> Views
    {
        get
        {
            Synchronize();
            return views;
        }
    }
    /// <summary>Gets the revision advanced when host state changes are committed.</summary>
    public long Revision { get; private set; }

    IModelState IModelState.Clone() => Clone();

    /// <summary>Creates an independent copy of the graph state.</summary>
    public PortableGraphState Clone()
    {
        Synchronize();
        return new PortableGraphState(ArchitectureId,
            views.Select(view => new Float32StateView(view.Name, view.Dimensions.ToArray(),
                (float[])view.Values.Clone())).ToArray());
    }

    /// <summary>Attaches the callback used to copy pending device state changes to host views.</summary>
    /// <param name="synchronize">The device-to-host synchronization callback or whether pending device writes should be synchronized.</param>
    public void AttachDeviceSynchronizer(Action<IReadOnlyList<Float32StateView>> synchronize)
    {
        ArgumentNullException.ThrowIfNull(synchronize);
        if (synchronizeFromDevice is not null)
            throw new InvalidOperationException("The graph state is already attached to a device session.");
        synchronizeFromDevice = synchronize;
    }

    /// <summary>Detaches the device synchronization callback, optionally copying pending device changes to host state first.</summary>
    /// <param name="synchronize">Whether to synchronize pending device changes before detaching.</param>
    public void DetachDeviceSynchronizer(bool synchronize = true)
    {
        if (deviceModified && synchronize)
            Synchronize();
        deviceModified = false;
        synchronizeFromDevice = null;
    }

    /// <summary>Marks host state views as stale after an attached device session modifies the state.</summary>
    public void MarkDeviceModified()
    {
        if (synchronizeFromDevice is null)
            throw new InvalidOperationException("The graph state has no attached device session.");
        deviceModified = true;
    }

    /// <summary>Clears all FP32 state values to zero and commits the host changes.</summary>
    public void Reset()
    {
        foreach (var view in views)
            Array.Clear(view.Values);
        CommitViews();
    }

    /// <summary>Copies synchronized state values into a flat destination in state-entry order.</summary>
    /// <param name="destination">The destination receiving the flattened state values.</param>
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

    /// <summary>Restores flattened state values in state-entry order and commits the host changes.</summary>
    /// <param name="source">The flattened state values in state-entry order.</param>
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

    /// <summary>Commits host view changes, clears pending device changes, and advances the state revision.</summary>
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
