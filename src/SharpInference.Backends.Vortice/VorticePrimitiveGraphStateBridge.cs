using SharpInference.Graphs;

namespace SharpInference.Backends.Vortice;

/// <summary>
/// Synchronizes a portable host state and one GPU session only at state boundaries.
/// Call BeforeToken before dispatch and AfterToken after a successful dispatch.
/// Dispose before disposing the GPU session to flush a pending state snapshot.
/// </summary>
public sealed class VorticePrimitiveGraphStateBridge : IProcessorStateExecutor, IDisposable
{
    private readonly VorticePrimitiveGraphSession session;
    private readonly PortableGraphState state;
    private readonly ExecutionGraph graph;
    private readonly IReadOnlyDictionary<string, GraphStateEntry> slots;
    private readonly bool writesState;
    private long uploadedRevision = -1;
    private bool disposed;

    public VorticePrimitiveGraphStateBridge(
        VorticePrimitiveGraphSession session,
        PortableGraphState state,
        ExecutionGraph graph)
    {
        this.session = session ?? throw new ArgumentNullException(nameof(session));
        this.state = state ?? throw new ArgumentNullException(nameof(state));
        this.graph = graph ?? throw new ArgumentNullException(nameof(graph));
        if (!ReferenceEquals(session.Graph, graph) ||
            state.ArchitectureId != graph.Identity.ArchitectureId ||
            graph.GraphState.Count != graph.Resources.Count(resource => resource.Kind == GraphResourceKind.SessionState))
            throw new ArgumentException("The host state does not match the graph session resources.", nameof(state));
        slots = graph.GraphState.Slots.ToDictionary(slot => slot.Name, StringComparer.Ordinal);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var views = state.Views;
        if (views.Count != slots.Count || views.Any(view =>
            !slots.TryGetValue(view.Name, out var slot) ||
            !resources[slot.Resource].Tensor.Dimensions.SequenceEqual(view.Dimensions) ||
            resources[slot.Resource].Tensor.Dimensions.Aggregate(1L, (size, dim) => checked(size * dim)) != view.Values.Length))
            throw new ArgumentException("The host state slots do not match the graph dimensions.", nameof(state));
        writesState = graph.Nodes.Any(node => node.Resources.Any(binding =>
            binding.Access != GraphResourceAccess.Read &&
            resources[binding.Resource].Kind == GraphResourceKind.SessionState));
        state.AttachDeviceSynchronizer(SynchronizeFromDevice);
    }

    /// <summary>Uploads only if Reset, Restore, CommitViews, or initial state changed the host revision.</summary>
    public void BeforeToken()
    {
        ThrowIfDisposed();
        if (state.Revision == uploadedRevision) return;
        var views = state.Views.ToDictionary(view => view.Name, StringComparer.Ordinal);
        foreach (var slot in graph.GraphState.Slots)
            session.UploadState(slot.Resource, views[slot.Name].Values);
        uploadedRevision = state.Revision;
    }

    /// <summary>Marks GPU state dirty without reading it back; Clone and state export trigger a lazy flush.</summary>
    public void AfterToken()
    {
        ThrowIfDisposed();
        if (writesState) state.MarkDeviceModified();
    }

    public IReadOnlyList<GraphStateValue> ReadState(GraphState requested)
    {
        ThrowIfDisposed();
        Validate(requested);
        return GraphSessionStateAccess.Read(requested, graph.Resources, state);
    }

    public void WriteState(GraphState requested, IReadOnlyList<GraphStateValue> values)
    {
        ThrowIfDisposed();
        Validate(requested);
        ArgumentNullException.ThrowIfNull(values);
        if (values.Count != requested.Count)
            throw new InvalidDataException("The imported state tensor count does not match the graph.");
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var length = checked((int)graph.GraphState.Slots.Sum(slot =>
            resources[slot.Resource].Tensor.Dimensions.Aggregate(1L, (size, dim) => checked(size * dim))));
        var flattened = new float[length];
        var offset = 0;
        for (var index = 0; index < values.Count; index++)
        {
            var slot = requested[index];
            var value = values[index];
            var descriptor = resources[slot.Resource].Tensor;
            if (value is null || value.Name != slot.Name ||
                !value.Dimensions.SequenceEqual(descriptor.Dimensions) ||
                value.Values.Length != descriptor.Dimensions.Aggregate(1L, (size, dim) => checked(size * dim)))
                throw new InvalidDataException($"Imported state tensor '{slot.Name}' does not match the graph.");
            value.Values.CopyTo(flattened, offset);
            offset += value.Values.Length;
        }
        state.Restore(flattened);
    }

    private void SynchronizeFromDevice(IReadOnlyList<Float32StateView> views)
    {
        foreach (var view in views)
            session.ReadState(slots[view.Name].Resource).CopyTo(view.Values, 0);
    }

    private void Validate(GraphState requested)
    {
        ArgumentNullException.ThrowIfNull(requested);
        if (!requested.Schema.Equals(graph.GraphState.Schema) ||
            !requested.Slots.SequenceEqual(graph.GraphState.Slots))
            throw new InvalidDataException("The requested state schema does not match this GPU session.");
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);

    public void Dispose()
    {
        if (disposed) return;
        state.DetachDeviceSynchronizer();
        disposed = true;
    }
}
