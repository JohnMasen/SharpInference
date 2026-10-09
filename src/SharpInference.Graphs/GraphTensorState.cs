namespace SharpInference.Graphs;

public sealed record GraphStateBuffer(string Name, TensorDescriptor Tensor, Memory<byte> Data);

/// <summary>Stores graph state using declared tensor descriptors and architecture-supplied initial bytes.</summary>
public sealed class GraphTensorState : IModelState
{
    private readonly GraphStateBuffer[] buffers;
    private readonly byte[][] initial;

    public GraphTensorState(LogicalGraph graph,
        IReadOnlyDictionary<string, ReadOnlyMemory<byte>>? initialState = null)
        : this(graph?.Identity ?? throw new ArgumentNullException(nameof(graph)), graph.Model,
            graph.GraphState, graph.Resources, initialState)
    {
        GraphValidator.Validate(graph);
    }

    public GraphTensorState(ExecutionGraph graph,
        IReadOnlyDictionary<string, ReadOnlyMemory<byte>>? initialState = null)
        : this(graph?.Identity ?? throw new ArgumentNullException(nameof(graph)), graph.Model,
            graph.GraphState, graph.Resources, initialState)
    {
        GraphValidator.Validate(graph);
    }

    private GraphTensorState(GraphIdentity identity, GraphModelSignature model, GraphState state,
        IReadOnlyList<GraphResource> resources,
        IReadOnlyDictionary<string, ReadOnlyMemory<byte>>? initialState)
    {
        ArchitectureId = identity.ArchitectureId;
        StateAbiId = model.StateAbiId;
        Schema = state.Schema;
        var descriptions = resources.ToDictionary(resource => resource.Id);
        if (resources.Count(resource => resource.Kind == GraphResourceKind.SessionState) != state.Count ||
            initialState is not null && initialState.Count != state.Count)
            throw new InvalidDataException("The state schema and initial buffers must cover every session-state resource.");
        buffers = new GraphStateBuffer[state.Count];
        initial = new byte[state.Count][];
        for (var index = 0; index < state.Count; index++)
        {
            var slot = state[index];
            if (!descriptions.TryGetValue(slot.Resource, out var resource) ||
                resource.Kind != GraphResourceKind.SessionState || resource.Tensor.Layout != "dense")
                throw new InvalidDataException($"State resource '{slot.Resource}' must be a dense session-state tensor.");
            var size = resource.Tensor.ElementType switch
            {
                GraphElementType.Byte => 1,
                GraphElementType.Float16 => 2,
                GraphElementType.Int32 or GraphElementType.UInt32 or GraphElementType.Float32 => 4,
                _ => throw new NotSupportedException($"Unsupported state element type '{resource.Tensor.ElementType}'."),
            };
            var length = resource.Tensor.Dimensions.Aggregate(size, (count, dimension) => checked(count * dimension));
            var bytes = new byte[length];
            if (initialState is not null)
            {
                if (!initialState.TryGetValue(slot.Name, out var supplied) || supplied.Length != length)
                    throw new InvalidDataException($"Initial state '{slot.Name}' does not match its tensor descriptor.");
                supplied.Span.CopyTo(bytes);
            }
            initial[index] = (byte[])bytes.Clone();
            buffers[index] = new GraphStateBuffer(slot.Name, resource.Tensor, bytes);
        }
        Buffers = Array.AsReadOnly(buffers);
    }

    private GraphTensorState(GraphTensorState source)
    {
        ArchitectureId = source.ArchitectureId;
        StateAbiId = source.StateAbiId;
        Schema = source.Schema;
        initial = source.initial.Select(value => (byte[])value.Clone()).ToArray();
        buffers = source.buffers.Select(buffer =>
            new GraphStateBuffer(buffer.Name, buffer.Tensor, buffer.Data.ToArray())).ToArray();
        Buffers = Array.AsReadOnly(buffers);
    }

    public string ArchitectureId { get; }
    public string StateAbiId { get; }
    public StateSchema Schema { get; }
    public IReadOnlyList<GraphStateBuffer> Buffers { get; }

    IModelState IModelState.Clone() => Clone();
    public GraphTensorState Clone() => new(this);

    public void Reset()
    {
        for (var index = 0; index < buffers.Length; index++)
            initial[index].AsSpan().CopyTo(buffers[index].Data.Span);
    }
}
