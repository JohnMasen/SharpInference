using SharpInference.Graphs;

namespace SharpInference.Runtime;

/// <summary>Adapts a processor-owned VM tensor session to resource-keyed graph execution.</summary>
public sealed class ProcessorGraphSession : ITensorProcessorSession
{
    private readonly Processor owner;
    private readonly VmTensorGraphSession session;

    internal ProcessorGraphSession(Processor owner, VmTensorGraphSession session)
    {
        this.owner = owner;
        this.session = session;
    }

    /// <inheritdoc/>
    public IProcessor Processor => owner;

    /// <summary>Executes the graph with descriptor-sized input byte buffers and returns its output buffers.</summary>
    /// <param name="inputs">Input resource identifiers mapped to descriptor-sized tensor byte buffers.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    public IReadOnlyDictionary<ResourceId, byte[]> Execute(
        IReadOnlyDictionary<ResourceId, ReadOnlyMemory<byte>> inputs, CancellationToken cancellationToken = default)
    {
        owner.ThrowIfDisposed();
        return session.Execute(inputs, cancellationToken);
    }

    IReadOnlyDictionary<string, byte[]> ITensorProcessorSession.Execute(
        IReadOnlyDictionary<string, ReadOnlyMemory<byte>> inputs, CancellationToken cancellationToken)
    {
        owner.ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(inputs);
        return session.Execute(inputs.ToDictionary(value => new ResourceId(value.Key), value => value.Value), cancellationToken)
            .ToDictionary(value => value.Key.Value, value => value.Value, StringComparer.Ordinal);
    }

    /// <summary>Resets graph state after verifying that the owning processor is still alive.</summary>
    public void Reset()
    {
        owner.ThrowIfDisposed();
        session.Reset();
    }

    /// <summary>Releases the underlying VM tensor session.</summary>
    public void Dispose() => session.Dispose();
}
