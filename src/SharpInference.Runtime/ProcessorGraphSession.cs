using SharpInference.Graphs;

namespace SharpInference.Runtime;

public sealed class ProcessorGraphSession : ITensorProcessorSession
{
    private readonly Processor owner;
    private readonly VmTensorGraphSession session;

    internal ProcessorGraphSession(Processor owner, VmTensorGraphSession session)
    {
        this.owner = owner;
        this.session = session;
    }

    public IProcessor Processor => owner;

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

    public void Reset()
    {
        owner.ThrowIfDisposed();
        session.Reset();
    }

    public void Dispose() => session.Dispose();
}
