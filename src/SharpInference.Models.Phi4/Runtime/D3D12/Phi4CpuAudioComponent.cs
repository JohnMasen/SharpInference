using System.Runtime.InteropServices;
using SharpInference.Backends.CpuVm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Phi4.Cpu;
using SharpInference.Runtime;
using SharpInference.Vm;

namespace SharpInference.Architectures.Phi4.D3D12;

public sealed class Phi4CpuAudioComponent : IDisposable
{
    private const int TextWidth = 3072;
    private readonly object gate = new();
    private readonly HostMemoryStorageAdapter storage;
    private readonly VmResourceManager resources;
    private readonly VmBindings bindings;
    private readonly IVmExecutable executor;
    private readonly int outputSlot;
    private bool disposed;

    public Phi4CpuAudioComponent(
        Phi4ModelPackage package,
        HostMemoryStorageAdapter storage,
        int maximumFrames = 352)
    {
        ArgumentNullException.ThrowIfNull(package);
        this.storage = storage ?? throw new ArgumentNullException(nameof(storage));
        FrameBucket = Phi4D3D12AudioComponent.GetFrameBucket(maximumFrames);
        TokenBucket = checked((FrameBucket + 7) / 8);
        var build = Phi4D3D12AudioComponent.Build(package.Omni, FrameBucket, VmTarget.Cpu);
        var initializers = build.Initializers.ToDictionary(
            initializer => initializer.Slot,
            initializer => initializer.Bytes,
            StringComparer.Ordinal);
        resources = new VmResourceManager(
            (slot, target) =>
            {
                if (!initializers.TryGetValue(slot.Id, out var bytes))
                    throw new InvalidDataException(
                        $"No initializer exists for Phi-4 Audio slot '{slot.Id}'.");
                target.Write(0, bytes);
            });
        try
        {
            bindings = resources.CreateBindings(build.Program);
            executor = new CpuVmCompiler(
                SharpInference.Runtime.Cpu.CpuInstructionCollections.Create()
                    .Append<IInstructionCollectionProvider>(
                        new Phi4CpuAudioInstructionCollection()))
                .Compile(build.Program)
                .CreateExecutor();
            outputSlot = build.Program.Slots.Select((slot, index) => (slot, index))
                .Single(value => value.slot.Id == "projected").index;
        }
        catch
        {
            bindings?.Dispose();
            resources.Dispose();
            throw;
        }
    }

    public int FrameBucket { get; }
    public int TokenBucket { get; }
    public StorageDomain Domain => storage.Domain;

    public ComponentPortDescriptor ProjectedEmbeddingsPort() =>
        new(
            "phi4.audio.projected_embeddings",
            Phi4D3D12AudioComponent.PortAbiVersion,
            new(GraphElementType.Float32, [TokenBucket, TextWidth], "token-major"),
            GraphResourceAccess.Write,
            GraphResourceLifetime.Invocation,
            [Domain]);

    public IStorageLease Encode(
        Phi4AudioFeatures features,
        Phi4AudioProjector projector,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(features);
        cancellationToken.ThrowIfCancellationRequested();
        if (features.FrameCount <= 0 || features.FrameCount > FrameBucket ||
            features.FeatureCount != 80 ||
            features.Values.Length != checked(features.FrameCount * features.FeatureCount))
            throw new ArgumentException("Audio features do not fit the configured CPU VM bucket.", nameof(features));
        if (features.EmbedSize != checked((features.FrameCount + 7) / 8))
            throw new ArgumentException("Audio embedding count does not match subsampling.", nameof(features));
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            using var execution = bindings.BeginExecution();
            var buffers = execution.GetBuffers();
            buffers[SlotIndex("audio_features")].AsSpan().Clear();
            MemoryMarshal.AsBytes(features.Values.AsSpan())
                .CopyTo(buffers[SlotIndex("audio_features")]);
            MemoryMarshal.Write(buffers[SlotIndex("frame_count")], features.FrameCount);
            executor.Execute(
                projector == Phi4AudioProjector.Vision ? "vision" : "speech",
                buffers);
            return storage.CreateLease(
                ProjectedEmbeddingsPort().Tensor,
                buffers[outputSlot]);
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return;
            executor.Dispose();
            bindings.Dispose();
            resources.Dispose();
            disposed = true;
        }
    }

    private int SlotIndex(string id) =>
        bindings.Program.Slots.Select((slot, index) => (slot, index))
            .Single(value => value.slot.Id == id).index;
}
