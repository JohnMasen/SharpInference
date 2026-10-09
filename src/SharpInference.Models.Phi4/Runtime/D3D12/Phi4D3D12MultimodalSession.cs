using SharpInference.Backends.D3D12Vm;
using SharpInference.Runtime;
using Vortice.Direct3D12;

namespace SharpInference.Architectures.Phi4.D3D12;

public sealed record Phi4D3D12GenerationResult(
    IReadOnlyList<int> InputTokenIds,
    IReadOnlyList<int> GeneratedTokenIds,
    string Text,
    ulong DecoderSubmissions);

public enum Phi4AudioExecutionBackend
{
    D3D12,
    Cpu,
}

public sealed class Phi4D3D12MultimodalSession : IDisposable
{
    private readonly object gate = new();
    private readonly Phi4ModelPackage package;
    private readonly ID3D12Device device;
    private readonly D3D12VmResourcePool pool;
    private readonly Phi4D3D12VisionComponent vision;
    private readonly Phi4D3D12AudioComponent audio;
    private readonly Phi4D3D12FusionComponent fusion;
    private readonly HostMemoryStorageAdapter hostStorage;
    private readonly Phi4CpuAudioComponent? cpuAudio;
    private readonly Phi4AudioExecutionBackend audioBackend;
    private readonly StorageDomain domain;
    private readonly int maximumContext;
    private bool disposed;

    public Phi4D3D12MultimodalSession(
        Phi4ModelPackage package,
        int maximumContext = 4096,
        int maximumAudioFrames = 4096,
        int adapterIndex = 0,
        Phi4AudioExecutionBackend audioBackend = Phi4AudioExecutionBackend.D3D12)
    {
        this.package = package ?? throw new ArgumentNullException(nameof(package));
        if (maximumContext <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumContext));
        if (!Enum.IsDefined(audioBackend))
            throw new ArgumentOutOfRangeException(nameof(audioBackend));
        this.maximumContext = maximumContext;
        this.audioBackend = audioBackend;
        hostStorage = new(new StorageDomain("cpu", "0", "phi4-host"));
        (device, _) = D3D12VmDeviceFactory.Create(adapterIndex);
        D3D12VmResourcePool? createdPool = null;
        Phi4D3D12VisionComponent? createdVision = null;
        Phi4D3D12AudioComponent? createdAudio = null;
        Phi4D3D12FusionComponent? createdFusion = null;
        try
        {
            createdPool = new D3D12VmResourcePool(device);
            domain = new StorageDomain(
                "d3d12",
                D3D12VmDeviceFactory.HardwareIdentity(adapterIndex),
                "phi4-multimodal");
            createdVision = new Phi4D3D12VisionComponent(package, createdPool, domain);
            createdAudio = new Phi4D3D12AudioComponent(
                package, createdPool, domain, maximumAudioFrames);
            createdFusion = new Phi4D3D12FusionComponent(package, createdPool, domain);
            pool = createdPool;
            vision = createdVision;
            audio = createdAudio;
            fusion = createdFusion;
            cpuAudio = audioBackend == Phi4AudioExecutionBackend.Cpu
                ? new Phi4CpuAudioComponent(package, hostStorage, maximumAudioFrames)
                : null;
        }
        catch
        {
            createdFusion?.Dispose();
            createdAudio?.Dispose();
            createdVision?.Dispose();
            createdPool?.Dispose();
            device.Dispose();
            throw;
        }
    }

    public Phi4D3D12GenerationResult Generate(
        Phi4PreparedInput input,
        int maximumNewTokens,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (maximumNewTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumNewTokens));
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (input.TokenIds.Length > maximumContext)
                throw new ArgumentException(
                    "The prepared prompt exceeds the GPU context.", nameof(input));
            cancellationToken.ThrowIfCancellationRequested();

            var imageTasks = input.Images.Select(image => Task.Run(
                () => vision.Encode(image), cancellationToken)).ToArray();
            var projector = input.AudioProjectionMode == "vision"
                ? Phi4AudioProjector.Vision
                : input.AudioProjectionMode == "speech"
                    ? Phi4AudioProjector.Speech
                    : throw new ArgumentException(
                        $"Unknown audio projection mode '{input.AudioProjectionMode}'.",
                        nameof(input));
            var d3dAdapter = new Phi4D3D12StorageAdapter(pool, domain);
            var resourceManager = new LogicalVmResourceManager(
                new HostStagingStorageTransferService(
                    [hostStorage, d3dAdapter],
                    new ArrayHostStagingAllocator()));
            var audioTasks = input.Audios.Select(features => Task.Run(
                () => EncodeAudio(
                    features,
                    projector,
                    input.AudioProjectionMode,
                    cancellationToken),
                cancellationToken)).ToArray();
            var mediaTasks = imageTasks.Cast<Task>().Concat(audioTasks).ToArray();
            try
            {
                Task.WhenAll(mediaTasks).GetAwaiter().GetResult();
            }
            catch
            {
                DisposeCompleted(imageTasks);
                DisposeCompleted(audioTasks);
                resourceManager.DisposeAsync().AsTask().GetAwaiter().GetResult();
                throw;
            }

            var imageLeases = imageTasks.Select(task => task.Result).ToArray();
            var audioResults = audioTasks.Select(task => task.Result).ToArray();
            var audioLeases = audioResults.Select(result => result.Lease).ToArray();
            try
            {
                var images = imageLeases.Select((lease, index) =>
                {
                    var port = vision.ProjectedEmbeddingsPort(
                        Phi4VisionBucket.FromFeatures(input.Images[index]));
                    var value = resourceManager.Publish(port, lease);
                    return new Phi4ResourceEmbedding(
                        value,
                        checked((int)port.Tensor.Dimensions[0]));
                }).ToArray();
                var audios = audioResults.Select(result =>
                {
                    var value = resourceManager.Publish(
                        result.Port,
                        result.Lease,
                        new TensorView([result.ValidTokenCount, 3072]));
                    return new Phi4ResourceEmbedding(value, result.ValidTokenCount);
                }).ToArray();
                using var fused = fusion.Fuse(
                    input.TokenIds,
                    resourceManager,
                    images,
                    audios,
                    cancellationToken);
                var fusedValue = resourceManager.Publish(
                    fusion.FusedEmbeddingsPort(input.TokenIds.Length),
                    fused);
                using var decoder = new Phi4D3D12TextSession(
                    package,
                    pool,
                    domain,
                    resourceManager,
                    fusedValue,
                    maximumContext,
                    input.Adapter);
                var inferenceSubmissions = decoder.QueueSubmissionCount;
                cancellationToken.ThrowIfCancellationRequested();
                var nextToken = decoder.PrefillFusedEmbeddings();

                var generated = new List<int>(maximumNewTokens);
                for (var index = 0; index < maximumNewTokens; index++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (nextToken == Phi4Tokenizer.EndOfTextTokenId)
                        break;
                    generated.Add(nextToken);
                    nextToken = decoder.ForwardToken(nextToken);
                }
                return new(
                    input.TokenIds,
                    generated,
                    package.Tokenizer.Decode(generated),
                    decoder.QueueSubmissionCount - inferenceSubmissions);
            }
            finally
            {
                resourceManager.DisposeAsync().AsTask().GetAwaiter().GetResult();
                foreach (var lease in audioLeases)
                    lease.Dispose();
                foreach (var lease in imageLeases)
                    lease.Dispose();
            }
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed)
                return;
            fusion.Dispose();
            cpuAudio?.Dispose();
            audio.Dispose();
            vision.Dispose();
            pool.Dispose();
            device.Dispose();
            disposed = true;
        }
    }

    private static void DisposeCompleted<T>(IEnumerable<Task<T>> tasks)
    {
        foreach (var task in tasks)
            if (task.IsCompletedSuccessfully)
            {
                if (task.Result is IStorageLease lease)
                    lease.Dispose();
                else if (task.Result is AudioEncoding audio)
                    audio.Lease.Dispose();
            }
    }

    private AudioEncoding EncodeAudio(
        Phi4AudioFeatures features,
        Phi4AudioProjector projector,
        string projectionMode,
        CancellationToken cancellationToken)
    {
        if (audioBackend == Phi4AudioExecutionBackend.D3D12)
        {
            var gpuLease = audio.EncodeResident(features, projector);
            var handle = gpuLease.Handle as IPhi4AudioEmbeddingHandle ??
                throw new InvalidDataException("D3D12 audio output is missing embedding metadata.");
            return new(gpuLease, handle.ValidTokenCount, audio.ProjectedEmbeddingsPort());
        }

        var cpuLease = cpuAudio!.Encode(features, projector, cancellationToken);
        return new(
            cpuLease,
            features.EmbedSize,
            cpuAudio.ProjectedEmbeddingsPort());
    }

    private sealed record AudioEncoding(
        IStorageLease Lease,
        int ValidTokenCount,
        ComponentPortDescriptor Port);
}
