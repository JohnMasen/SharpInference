using System.Runtime.InteropServices;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Gguf;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime;
using SharpInference.Vm;

namespace SharpInference.Architectures.Phi4.D3D12;

/// <summary>Pairs a resident vision embedding lease with the component that produced it.</summary>
/// <param name="Component">The vision component that owns the embedding.</param>
/// <param name="Lease">The lease for the resident projected image embeddings.</param>
public sealed record Phi4D3D12VisionEmbedding(
    Phi4D3D12VisionComponent Component,
    IStorageLease Lease);

/// <summary>Pairs a resident audio embedding lease with the component that produced it.</summary>
/// <param name="Component">The audio component that owns the embedding.</param>
/// <param name="Lease">The lease for the resident projected audio embeddings.</param>
public sealed record Phi4D3D12AudioEmbedding(
    Phi4D3D12AudioComponent Component,
    IStorageLease Lease);

/// <summary>Identifies a logical VM media embedding resource and the number of valid token rows it contains.</summary>
/// <param name="Resource">The logical VM reference to the resident embedding buffer.</param>
/// <param name="ValidTokenCount">The number of valid token rows, excluding bucket padding.</param>
public sealed record Phi4ResourceEmbedding(
    VmDataReference Resource,
    int ValidTokenCount);

/// <summary>Describes a resident prompt embedding sequence assembled from text and media.</summary>
public interface IPhi4FusedEmbeddingHandle : IStorageHandle
{
    /// <summary>Gets the number of token rows in the fused prompt embedding sequence.</summary>
    int TokenCount { get; }
}

/// <summary>Combines text token embeddings and resident media embeddings into a Direct3D 12 prompt buffer.</summary>
public sealed class Phi4D3D12FusionComponent : IDisposable
{
    /// <summary>The ABI version of this component's exposed tensor ports.</summary>
    public const int PortAbiVersion = 1;
    private const int EmbeddingSize = 3072;
    private const int VocabularySize = 200064;
    private readonly object gate = new();
    private readonly Guid componentId = Guid.NewGuid();
    private readonly D3D12VmResourcePool pool;
    private readonly VmResourceManager manager;
    private readonly GgufModelTensor tokenEmbedding;
    private bool tokenEmbeddingUploaded;
    private bool disposed;

    /// <summary>Initializes embedding fusion using the package token weights and the supplied GPU resource pool and storage domain. Uses the supplied shared GPU resource pool.</summary>
    public Phi4D3D12FusionComponent(
        Phi4ModelPackage package,
        D3D12VmResourcePool pool,
        StorageDomain domain)
    {
        ArgumentNullException.ThrowIfNull(package);
        this.pool = pool ?? throw new ArgumentNullException(nameof(pool));
        Domain = domain ?? throw new ArgumentNullException(nameof(domain));
        tokenEmbedding = package.Text.GetTensor("token_embd.weight");
        manager = new VmResourceManager(InitializeGlobal, pool.Allocate);
    }

    /// <summary>Gets the storage domain in which this component allocates its tensors.</summary>
    public StorageDomain Domain { get; }

    /// <summary>Returns the FP32 token-major fused prompt output port descriptor for the specified token count.</summary>
    public ComponentPortDescriptor FusedEmbeddingsPort(int tokenCount) =>
        new(
            "phi4.fused_embeddings",
            PortAbiVersion,
            new(GraphElementType.Float32, [tokenCount, EmbeddingSize], "token-major"),
            GraphResourceAccess.Write,
            GraphResourceLifetime.Invocation,
            [Domain]);

    /// <summary>Replaces expanded media placeholders with resident image and audio embeddings and gathers text token embeddings into one GPU buffer.</summary>
    public IStorageLease Fuse(
        IReadOnlyList<int> tokenIds,
        IReadOnlyList<Phi4D3D12VisionEmbedding>? images = null,
        IReadOnlyList<Phi4D3D12AudioEmbedding>? audios = null)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (tokenIds.Count == 0)
            throw new ArgumentException("At least one token is required.", nameof(tokenIds));
        images ??= [];
        audios ??= [];
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var plan = BuildPlan(tokenIds, images, audios);
            return FuseCore(tokenIds, plan);
        }
    }

    /// <summary>Replaces expanded media placeholders with resident image and audio embeddings and gathers text token embeddings into one GPU buffer.</summary>
    public IStorageLease Fuse(
        IReadOnlyList<int> tokenIds,
        LogicalVmResourceManager resources,
        IReadOnlyList<Phi4ResourceEmbedding>? images = null,
        IReadOnlyList<Phi4ResourceEmbedding>? audios = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(resources);
        if (tokenIds.Count == 0)
            throw new ArgumentException("At least one token is required.", nameof(tokenIds));
        images ??= [];
        audios ??= [];
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var plan = BuildPlan(
                tokenIds, resources, images, audios, cancellationToken);
            return FuseCore(tokenIds, plan);
        }
    }

    private IStorageLease FuseCore(
        IReadOnlyList<int> tokenIds,
        FusionPlan plan)
    {
            var build = BuildProgram(tokenIds.Count, plan);
            using var bindings = manager.CreateBindings(build.Program);
            foreach (var media in plan.Images.Concat(plan.Audios))
                media.Bind(bindings);
            var outputSlot = build.Program.Slots.Single(slot => slot.Id == "fused_embeddings");
            var outputResource = new VmResource(
                outputSlot.Tensor,
                outputSlot.Scope,
                VmAccess.ReadWrite,
                pool.Allocate(outputSlot));
            using var outputLease = outputResource.Acquire();
            outputResource.Dispose();
            bindings.Bind(outputSlot.Id, outputLease);

            var executor = new D3D12VmCompiler(SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create())
                .Compile(build.Program)
                .CreateExecutor(pool);
            try
            {
                using (var execution = bindings.BeginExecution())
                {
                    execution.Write(
                        "token_ids",
                        0,
                        MemoryMarshal.AsBytes(tokenIds.ToArray().AsSpan()));
                    var buffers = execution.GetBuffers();
                    var uploads = new List<int>
                    {
                        build.TokenIdsSlot,
                        build.OutputSlot,
                    };
                    if (!tokenEmbeddingUploaded)
                        uploads.Add(build.TokenEmbeddingSlot);
                    executor.UploadSlots(buffers, uploads);
                    executor.InitializeSlots(
                        buffers,
                        plan.Images.Concat(plan.Audios).Select(media =>
                            build.Program.Slots.ToList().FindIndex(slot => slot.Id == media.Slot)));
                    tokenEmbeddingUploaded = true;
                    executor.Execute("fuse");
                }

                var retained = outputLease.Retain();
                return StorageLease.Create(
                    new FusionHandle(
                        componentId,
                        Domain,
                        FusedEmbeddingsPort(tokenIds.Count).Tensor,
                        retained,
                        executor,
                        tokenIds.Count),
                    static handle => ((FusionHandle)handle).Dispose());
            }
            catch
            {
                executor.Dispose();
                throw;
            }
    }

    /// <summary>Downloads the component's resident embedding output into a host FP32 array.</summary>
    public float[] Readback(IStorageLease output)
    {
        lock (gate)
        {
            var handle = RequireOutput(output);
            return MemoryMarshal.Cast<byte, float>(
                handle.Executor.Readback("fused_embeddings")).ToArray();
        }
    }

    /// <summary>Binds a compatible resident output lease to the specified VM resource slot.</summary>
    public void BindOutput(
        VmBindings bindings,
        string slotId,
        IStorageLease output)
    {
        ArgumentNullException.ThrowIfNull(bindings);
        ArgumentException.ThrowIfNullOrWhiteSpace(slotId);
        lock (gate)
            RequireOutput(output).Bind(bindings, slotId);
    }

    /// <summary>Creates a VM executor for the artifact using this component's GPU device and resource pool.</summary>
    public D3D12VmExecutor CreateExecutor(D3D12VmArtifact artifact)
    {
        ArgumentNullException.ThrowIfNull(artifact);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return artifact.CreateExecutor(pool);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (gate)
        {
            if (disposed)
                return;
            manager.Dispose();
            disposed = true;
        }
    }

    private FusionHandle RequireOutput(IStorageLease output)
    {
        ArgumentNullException.ThrowIfNull(output);
        ObjectDisposedException.ThrowIf(disposed, this);
        return output.Handle is FusionHandle handle &&
            handle.ComponentId == componentId &&
            handle.Domain == Domain
            ? handle
            : throw new ArgumentException(
                "The fused output was not created by this component.", nameof(output));
    }

    private void InitializeGlobal(VmSlot slot, IVmStorage storage)
    {
        if (slot.Id != "token_embedding")
            throw new InvalidDataException($"Unexpected fusion global '{slot.Id}'.");
        storage.Write(0, tokenEmbedding.Bytes);
    }

    private FusionPlan BuildPlan(
        IReadOnlyList<int> tokenIds,
        IReadOnlyList<Phi4D3D12VisionEmbedding> images,
        IReadOnlyList<Phi4D3D12AudioEmbedding> audios)
    {
        var imagePlans = new List<MediaPlan>();
        var audioPlans = new List<MediaPlan>();
        var position = 0;
        var imageIndex = 0;
        var audioIndex = 0;
        while (position < tokenIds.Count)
        {
            if (tokenIds[position] == Phi4Tokenizer.ImageTokenId)
            {
                if (imageIndex >= images.Count)
                    throw new InvalidDataException(
                        "The token sequence has more image placeholders than image embeddings.");
                var source = images[imageIndex++];
                var rows = checked((int)source.Lease.Handle.Descriptor.Dimensions[0]);
                RequirePlaceholderRun(tokenIds, position, rows, Phi4Tokenizer.ImageTokenId, "image");
                var slot = $"image_embeddings_{imagePlans.Count}";
                imagePlans.Add(new(
                    slot,
                    position,
                    rows,
                    source.Lease.Handle.Descriptor,
                    bindings => source.Component.BindOutput(bindings, slot, source.Lease)));
                position += rows;
            }
            else if (tokenIds[position] == Phi4Tokenizer.AudioTokenId)
            {
                if (audioIndex >= audios.Count)
                    throw new InvalidDataException(
                        "The token sequence has more audio placeholders than audio embeddings.");
                var source = audios[audioIndex++];
                if (source.Lease.Handle is not IPhi4AudioEmbeddingHandle handle)
                    throw new InvalidDataException("Audio output does not expose its valid token count.");
                RequirePlaceholderRun(
                    tokenIds, position, handle.ValidTokenCount,
                    Phi4Tokenizer.AudioTokenId, "audio");
                var slot = $"audio_embeddings_{audioPlans.Count}";
                audioPlans.Add(new(
                    slot,
                    position,
                    handle.ValidTokenCount,
                    source.Lease.Handle.Descriptor,
                    bindings => source.Component.BindOutput(bindings, slot, source.Lease)));
                position += handle.ValidTokenCount;
            }
            else
            {
                if ((uint)tokenIds[position] >= VocabularySize)
                    throw new ArgumentOutOfRangeException(
                        nameof(tokenIds), $"Token {tokenIds[position]} is outside the vocabulary.");
                position++;
            }
        }
        if (imageIndex != images.Count || audioIndex != audios.Count)
            throw new InvalidDataException(
                "The number of media embeddings does not match the token placeholders.");
        return new(imagePlans, audioPlans);
    }

    private FusionPlan BuildPlan(
        IReadOnlyList<int> tokenIds,
        LogicalVmResourceManager resources,
        IReadOnlyList<Phi4ResourceEmbedding> images,
        IReadOnlyList<Phi4ResourceEmbedding> audios,
        CancellationToken cancellationToken)
    {
        var imagePlans = new List<MediaPlan>();
        var audioPlans = new List<MediaPlan>();
        var position = 0;
        var imageIndex = 0;
        var audioIndex = 0;
        while (position < tokenIds.Count)
        {
            if (tokenIds[position] == Phi4Tokenizer.ImageTokenId)
            {
                if (imageIndex >= images.Count)
                    throw new InvalidDataException(
                        "The token sequence has more image placeholders than image embeddings.");
                var source = images[imageIndex++];
                var rows = source.ValidTokenCount;
                RequirePlaceholderRun(tokenIds, position, rows, Phi4Tokenizer.ImageTokenId, "image");
                imagePlans.Add(ResourcePlan(
                    $"image_embeddings_{imagePlans.Count}", position, rows, source.Resource));
                position += rows;
            }
            else if (tokenIds[position] == Phi4Tokenizer.AudioTokenId)
            {
                if (audioIndex >= audios.Count)
                    throw new InvalidDataException(
                        "The token sequence has more audio placeholders than audio embeddings.");
                var source = audios[audioIndex++];
                var rows = source.ValidTokenCount;
                RequirePlaceholderRun(tokenIds, position, rows, Phi4Tokenizer.AudioTokenId, "audio");
                audioPlans.Add(ResourcePlan(
                    $"audio_embeddings_{audioPlans.Count}", position, rows, source.Resource));
                position += rows;
            }
            else
            {
                if ((uint)tokenIds[position] >= VocabularySize)
                    throw new ArgumentOutOfRangeException(
                        nameof(tokenIds), $"Token {tokenIds[position]} is outside the vocabulary.");
                position++;
            }
        }
        if (imageIndex != images.Count || audioIndex != audios.Count)
            throw new InvalidDataException(
                "The number of media embeddings does not match the token placeholders.");
        return new(imagePlans, audioPlans);

        MediaPlan ResourcePlan(
            string slot,
            int outputStart,
            int rows,
            VmDataReference source)
        {
            var target = new ComponentPortDescriptor(
                source.Port.SemanticName,
                source.Port.AbiVersion,
                source.Port.Tensor,
                GraphResourceAccess.Read,
                GraphResourceLifetime.Invocation,
                [Domain]);
            return new(
                slot,
                outputStart,
                rows,
                source.Port.Tensor,
                bindings => resources.BindAsync(
                    source,
                    Domain,
                    target,
                    bindings,
                    slot,
                    cancellationToken).AsTask().GetAwaiter().GetResult());
        }
    }

    private static void RequirePlaceholderRun(
        IReadOnlyList<int> tokenIds,
        int start,
        int count,
        int token,
        string modality)
    {
        if (count <= 0 || start > tokenIds.Count - count ||
            Enumerable.Range(start, count).Any(index => tokenIds[index] != token))
        {
            throw new InvalidDataException(
                $"The {modality} placeholder run does not match its projected embedding count.");
        }
    }

    private static FusionBuild BuildProgram(int tokenCount, FusionPlan plan)
    {
        static VmThreadGroup Grid(int elements)
        {
            var groups = checked((uint)((elements + 63) / 64));
            var x = Math.Min(groups, 65535u);
            var y = checked((groups + x - 1) / x);
            return new(x, y);
        }

        var slots = new List<VmSlot>
        {
            new("token_embedding", VmSlotScope.Global, VmAccess.ReadOnly,
                new(VmElementType.Float16, [VocabularySize, EmbeddingSize]),
                "phi4.token_embedding"),
            new("token_ids", VmSlotScope.Local, VmAccess.ReadWrite,
                new(VmElementType.Int32, [tokenCount])),
            new("fused_embeddings", VmSlotScope.Local, VmAccess.ReadWrite,
                new(VmElementType.Float32, [tokenCount, EmbeddingSize])),
        };
        foreach (var image in plan.Images)
            slots.Add(new(
                image.Slot,
                VmSlotScope.Local,
                VmAccess.ReadOnly,
                Tensor(image.Descriptor)));
        foreach (var audio in plan.Audios)
            slots.Add(new(
                audio.Slot,
                VmSlotScope.Local,
                VmAccess.ReadOnly,
                Tensor(audio.Descriptor)));

        var definitions = new List<VmDefinition>();
        var nodes = new List<VmNode>();
        var gatherParameters = new[]
        {
            new VmParameter("table", VmAccess.ReadOnly, slots[0].Tensor),
            new VmParameter("indices", VmAccess.ReadOnly, slots[1].Tensor),
            new VmParameter("output", VmAccess.ReadWrite, slots[2].Tensor),
        };
        var gather = new VmDefinition(
            "gather_embeddings",
            VmDefinitionKind.Kernel,
            gatherParameters,
            [new("body", Operator(
                "transformer.embedding-gather-batch", gatherParameters))],
            new(64));
        definitions.Add(gather);
        nodes.Add(new(
            "gather",
            new VmDispatch(
                gather.Id,
                [
                    new("table", "token_embedding"),
                    new("indices", "token_ids"),
                    new("output", "fused_embeddings"),
                ],
                Grid(checked(tokenCount * EmbeddingSize)))));
        nodes.Add(new("gather_barrier", new VmBarrier(["fused_embeddings"]), ["gather"]));

        var nodeIndex = 0;
        foreach (var media in plan.Images.Concat(plan.Audios))
        {
            var input = slots.Single(slot => slot.Id == media.Slot);
            var parameters = new[]
            {
                new VmParameter("input", VmAccess.ReadOnly, input.Tensor),
                new VmParameter("output", VmAccess.ReadWrite, slots[2].Tensor),
            };
            var kernel = new VmDefinition(
                $"copy_media_{nodeIndex}",
                VmDefinitionKind.Kernel,
                parameters,
                [new("body", Operator(
                    "transformer.copy-embedding-rows",
                    parameters,
                    new Dictionary<string, string>
                    {
                        ["rows"] = media.Rows.ToString(
                            System.Globalization.CultureInfo.InvariantCulture),
                        ["output_start"] = media.OutputStart.ToString(
                            System.Globalization.CultureInfo.InvariantCulture),
                    }))],
                new(64));
            definitions.Add(kernel);
            var dispatchId = $"copy_{nodeIndex}";
            nodes.Add(new(
                dispatchId,
                new VmDispatch(
                    kernel.Id,
                    [
                        new("input", media.Slot),
                        new("output", "fused_embeddings"),
                    ],
                    Grid(checked(media.Rows * EmbeddingSize)))));
            nodes.Add(new(
                $"copy_barrier_{nodeIndex}",
                new VmBarrier(["fused_embeddings"]),
                [dispatchId]));
            nodeIndex++;
        }

        var rootParameters = slots.Select(
            slot => new VmParameter(slot.Id, slot.Access, slot.Tensor)).ToArray();
        var fuse = new VmDefinition(
            "fuse_definition",
            VmDefinitionKind.Orchestration,
            rootParameters,
            nodes);
        definitions.Add(fuse);
        var arguments = slots.Select(slot => new VmArgument(slot.Id, slot.Id)).ToArray();
        var program = new VmProgram(
            $"phi4-fusion-{tokenCount}",
            "phi4.multimodal.fusion",
            VmTarget.Direct3D12,
            slots,
            definitions,
            [new("fuse", fuse.Id, arguments)],
            new("none", 1, []));
        return new(
            program,
            slots.FindIndex(slot => slot.Id == "token_embedding"),
            slots.FindIndex(slot => slot.Id == "token_ids"),
            slots.FindIndex(slot => slot.Id == "fused_embeddings"));
    }

    private static VmOperator Operator(
        string name,
        IReadOnlyList<VmParameter> parameters,
        IReadOnlyDictionary<string, string>? attributes = null) =>
        new(
            new(GraphElementType.Float32, GraphElementType.Float32),
            InstructionCollectionIds.TransformerFloat32,
            name,
            parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)),
            attributes);

    private static VmTensor Tensor(TensorDescriptor descriptor) =>
        new(
            descriptor.ElementType switch
            {
                GraphElementType.Float32 => VmElementType.Float32,
                _ => throw new NotSupportedException(
                    $"Fusion does not support {descriptor.ElementType} media embeddings."),
            },
            descriptor.Dimensions.ToArray());

    private sealed record MediaPlan(
        string Slot,
        int OutputStart,
        int Rows,
        TensorDescriptor Descriptor,
        Action<VmBindings> Bind);

    private sealed record FusionPlan(
        IReadOnlyList<MediaPlan> Images,
        IReadOnlyList<MediaPlan> Audios);

    private sealed record FusionBuild(
        VmProgram Program,
        int TokenEmbeddingSlot,
        int TokenIdsSlot,
        int OutputSlot);

    private sealed class FusionHandle(
        Guid componentId,
        StorageDomain domain,
        TensorDescriptor descriptor,
        VmResourceLease resource,
        D3D12VmExecutor executor,
        int tokenCount) : IPhi4FusedEmbeddingHandle, IVmBindableStorageHandle
    {
        public Guid Id { get; } = Guid.NewGuid();
        public Guid ComponentId { get; } = componentId;
        public StorageDomain Domain { get; } = domain;
        public TensorDescriptor Descriptor { get; } = descriptor;
        public long ByteLength { get; } = checked(tokenCount * EmbeddingSize * sizeof(float));
        public int TokenCount { get; } = tokenCount;
        public VmResourceLease Resource { get; } = resource;
        public D3D12VmExecutor Executor { get; } = executor;

        public void Bind(VmBindings bindings, string slotId)
        {
            using var retained = Resource.Retain();
            bindings.Bind(slotId, retained);
        }

        public void Dispose()
        {
            Executor.Dispose();
            Resource.Dispose();
        }
    }
}
