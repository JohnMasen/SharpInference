using System.Runtime.InteropServices;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Gguf;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Phi4;
using SharpInference.Instructions.Phi4.D3D12;
using SharpInference.Runtime;
using SharpInference.Vm;
using Vortice.Direct3D12;

namespace SharpInference.Architectures.Phi4.D3D12;

public readonly record struct Phi4VisionBucket(
    int CropCount,
    int TargetHeight,
    int TargetWidth,
    int UsefulHeight,
    int UsefulWidth,
    int ImageTokenCount)
{
    public static Phi4VisionBucket FromFeatures(Phi4ImageFeatures image)
    {
        ArgumentNullException.ThrowIfNull(image);
        const int grid = Phi4ImageFeatures.MaskSize;
        const int cropSize = Phi4ImageFeatures.CropSize;
        if (image.CropCount <= 1 ||
            image.TargetHeight <= 0 || image.TargetHeight % cropSize != 0 ||
            image.TargetWidth <= 0 || image.TargetWidth % cropSize != 0 ||
            image.PixelValues.Length != checked(image.CropCount * 3 * cropSize * cropSize) ||
            image.AttentionMask.Length != checked(image.CropCount * grid * grid))
            throw new InvalidDataException("Phi-4 image features do not match a fixed HD crop bucket.");

        var horizontalTiles = image.TargetWidth / cropSize;
        var verticalTiles = image.TargetHeight / cropSize;
        if (image.CropCount != checked(horizontalTiles * verticalTiles + 1))
            throw new InvalidDataException("Phi-4 crop count does not match the HD tile grid.");

        var usefulHeight = 0;
        for (var tileY = 0; tileY < verticalTiles; tileY++)
        for (var y = 0; y < grid / 2; y++)
        {
            var crop = 1 + tileY * horizontalTiles;
            if (image.AttentionMask[crop * grid * grid + y * 2 * grid] != 0)
                usefulHeight = tileY * (grid / 2) + y + 1;
        }
        var usefulWidth = 0;
        for (var tileX = 0; tileX < horizontalTiles; tileX++)
        for (var x = 0; x < grid / 2; x++)
        {
            var crop = 1 + tileX;
            if (image.AttentionMask[crop * grid * grid + x * 2] != 0)
                usefulWidth = tileX * (grid / 2) + x + 1;
        }
        if (usefulHeight == 0 || usefulWidth == 0)
            throw new InvalidDataException("Phi-4 image attention mask has no useful HD patches.");

        var expectedTokens = checked(
            usefulHeight * usefulWidth + usefulHeight + 1 + 16 * 16 + 16);
        if (image.ImageTokenCount != expectedTokens)
            throw new InvalidDataException(
                $"Phi-4 image token count is {image.ImageTokenCount}; bucket requires {expectedTokens}.");
        return new(
            image.CropCount,
            image.TargetHeight,
            image.TargetWidth,
            usefulHeight,
            usefulWidth,
            image.ImageTokenCount);
    }
}

public sealed class Phi4D3D12VisionComponent : IDisposable
{
    public const int PortAbiVersion = 1;
    private const int CropSize = 448;
    private const int PatchSize = 14;
    private const int PatchGrid = 32;
    private const int CompressedGrid = 16;
    private const int VisionWidth = 1152;
    private const int IntermediateWidth = 4304;
    private const int TextWidth = 3072;
    private const int HeadCount = 16;
    private const int LayerCount = 26;
    private readonly object gate = new();
    private readonly GgufModelFile omni;
    private readonly ID3D12Device? ownedDevice;
    private readonly D3D12VmResourcePool pool;
    private readonly bool ownsPool;
    private readonly VmResourceManager resources;
    private readonly IReadOnlyDictionary<string, GgufModelTensor> weights;
    private readonly Dictionary<Phi4VisionBucket, BucketRuntime> runtimes = [];
    private readonly Guid componentId = Guid.NewGuid();
    private bool disposed;

    public Phi4D3D12VisionComponent(Phi4ModelPackage package, int adapterIndex = 0)
    {
        ArgumentNullException.ThrowIfNull(package);
        omni = package.Omni;
        var (device, name) = D3D12VmDeviceFactory.Create(adapterIndex);
        ownedDevice = device;
        try
        {
            pool = new D3D12VmResourcePool(device);
            ownsPool = true;
            Domain = new StorageDomain(
                "d3d12", $"{name}:{adapterIndex}", $"phi4-vision-{componentId:N}");
            weights = CollectWeights(omni);
            resources = new VmResourceManager(InitializeGlobal, pool.Allocate);
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    public Phi4D3D12VisionComponent(
        Phi4ModelPackage package,
        D3D12VmResourcePool pool,
        StorageDomain domain)
    {
        ArgumentNullException.ThrowIfNull(package);
        omni = package.Omni;
        this.pool = pool ?? throw new ArgumentNullException(nameof(pool));
        Domain = domain ?? throw new ArgumentNullException(nameof(domain));
        weights = CollectWeights(omni);
        resources = new VmResourceManager(InitializeGlobal, pool.Allocate);
    }

    public StorageDomain Domain { get; }
    public int CompiledBucketCount
    {
        get
        {
            lock (gate)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                return runtimes.Count;
            }
        }
    }

    public ComponentPortDescriptor PixelValuesPort(Phi4VisionBucket bucket)
    {
        ValidateBucket(bucket);
        return new(
            "phi4.vision.pixel_values",
            PortAbiVersion,
            new(GraphElementType.Float32, [bucket.CropCount, 3, CropSize, CropSize], "nchw"),
            GraphResourceAccess.Read,
            GraphResourceLifetime.Invocation,
            [Domain]);
    }

    public ComponentPortDescriptor AttentionMaskPort(Phi4VisionBucket bucket)
    {
        ValidateBucket(bucket);
        return new(
            "phi4.vision.attention_mask",
            PortAbiVersion,
            new(GraphElementType.Float32, [bucket.CropCount, PatchGrid, PatchGrid], "dense"),
            GraphResourceAccess.Read,
            GraphResourceLifetime.Invocation,
            [Domain]);
    }

    public ComponentPortDescriptor ProjectedEmbeddingsPort(Phi4VisionBucket bucket)
    {
        ValidateBucket(bucket);
        return new(
            "phi4.vision.projected_embeddings",
            PortAbiVersion,
            new(GraphElementType.Float32, [bucket.ImageTokenCount, TextWidth], "token-major"),
            GraphResourceAccess.Write,
            GraphResourceLifetime.Invocation,
            [Domain]);
    }

    public IStorageLease Upload(
        ComponentPortDescriptor port,
        ReadOnlySpan<float> values)
    {
        ArgumentNullException.ThrowIfNull(port);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (!port.StorageDomains.Contains(Domain) ||
                port.Access != GraphResourceAccess.Read ||
                port.Tensor.ElementType != GraphElementType.Float32)
                throw new InvalidDataException("The input port is not compatible with this vision component.");
            var expected = port.Tensor.Dimensions.Aggregate(
                1, (product, dimension) => checked(product * dimension));
            if (values.Length != expected)
                throw new ArgumentException($"Input contains {values.Length} values; expected {expected}.",
                    nameof(values));
            var handle = AllocateHandle(
                port.SemanticName, port.Tensor, ToVmTensor(port.Tensor),
                VmAccess.ReadOnly, bucket: null);
            try
            {
                handle.Resource.Write(0, MemoryMarshal.AsBytes(values));
                return CreateLease(handle);
            }
            catch
            {
                handle.Dispose();
                throw;
            }
        }
    }

    public IStorageLease Encode(Phi4ImageFeatures image)
    {
        ArgumentNullException.ThrowIfNull(image);
        var bucket = Phi4VisionBucket.FromFeatures(image);
        using var pixels = Upload(PixelValuesPort(bucket), image.PixelValues);
        using var mask = Upload(AttentionMaskPort(bucket), image.AttentionMask);
        return Encode(bucket, pixels, mask);
    }

    public IStorageLease Encode(
        Phi4VisionBucket bucket,
        IStorageLease pixelValues,
        IStorageLease attentionMask)
    {
        ArgumentNullException.ThrowIfNull(pixelValues);
        ArgumentNullException.ThrowIfNull(attentionMask);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ValidateBucket(bucket);
            var pixelHandle = RequireHandle(pixelValues, PixelValuesPort(bucket));
            var maskHandle = RequireHandle(attentionMask, AttentionMaskPort(bucket));
            var runtime = GetRuntime(bucket);
            var outputPort = ProjectedEmbeddingsPort(bucket);
            var output = AllocateHandle(
                outputPort.SemanticName, outputPort.Tensor, ToVmTensor(outputPort.Tensor),
                VmAccess.ReadWrite, bucket);
            try
            {
                runtime.Execute(pixelHandle, maskHandle, output, CreateHdMapping(bucket));
                return CreateLease(output);
            }
            catch
            {
                output.Dispose();
                throw;
            }
        }
    }

    public float[] Readback(IStorageLease output)
    {
        ArgumentNullException.ThrowIfNull(output);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var handle = output.Handle as VisionStorageHandle ??
                throw new InvalidDataException("The output is not a Phi-4 D3D12 storage handle.");
            if (handle.ComponentId != componentId ||
                handle.SemanticName != "phi4.vision.projected_embeddings")
                throw new InvalidDataException("The output belongs to a different component or port.");
            if (handle.Bucket is not { } bucket ||
                !runtimes.TryGetValue(bucket, out var runtime))
                throw new InvalidDataException("No compiled bucket matches the output descriptor.");
            var bytes = runtime.Readback(handle);
            var result = new float[bytes.Length / sizeof(float)];
            MemoryMarshal.Cast<byte, float>(bytes).CopyTo(result);
            return result;
        }
    }

    public void BindOutput(VmBindings bindings, string slotId, IStorageLease output)
    {
        ArgumentNullException.ThrowIfNull(bindings);
        ArgumentException.ThrowIfNullOrWhiteSpace(slotId);
        ArgumentNullException.ThrowIfNull(output);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var handle = output.Handle as VisionStorageHandle ??
                throw new InvalidDataException("The output is not a Phi-4 D3D12 storage handle.");
            if (handle.ComponentId != componentId ||
                handle.SemanticName != "phi4.vision.projected_embeddings")
                throw new InvalidDataException("The output belongs to a different component or port.");
            handle.Bind(bindings, slotId);
        }
    }

    public D3D12VmExecutor CreateExecutor(D3D12VmArtifact artifact)
    {
        ArgumentNullException.ThrowIfNull(artifact);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return artifact.CreateExecutor(pool);
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed)
                return;
            foreach (var runtime in runtimes.Values)
                runtime.Dispose();
            runtimes.Clear();
            resources.Dispose();
            if (ownsPool)
                pool.Dispose();
            ownedDevice?.Dispose();
            disposed = true;
        }
    }

    private BucketRuntime GetRuntime(Phi4VisionBucket bucket)
    {
        if (runtimes.TryGetValue(bucket, out var runtime))
            return runtime;
        var program = BuildProgram(bucket, omni);
        var providers = SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create()
            .Append<IInstructionCollectionProvider>(new Phi4D3D12VisionInstructionCollection())
            .ToArray();
        var artifact = new D3D12VmCompiler(providers).Compile(program);
        runtime = new BucketRuntime(this, bucket, artifact);
        runtimes.Add(bucket, runtime);
        return runtime;
    }

    private void InitializeGlobal(VmSlot slot, IVmStorage storage)
    {
        var name = slot.BindingKey ?? slot.Id;
        if (!weights.TryGetValue(name, out var tensor))
            throw new InvalidDataException($"No Phi-4 vision initializer exists for '{name}'.");
        storage.Write(0, tensor.Bytes);
    }

    private VisionStorageHandle AllocateHandle(
        string semanticName,
        TensorDescriptor descriptor,
        VmTensor tensor,
        VmAccess slotAccess,
        Phi4VisionBucket? bucket)
    {
        var slot = new VmSlot(
            $"external_{Guid.NewGuid():N}", VmSlotScope.Local, slotAccess, tensor);
        var storage = pool.Allocate(slot);
        VmResource? resource = null;
        VmResourceLease? lease = null;
        try
        {
            resource = new VmResource(tensor, VmSlotScope.Local, VmAccess.ReadWrite, storage);
            lease = resource.Acquire();
            resource.Dispose();
            resource = null;
            return new VisionStorageHandle(
                componentId, semanticName, Domain, descriptor, checked((long)tensor.ByteLength),
                lease, bucket);
        }
        catch
        {
            lease?.Dispose();
            resource?.Dispose();
            if (resource is null && lease is null)
                storage.Dispose();
            throw;
        }
    }

    private VisionStorageHandle RequireHandle(
        IStorageLease lease,
        ComponentPortDescriptor port)
    {
        var handle = lease.Handle as VisionStorageHandle ??
            throw new InvalidDataException("The input is not a Phi-4 D3D12 storage handle.");
        if (handle.ComponentId != componentId ||
            handle.SemanticName != port.SemanticName ||
            handle.Domain != Domain ||
            !ComponentPortDescriptor.TensorEquals(handle.Descriptor, port.Tensor))
            throw new InvalidDataException(
                $"Storage handle is incompatible with '{port.SemanticName}@{port.AbiVersion}'.");
        return handle;
    }

    private static IStorageLease CreateLease(VisionStorageHandle handle) =>
        StorageLease.Create(handle, released => ((VisionStorageHandle)released).Dispose());

    private static VmTensor ToVmTensor(TensorDescriptor descriptor) =>
        new(descriptor.ElementType switch
        {
            GraphElementType.Float32 => VmElementType.Float32,
            GraphElementType.Float16 => VmElementType.Float16,
            GraphElementType.Int32 => VmElementType.Int32,
            _ => throw new NotSupportedException(
                $"Vision storage does not support '{descriptor.ElementType}'."),
        }, descriptor.Dimensions);

    private static IReadOnlyDictionary<string, GgufModelTensor> CollectWeights(GgufModelFile omni)
    {
        var names = new List<string>
        {
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.sub_GN",
            "v.glb_GN",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
        };
        for (var layer = 0; layer < LayerCount; layer++)
        {
            var prefix = $"v.blk.{layer}.";
            names.AddRange(
            [
                prefix + "ln1.weight",
                prefix + "ln1.bias",
                prefix + "attn_q.weight",
                prefix + "attn_q.bias",
                prefix + "attn_k.weight",
                prefix + "attn_k.bias",
                prefix + "attn_v.weight",
                prefix + "attn_v.bias",
                prefix + "attn_out.weight",
                prefix + "attn_out.bias",
                prefix + "ln2.weight",
                prefix + "ln2.bias",
                prefix + "ffn_up.weight",
                prefix + "ffn_up.bias",
                prefix + "ffn_down.weight",
                prefix + "ffn_down.bias",
            ]);
        }
        return names.ToDictionary(name => name, omni.GetTensor, StringComparer.Ordinal);
    }

    private static int[] CreateHdMapping(Phi4VisionBucket bucket)
    {
        var horizontalTiles = bucket.TargetWidth / CropSize;
        var mapping = new List<int>(bucket.ImageTokenCount);
        for (var y = 0; y < bucket.UsefulHeight; y++)
        {
            var tileY = y / CompressedGrid;
            var localY = y % CompressedGrid;
            for (var x = 0; x < bucket.UsefulWidth; x++)
            {
                var tileX = x / CompressedGrid;
                var localX = x % CompressedGrid;
                var crop = 1 + tileY * horizontalTiles + tileX;
                mapping.Add((crop * CompressedGrid + localY) * CompressedGrid + localX);
            }
            mapping.Add(-1);
        }
        mapping.Add(-2);
        for (var y = 0; y < CompressedGrid; y++)
        {
            for (var x = 0; x < CompressedGrid; x++)
                mapping.Add(y * CompressedGrid + x);
            mapping.Add(-1);
        }
        if (mapping.Count != bucket.ImageTokenCount)
            throw new InvalidDataException("HD mapping does not match the fixed bucket token count.");
        return mapping.ToArray();
    }

    private static void ValidateBucket(Phi4VisionBucket bucket)
    {
        if (bucket.CropCount <= 1 ||
            bucket.TargetHeight <= 0 || bucket.TargetHeight % CropSize != 0 ||
            bucket.TargetWidth <= 0 || bucket.TargetWidth % CropSize != 0 ||
            bucket.CropCount != checked(
                bucket.TargetHeight / CropSize * (bucket.TargetWidth / CropSize) + 1) ||
            bucket.UsefulHeight <= 0 ||
            bucket.UsefulHeight > bucket.TargetHeight / 2 / PatchSize ||
            bucket.UsefulWidth <= 0 ||
            bucket.UsefulWidth > bucket.TargetWidth / 2 / PatchSize ||
            bucket.ImageTokenCount != checked(
                bucket.UsefulHeight * bucket.UsefulWidth + bucket.UsefulHeight +
                1 + CompressedGrid * CompressedGrid + CompressedGrid))
            throw new ArgumentOutOfRangeException(nameof(bucket), "Invalid Phi-4 vision bucket.");
    }

    private static VmProgram BuildProgram(Phi4VisionBucket bucket, GgufModelFile omni)
    {
        ValidateBucket(bucket);
        var slots = new List<VmSlot>();
        var definitions = new List<VmDefinition>();
        var nodes = new List<VmNode>();
        var nodeIndex = 0;

        VmTensor F(params int[] dimensions) => new(VmElementType.Float32, dimensions);
        VmTensor H(params int[] dimensions) => new(VmElementType.Float16, dimensions);
        VmTensor I(params int[] dimensions) => new(VmElementType.Int32, dimensions);
        void Slot(string id, VmSlotScope scope, VmAccess access, VmTensor tensor,
            string? bindingKey = null) =>
            slots.Add(new(id, scope, access, tensor, bindingKey));
        void Weight(string name, VmTensor descriptor)
        {
            var tensor = omni.GetTensor(name);
            if (tensor.Type != GgufModelTensorType.Float16 ||
                checked((ulong)tensor.ByteLength) != descriptor.ByteLength)
                throw new InvalidDataException(
                    $"Phi-4 vision tensor '{name}' does not match its VM descriptor.");
            Slot(WeightSlot(name), VmSlotScope.Global, VmAccess.ReadOnly, descriptor, name);
        }
        VmParameter[] Parameters(
            params (string Name, VmAccess Access, VmTensor Tensor)[] values) =>
            values.Select(value => new VmParameter(value.Name, value.Access, value.Tensor)).ToArray();
        VmArgument[] Arguments(params (string Parameter, string Source)[] values) =>
            values.Select(value => new VmArgument(value.Parameter, value.Source)).ToArray();
        VmArgument[] SliceArguments(
            params (string Parameter, string Source, ulong ByteOffset)[] values) =>
            values.Select(value =>
                new VmArgument(value.Parameter, value.Source, value.ByteOffset)).ToArray();
        string Kernel(
            string id,
            VmParameter[] parameters,
            string operation,
            IReadOnlyDictionary<string, string>? attributes = null,
            Guid? collection = null,
            InstructionExecutionConfiguration? executionConfiguration = null,
            VmThreadGroup? threads = null)
        {
            var instruction = new VmOperator(
                new(GraphElementType.Float32, GraphElementType.Float32),
                collection ?? Phi4InstructionCollectionIds.VisionFloat32,
                operation,
                parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)),
                attributes,
                executionConfiguration: executionConfiguration);
            definitions.Add(new(
                id, VmDefinitionKind.Kernel, parameters,
                [new("body", instruction)], threads ?? new(64)));
            return id;
        }
        void Dispatch(
            string kernel,
            ulong groups,
            VmArgument[] arguments,
            params string[] writes) =>
            DispatchGrid(kernel, DispatchGroups(groups), arguments, writes);
        void DispatchGrid(
            string kernel,
            VmThreadGroup groups,
            VmArgument[] arguments,
            params string[] writes)
        {
            var dispatchId = $"dispatch_{nodeIndex++}";
            nodes.Add(new(dispatchId, new VmDispatch(
                kernel, arguments, groups)));
            nodes.Add(new(
                $"barrier_{nodeIndex++}", new VmBarrier(writes), [dispatchId]));
        }

        var cropTokens = checked(bucket.CropCount * PatchGrid * PatchGrid);
        var visionValues = checked(cropTokens * VisionWidth);
        var intermediateValues = checked(cropTokens * IntermediateWidth);
        var compressedValues = checked(
            bucket.CropCount * CompressedGrid * CompressedGrid * VisionWidth);
        var hdValues = checked(bucket.ImageTokenCount * VisionWidth);
        var projectedValues = checked(bucket.ImageTokenCount * TextWidth);

        Slot("pixels", VmSlotScope.Local, VmAccess.ReadOnly,
            F(bucket.CropCount, 3, CropSize, CropSize));
        Slot("mask", VmSlotScope.Local, VmAccess.ReadOnly,
            F(bucket.CropCount, PatchGrid, PatchGrid));
        Slot("mapping", VmSlotScope.Local, VmAccess.ReadOnly, I(bucket.ImageTokenCount));
        Slot("hidden_a", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.CropCount, PatchGrid, PatchGrid, VisionWidth));
        Slot("hidden_b", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.CropCount, PatchGrid, PatchGrid, VisionWidth));
        Slot("normalized", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.CropCount, PatchGrid, PatchGrid, VisionWidth));
        Slot("query", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.CropCount, PatchGrid, PatchGrid, VisionWidth));
        Slot("key", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.CropCount, PatchGrid, PatchGrid, VisionWidth));
        Slot("value", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.CropCount, PatchGrid, PatchGrid, VisionWidth));
        Slot("attention", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.CropCount, PatchGrid, PatchGrid, VisionWidth));
        Slot("projected", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.CropCount, PatchGrid, PatchGrid, VisionWidth));
        Slot("ffn_up", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.CropCount, PatchGrid, PatchGrid, IntermediateWidth));
        Slot("ffn_activated", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.CropCount, PatchGrid, PatchGrid, IntermediateWidth));
        Slot("compressed", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.CropCount, CompressedGrid, CompressedGrid, VisionWidth));
        Slot("hd", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.ImageTokenCount, VisionWidth));
        Slot("project_hidden", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.ImageTokenCount, TextWidth));
        Slot("project_activated", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.ImageTokenCount, TextWidth));
        Slot("output", VmSlotScope.Local, VmAccess.ReadWrite,
            F(bucket.ImageTokenCount, TextWidth));

        Weight("v.patch_embd.weight", H(VisionWidth, 3, PatchSize, PatchSize));
        Weight("v.patch_embd.bias", H(VisionWidth));
        Weight("v.position_embd.weight", H(PatchGrid * PatchGrid, VisionWidth));
        Weight("v.sub_GN", H(VisionWidth));
        Weight("v.glb_GN", H(VisionWidth));
        Weight("mm.0.weight", H(TextWidth, VisionWidth));
        Weight("mm.0.bias", H(TextWidth));
        Weight("mm.2.weight", H(TextWidth, TextWidth));
        Weight("mm.2.bias", H(TextWidth));
        for (var layer = 0; layer < LayerCount; layer++)
        {
            var prefix = $"v.blk.{layer}.";
            Weight(prefix + "ln1.weight", H(VisionWidth));
            Weight(prefix + "ln1.bias", H(VisionWidth));
            Weight(prefix + "attn_q.weight", H(VisionWidth, VisionWidth));
            Weight(prefix + "attn_q.bias", H(VisionWidth));
            Weight(prefix + "attn_k.weight", H(VisionWidth, VisionWidth));
            Weight(prefix + "attn_k.bias", H(VisionWidth));
            Weight(prefix + "attn_v.weight", H(VisionWidth, VisionWidth));
            Weight(prefix + "attn_v.bias", H(VisionWidth));
            Weight(prefix + "attn_out.weight", H(VisionWidth, VisionWidth));
            Weight(prefix + "attn_out.bias", H(VisionWidth));
            Weight(prefix + "ln2.weight", H(VisionWidth));
            Weight(prefix + "ln2.bias", H(VisionWidth));
            Weight(prefix + "ffn_up.weight", H(IntermediateWidth, VisionWidth));
            Weight(prefix + "ffn_up.bias", H(IntermediateWidth));
            Weight(prefix + "ffn_down.weight", H(VisionWidth, IntermediateWidth));
            Weight(prefix + "ffn_down.bias", H(VisionWidth));
        }

        var patchParameters = Parameters(
            ("pixels", VmAccess.ReadOnly, F(1, 3, CropSize, CropSize)),
            ("mask", VmAccess.ReadOnly, F(1, PatchGrid, PatchGrid)),
            ("weight", VmAccess.ReadOnly, H(VisionWidth, 3, PatchSize, PatchSize)),
            ("bias", VmAccess.ReadOnly, H(VisionWidth)),
            ("position", VmAccess.ReadOnly, H(PatchGrid * PatchGrid, VisionWidth)),
            ("output", VmAccess.ReadWrite,
                F(1, PatchGrid, PatchGrid, VisionWidth)));
        var patchKernel = Kernel(
            "patch_embedding", patchParameters, Phi4VisionInstructionNames.PatchEmbedding,
            new Dictionary<string, string>
            {
                ["crop_size"] = CropSize.ToString(),
                ["patch_size"] = PatchSize.ToString(),
                ["width"] = VisionWidth.ToString(),
            });
        var layerNormParameters = Parameters(
            ("input", VmAccess.ReadOnly,
                F(1, PatchGrid, PatchGrid, VisionWidth)),
            ("weight", VmAccess.ReadOnly, H(VisionWidth)),
            ("bias", VmAccess.ReadOnly, H(VisionWidth)),
            ("output", VmAccess.ReadWrite,
                F(1, PatchGrid, PatchGrid, VisionWidth)));
        var layerNormKernel = Kernel(
            "layer_norm", layerNormParameters, Phi4VisionInstructionNames.LayerNorm,
            new Dictionary<string, string> { ["epsilon"] = "1E-06" });

        string LinearKernel(
            string id,
            int rows,
            int inputWidth,
            int outputWidth)
        {
            var parameters = Parameters(
                ("left", VmAccess.ReadOnly, F(rows, inputWidth)),
                ("right", VmAccess.ReadOnly, H(outputWidth, inputWidth)),
                ("bias", VmAccess.ReadOnly, H(outputWidth)),
                ("output", VmAccess.ReadWrite, F(rows, outputWidth)));
            return Kernel(
                id,
                parameters,
                "core.affine",
                new Dictionary<string, string>
                {
                    ["transpose_left"] = "false",
                    ["transpose_right"] = "true",
                },
                InstructionCollectionIds.TierZeroFloat32,
                GpuMatrixMultiplyExecution.Tiled,
                new(8, 8));
        }
        var visionLinear = LinearKernel(
            "linear_vision", PatchGrid * PatchGrid, VisionWidth, VisionWidth);
        var ffnUpLinear = LinearKernel(
            "linear_ffn_up", PatchGrid * PatchGrid, VisionWidth, IntermediateWidth);
        var ffnDownLinear = LinearKernel(
            "linear_ffn_down", PatchGrid * PatchGrid, IntermediateWidth, VisionWidth);
        void DispatchLinear(
            string kernel,
            int rows,
            int outputWidth,
            string input,
            ulong inputOffset,
            string weight,
            string bias,
            string output,
            ulong outputOffset)
        {
            var groups = GpuMatrixMultiplyExecution.TileGroups(rows, outputWidth);
            DispatchGrid(
                kernel,
                new(groups.X, groups.Y),
                SliceArguments(
                    ("left", input, inputOffset),
                    ("right", weight, 0),
                    ("bias", bias, 0),
                    ("output", output, outputOffset)),
                output);
        }

        var attentionParameters = Parameters(
            ("query", VmAccess.ReadOnly,
                F(1, PatchGrid * PatchGrid, VisionWidth)),
            ("key", VmAccess.ReadOnly,
                F(1, PatchGrid * PatchGrid, VisionWidth)),
            ("value", VmAccess.ReadOnly,
                F(1, PatchGrid * PatchGrid, VisionWidth)),
            ("mask", VmAccess.ReadOnly, F(1, PatchGrid * PatchGrid)),
            ("output", VmAccess.ReadWrite,
                F(1, PatchGrid * PatchGrid, VisionWidth)));
        var attentionKernel = Kernel(
            "self_attention", attentionParameters, Phi4VisionInstructionNames.Attention,
            new Dictionary<string, string> { ["heads"] = HeadCount.ToString() });
        var addParameters = Parameters(
            ("left", VmAccess.ReadOnly,
                F(1, PatchGrid, PatchGrid, VisionWidth)),
            ("right", VmAccess.ReadOnly,
                F(1, PatchGrid, PatchGrid, VisionWidth)),
            ("output", VmAccess.ReadWrite,
                F(1, PatchGrid, PatchGrid, VisionWidth)));
        var addKernel = Kernel(
            "residual_add", addParameters, "core.add",
            collection: InstructionCollectionIds.TierZeroFloat32);
        var geluVisionParameters = Parameters(
            ("input", VmAccess.ReadOnly, F(PatchGrid * PatchGrid, IntermediateWidth)),
            ("output", VmAccess.ReadWrite, F(PatchGrid * PatchGrid, IntermediateWidth)));
        var geluVisionKernel = Kernel(
            "gelu_vision", geluVisionParameters, Phi4VisionInstructionNames.Gelu);
        var poolParameters = Parameters(
            ("input", VmAccess.ReadOnly,
                F(1, PatchGrid, PatchGrid, VisionWidth)),
            ("output", VmAccess.ReadWrite,
                F(1, CompressedGrid, CompressedGrid, VisionWidth)));
        var poolKernel = Kernel("pool_2x2", poolParameters, Phi4VisionInstructionNames.Pool2x2);
        var gatherParameters = Parameters(
            ("input", VmAccess.ReadOnly,
                F(checked(bucket.CropCount * CompressedGrid * CompressedGrid), VisionWidth)),
            ("mapping", VmAccess.ReadOnly, I(bucket.ImageTokenCount)),
            ("sub_separator", VmAccess.ReadOnly, H(VisionWidth)),
            ("global_separator", VmAccess.ReadOnly, H(VisionWidth)),
            ("output", VmAccess.ReadWrite, F(bucket.ImageTokenCount, VisionWidth)));
        var gatherKernel = Kernel("hd_gather", gatherParameters, Phi4VisionInstructionNames.HdGather);

        var entryRanges = new List<(string Id, int Start, int Count)>();
        var pixelCropBytes = checked((ulong)(3 * CropSize * CropSize * sizeof(float)));
        var maskCropBytes = checked((ulong)(PatchGrid * PatchGrid * sizeof(float)));
        var visionCropBytes = checked(
            (ulong)(PatchGrid * PatchGrid * VisionWidth * sizeof(float)));
        var intermediateCropBytes = checked(
            (ulong)(PatchGrid * PatchGrid * IntermediateWidth * sizeof(float)));
        var compressedCropBytes = checked(
            (ulong)(CompressedGrid * CompressedGrid * VisionWidth * sizeof(float)));
        var cropVisionValues = PatchGrid * PatchGrid * VisionWidth;
        var cropIntermediateValues = PatchGrid * PatchGrid * IntermediateWidth;
        var cropCompressedValues = CompressedGrid * CompressedGrid * VisionWidth;

        for (var crop = 0; crop < bucket.CropCount; crop++)
        {
            var start = nodes.Count;
            Dispatch(
                patchKernel, DivideRoundUp((ulong)cropVisionValues, 64),
                SliceArguments(
                    ("pixels", "pixels", (ulong)crop * pixelCropBytes),
                    ("mask", "mask", (ulong)crop * maskCropBytes),
                    ("weight", WeightSlot("v.patch_embd.weight"), 0),
                    ("bias", WeightSlot("v.patch_embd.bias"), 0),
                    ("position", WeightSlot("v.position_embd.weight"), 0),
                    ("output", "hidden_a", (ulong)crop * visionCropBytes)),
                "hidden_a");
            entryRanges.Add(($"patch_{crop}", start, nodes.Count - start));
        }
        for (var layer = 0; layer < LayerCount; layer++)
        {
            var prefix = $"v.blk.{layer}.";
            for (var crop = 0; crop < bucket.CropCount; crop++)
            {
                var start = nodes.Count;
                var visionOffset = (ulong)crop * visionCropBytes;
                var intermediateOffset = (ulong)crop * intermediateCropBytes;
                var maskOffset = (ulong)crop * maskCropBytes;
                Dispatch(
                    layerNormKernel, PatchGrid * PatchGrid,
                    SliceArguments(
                        ("input", "hidden_a", visionOffset),
                        ("weight", WeightSlot(prefix + "ln1.weight"), 0),
                        ("bias", WeightSlot(prefix + "ln1.bias"), 0),
                        ("output", "normalized", visionOffset)),
                    "normalized");
                foreach (var (name, destination) in new[]
                {
                    ("attn_q", "query"),
                    ("attn_k", "key"),
                    ("attn_v", "value"),
                })
                {
                    DispatchLinear(
                        visionLinear,
                        PatchGrid * PatchGrid,
                        VisionWidth,
                        "normalized",
                        visionOffset,
                        WeightSlot(prefix + name + ".weight"),
                        WeightSlot(prefix + name + ".bias"),
                        destination,
                        visionOffset);
                }
                Dispatch(
                    attentionKernel,
                    checked((ulong)PatchGrid * PatchGrid * HeadCount),
                    SliceArguments(
                        ("query", "query", visionOffset),
                        ("key", "key", visionOffset),
                        ("value", "value", visionOffset),
                        ("mask", "mask", maskOffset),
                        ("output", "attention", visionOffset)),
                    "attention");
                DispatchLinear(
                    visionLinear,
                    PatchGrid * PatchGrid,
                    VisionWidth,
                    "attention",
                    visionOffset,
                    WeightSlot(prefix + "attn_out.weight"),
                    WeightSlot(prefix + "attn_out.bias"),
                    "projected",
                    visionOffset);
                Dispatch(
                    addKernel, DivideRoundUp((ulong)cropVisionValues, 64),
                    SliceArguments(
                        ("left", "hidden_a", visionOffset),
                        ("right", "projected", visionOffset),
                        ("output", "hidden_b", visionOffset)),
                    "hidden_b");
                Dispatch(
                    layerNormKernel, PatchGrid * PatchGrid,
                    SliceArguments(
                        ("input", "hidden_b", visionOffset),
                        ("weight", WeightSlot(prefix + "ln2.weight"), 0),
                        ("bias", WeightSlot(prefix + "ln2.bias"), 0),
                        ("output", "normalized", visionOffset)),
                    "normalized");
                DispatchLinear(
                    ffnUpLinear,
                    PatchGrid * PatchGrid,
                    IntermediateWidth,
                    "normalized",
                    visionOffset,
                    WeightSlot(prefix + "ffn_up.weight"),
                    WeightSlot(prefix + "ffn_up.bias"),
                    "ffn_up",
                    intermediateOffset);
                Dispatch(
                    geluVisionKernel, DivideRoundUp((ulong)cropIntermediateValues, 64),
                    SliceArguments(
                        ("input", "ffn_up", intermediateOffset),
                        ("output", "ffn_activated", intermediateOffset)),
                    "ffn_activated");
                DispatchLinear(
                    ffnDownLinear,
                    PatchGrid * PatchGrid,
                    VisionWidth,
                    "ffn_activated",
                    intermediateOffset,
                    WeightSlot(prefix + "ffn_down.weight"),
                    WeightSlot(prefix + "ffn_down.bias"),
                    "projected",
                    visionOffset);
                Dispatch(
                    addKernel, DivideRoundUp((ulong)cropVisionValues, 64),
                    SliceArguments(
                        ("left", "hidden_b", visionOffset),
                        ("right", "projected", visionOffset),
                        ("output", "hidden_a", visionOffset)),
                    "hidden_a");
                entryRanges.Add((
                    $"layer_{layer}_crop_{crop}", start, nodes.Count - start));
            }
        }
        for (var crop = 0; crop < bucket.CropCount; crop++)
        {
            var start = nodes.Count;
            Dispatch(
                poolKernel, DivideRoundUp((ulong)cropCompressedValues, 64),
                SliceArguments(
                    ("input", "hidden_a", (ulong)crop * visionCropBytes),
                    ("output", "compressed", (ulong)crop * compressedCropBytes)),
                "compressed");
            entryRanges.Add(($"pool_{crop}", start, nodes.Count - start));
        }
        var gatherStart = nodes.Count;
        Dispatch(
            gatherKernel, DivideRoundUp((ulong)hdValues, 64),
            Arguments(
                ("input", "compressed"),
                ("mapping", "mapping"),
                ("sub_separator", WeightSlot("v.sub_GN")),
                ("global_separator", WeightSlot("v.glb_GN")),
                ("output", "hd")),
            "hd");
        entryRanges.Add(("hd_gather_entry", gatherStart, nodes.Count - gatherStart));

        const int projectorChunkSize = 32;
        var projectorKernels =
            new Dictionary<int, (string Input, string Gelu, string Output)>();
        for (var token = 0; token < bucket.ImageTokenCount; token += projectorChunkSize)
        {
            var count = Math.Min(projectorChunkSize, bucket.ImageTokenCount - token);
            if (!projectorKernels.TryGetValue(count, out var kernels))
            {
                var inputKernel = LinearKernel(
                    $"linear_projector_input_{count}", count, VisionWidth, TextWidth);
                var geluParameters = Parameters(
                    ("input", VmAccess.ReadOnly, F(count, TextWidth)),
                    ("output", VmAccess.ReadWrite, F(count, TextWidth)));
                var geluKernel = Kernel(
                    $"gelu_projector_{count}", geluParameters, Phi4VisionInstructionNames.Gelu);
                var outputKernel = LinearKernel(
                    $"linear_projector_output_{count}", count, TextWidth, TextWidth);
                kernels = (inputKernel, geluKernel, outputKernel);
                projectorKernels.Add(count, kernels);
            }
            var start = nodes.Count;
            var hdOffset = checked((ulong)token * VisionWidth * sizeof(float));
            var projectedOffset = checked((ulong)token * TextWidth * sizeof(float));
            var values = checked((ulong)count * TextWidth);
            DispatchLinear(
                kernels.Input,
                count,
                TextWidth,
                "hd",
                hdOffset,
                WeightSlot("mm.0.weight"),
                WeightSlot("mm.0.bias"),
                "project_hidden",
                projectedOffset);
            Dispatch(
                kernels.Gelu, DivideRoundUp(values, 64),
                SliceArguments(
                    ("input", "project_hidden", projectedOffset),
                    ("output", "project_activated", projectedOffset)),
                "project_activated");
            DispatchLinear(
                kernels.Output,
                count,
                TextWidth,
                "project_activated",
                projectedOffset,
                WeightSlot("mm.2.weight"),
                WeightSlot("mm.2.bias"),
                "output",
                projectedOffset);
            entryRanges.Add((
                $"project_{token}_{count}", start, nodes.Count - start));
        }

        var rootParameters = slots.Select(
            slot => new VmParameter(slot.Id, slot.Access, slot.Tensor)).ToArray();
        var entryArguments = rootParameters.Select(
            parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var entries = new List<VmEntry>();
        void Entry(string id, IEnumerable<VmNode> entryNodes)
        {
            definitions.Add(new(
                id, VmDefinitionKind.Orchestration, rootParameters, entryNodes));
            entries.Add(new(id, id, entryArguments));
        }
        foreach (var range in entryRanges)
            Entry(range.Id, nodes.Skip(range.Start).Take(range.Count));
        return new(
            $"phi4-vision-{bucket.CropCount}-{bucket.ImageTokenCount}",
            "phi4.vision.fp16.v1",
            VmTarget.Direct3D12,
            slots,
            definitions,
            entries,
            new("none", 1, []));
    }

    private static string WeightSlot(string name) =>
        "weight_" + name.Replace('.', '_');

    private static VmThreadGroup DispatchGroups(ulong groups)
    {
        if (groups == 0)
            throw new ArgumentOutOfRangeException(nameof(groups));
        const ulong maximum = 65535;
        var x = Math.Min(groups, maximum);
        var y = DivideRoundUp(groups, x);
        if (y > maximum)
            throw new NotSupportedException("Vision dispatch exceeds the D3D12 two-dimensional limit.");
        return new(checked((uint)x), checked((uint)y));
    }

    private static ulong DivideRoundUp(ulong value, ulong divisor) =>
        checked((value + divisor - 1) / divisor);

    private sealed class BucketRuntime : IDisposable
    {
        private readonly Phi4D3D12VisionComponent owner;
        private readonly Phi4VisionBucket bucket;
        private readonly D3D12VmExecutor executor;
        private readonly VmBindings bindings;
        private readonly IReadOnlyList<string> entryNames;
        private readonly int pixelSlot;
        private readonly int maskSlot;
        private readonly int mappingSlot;
        private readonly int outputSlot;
        private bool initialized;

        public BucketRuntime(
            Phi4D3D12VisionComponent owner,
            Phi4VisionBucket bucket,
            D3D12VmArtifact artifact)
        {
            this.owner = owner;
            this.bucket = bucket;
            executor = artifact.CreateExecutor(owner.pool);
            try
            {
                bindings = owner.resources.CreateBindings(artifact.Program);
            }
            catch
            {
                executor.Dispose();
                throw;
            }
            entryNames = artifact.Program.Entries.Select(entry => entry.Name).ToArray();
            pixelSlot = SlotIndex("pixels");
            maskSlot = SlotIndex("mask");
            mappingSlot = SlotIndex("mapping");
            outputSlot = SlotIndex("output");
        }

        public void Execute(
            VisionStorageHandle pixels,
            VisionStorageHandle mask,
            VisionStorageHandle output,
            int[] mapping)
        {
            using var pixelResource = pixels.Resource.Retain();
            using var maskResource = mask.Resource.Retain();
            using var outputResource = output.Resource.Retain();
            bindings.Bind("pixels", pixelResource);
            bindings.Bind("mask", maskResource);
            bindings.Bind("output", outputResource);
            Exception? executionFailure = null;
            try
            {
                using var execution = bindings.BeginExecution();
                var context = execution.GetBuffers();
                MemoryMarshal.AsBytes(mapping.AsSpan()).CopyTo(context[mappingSlot]);
                if (!initialized)
                {
                    executor.UploadSlots(context, Enumerable.Range(0, context.Length).ToArray());
                    initialized = true;
                }
                else
                {
                    executor.UploadSlots(context, [pixelSlot, maskSlot, mappingSlot, outputSlot]);
                }
                foreach (var entryName in entryNames)
                {
                    try
                    {
                        executor.Execute(entryName);
                    }
                    catch (Exception error)
                    {
                        executionFailure = new InvalidOperationException(
                            $"Phi-4 vision GPU entry '{entryName}' failed.", error);
                        throw executionFailure;
                    }
                }
            }
            finally
            {
                if (executionFailure is null)
                {
                    BindPlaceholder("pixels");
                    BindPlaceholder("mask");
                    BindPlaceholder("output");
                }
            }
        }

        public byte[] Readback(VisionStorageHandle output)
        {
            using var resource = output.Resource.Retain();
            bindings.Bind("output", resource);
            try
            {
                using var execution = bindings.BeginExecution();
                var context = execution.GetBuffers();
                executor.ReadbackSlots(context, [outputSlot]);
                return context[outputSlot].ToArray();
            }
            finally
            {
                BindPlaceholder("output");
            }
        }

        public void Dispose()
        {
            executor.Dispose();
            bindings.Dispose();
        }

        private int SlotIndex(string id) =>
            bindings.Program.Slots.Select((slot, index) => (slot, index))
                .Single(value => value.slot.Id == id).index;

        private void BindPlaceholder(string slotId)
        {
            var slot = bindings.Program.Slots.Single(candidate => candidate.Id == slotId);
            var storage = owner.pool.Allocate(
                new VmSlot($"placeholder_{Guid.NewGuid():N}",
                    VmSlotScope.Local, slot.Access, slot.Tensor));
            using var resource = new VmResource(
                slot.Tensor, VmSlotScope.Local, VmAccess.ReadWrite, storage);
            using var lease = resource.Acquire();
            bindings.Bind(slotId, lease);
        }
    }

    private sealed class VisionStorageHandle : IVmBindableStorageHandle, IDisposable
    {
        private VmResourceLease? resource;

        public VisionStorageHandle(
            Guid componentId,
            string semanticName,
            StorageDomain domain,
            TensorDescriptor descriptor,
            long byteLength,
            VmResourceLease resource,
            Phi4VisionBucket? bucket)
        {
            ComponentId = componentId;
            SemanticName = semanticName;
            Domain = domain;
            Descriptor = descriptor;
            ByteLength = byteLength;
            this.resource = resource;
            Bucket = bucket;
        }

        public Guid Id { get; } = Guid.NewGuid();
        public Guid ComponentId { get; }
        public string SemanticName { get; }
        public StorageDomain Domain { get; }
        public TensorDescriptor Descriptor { get; }
        public long ByteLength { get; }
        public Phi4VisionBucket? Bucket { get; }
        public VmResourceLease Resource =>
            resource ?? throw new ObjectDisposedException(nameof(VisionStorageHandle));

        public void Bind(VmBindings bindings, string slotId)
        {
            using var retained = Resource.Retain();
            bindings.Bind(slotId, retained);
        }

        public void Dispose() => Interlocked.Exchange(ref resource, null)?.Dispose();
    }
}
