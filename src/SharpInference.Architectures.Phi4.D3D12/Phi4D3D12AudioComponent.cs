using System.Globalization;
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

public enum Phi4AudioProjector
{
    Speech,
    Vision,
}

public interface IPhi4AudioEmbeddingHandle : IStorageHandle
{
    int ValidTokenCount { get; }
    int TokenBucket { get; }
    Phi4AudioProjector Projector { get; }
}

public sealed class Phi4D3D12AudioComponent : IDisposable
{
    public const int PortAbiVersion = 1;
    private const int FeatureCount = 80;
    private const int AudioWidth = 1024;
    private const int TextWidth = 3072;
    private static readonly int[] FrameBuckets = [128, 256, 352, 512, 1024, 2048, 4096];
    private readonly object gate = new();
    private readonly ID3D12Device? ownedDevice;
    private readonly D3D12VmResourcePool pool;
    private readonly bool ownsPool;
    private readonly VmResourceManager resources;
    private readonly Dictionary<string, byte[]> initializers;
    private readonly AudioRuntime runtime;
    private readonly Guid componentId = Guid.NewGuid();
    private bool disposed;

    public Phi4D3D12AudioComponent(
        Phi4ModelPackage package,
        int maximumFrames = 352,
        int adapterIndex = 0)
    {
        ArgumentNullException.ThrowIfNull(package);
        FrameBucket = GetFrameBucket(maximumFrames);
        TokenBucket = DivideRoundUp(FrameBucket, 8);
        var (device, name) = D3D12VmDeviceFactory.Create(adapterIndex);
        ownedDevice = device;
        try
        {
            pool = new D3D12VmResourcePool(device);
            ownsPool = true;
            Domain = new StorageDomain(
                "d3d12", $"{name}:{adapterIndex}", $"phi4-audio-{componentId:N}");
            var build = Build(package.Omni, FrameBucket, VmTarget.Direct3D12);
            initializers = build.Initializers.ToDictionary(
                value => value.Slot, value => value.Bytes, StringComparer.Ordinal);
            resources = new VmResourceManager(InitializeGlobal, pool.Allocate);
            runtime = CreateRuntime(build.Program);
            initializers.Clear();
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    public Phi4D3D12AudioComponent(
        Phi4ModelPackage package,
        D3D12VmResourcePool pool,
        StorageDomain domain,
        int maximumFrames = 352)
    {
        ArgumentNullException.ThrowIfNull(package);
        this.pool = pool ?? throw new ArgumentNullException(nameof(pool));
        Domain = domain ?? throw new ArgumentNullException(nameof(domain));
        FrameBucket = GetFrameBucket(maximumFrames);
        TokenBucket = DivideRoundUp(FrameBucket, 8);
        var build = Build(package.Omni, FrameBucket, VmTarget.Direct3D12);
        initializers = build.Initializers.ToDictionary(
            value => value.Slot, value => value.Bytes, StringComparer.Ordinal);
        resources = new VmResourceManager(InitializeGlobal, pool.Allocate);
        try
        {
            runtime = CreateRuntime(build.Program);
            initializers.Clear();
        }
        catch
        {
            resources.Dispose();
            throw;
        }
    }

    public static IReadOnlyList<int> SupportedFrameBuckets { get; } =
        Array.AsReadOnly(FrameBuckets);

    public int FrameBucket { get; }
    public int TokenBucket { get; }
    public StorageDomain Domain { get; }
    public ulong QueueSubmissionCount
    {
        get
        {
            lock (gate)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                return runtime.SubmissionCount;
            }
        }
    }

    public static int GetFrameBucket(int frameCount)
    {
        if (frameCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(frameCount));
        foreach (var bucket in FrameBuckets)
            if (frameCount <= bucket)
                return bucket;
        throw new ArgumentOutOfRangeException(
            nameof(frameCount),
            $"Phi-4 D3D12 audio supports at most {FrameBuckets[^1]} feature frames.");
    }

    public float[] Encode(Phi4AudioFeatures features, Phi4AudioProjector projector)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            using var output = EncodeResident(features, projector);
            return Readback(output);
        }
    }

    public IStorageLease EncodeResident(
        Phi4AudioFeatures features,
        Phi4AudioProjector projector)
    {
        ArgumentNullException.ThrowIfNull(features);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var tokenCount = ValidateFeatures(features, projector);
            var outputPort = ProjectedEmbeddingsPort();
            var output = AllocateHandle(
                outputPort.SemanticName,
                outputPort.Tensor,
                new(VmElementType.Float32, [TokenBucket, TextWidth]),
                VmAccess.ReadWrite,
                tokenCount,
                projector);
            try
            {
                runtime.Execute(features, projector, output);
                return CreateLease(output);
            }
            catch
            {
                output.Dispose();
                throw;
            }
        }
    }

    public ComponentPortDescriptor ProjectedEmbeddingsPort() =>
        new(
            "phi4.audio.projected_embeddings",
            PortAbiVersion,
            new(GraphElementType.Float32, [TokenBucket, TextWidth], "token-major"),
            GraphResourceAccess.Write,
            GraphResourceLifetime.Invocation,
            [Domain]);

    public float[] Readback(IStorageLease output)
    {
        ArgumentNullException.ThrowIfNull(output);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var handle = RequireOutputHandle(output);
            var bytes = runtime.Readback(handle);
            var values = MemoryMarshal.Cast<byte, float>(bytes);
            return values[..checked(handle.ValidTokenCount * TextWidth)].ToArray();
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
            var handle = RequireOutputHandle(output);
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
            runtime.Dispose();
            resources.Dispose();
            if (ownsPool)
                pool.Dispose();
            ownedDevice?.Dispose();
            disposed = true;
        }
    }

    private int ValidateFeatures(Phi4AudioFeatures features, Phi4AudioProjector projector)
    {
        if (!Enum.IsDefined(projector))
            throw new ArgumentOutOfRangeException(nameof(projector));
        if (features.FrameCount <= 0 ||
            features.FrameCount > FrameBucket ||
            features.FeatureCount != FeatureCount ||
            features.Values.Length != checked(features.FrameCount * FeatureCount))
            throw new ArgumentException(
                $"Audio must contain 1..{FrameBucket} frames of {FeatureCount} features.",
                nameof(features));
        var tokenCount = DivideRoundUp(features.FrameCount, 8);
        if (features.EmbedSize != tokenCount)
            throw new ArgumentException(
                $"Audio embed size must be {tokenCount} for {features.FrameCount} frames.",
                nameof(features));
        return tokenCount;
    }

    private AudioRuntime CreateRuntime(VmProgram program)
    {
        var providers = DefaultInstructionCollections.Create()
            .Append<IInstructionCollectionProvider>(
                new Phi4D3D12AudioInstructionCollection());
        var artifact = new D3D12VmCompiler(providers).Compile(program);
        return new(this, artifact);
    }

    private void InitializeGlobal(VmSlot slot, IVmStorage storage)
    {
        var name = slot.BindingKey ?? slot.Id;
        if (!initializers.TryGetValue(name, out var bytes))
            throw new InvalidDataException($"No Phi-4 audio initializer exists for '{name}'.");
        storage.Write(0, bytes);
    }

    private AudioStorageHandle AllocateHandle(
        string semanticName,
        TensorDescriptor descriptor,
        VmTensor tensor,
        VmAccess slotAccess,
        int validTokenCount,
        Phi4AudioProjector projector)
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
            return new(
                componentId,
                semanticName,
                Domain,
                descriptor,
                checked((long)tensor.ByteLength),
                lease,
                validTokenCount,
                TokenBucket,
                projector);
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

    private AudioStorageHandle RequireOutputHandle(IStorageLease output)
    {
        var handle = output.Handle as AudioStorageHandle ??
            throw new InvalidDataException("The output is not a Phi-4 D3D12 audio storage handle.");
        var port = ProjectedEmbeddingsPort();
        if (handle.ComponentId != componentId ||
            handle.SemanticName != port.SemanticName ||
            handle.Domain != Domain ||
            !ComponentPortDescriptor.TensorEquals(handle.Descriptor, port.Tensor) ||
            handle.TokenBucket != TokenBucket)
            throw new InvalidDataException(
                $"Storage handle is incompatible with '{port.SemanticName}@{port.AbiVersion}'.");
        return handle;
    }

    private static IStorageLease CreateLease(AudioStorageHandle handle) =>
        StorageLease.Create(handle, released => ((AudioStorageHandle)released).Dispose());

    internal static BuildResult Build(GgufModelFile omni, int frames, VmTarget target)
    {
        const int heads = 16;
        const int intermediate = 1536;
        var tokens = DivideRoundUp(frames, 8);
        var slots = new List<VmSlot>();
        var initializers = new List<Initializer>();
        var definitions = new List<VmDefinition>();
        var nodes = new List<VmNode>();
        var slotIds = new HashSet<string>(StringComparer.Ordinal);
        var kernelCache = new Dictionary<string, string>(StringComparer.Ordinal);
        var kernelIndex = 0;
        var nodeIndex = 0;
        string? previousNode = null;

        VmTensor F(params int[] dimensions) => new(VmElementType.Float32, dimensions);
        VmTensor I(params int[] dimensions) => new(VmElementType.Int32, dimensions);
        void Slot(string id, VmSlotScope scope, VmAccess access, VmTensor tensor)
        {
            if (!slotIds.Add(id))
                throw new InvalidOperationException($"Duplicate Phi-4 audio VM slot '{id}'.");
            slots.Add(new(id, scope, access, tensor));
        }
        string TensorId(string name) => "w_" + name.Replace('.', '_');
        string Weight(string name, bool matrix = false)
        {
            var id = TensorId(name);
            if (slotIds.Contains(id))
                return id;
            var tensor = omni.GetTensor(name);
            if (matrix)
            {
                if (tensor.Type is not (GgufModelTensorType.Float16 or GgufModelTensorType.Float32) ||
                    tensor.Dimensions.Count < 2)
                    throw new InvalidDataException(
                        $"Phi-4 audio matrix '{name}' must be Float16 or Float32.");
                Slot(id, VmSlotScope.Global, VmAccess.ReadOnly,
                    new(
                        tensor.Type == GgufModelTensorType.Float16
                            ? VmElementType.Float16
                            : VmElementType.Float32,
                        tensor.Dimensions.Reverse().Select(CheckedInt).ToArray()));
                initializers.Add(new(id, tensor.Bytes.ToArray()));
            }
            else
            {
                var values = ToFloat32(tensor);
                Slot(id, VmSlotScope.Global, VmAccess.ReadOnly, F(values.Length));
                initializers.Add(new(id, MemoryMarshal.AsBytes(values.AsSpan()).ToArray()));
            }
            return id;
        }
        VmParameter[] Parameters(
            params (string Name, VmAccess Access, VmTensor Tensor)[] values) =>
            values.Select(value => new VmParameter(value.Name, value.Access, value.Tensor)).ToArray();
        VmArgument[] Arguments(params (string Parameter, string Source, ulong Offset)[] values) =>
            values.Select(value => new VmArgument(value.Parameter, value.Source, value.Offset)).ToArray();
        VmOperator Operation(
            string name,
            VmParameter[] parameters,
            IReadOnlyDictionary<string, string> attributes,
            Guid? collection = null,
            InstructionExecutionConfiguration? configuration = null) =>
            new(new(GraphElementType.Float32, GraphElementType.Float32),
                collection ?? Phi4InstructionCollectionIds.AudioFloat32,
                name,
                parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)),
                attributes,
                executionConfiguration: configuration);
        string Kernel(
            string name,
            VmParameter[] parameters,
            VmOperator operation,
            VmThreadGroup? threads = null)
        {
            var threadGroup = threads ?? new(64);
            var key = string.Join(
                "|",
                name,
                operation.InstructionCollectionId,
                operation.InstructionName,
                threadGroup,
                string.Join(";", parameters.Select(parameter =>
                    $"{parameter.Name}:{parameter.Access}:{parameter.Tensor.ElementType}:" +
                    string.Join(",", parameter.Tensor.Dimensions))),
                string.Join(";", operation.Attributes.OrderBy(value => value.Key, StringComparer.Ordinal)
                    .Select(value => $"{value.Key}={value.Value}")),
                operation.ExecutionConfiguration?.ToString() ?? "");
            if (kernelCache.TryGetValue(key, out var existing))
                return existing;
            var id = $"{name}_{kernelIndex++}";
            definitions.Add(target == VmTarget.Direct3D12
                ? new(
                    id, VmDefinitionKind.Kernel, parameters,
                    [new("body", operation)], threadGroup)
                : new(
                    id, VmDefinitionKind.Function, parameters,
                    [new("body", operation)]));
            kernelCache.Add(key, id);
            return id;
        }
        void Dispatch(
            string kernel,
            VmThreadGroup groups,
            VmArgument[] arguments,
            params string[] writes)
        {
            var dispatchId = $"dispatch_{nodeIndex++}";
            nodes.Add(new(
                dispatchId,
                target == VmTarget.Direct3D12
                    ? new VmDispatch(kernel, arguments, groups)
                    : new VmCall(kernel, arguments),
                previousNode is null ? [] : [previousNode]));
            if (target == VmTarget.Direct3D12)
            {
                var barrierId = $"barrier_{nodeIndex++}";
                nodes.Add(new(barrierId, new VmBarrier(writes), [dispatchId]));
                previousNode = barrierId;
            }
            else
            {
                previousNode = dispatchId;
            }
        }
        static IReadOnlyDictionary<string, string> Attr(
            params (string Name, object Value)[] values) =>
            values.ToDictionary(
                value => value.Name,
                value => Convert.ToString(value.Value, CultureInfo.InvariantCulture)!,
                StringComparer.Ordinal);
        static VmThreadGroup Grid(int count)
        {
            var groups = checked((uint)((count + 63) / 64));
            var x = Math.Min(groups, 65535u);
            return new(x, checked((groups + x - 1) / x));
        }

        Slot("audio_features", VmSlotScope.Local, VmAccess.ReadOnly, F(frames, FeatureCount));
        Slot("frame_count", VmSlotScope.Local, VmAccess.ReadOnly, I(1));
        Slot("normalized_features", VmSlotScope.Local, VmAccess.ReadWrite, F(frames, FeatureCount));
        var height2 = DivideRoundUp(frames, 2);
        var height4 = DivideRoundUp(frames, 4);
        var height8 = DivideRoundUp(frames, 8);
        Slot("subsample_1", VmSlotScope.Local, VmAccess.ReadWrite, F(height2, 40, AudioWidth));
        Slot("subsample_2_depthwise", VmSlotScope.Local, VmAccess.ReadWrite, F(height4, 20, AudioWidth));
        Slot("subsample_2", VmSlotScope.Local, VmAccess.ReadWrite, F(height4, 20, AudioWidth));
        Slot("subsample_3_depthwise", VmSlotScope.Local, VmAccess.ReadWrite, F(height8, 10, AudioWidth));
        Slot("subsample_3", VmSlotScope.Local, VmAccess.ReadWrite, F(height8, 10, AudioWidth));
        Slot("flattened", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, AudioWidth * 10));
        Slot("hidden", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, AudioWidth));
        Slot("normalized", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, AudioWidth));
        Slot("wide", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, intermediate * 2));
        Slot("intermediate", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, intermediate));
        Slot("update", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, AudioWidth));
        Slot("query", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, AudioWidth));
        Slot("key", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, AudioWidth));
        Slot("value", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, AudioWidth));
        Slot("attention", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, AudioWidth));
        Slot("conv_wide", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, AudioWidth * 2));
        Slot("conv_hidden", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, AudioWidth));
        Slot("conv_depthwise", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens + 2, AudioWidth));
        Slot("projector_hidden", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, TextWidth));
        Slot("projected", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, TextWidth));
        Slot("activation", VmSlotScope.Local, VmAccess.ReadWrite, F(tokens, TextWidth));

        var normalizeParameters = Parameters(
            ("input", VmAccess.ReadOnly, F(frames, FeatureCount)),
            ("mean", VmAccess.ReadOnly, F(FeatureCount)),
            ("inverse_standard_deviation", VmAccess.ReadOnly, F(FeatureCount)),
            ("frame_count", VmAccess.ReadOnly, I(1)),
            ("output", VmAccess.ReadWrite, F(frames, FeatureCount)));
        var normalizeKernel = Kernel(
            "normalize_features",
            normalizeParameters,
            Operation(Phi4AudioInstructionNames.NormalizeFeatures, normalizeParameters,
                Attr(("frames", frames), ("features", FeatureCount))));
        Dispatch(
            normalizeKernel,
            Grid(frames * FeatureCount),
            Arguments(
                ("input", "audio_features", 0),
                ("mean", Weight("a.global_mean"), 0),
                ("inverse_standard_deviation", Weight("a.global_invstd"), 0),
                ("frame_count", "frame_count", 0),
                ("output", "normalized_features", 0)),
            "normalized_features");

        void Conv2D(
            string prefix,
            string source,
            string destination,
            int inputHeight,
            int inputWidth,
            int inputChannels,
            int outputHeight,
            int outputWidth,
            int outputChannels,
            int kernelHeight,
            int kernelWidth,
            int stride,
            int padding,
            int groups,
            bool relu)
        {
            var parameters = Parameters(
                ("input", VmAccess.ReadOnly, F(inputHeight, inputWidth, inputChannels)),
                ("weight", VmAccess.ReadOnly,
                    F(outputChannels, inputChannels / groups, kernelHeight, kernelWidth)),
                ("bias", VmAccess.ReadOnly, F(outputChannels)),
                ("output", VmAccess.ReadWrite, F(outputHeight, outputWidth, outputChannels)));
            var kernel = Kernel(
                "conv2d",
                parameters,
                Operation(Phi4AudioInstructionNames.Conv2D, parameters,
                    Attr(
                        ("input_height", inputHeight), ("input_width", inputWidth),
                        ("input_channels", inputChannels), ("output_height", outputHeight),
                        ("output_width", outputWidth), ("output_channels", outputChannels),
                        ("kernel_height", kernelHeight), ("kernel_width", kernelWidth),
                        ("padding", padding), ("stride", stride), ("groups", groups),
                        ("activation", relu ? "relu" : "none"))));
            Dispatch(
                kernel,
                Grid(outputHeight * outputWidth * outputChannels),
                Arguments(
                    ("input", source, 0),
                    ("weight", Weight(prefix + ".weight", matrix: true), 0),
                    ("bias", Weight(prefix + ".bias"), 0),
                    ("output", destination, 0)),
                destination);
        }

        var zeroVectors = new Dictionary<int, string>();
        string Zero(int width)
        {
            if (zeroVectors.TryGetValue(width, out var existing))
                return existing;
            var id = $"zero_{width}";
            Slot(id, VmSlotScope.Global, VmAccess.ReadOnly, F(width));
            initializers.Add(new(id, new byte[checked(width * sizeof(float))]));
            zeroVectors.Add(width, id);
            return id;
        }

        void BiasActivation(
            string values,
            string bias,
            int width,
            int count,
            string activation)
        {
            var parameters = Parameters(
                ("input", VmAccess.ReadOnly, F(count)),
                ("bias", VmAccess.ReadOnly, F(width)),
                ("output", VmAccess.ReadWrite, F(count)));
            var kernel = Kernel(
                "bias_activation",
                parameters,
                Operation(Phi4AudioInstructionNames.BiasActivation, parameters,
                    Attr(("count", count), ("width", width), ("activation", activation))));
            Dispatch(
                kernel,
                Grid(count),
                Arguments(
                    ("input", values, 0),
                    ("bias", bias, 0),
                    ("output", "activation", 0)),
                "activation");
            Copy("activation", values, count);
        }

        Conv2D("a.conv1d.0", "normalized_features", "subsample_1",
            frames, FeatureCount, 1, height2, 40, AudioWidth, 3, 3, 2, 1, 1, relu: true);
        Conv2D("a.conv1d.2", "subsample_1", "subsample_2_depthwise",
            height2, 40, AudioWidth, height4, 20, AudioWidth, 3, 3, 2, 1,
            AudioWidth, relu: false);
        Conv2D("a.conv1d.3", "subsample_2_depthwise", "subsample_2",
            height4, 20, AudioWidth, height4, 20, AudioWidth, 1, 1, 1, 0, 1, relu: true);
        Conv2D("a.conv1d.5", "subsample_2", "subsample_3_depthwise",
            height4, 20, AudioWidth, height8, 10, AudioWidth, 3, 3, 2, 1,
            AudioWidth, relu: false);
        Conv2D("a.conv1d.6", "subsample_3_depthwise", "subsample_3",
            height8, 10, AudioWidth, height8, 10, AudioWidth, 1, 1, 1, 0, 1, relu: true);

        var flattenParameters = Parameters(
            ("input", VmAccess.ReadOnly, F(tokens, 10, AudioWidth)),
            ("output", VmAccess.ReadWrite, F(tokens, AudioWidth * 10)));
        var flattenKernel = Kernel(
            "flatten_subsampling",
            flattenParameters,
            Operation(Phi4AudioInstructionNames.FlattenSubsampling, flattenParameters,
                Attr(("tokens", tokens), ("frequency", 10), ("channels", AudioWidth))));
        Dispatch(
            flattenKernel,
            Grid(tokens * AudioWidth * 10),
            Arguments(("input", "subsample_3", 0), ("output", "flattened", 0)),
            "flattened");

        var matrixMultiplyKernels =
            new Dictionary<
                (int Rows, int Input, int Output, VmElementType Type, bool Affine),
                string>();
        string MatrixMultiplyKernel(
            int rows,
            int inputWidth,
            int outputWidth,
            VmElementType weightType,
            bool affine)
        {
            if (matrixMultiplyKernels.TryGetValue(
                    (rows, inputWidth, outputWidth, weightType, affine), out var existing))
                return existing;
            var parameterList = new List<VmParameter>
            {
                new("left", VmAccess.ReadOnly, F(rows, inputWidth)),
                new("right", VmAccess.ReadOnly,
                    new VmTensor(weightType, [outputWidth, inputWidth])),
            };
            if (affine)
                parameterList.Add(new("bias", VmAccess.ReadOnly, F(outputWidth)));
            parameterList.Add(new("output", VmAccess.ReadWrite, F(rows, outputWidth)));
            var parameters = parameterList.ToArray();
            var operation = new VmOperator(
                new(GraphElementType.Float32, GraphElementType.Float32),
                InstructionCollectionIds.TierZeroFloat32,
                affine ? "core.affine" : "core.matrix-multiply",
                parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)),
                new Dictionary<string, string>
                {
                    ["transpose_left"] = "false",
                    ["transpose_right"] = "true",
                },
                executionConfiguration: target == VmTarget.Direct3D12
                    ? GpuMatrixMultiplyExecution.Tiled
                    : null);
            var kernel = Kernel(
                affine ? "affine" : "matrix_multiply",
                parameters,
                operation,
                target == VmTarget.Direct3D12 ? new(8, 8) : null);
            matrixMultiplyKernels.Add(
                (rows, inputWidth, outputWidth, weightType, affine), kernel);
            return kernel;
        }

        void Linear(
            string source,
            string destination,
            string prefix,
            int rows,
            int inputWidth,
            int outputWidth,
            string activation = "none")
        {
            var tensor = omni.GetTensor(prefix + ".weight");
            var weightType = tensor.Type == GgufModelTensorType.Float16
                ? VmElementType.Float16
                : VmElementType.Float32;
            var affine = activation == "none";
            var kernel = MatrixMultiplyKernel(
                rows, inputWidth, outputWidth, weightType, affine);
            var weight = Weight(tensor.Name, matrix: true);
            var groups = GpuMatrixMultiplyExecution.TileGroups(rows, outputWidth);
            var arguments = new List<(string Parameter, string Source, ulong Offset)>
            {
                ("left", source, 0),
                ("right", weight, 0),
            };
            if (affine)
                arguments.Add(("bias", Weight(prefix + ".bias"), 0));
            arguments.Add(("output", destination, 0));
            Dispatch(
                kernel,
                new(groups.X, groups.Y),
                Arguments(arguments.ToArray()),
                destination);
            if (!affine)
                BiasActivation(
                    destination,
                    Weight(prefix + ".bias"),
                    outputWidth,
                    checked(rows * outputWidth),
                    activation);
        }

        void LayerNorm(string source, string destination, string prefix)
        {
            var parameters = Parameters(
                ("input", VmAccess.ReadOnly, F(tokens, AudioWidth)),
                ("weight", VmAccess.ReadOnly, F(AudioWidth)),
                ("bias", VmAccess.ReadOnly, F(AudioWidth)),
                ("output", VmAccess.ReadWrite, F(tokens, AudioWidth)));
            var kernel = Kernel(
                "layer_norm",
                parameters,
                Operation(Phi4AudioInstructionNames.LayerNorm, parameters,
                    Attr(("rows", tokens), ("width", AudioWidth), ("epsilon", 1e-5f))));
            Dispatch(
                kernel,
                new(checked((uint)tokens)),
                Arguments(
                    ("input", source, 0),
                    ("weight", Weight(prefix + ".weight"), 0),
                    ("bias", Weight(prefix + ".bias"), 0),
                    ("output", destination, 0)),
                destination);
        }

        void Residual(string update, float scale)
        {
            var count = checked(tokens * AudioWidth);
            var parameters = Parameters(
                ("hidden", VmAccess.ReadOnly, F(count)),
                ("update", VmAccess.ReadOnly, F(count)),
                ("output", VmAccess.ReadWrite, F(count)));
            var kernel = Kernel(
                "residual",
                parameters,
                Operation(Phi4AudioInstructionNames.Residual, parameters,
                    Attr(("count", count), ("scale", scale))));
            Dispatch(
                kernel,
                Grid(count),
                Arguments(
                    ("hidden", "hidden", 0),
                    ("update", update, 0),
                    ("output", "normalized", 0)),
                "normalized");
            Copy("normalized", "hidden", count);
        }

        void FeedForward(string prefix)
        {
            LayerNorm("hidden", "normalized", prefix + ".ln");
            Linear("normalized", "wide", prefix + ".up",
                tokens, AudioWidth, intermediate * 2);
            var parameters = Parameters(
                ("input", VmAccess.ReadOnly, F(tokens, intermediate * 2)),
                ("bias_first", VmAccess.ReadOnly, F(intermediate)),
                ("bias_second", VmAccess.ReadOnly, F(intermediate)),
                ("output", VmAccess.ReadWrite, F(tokens, intermediate)));
            var kernel = Kernel(
                "swi_glu",
                parameters,
                Operation(Phi4AudioInstructionNames.SwiGlu, parameters,
                    Attr(("rows", tokens), ("width", intermediate))));
            Dispatch(
                kernel,
                Grid(tokens * intermediate),
                Arguments(
                    ("input", "wide", 0),
                    ("bias_first", Zero(intermediate), 0),
                    ("bias_second", Zero(intermediate), 0),
                    ("output", "intermediate", 0)),
                "intermediate");
            // Linear already adds the packed 2*intermediate bias. SwiGlu therefore receives zero
            // bias vectors and only performs the split activation.
            Linear("intermediate", "update", prefix + ".down",
                tokens, intermediate, AudioWidth);
        }

        Linear("flattened", "hidden", "a.conv1d.out",
            tokens, AudioWidth * 10, AudioWidth);

        var relativeBias = Weight("a.rel_attn_bias");
        for (var layer = 0; layer < 24; layer++)
        {
            var prefix = $"a.blk.{layer}.";
            FeedForward(prefix + "ffn_in");
            Residual("update", 0.5f);

            LayerNorm("hidden", "normalized", prefix + "ln_att");
            Linear("normalized", "query", prefix + "attn_q", tokens, AudioWidth, AudioWidth);
            Linear("normalized", "key", prefix + "attn_k", tokens, AudioWidth, AudioWidth);
            Linear("normalized", "value", prefix + "attn_v", tokens, AudioWidth, AudioWidth);
            var attentionParameters = Parameters(
                ("query", VmAccess.ReadOnly, F(tokens, AudioWidth)),
                ("key", VmAccess.ReadOnly, F(tokens, AudioWidth)),
                ("value", VmAccess.ReadOnly, F(tokens, AudioWidth)),
                ("relative_bias", VmAccess.ReadOnly, F(1000, heads)),
                ("frame_count", VmAccess.ReadOnly, I(1)),
                ("output", VmAccess.ReadWrite, F(tokens, AudioWidth)));
            var attentionKernel = Kernel(
                "relative_attention",
                attentionParameters,
                Operation(Phi4AudioInstructionNames.RelativeAttention, attentionParameters,
                    Attr(
                        ("tokens", tokens), ("width", AudioWidth),
                        ("heads", heads), ("subsampling", 8))));
            var attentionGroups = GpuMatVecExecution.Groups(
                checked((ulong)tokens * (ulong)heads));
            Dispatch(
                attentionKernel,
                new(attentionGroups.X, attentionGroups.Y),
                Arguments(
                    ("query", "query", 0),
                    ("key", "key", 0),
                    ("value", "value", 0),
                    ("relative_bias", relativeBias, 0),
                    ("frame_count", "frame_count", 0),
                    ("output", "attention", 0)),
                "attention");
            Linear("attention", "update", prefix + "attn_out",
                tokens, AudioWidth, AudioWidth);
            Residual("update", 1f);

            LayerNorm("hidden", "normalized", prefix + "conv.ln");
            Linear("normalized", "conv_wide", prefix + "conv.glu.pw",
                tokens, AudioWidth, AudioWidth * 2);
            var convGluParameters = Parameters(
                ("input", VmAccess.ReadOnly, F(tokens, AudioWidth * 2)),
                ("bias_first", VmAccess.ReadOnly, F(AudioWidth)),
                ("bias_second", VmAccess.ReadOnly, F(AudioWidth)),
                ("output", VmAccess.ReadWrite, F(tokens, AudioWidth)));
            var convGluKernel = Kernel(
                "conv_glu",
                convGluParameters,
                Operation(Phi4AudioInstructionNames.SwiGlu, convGluParameters,
                    Attr(("rows", tokens), ("width", AudioWidth))));
            Dispatch(
                convGluKernel,
                Grid(tokens * AudioWidth),
                Arguments(
                    ("input", "conv_wide", 0),
                    ("bias_first", Weight(prefix + "conv.glu.b1"), 0),
                    ("bias_second", Weight(prefix + "conv.glu.b2"), 0),
                    ("output", "conv_hidden", 0)),
                "conv_hidden");
            Conv1D(prefix + "conv.dw", "conv_hidden", "conv_depthwise",
                tokens, AudioWidth, AudioWidth, 3, 2, AudioWidth);
            Conv1D(prefix + "conv.pw_mid", "conv_depthwise", "conv_hidden",
                tokens, AudioWidth, AudioWidth, 1, 0, 1, activation: "swish");
            Conv1D(prefix + "conv.pw_ext", "conv_hidden", "update",
                tokens, AudioWidth, AudioWidth, 1, 0, 1);
            Residual("update", 1f);

            FeedForward(prefix + "ffn_out");
            Residual("update", 0.5f);
            LayerNorm("hidden", "normalized", prefix + "ln");
            Copy("normalized", "hidden", tokens * AudioWidth);
        }

        void Conv1D(
            string prefix,
            string source,
            string destination,
            int length,
            int inputChannels,
            int outputChannels,
            int kernelWidth,
            int padding,
            int groups,
            string activation = "none")
        {
            var outputLength = checked((length + 2 * padding - kernelWidth) + 1);
            var parameters = Parameters(
                ("input", VmAccess.ReadOnly, F(length, inputChannels)),
                ("weight", VmAccess.ReadOnly,
                    F(outputChannels, inputChannels / groups, kernelWidth)),
                ("bias", VmAccess.ReadOnly, F(outputChannels)),
                ("output", VmAccess.ReadWrite, F(outputLength, outputChannels)));
            var kernel = Kernel(
                "conv1d",
                parameters,
                Operation(Phi4AudioInstructionNames.Conv1D, parameters,
                    Attr(
                        ("input_length", length), ("input_channels", inputChannels),
                        ("output_length", outputLength), ("output_channels", outputChannels),
                        ("kernel", kernelWidth), ("padding", padding),
                        ("stride", 1), ("groups", groups), ("activation", "none"))));
            Dispatch(
                kernel,
                Grid(outputLength * outputChannels),
                Arguments(
                    ("input", source, 0),
                    ("weight", Weight(prefix + ".weight", matrix: true), 0),
                    ("bias", Weight(prefix + ".bias"), 0),
                    ("output", destination, 0)),
                destination);
            if (activation != "none")
                BiasActivation(
                    destination, Zero(outputChannels), outputChannels,
                    outputLength * outputChannels, activation);
        }

        void Copy(string source, string destination, int count)
        {
            var parameters = Parameters(
                ("input", VmAccess.ReadOnly, F(count)),
                ("output", VmAccess.ReadWrite, F(count)));
            var operation = new VmOperator(
                new(GraphElementType.Float32, GraphElementType.Float32),
                InstructionCollectionIds.TierZeroFloat32,
                "core.copy",
                parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)));
            var kernel = Kernel("copy", parameters, operation);
            Dispatch(
                kernel,
                Grid(count),
                Arguments(("input", source, 0), ("output", destination, 0)),
                destination);
        }

        VmNode[] Project(string prefix)
        {
            var first = omni.GetTensor(prefix + ".0.weight");
            var hiddenWidth = CheckedInt(first.Dimensions[1]);
            Linear("hidden", "projector_hidden", prefix + ".0",
                tokens, AudioWidth, hiddenWidth, "gelu");
            Linear("projector_hidden", "projected", prefix + ".2",
                tokens, hiddenWidth, TextWidth);
            return nodes.ToArray();
        }

        var conformerNodes = nodes.ToArray();
        var speechNodes = Project("mm.a.mlp");
        nodes.Clear();
        nodes.AddRange(conformerNodes);
        previousNode = nodes.Count == 0 ? null : nodes[^1].Id;
        var visionNodes = Project("mm.a.vis");

        var allParameters = slots.Select(
            slot => new VmParameter(slot.Id, slot.Access, slot.Tensor)).ToArray();
        definitions.Add(new(
            "speech_forward",
            VmDefinitionKind.Orchestration,
            allParameters,
            speechNodes));
        definitions.Add(new(
            "vision_forward",
            VmDefinitionKind.Orchestration,
            allParameters,
            visionNodes));
        var allArguments = slots.Select(slot => new VmArgument(slot.Id, slot.Id)).ToArray();
        var entries = new[]
        {
            new VmEntry("speech", "speech_forward", allArguments),
            new VmEntry("vision", "vision_forward", allArguments),
        };
        return new(
            new VmProgram(
                $"phi4-audio-{frames}",
                "phi4.audio.d3d12",
                target,
                slots,
                definitions,
                entries,
                new("none", 1, [])),
            initializers);
    }

    private static float[] ToFloat32(GgufModelTensor tensor) =>
        tensor.Type switch
        {
            GgufModelTensorType.Float32 => tensor.FloatValues.ToArray(),
            GgufModelTensorType.Float16 =>
                tensor.HalfValues.ToArray().Select(value => (float)value).ToArray(),
            _ => throw new InvalidDataException(
                $"Phi-4 audio tensor '{tensor.Name}' must be Float16 or Float32."),
        };

    private static int CheckedInt(ulong value) =>
        value <= int.MaxValue
            ? (int)value
            : throw new InvalidDataException("Phi-4 tensor dimension exceeds Int32 capacity.");

    private static int DivideRoundUp(int value, int divisor) =>
        checked((value + divisor - 1) / divisor);

    private sealed class AudioRuntime : IDisposable
    {
        private readonly Phi4D3D12AudioComponent owner;
        private readonly D3D12VmExecutor executor;
        private readonly VmBindings bindings;
        private readonly int featuresSlot;
        private readonly int frameCountSlot;
        private readonly int outputSlot;
        private bool initialized;

        public AudioRuntime(
            Phi4D3D12AudioComponent owner,
            D3D12VmArtifact artifact)
        {
            this.owner = owner;
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
            featuresSlot = SlotIndex("audio_features");
            frameCountSlot = SlotIndex("frame_count");
            outputSlot = SlotIndex("projected");
        }

        public ulong SubmissionCount => executor.SubmissionCount;

        public void Execute(
            Phi4AudioFeatures features,
            Phi4AudioProjector projector,
            AudioStorageHandle output)
        {
            using var outputResource = output.Resource.Retain();
            bindings.Bind("projected", outputResource);
            try
            {
                using var execution = bindings.BeginExecution();
                var context = execution.GetBuffers();
                context[featuresSlot].AsSpan().Clear();
                MemoryMarshal.AsBytes(features.Values.AsSpan())
                    .CopyTo(context[featuresSlot]);
                var actualFrameCount = features.FrameCount;
                MemoryMarshal.Write(context[frameCountSlot], in actualFrameCount);
                if (!initialized)
                {
                    executor.UploadSlots(
                        context, Enumerable.Range(0, context.Length).ToArray());
                    initialized = true;
                }
                else
                {
                    executor.UploadSlots(
                        context, [featuresSlot, frameCountSlot, outputSlot]);
                }
                executor.Execute(
                    projector == Phi4AudioProjector.Vision ? "vision" : "speech");
            }
            finally
            {
                BindPlaceholder("projected");
            }
        }

        public byte[] Readback(AudioStorageHandle output)
        {
            using var outputResource = output.Resource.Retain();
            bindings.Bind("projected", outputResource);
            try
            {
                using var execution = bindings.BeginExecution();
                var context = execution.GetBuffers();
                executor.ReadbackSlots(context, [outputSlot]);
                return context[outputSlot].ToArray();
            }
            finally
            {
                BindPlaceholder("projected");
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
                new VmSlot(
                    $"placeholder_{Guid.NewGuid():N}",
                    VmSlotScope.Local,
                    slot.Access,
                    slot.Tensor));
            using var resource = new VmResource(
                slot.Tensor, VmSlotScope.Local, VmAccess.ReadWrite, storage);
            using var lease = resource.Acquire();
            bindings.Bind(slotId, lease);
        }
    }

    private sealed class AudioStorageHandle : IPhi4AudioEmbeddingHandle, IVmBindableStorageHandle, IDisposable
    {
        private VmResourceLease? resource;

        public AudioStorageHandle(
            Guid componentId,
            string semanticName,
            StorageDomain domain,
            TensorDescriptor descriptor,
            long byteLength,
            VmResourceLease resource,
            int validTokenCount,
            int tokenBucket,
            Phi4AudioProjector projector)
        {
            ComponentId = componentId;
            SemanticName = semanticName;
            Domain = domain;
            Descriptor = descriptor;
            ByteLength = byteLength;
            this.resource = resource;
            ValidTokenCount = validTokenCount;
            TokenBucket = tokenBucket;
            Projector = projector;
        }

        public Guid Id { get; } = Guid.NewGuid();
        public Guid ComponentId { get; }
        public string SemanticName { get; }
        public StorageDomain Domain { get; }
        public TensorDescriptor Descriptor { get; }
        public long ByteLength { get; }
        public int ValidTokenCount { get; }
        public int TokenBucket { get; }
        public Phi4AudioProjector Projector { get; }
        public VmResourceLease Resource =>
            resource ?? throw new ObjectDisposedException(nameof(AudioStorageHandle));

        public void Bind(VmBindings bindings, string slotId)
        {
            using var retained = Resource.Retain();
            bindings.Bind(slotId, retained);
        }

        public void Dispose() =>
            Interlocked.Exchange(ref resource, null)?.Dispose();
    }

    internal sealed record Initializer(string Slot, byte[] Bytes);
    internal sealed record BuildResult(VmProgram Program, IReadOnlyList<Initializer> Initializers);
}
