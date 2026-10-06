using System.Buffers.Binary;
using System.Runtime.InteropServices;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Gguf;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime;
using SharpInference.Vm;

namespace SharpInference.Architectures.Phi4.D3D12;

public sealed class Phi4D3D12TextSession : IDisposable
{
    private const int EmbeddingSize = 3072;
    private const int QuerySize = 3072;
    private const int KeyValueSize = 1024;
    private const int QkvSize = QuerySize + 2 * KeyValueSize;
    private const int IntermediateSize = 8192;
    private const int VocabularySize = 200064;
    private const int LayerCount = 32;
    private const int QueryHeads = 24;
    private const int KeyValueHeads = 8;
    private const int HeadSize = 128;
    private const int RotarySize = 96;
    private const int PrefillBatchSize = 16;
    private readonly D3D12VmExecutor executor = null!;
    private readonly byte[] control = new byte[2 * sizeof(int)];
    private readonly byte[] embeddingControl =
        new byte[EmbeddingSize * sizeof(float) + sizeof(int)];
    private readonly byte[] tokenOutput = new byte[sizeof(int)];
    private readonly VmResourceManager? resourceManager;
    private readonly VmSessionResources? sessionResources;
    private readonly VmBindings? bindings;
    private readonly VmExecutionLease? execution;
    private readonly byte[][]? pooledBuffers;
    private readonly int fusedTokenCount;
    private bool disposed;

    public Phi4D3D12TextSession(
        Phi4ModelPackage package,
        int maximumContext = 4096,
        int adapterIndex = 0,
        Phi4Adapter adapter = Phi4Adapter.None)
    {
        ArgumentNullException.ThrowIfNull(package);
        if (maximumContext <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumContext));
        MaximumContext = maximumContext;
        var build = Build(package, maximumContext, adapter, fusedTokenCount: 0);
        executor = new D3D12VmCompiler(DefaultInstructionCollections.Create())
            .Compile(build.Program)
            .CreateExecutorForExplicitInitialization(adapterIndex);
        try
        {
            foreach (var (slot, tensor) in build.Weights)
                executor.Upload(slot, tensor.Bytes);
            foreach (var (slot, bytes) in build.Initializers)
                executor.Upload(slot, bytes);
        }
        catch
        {
            executor.Dispose();
            throw;
        }
    }

    public Phi4D3D12TextSession(
        Phi4ModelPackage package,
        D3D12VmResourcePool pool,
        Phi4D3D12FusionComponent fusion,
        IStorageLease fusedEmbeddings,
        int maximumContext = 4096,
        Phi4Adapter adapter = Phi4Adapter.None)
    {
        ArgumentNullException.ThrowIfNull(package);
        ArgumentNullException.ThrowIfNull(pool);
        ArgumentNullException.ThrowIfNull(fusion);
        ArgumentNullException.ThrowIfNull(fusedEmbeddings);
        if (maximumContext <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumContext));
        if (fusedEmbeddings.Handle is not IPhi4FusedEmbeddingHandle fusedHandle)
            throw new ArgumentException(
                "A Phi-4 fused embedding lease is required.", nameof(fusedEmbeddings));
        MaximumContext = maximumContext;
        fusedTokenCount = fusedHandle.TokenCount;
        if (fusedTokenCount > maximumContext)
            throw new ArgumentException(
                "The fused prompt exceeds the decoder context.", nameof(fusedEmbeddings));
        var build = Build(package, maximumContext, adapter, fusedTokenCount);
        var weights = build.Weights.ToDictionary(
            item => item.Slot,
            item => item.Tensor,
            StringComparer.Ordinal);
        var initializers = build.Initializers.ToDictionary(
            item => item.Slot,
            item => item.Bytes,
            StringComparer.Ordinal);
        resourceManager = new VmResourceManager(
            (slot, storage) =>
            {
                if (weights.TryGetValue(slot.Id, out var tensor))
                    storage.Write(0, tensor.Bytes);
                else if (initializers.TryGetValue(slot.Id, out var bytes))
                    storage.Write(0, bytes);
                else
                    throw new InvalidDataException(
                        $"No initializer exists for global slot '{slot.Id}'.");
            },
            pool.Allocate);
        try
        {
            bindings = resourceManager.CreateBindings(build.Program);
            sessionResources = resourceManager.CreateSession(build.Program);
            resourceManager.BindSession(bindings, sessionResources);
            fusion.BindOutput(bindings, "fused_embeddings", fusedEmbeddings);
            executor = new D3D12VmCompiler(DefaultInstructionCollections.Create())
                .Compile(build.Program)
                .CreateExecutor(pool);
            execution = bindings.BeginExecution();
            pooledBuffers = execution.GetBuffers();
            var fusedSlot = build.Program.Slots.ToList().FindIndex(
                slot => slot.Id == "fused_embeddings");
            executor.UploadSlots(
                pooledBuffers,
                Enumerable.Range(0, pooledBuffers.Length)
                    .Where(index => index != fusedSlot)
                    .ToArray());
        }
        catch
        {
            execution?.Dispose();
            executor?.Dispose();
            bindings?.Dispose();
            sessionResources?.Dispose();
            resourceManager.Dispose();
            throw;
        }
    }

    public int MaximumContext { get; }
    public int Position { get; private set; }
    public ulong QueueSubmissionCount => executor.SubmissionCount;

    public int ForwardToken(int token)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if ((uint)token >= VocabularySize)
            throw new ArgumentOutOfRangeException(nameof(token));
        if (Position >= MaximumContext)
            throw new InvalidOperationException(
                $"The GPU KV cache capacity of {MaximumContext} tokens has been reached.");
        BinaryPrimitives.WriteInt32LittleEndian(control, token);
        BinaryPrimitives.WriteInt32LittleEndian(control.AsSpan(sizeof(int)), Position);
        executor.ExecuteWithTransfers(
            "decode", "control", control, "next_token", tokenOutput);
        Position++;
        return BinaryPrimitives.ReadInt32LittleEndian(tokenOutput);
    }

    public int ForwardEmbedding(ReadOnlySpan<float> embedding)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (embedding.Length != EmbeddingSize)
            throw new ArgumentException(
                $"Expected {EmbeddingSize} embedding values.", nameof(embedding));
        if (Position >= MaximumContext)
            throw new InvalidOperationException(
                $"The GPU KV cache capacity of {MaximumContext} tokens has been reached.");
        MemoryMarshal.AsBytes(embedding).CopyTo(embeddingControl);
        BinaryPrimitives.WriteInt32LittleEndian(
            embeddingControl.AsSpan(EmbeddingSize * sizeof(float)), Position);
        executor.ExecuteWithTransfers(
            "decode_embedding",
            "embedding_control",
            embeddingControl,
            "next_token",
            tokenOutput);
        Position++;
        return BinaryPrimitives.ReadInt32LittleEndian(tokenOutput);
    }

    public int ForwardFusedEmbedding()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (pooledBuffers is null)
            throw new InvalidOperationException(
                "This session was not created with fused GPU embeddings.");
        if (Position >= fusedTokenCount)
            throw new InvalidOperationException(
                "All fused prompt embeddings have already been consumed.");
        BinaryPrimitives.WriteInt32LittleEndian(control, 0);
        BinaryPrimitives.WriteInt32LittleEndian(control.AsSpan(sizeof(int)), Position);
        executor.ExecuteWithTransfers(
            "decode_fused", "control", control, "next_token", tokenOutput);
        Position++;
        return BinaryPrimitives.ReadInt32LittleEndian(tokenOutput);
    }

    public int PrefillFusedEmbeddings()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (pooledBuffers is null)
            throw new InvalidOperationException(
                "This session was not created with fused GPU embeddings.");
        if (Position >= fusedTokenCount)
            throw new InvalidOperationException(
                "All fused prompt embeddings have already been consumed.");
        while (Position < fusedTokenCount)
        {
            var count = Math.Min(PrefillBatchSize, fusedTokenCount - Position);
            BinaryPrimitives.WriteInt32LittleEndian(control, Position);
            BinaryPrimitives.WriteInt32LittleEndian(
                control.AsSpan(sizeof(int)), count);
            executor.ExecuteWithTransfers(
                "prefill_fused_16", "control", control, "next_token", tokenOutput);
            Position += count;
        }
        return BinaryPrimitives.ReadInt32LittleEndian(tokenOutput);
    }

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        Position = 0;
    }

    public void Dispose()
    {
        if (disposed)
            return;
        execution?.Dispose();
        executor.Dispose();
        bindings?.Dispose();
        sessionResources?.Dispose();
        resourceManager?.Dispose();
        disposed = true;
    }

    private static BuildResult Build(
        Phi4ModelPackage package,
        int context,
        Phi4Adapter adapter,
        int fusedTokenCount)
    {
        var slots = new List<VmSlot>();
        var weights = new List<(string Slot, GgufModelTensor Tensor)>();
        var initializers = new List<(string Slot, byte[] Bytes)>();
        var definitions = new List<VmDefinition>();
        var adapterFile = adapter switch
        {
            Phi4Adapter.None => null,
            Phi4Adapter.Vision => package.VisionAdapter,
            Phi4Adapter.Speech => package.SpeechAdapter,
            _ => throw new ArgumentOutOfRangeException(nameof(adapter)),
        };
        var adapterRank = adapter switch
        {
            Phi4Adapter.None => 0,
            Phi4Adapter.Vision => 256,
            Phi4Adapter.Speech => 320,
            _ => throw new ArgumentOutOfRangeException(nameof(adapter)),
        };
        var nodes = new List<VmNode>();
        var state = new List<VmStateEntry>();
        var nodeIndex = 0;

        VmTensor F(params int[] dimensions) => new(VmElementType.Float32, dimensions);
        VmTensor H(params int[] dimensions) => new(VmElementType.Float16, dimensions);
        VmTensor I(params int[] dimensions) => new(VmElementType.Int32, dimensions);
        void Slot(string id, VmSlotScope scope, VmAccess access, VmTensor tensor) =>
            slots.Add(new(id, scope, access, tensor));
        void Weight(string id, GgufModelTensor tensor, VmTensor descriptor)
        {
            Slot(id, VmSlotScope.Global, VmAccess.ReadOnly, descriptor);
            weights.Add((id, tensor));
        }
        void AdapterWeights(string id, string baseName, int input, int output)
        {
            var file = adapterFile ??
                throw new InvalidOperationException("An adapter file is required.");
            Weight($"{id}_lora_a", file.GetTensor(baseName + ".lora_a"),
                H(adapterRank, input));
            Weight($"{id}_lora_b", file.GetTensor(baseName + ".lora_b"),
                H(output, adapterRank));
        }
        VmParameter[] Parameters(params (string Name, VmAccess Access, VmTensor Tensor)[] values) =>
            values.Select(value => new VmParameter(value.Name, value.Access, value.Tensor)).ToArray();
        VmArgument[] Arguments(params (string Parameter, string Source)[] values) =>
            values.Select(value => new VmArgument(value.Parameter, value.Source)).ToArray();
        string Kernel(
            string id,
            VmParameter[] parameters,
            VmOperator operation,
            uint threads = 64)
        {
            definitions.Add(new(id, VmDefinitionKind.Kernel, parameters,
                [new("body", operation)], new(threads)));
            return id;
        }
        void Dispatch(
            string kernel,
            VmThreadGroup groups,
            VmArgument[] arguments,
            params string[] writes)
        {
            var dispatchId = $"dispatch_{nodeIndex++}";
            nodes.Add(new(dispatchId, new VmDispatch(kernel, arguments, groups)));
            nodes.Add(new($"barrier_{nodeIndex++}", new VmBarrier(writes), [dispatchId]));
        }
        VmOperator Operation(
            string name,
            VmParameter[] parameters,
            IReadOnlyDictionary<string, string>? attributes = null,
            InstructionExecutionConfiguration? configuration = null,
            Guid? collection = null) =>
            new(new(GraphElementType.Float32, GraphElementType.Float32),
                collection ?? InstructionCollectionIds.TransformerFloat32,
                name,
                parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)),
                attributes,
                executionConfiguration: configuration);

        var tokenEmbedding = package.Text.GetTensor("token_embd.weight");
        Weight("token_embedding", tokenEmbedding, H(VocabularySize, EmbeddingSize));
        Weight("output_norm", package.Text.GetTensor("output_norm.weight"), F(EmbeddingSize));
        var ropeBase = package.Text.GetMetadata<float>("phi3.rope.freq_base");
        var frequencies = Enumerable.Range(0, RotarySize / 2)
            .Select(index => 1f / MathF.Pow(ropeBase, 2f * index / RotarySize))
            .ToArray();
        Slot("rope_frequencies", VmSlotScope.Global, VmAccess.ReadOnly, F(RotarySize / 2));
        initializers.Add(("rope_frequencies",
            MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray()));
        for (var layer = 0; layer < LayerCount; layer++)
        {
            Weight($"attn_norm_{layer}", package.Text.GetTensor($"blk.{layer}.attn_norm.weight"), F(EmbeddingSize));
            Weight($"ffn_norm_{layer}", package.Text.GetTensor($"blk.{layer}.ffn_norm.weight"), F(EmbeddingSize));
            Weight($"qkv_weight_{layer}", package.Text.GetTensor($"blk.{layer}.attn_qkv.weight"), H(QkvSize, EmbeddingSize));
            Weight($"attn_output_weight_{layer}", package.Text.GetTensor($"blk.{layer}.attn_output.weight"), H(EmbeddingSize, EmbeddingSize));
            Weight($"ffn_up_weight_{layer}", package.Text.GetTensor($"blk.{layer}.ffn_up.weight"), H(IntermediateSize * 2, EmbeddingSize));
            Weight($"ffn_down_weight_{layer}", package.Text.GetTensor($"blk.{layer}.ffn_down.weight"), H(EmbeddingSize, IntermediateSize));
            if (adapterFile is not null)
            {
                AdapterWeights($"qkv_{layer}", $"blk.{layer}.attn_qkv.weight",
                    EmbeddingSize, QkvSize);
                AdapterWeights($"attn_output_{layer}", $"blk.{layer}.attn_output.weight",
                    EmbeddingSize, EmbeddingSize);
                AdapterWeights($"ffn_up_{layer}", $"blk.{layer}.ffn_up.weight",
                    EmbeddingSize, IntermediateSize * 2);
                AdapterWeights($"ffn_down_{layer}", $"blk.{layer}.ffn_down.weight",
                    IntermediateSize, EmbeddingSize);
            }
            Slot($"key_cache_{layer}", VmSlotScope.Session, VmAccess.ReadWrite,
                F(context, KeyValueHeads, HeadSize));
            Slot($"value_cache_{layer}", VmSlotScope.Session, VmAccess.ReadWrite,
                F(context, KeyValueHeads, HeadSize));
            state.Add(new($"key_cache_{layer}", $"key_cache_{layer}"));
            state.Add(new($"value_cache_{layer}", $"value_cache_{layer}"));
        }

        Slot("control", VmSlotScope.Local, VmAccess.ReadOnly, I(2));
        Slot("embedding_control", VmSlotScope.Local, VmAccess.ReadOnly,
            new(VmElementType.Byte, [EmbeddingSize * sizeof(float) + sizeof(int)]));
        if (fusedTokenCount > 0)
            Slot("fused_embeddings", VmSlotScope.Local, VmAccess.ReadWrite,
                F(fusedTokenCount, EmbeddingSize));
        Slot("hidden", VmSlotScope.Local, VmAccess.ReadWrite, F(EmbeddingSize));
        Slot("hidden_after_attention", VmSlotScope.Local, VmAccess.ReadWrite, F(EmbeddingSize));
        Slot("normalized", VmSlotScope.Local, VmAccess.ReadWrite, F(EmbeddingSize));
        Slot("qkv", VmSlotScope.Local, VmAccess.ReadWrite, F(QkvSize));
        Slot("query", VmSlotScope.Local, VmAccess.ReadWrite, F(QuerySize));
        Slot("scores", VmSlotScope.Local, VmAccess.ReadWrite, F(QueryHeads, context));
        Slot("probabilities", VmSlotScope.Local, VmAccess.ReadWrite, F(QueryHeads, context));
        Slot("attention", VmSlotScope.Local, VmAccess.ReadWrite, F(EmbeddingSize));
        Slot("projected", VmSlotScope.Local, VmAccess.ReadWrite, F(EmbeddingSize));
        Slot("gate_up", VmSlotScope.Local, VmAccess.ReadWrite, F(IntermediateSize * 2));
        Slot("activated", VmSlotScope.Local, VmAccess.ReadWrite, F(IntermediateSize));
        Slot("down", VmSlotScope.Local, VmAccess.ReadWrite, F(EmbeddingSize));
        if (adapterFile is not null)
        {
            Slot("lora_low", VmSlotScope.Local, VmAccess.ReadWrite, F(adapterRank));
            Slot("qkv_base", VmSlotScope.Local, VmAccess.ReadWrite, F(QkvSize));
            Slot("qkv_update", VmSlotScope.Local, VmAccess.ReadWrite, F(QkvSize));
            Slot("projected_base", VmSlotScope.Local, VmAccess.ReadWrite, F(EmbeddingSize));
            Slot("projected_update", VmSlotScope.Local, VmAccess.ReadWrite, F(EmbeddingSize));
            Slot("gate_up_base", VmSlotScope.Local, VmAccess.ReadWrite, F(IntermediateSize * 2));
            Slot("gate_up_update", VmSlotScope.Local, VmAccess.ReadWrite, F(IntermediateSize * 2));
            Slot("down_base", VmSlotScope.Local, VmAccess.ReadWrite, F(EmbeddingSize));
            Slot("down_update", VmSlotScope.Local, VmAccess.ReadWrite, F(EmbeddingSize));
        }
        if (fusedTokenCount > 0)
        {
            Slot("batch_hidden", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, EmbeddingSize));
            Slot("batch_hidden_after_attention", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, EmbeddingSize));
            Slot("batch_normalized", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, EmbeddingSize));
            Slot("batch_qkv", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, QkvSize));
            Slot("batch_query", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, QuerySize));
            Slot("batch_scores", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, QueryHeads, context));
            Slot("batch_probabilities", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, QueryHeads, context));
            Slot("batch_attention", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, EmbeddingSize));
            Slot("batch_projected", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, EmbeddingSize));
            Slot("batch_gate_up", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, IntermediateSize * 2));
            Slot("batch_activated", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, IntermediateSize));
            Slot("batch_down", VmSlotScope.Local, VmAccess.ReadWrite,
                F(PrefillBatchSize, EmbeddingSize));
            Slot("batch_last", VmSlotScope.Local, VmAccess.ReadWrite, F(EmbeddingSize));
            if (adapterFile is not null)
            {
                Slot("batch_lora_low", VmSlotScope.Local, VmAccess.ReadWrite,
                    F(PrefillBatchSize, adapterRank));
                Slot("batch_qkv_base", VmSlotScope.Local, VmAccess.ReadWrite,
                    F(PrefillBatchSize, QkvSize));
                Slot("batch_qkv_update", VmSlotScope.Local, VmAccess.ReadWrite,
                    F(PrefillBatchSize, QkvSize));
                Slot("batch_projected_base", VmSlotScope.Local, VmAccess.ReadWrite,
                    F(PrefillBatchSize, EmbeddingSize));
                Slot("batch_projected_update", VmSlotScope.Local, VmAccess.ReadWrite,
                    F(PrefillBatchSize, EmbeddingSize));
                Slot("batch_gate_up_base", VmSlotScope.Local, VmAccess.ReadWrite,
                    F(PrefillBatchSize, IntermediateSize * 2));
                Slot("batch_gate_up_update", VmSlotScope.Local, VmAccess.ReadWrite,
                    F(PrefillBatchSize, IntermediateSize * 2));
                Slot("batch_down_base", VmSlotScope.Local, VmAccess.ReadWrite,
                    F(PrefillBatchSize, EmbeddingSize));
                Slot("batch_down_update", VmSlotScope.Local, VmAccess.ReadWrite,
                    F(PrefillBatchSize, EmbeddingSize));
            }
        }
        Slot("logits", VmSlotScope.Local, VmAccess.ReadWrite, F(VocabularySize));
        Slot("next_token", VmSlotScope.Local, VmAccess.ReadWrite, I(1));

        var embeddingParameters = Parameters(
            ("table", VmAccess.ReadOnly, H(VocabularySize, EmbeddingSize)),
            ("index", VmAccess.ReadOnly, I(1)),
            ("output", VmAccess.ReadWrite, F(EmbeddingSize)));
        var embeddingKernel = Kernel("embedding", embeddingParameters,
            Operation("core.gather-row", embeddingParameters,
                collection: InstructionCollectionIds.TierZeroFloat32));
        string? fusedEmbeddingKernel = null;
        if (fusedTokenCount > 0)
        {
            var fusedEmbeddingParameters = Parameters(
                ("table", VmAccess.ReadOnly, F(fusedTokenCount, EmbeddingSize)),
                ("index", VmAccess.ReadOnly, I(1)),
                ("output", VmAccess.ReadWrite, F(EmbeddingSize)));
            fusedEmbeddingKernel = Kernel("fused_embedding", fusedEmbeddingParameters,
                Operation("core.gather-row", fusedEmbeddingParameters,
                    collection: InstructionCollectionIds.TierZeroFloat32));
        }
        var copyParameters = Parameters(
            ("input", VmAccess.ReadOnly, F(EmbeddingSize)),
            ("output", VmAccess.ReadWrite, F(EmbeddingSize)));
        var copyKernel = Kernel("copy_embedding", copyParameters,
            Operation("core.copy", copyParameters,
                collection: InstructionCollectionIds.TierZeroFloat32));
        var rmsParameters = Parameters(
            ("input", VmAccess.ReadOnly, F(EmbeddingSize)),
            ("weight", VmAccess.ReadOnly, F(EmbeddingSize)),
            ("output", VmAccess.ReadWrite, F(EmbeddingSize)));
        var rmsKernel = Kernel("rms_norm", rmsParameters,
            Operation("transformer.rms-norm", rmsParameters,
                new Dictionary<string, string>
                {
                    ["epsilon"] = package.Text.GetMetadata<float>(
                        "phi3.attention.layer_norm_rms_epsilon").ToString(
                            "R", System.Globalization.CultureInfo.InvariantCulture),
                }));
        var residualRmsParameters = Parameters(
            ("hidden", VmAccess.ReadOnly, F(EmbeddingSize)),
            ("residual", VmAccess.ReadOnly, F(EmbeddingSize)),
            ("weight", VmAccess.ReadOnly, F(EmbeddingSize)),
            ("updated", VmAccess.ReadWrite, F(EmbeddingSize)),
            ("output", VmAccess.ReadWrite, F(EmbeddingSize)));
        var residualRmsKernel = Kernel("residual_rms_norm", residualRmsParameters,
            Operation("transformer.residual-rms-norm", residualRmsParameters,
                new Dictionary<string, string>
                {
                    ["epsilon"] = package.Text.GetMetadata<float>(
                        "phi3.attention.layer_norm_rms_epsilon").ToString(
                            "R", System.Globalization.CultureInfo.InvariantCulture),
                }));

        string MatVecKernel(string id, int input, int output)
        {
            var parameters = Parameters(
                ("weight", VmAccess.ReadOnly, H(output, input)),
                ("input", VmAccess.ReadOnly, F(input)),
                ("output", VmAccess.ReadWrite, F(output)));
            return Kernel(id, parameters,
                Operation("core.mat-vec", parameters,
                    configuration: GpuMatVecExecution.Cooperative,
                    collection: InstructionCollectionIds.TierZeroFloat32));
        }
        var qkvKernel = MatVecKernel("qkv_matvec", EmbeddingSize, QkvSize);
        var attentionOutputKernel = MatVecKernel("attention_output_matvec", EmbeddingSize, EmbeddingSize);
        var ffnUpKernel = MatVecKernel("ffn_up_matvec", EmbeddingSize, IntermediateSize * 2);
        var ffnDownKernel = MatVecKernel("ffn_down_matvec", IntermediateSize, EmbeddingSize);
        var vocabularyKernel = MatVecKernel("vocabulary_matvec", EmbeddingSize, VocabularySize);
        var loraDownKernels = adapterFile is null
            ? new Dictionary<int, string>()
            : new[] { EmbeddingSize, IntermediateSize }
                .Distinct()
                .ToDictionary(size => size,
                    size => MatVecKernel($"lora_down_{size}", size, adapterRank));
        var loraUpKernels = adapterFile is null
            ? new Dictionary<int, string>()
            : new[] { QkvSize, EmbeddingSize, IntermediateSize * 2 }
                .Distinct()
                .ToDictionary(size => size,
                    size => MatVecKernel($"lora_up_{size}", adapterRank, size));

        string ScaledAddKernel(string id, int size)
        {
            var parameters = Parameters(
                ("input", VmAccess.ReadOnly, F(size)),
                ("update", VmAccess.ReadOnly, F(size)),
                ("output", VmAccess.ReadWrite, F(size)));
            return Kernel(id, parameters,
                Operation("transformer.scaled-add", parameters,
                    new Dictionary<string, string> { ["scale"] = "2" }));
        }
        var scaledAddKernels = adapterFile is null
            ? new Dictionary<int, string>()
            : new[] { QkvSize, EmbeddingSize, IntermediateSize * 2 }
                .Distinct()
                .ToDictionary(size => size,
                    size => ScaledAddKernel($"lora_add_{size}", size));

        var ropeParameters = Parameters(
            ("qkv", VmAccess.ReadOnly, F(QkvSize)),
            ("position", VmAccess.ReadOnly, I(1)),
            ("frequencies", VmAccess.ReadOnly, F(RotarySize / 2)),
            ("query", VmAccess.ReadWrite, F(QuerySize)),
            ("key_cache", VmAccess.ReadWrite, F(context, KeyValueHeads, HeadSize)),
            ("value_cache", VmAccess.ReadWrite, F(context, KeyValueHeads, HeadSize)));
        var ropeKernel = Kernel("rope_kv_write", ropeParameters,
            Operation("transformer.rope-kv-write", ropeParameters,
                new Dictionary<string, string>
                {
                    ["head_size"] = HeadSize.ToString(),
                    ["rotary_size"] = RotarySize.ToString(),
                    ["rope_scale"] = package.Text.GetMetadata<float>("phi3.rope.scaling.attn_factor")
                        .ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                }));
        var scoreParameters = Parameters(
            ("query", VmAccess.ReadOnly, F(QuerySize)),
            ("key_cache", VmAccess.ReadOnly, F(context, KeyValueHeads, HeadSize)),
            ("position", VmAccess.ReadOnly, I(1)),
            ("scores", VmAccess.ReadWrite, F(QueryHeads, context)));
        var scoreKernel = Kernel("gqa_scores", scoreParameters,
            Operation("transformer.gqa-scores", scoreParameters));
        var softmaxParameters = Parameters(
            ("scores", VmAccess.ReadOnly, F(QueryHeads, context)),
            ("position", VmAccess.ReadOnly, I(1)),
            ("probabilities", VmAccess.ReadWrite, F(QueryHeads, context)));
        var softmaxKernel = Kernel("causal_softmax", softmaxParameters,
            Operation("transformer.causal-softmax", softmaxParameters));
        var valueParameters = Parameters(
            ("probabilities", VmAccess.ReadOnly, F(QueryHeads, context)),
            ("value_cache", VmAccess.ReadOnly, F(context, KeyValueHeads, HeadSize)),
            ("position", VmAccess.ReadOnly, I(1)),
            ("output", VmAccess.ReadWrite, F(EmbeddingSize)));
        var valueKernel = Kernel("gqa_values", valueParameters,
            Operation("transformer.gqa-values", valueParameters));
        var swigluParameters = Parameters(
            ("gate_up", VmAccess.ReadOnly, F(IntermediateSize * 2)),
            ("output", VmAccess.ReadWrite, F(IntermediateSize)));
        var swigluKernel = Kernel("swiglu", swigluParameters,
            Operation("transformer.swiglu", swigluParameters));
        var argmaxParameters = Parameters(
            ("input", VmAccess.ReadOnly, F(VocabularySize)),
            ("output", VmAccess.ReadWrite, I(1)));
        var argmaxKernel = Kernel("argmax", argmaxParameters,
            Operation("transformer.argmax", argmaxParameters));

        string? batchGatherKernel = null;
        string? batchRmsKernel = null;
        string? batchResidualRmsKernel = null;
        string? batchSwiGluKernel = null;
        string? batchRopeKernel = null;
        string? batchScoreKernel = null;
        string? batchSoftmaxKernel = null;
        string? batchValueKernel = null;
        string? batchSelectLastKernel = null;
        var batchMatVecKernels = new Dictionary<(int Input, int Output), string>();
        var batchScaledAddKernels = new Dictionary<int, string>();
        if (fusedTokenCount > 0)
        {
            var epsilon = package.Text.GetMetadata<float>(
                    "phi3.attention.layer_norm_rms_epsilon")
                .ToString("R", System.Globalization.CultureInfo.InvariantCulture);
            var batchGatherParameters = Parameters(
                ("table", VmAccess.ReadOnly, F(fusedTokenCount, EmbeddingSize)),
                ("control", VmAccess.ReadOnly, I(2)),
                ("output", VmAccess.ReadWrite, F(PrefillBatchSize, EmbeddingSize)));
            batchGatherKernel = Kernel("batch_gather", batchGatherParameters,
                Operation("transformer.batch-gather-rows", batchGatherParameters));
            var batchRmsParameters = Parameters(
                ("input", VmAccess.ReadOnly, F(PrefillBatchSize, EmbeddingSize)),
                ("weight", VmAccess.ReadOnly, F(EmbeddingSize)),
                ("control", VmAccess.ReadOnly, I(2)),
                ("output", VmAccess.ReadWrite, F(PrefillBatchSize, EmbeddingSize)));
            batchRmsKernel = Kernel("batch_rms_norm", batchRmsParameters,
                Operation("transformer.batch-rms-norm", batchRmsParameters,
                    new Dictionary<string, string> { ["epsilon"] = epsilon }));
            var batchResidualParameters = Parameters(
                ("hidden", VmAccess.ReadOnly, F(PrefillBatchSize, EmbeddingSize)),
                ("residual", VmAccess.ReadOnly, F(PrefillBatchSize, EmbeddingSize)),
                ("weight", VmAccess.ReadOnly, F(EmbeddingSize)),
                ("control", VmAccess.ReadOnly, I(2)),
                ("updated", VmAccess.ReadWrite, F(PrefillBatchSize, EmbeddingSize)),
                ("output", VmAccess.ReadWrite, F(PrefillBatchSize, EmbeddingSize)));
            batchResidualRmsKernel = Kernel(
                "batch_residual_rms_norm",
                batchResidualParameters,
                Operation(
                    "transformer.batch-residual-rms-norm",
                    batchResidualParameters,
                    new Dictionary<string, string> { ["epsilon"] = epsilon }));

            string BatchMatVecKernel(int input, int output)
            {
                if (batchMatVecKernels.TryGetValue((input, output), out var existing))
                    return existing;
                var parameters = Parameters(
                    ("weight", VmAccess.ReadOnly, H(output, input)),
                    ("input", VmAccess.ReadOnly, F(PrefillBatchSize, input)),
                    ("control", VmAccess.ReadOnly, I(2)),
                    ("output", VmAccess.ReadWrite, F(PrefillBatchSize, output)));
                var created = Kernel(
                    $"batch_matvec_{input}_{output}",
                    parameters,
                    Operation("transformer.batch-shared-matvec", parameters));
                batchMatVecKernels.Add((input, output), created);
                return created;
            }

            BatchMatVecKernel(EmbeddingSize, QkvSize);
            BatchMatVecKernel(EmbeddingSize, EmbeddingSize);
            BatchMatVecKernel(EmbeddingSize, IntermediateSize * 2);
            BatchMatVecKernel(IntermediateSize, EmbeddingSize);
            if (adapterFile is not null)
            {
                BatchMatVecKernel(EmbeddingSize, adapterRank);
                BatchMatVecKernel(IntermediateSize, adapterRank);
                BatchMatVecKernel(adapterRank, QkvSize);
                BatchMatVecKernel(adapterRank, EmbeddingSize);
                BatchMatVecKernel(adapterRank, IntermediateSize * 2);
            }

            var batchSwiGluParameters = Parameters(
                ("gate_up", VmAccess.ReadOnly,
                    F(PrefillBatchSize, IntermediateSize * 2)),
                ("control", VmAccess.ReadOnly, I(2)),
                ("output", VmAccess.ReadWrite,
                    F(PrefillBatchSize, IntermediateSize)));
            batchSwiGluKernel = Kernel("batch_swiglu", batchSwiGluParameters,
                Operation("transformer.batch-swiglu", batchSwiGluParameters));
            var batchRopeParameters = Parameters(
                ("qkv", VmAccess.ReadOnly, F(PrefillBatchSize, QkvSize)),
                ("control", VmAccess.ReadOnly, I(2)),
                ("frequencies", VmAccess.ReadOnly, F(RotarySize / 2)),
                ("query", VmAccess.ReadWrite, F(PrefillBatchSize, QuerySize)),
                ("key_cache", VmAccess.ReadWrite,
                    F(context, KeyValueHeads, HeadSize)),
                ("value_cache", VmAccess.ReadWrite,
                    F(context, KeyValueHeads, HeadSize)));
            batchRopeKernel = Kernel("batch_rope_kv_write", batchRopeParameters,
                Operation(
                    "transformer.batch-rope-kv-write",
                    batchRopeParameters,
                    new Dictionary<string, string>
                    {
                        ["head_size"] = HeadSize.ToString(),
                        ["rotary_size"] = RotarySize.ToString(),
                        ["rope_scale"] = package.Text.GetMetadata<float>(
                                "phi3.rope.scaling.attn_factor")
                            .ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                    }));
            var batchScoreParameters = Parameters(
                ("query", VmAccess.ReadOnly, F(PrefillBatchSize, QuerySize)),
                ("key_cache", VmAccess.ReadOnly,
                    F(context, KeyValueHeads, HeadSize)),
                ("control", VmAccess.ReadOnly, I(2)),
                ("scores", VmAccess.ReadWrite,
                    F(PrefillBatchSize, QueryHeads, context)));
            batchScoreKernel = Kernel("batch_gqa_scores", batchScoreParameters,
                Operation("transformer.batch-gqa-scores", batchScoreParameters));
            var batchSoftmaxParameters = Parameters(
                ("scores", VmAccess.ReadOnly,
                    F(PrefillBatchSize, QueryHeads, context)),
                ("control", VmAccess.ReadOnly, I(2)),
                ("probabilities", VmAccess.ReadWrite,
                    F(PrefillBatchSize, QueryHeads, context)));
            batchSoftmaxKernel = Kernel(
                "batch_causal_softmax",
                batchSoftmaxParameters,
                Operation("transformer.batch-causal-softmax", batchSoftmaxParameters));
            var batchValueParameters = Parameters(
                ("probabilities", VmAccess.ReadOnly,
                    F(PrefillBatchSize, QueryHeads, context)),
                ("value_cache", VmAccess.ReadOnly,
                    F(context, KeyValueHeads, HeadSize)),
                ("control", VmAccess.ReadOnly, I(2)),
                ("output", VmAccess.ReadWrite,
                    F(PrefillBatchSize, EmbeddingSize)));
            batchValueKernel = Kernel("batch_gqa_values", batchValueParameters,
                Operation("transformer.batch-gqa-values", batchValueParameters));
            var batchSelectParameters = Parameters(
                ("input", VmAccess.ReadOnly,
                    F(PrefillBatchSize, EmbeddingSize)),
                ("control", VmAccess.ReadOnly, I(2)),
                ("output", VmAccess.ReadWrite, F(EmbeddingSize)));
            batchSelectLastKernel = Kernel(
                "batch_select_last",
                batchSelectParameters,
                Operation("transformer.batch-select-last", batchSelectParameters));

            foreach (var size in new[]
                     {
                         QkvSize, EmbeddingSize, IntermediateSize * 2,
                     }.Distinct())
            {
                var parameters = Parameters(
                    ("input", VmAccess.ReadOnly, F(PrefillBatchSize, size)),
                    ("update", VmAccess.ReadOnly, F(PrefillBatchSize, size)),
                    ("output", VmAccess.ReadWrite, F(PrefillBatchSize, size)));
                batchScaledAddKernels.Add(
                    size,
                    Kernel(
                        $"batch_lora_add_{size}",
                        parameters,
                        Operation(
                            "transformer.scaled-add",
                            parameters,
                            new Dictionary<string, string> { ["scale"] = "2" })));
            }
        }

        void DispatchProjection(
            string baseKernel,
            string baseWeight,
            string adapterWeightPrefix,
            string input,
            int inputSize,
            string output,
            string baseOutput,
            string updateOutput,
            int outputSize)
        {
            var groups = GpuMatVecExecution.Groups((ulong)outputSize);
            if (adapterFile is null)
            {
                Dispatch(baseKernel, new(groups.X, groups.Y),
                    Arguments(("weight", baseWeight), ("input", input), ("output", output)),
                    output);
                return;
            }

            Dispatch(baseKernel, new(groups.X, groups.Y),
                Arguments(("weight", baseWeight), ("input", input), ("output", baseOutput)),
                baseOutput);
            var lowGroups = GpuMatVecExecution.Groups((ulong)adapterRank);
            Dispatch(loraDownKernels[inputSize], new(lowGroups.X, lowGroups.Y),
                Arguments(("weight", adapterWeightPrefix + "_lora_a"),
                    ("input", input), ("output", "lora_low")),
                "lora_low");
            Dispatch(loraUpKernels[outputSize], new(groups.X, groups.Y),
                Arguments(("weight", adapterWeightPrefix + "_lora_b"),
                    ("input", "lora_low"), ("output", updateOutput)),
                updateOutput);
            Dispatch(scaledAddKernels[outputSize],
                new(checked((uint)((outputSize + 63) / 64))),
                Arguments(("input", baseOutput), ("update", updateOutput), ("output", output)),
                output);
        }

        Dispatch(embeddingKernel, new(48),
            Arguments(("table", "token_embedding"), ("index", "token"), ("output", "hidden")),
            "hidden");
        var commonStart = nodes.Count;
        Dispatch(rmsKernel, new(1),
            Arguments(("input", "hidden"), ("weight", "attn_norm_0"), ("output", "normalized")),
            "normalized");
        for (var layer = 0; layer < LayerCount; layer++)
        {
            DispatchProjection(qkvKernel, $"qkv_weight_{layer}", $"qkv_{layer}",
                "normalized", EmbeddingSize, "qkv", "qkv_base", "qkv_update", QkvSize);
            Dispatch(ropeKernel, new(80),
                Arguments(("qkv", "qkv"), ("position", "position"),
                    ("frequencies", "rope_frequencies"), ("query", "query"),
                    ("key_cache", $"key_cache_{layer}"), ("value_cache", $"value_cache_{layer}")),
                "query", $"key_cache_{layer}", $"value_cache_{layer}");
            Dispatch(scoreKernel, new(checked((uint)((QueryHeads * context + 63) / 64))),
                Arguments(("query", "query"), ("key_cache", $"key_cache_{layer}"),
                    ("position", "position"), ("scores", "scores")),
                "scores");
            Dispatch(softmaxKernel, new(QueryHeads),
                Arguments(("scores", "scores"), ("position", "position"),
                    ("probabilities", "probabilities")),
                "probabilities");
            Dispatch(valueKernel, new(48),
                Arguments(("probabilities", "probabilities"), ("value_cache", $"value_cache_{layer}"),
                    ("position", "position"), ("output", "attention")),
                "attention");
            DispatchProjection(attentionOutputKernel, $"attn_output_weight_{layer}",
                $"attn_output_{layer}", "attention", EmbeddingSize,
                "projected", "projected_base", "projected_update", EmbeddingSize);
            Dispatch(residualRmsKernel, new(1),
                Arguments(("hidden", "hidden"), ("residual", "projected"),
                    ("weight", $"ffn_norm_{layer}"), ("updated", "hidden_after_attention"),
                    ("output", "normalized")),
                "hidden_after_attention", "normalized");
            DispatchProjection(ffnUpKernel, $"ffn_up_weight_{layer}", $"ffn_up_{layer}",
                "normalized", EmbeddingSize, "gate_up", "gate_up_base", "gate_up_update",
                IntermediateSize * 2);
            Dispatch(swigluKernel, new(128),
                Arguments(("gate_up", "gate_up"), ("output", "activated")),
                "activated");
            DispatchProjection(ffnDownKernel, $"ffn_down_weight_{layer}", $"ffn_down_{layer}",
                "activated", IntermediateSize, "down", "down_base", "down_update",
                EmbeddingSize);
            Dispatch(residualRmsKernel, new(1),
                Arguments(("hidden", "hidden_after_attention"), ("residual", "down"),
                    ("weight", layer == LayerCount - 1 ? "output_norm" : $"attn_norm_{layer + 1}"),
                    ("updated", "hidden"), ("output", "normalized")),
                "hidden", "normalized");
        }
        var vocabularyGroups = GpuMatVecExecution.Groups(VocabularySize);
        Dispatch(vocabularyKernel, new(vocabularyGroups.X, vocabularyGroups.Y),
            Arguments(("weight", "token_embedding"), ("input", "normalized"), ("output", "logits")),
            "logits");
        Dispatch(argmaxKernel, new(1),
            Arguments(("input", "logits"), ("output", "next_token")),
            "next_token");

        List<VmNode>? batchNodes = null;
        if (fusedTokenCount > 0)
        {
            batchNodes = [];
            static VmThreadGroup BatchGrid(int elements)
            {
                var groups = checked((uint)((elements + 63) / 64));
                var x = Math.Min(groups, 65535u);
                return new(x, checked((groups + x - 1) / x));
            }
            static VmThreadGroup BatchReductionGrid(int outputs)
            {
                var groups = checked((uint)outputs);
                var x = Math.Min(groups, 65535u);
                return new(x, checked((groups + x - 1) / x));
            }
            void DispatchBatch(
                string kernel,
                VmThreadGroup groups,
                VmArgument[] arguments,
                params string[] writes)
            {
                var dispatchId = $"batch_dispatch_{nodeIndex++}";
                batchNodes.Add(new(
                    dispatchId,
                    new VmDispatch(kernel, arguments, groups)));
                batchNodes.Add(new(
                    $"batch_barrier_{nodeIndex++}",
                    new VmBarrier(writes),
                    [dispatchId]));
            }
            void DispatchBatchProjection(
                string baseWeight,
                string adapterWeightPrefix,
                string input,
                int inputSize,
                string output,
                string baseOutput,
                string updateOutput,
                int outputSize)
            {
                var kernel = batchMatVecKernels[(inputSize, outputSize)];
                var groups = BatchReductionGrid(
                    checked(PrefillBatchSize * outputSize));
                if (adapterFile is null)
                {
                    DispatchBatch(
                        kernel,
                        groups,
                        Arguments(
                            ("weight", baseWeight),
                            ("input", input),
                            ("control", "control"),
                            ("output", output)),
                        output);
                    return;
                }
                DispatchBatch(
                    kernel,
                    groups,
                    Arguments(
                        ("weight", baseWeight),
                        ("input", input),
                        ("control", "control"),
                        ("output", baseOutput)),
                    baseOutput);
                var lowKernel = batchMatVecKernels[(inputSize, adapterRank)];
                DispatchBatch(
                    lowKernel,
                    BatchReductionGrid(
                        checked(PrefillBatchSize * adapterRank)),
                    Arguments(
                        ("weight", adapterWeightPrefix + "_lora_a"),
                        ("input", input),
                        ("control", "control"),
                        ("output", "batch_lora_low")),
                    "batch_lora_low");
                var upKernel = batchMatVecKernels[(adapterRank, outputSize)];
                DispatchBatch(
                    upKernel,
                    groups,
                    Arguments(
                        ("weight", adapterWeightPrefix + "_lora_b"),
                        ("input", "batch_lora_low"),
                        ("control", "control"),
                        ("output", updateOutput)),
                    updateOutput);
                DispatchBatch(
                    batchScaledAddKernels[outputSize],
                    groups,
                    Arguments(
                        ("input", baseOutput),
                        ("update", updateOutput),
                        ("output", output)),
                    output);
            }

            DispatchBatch(
                batchGatherKernel!,
                BatchGrid(PrefillBatchSize * EmbeddingSize),
                Arguments(
                    ("table", "fused_embeddings"),
                    ("control", "control"),
                    ("output", "batch_hidden")),
                "batch_hidden");
            DispatchBatch(
                batchRmsKernel!,
                new(PrefillBatchSize),
                Arguments(
                    ("input", "batch_hidden"),
                    ("weight", "attn_norm_0"),
                    ("control", "control"),
                    ("output", "batch_normalized")),
                "batch_normalized");
            for (var layer = 0; layer < LayerCount; layer++)
            {
                DispatchBatchProjection(
                    $"qkv_weight_{layer}",
                    $"qkv_{layer}",
                    "batch_normalized",
                    EmbeddingSize,
                    "batch_qkv",
                    "batch_qkv_base",
                    "batch_qkv_update",
                    QkvSize);
                DispatchBatch(
                    batchRopeKernel!,
                    BatchGrid(PrefillBatchSize * QkvSize),
                    Arguments(
                        ("qkv", "batch_qkv"),
                        ("control", "control"),
                        ("frequencies", "rope_frequencies"),
                        ("query", "batch_query"),
                        ("key_cache", $"key_cache_{layer}"),
                        ("value_cache", $"value_cache_{layer}")),
                    "batch_query",
                    $"key_cache_{layer}",
                    $"value_cache_{layer}");
                DispatchBatch(
                    batchScoreKernel!,
                    BatchGrid(checked(PrefillBatchSize * QueryHeads * context)),
                    Arguments(
                        ("query", "batch_query"),
                        ("key_cache", $"key_cache_{layer}"),
                        ("control", "control"),
                        ("scores", "batch_scores")),
                    "batch_scores");
                DispatchBatch(
                    batchSoftmaxKernel!,
                    new(PrefillBatchSize * QueryHeads),
                    Arguments(
                        ("scores", "batch_scores"),
                        ("control", "control"),
                        ("probabilities", "batch_probabilities")),
                    "batch_probabilities");
                DispatchBatch(
                    batchValueKernel!,
                    BatchGrid(PrefillBatchSize * EmbeddingSize),
                    Arguments(
                        ("probabilities", "batch_probabilities"),
                        ("value_cache", $"value_cache_{layer}"),
                        ("control", "control"),
                        ("output", "batch_attention")),
                    "batch_attention");
                DispatchBatchProjection(
                    $"attn_output_weight_{layer}",
                    $"attn_output_{layer}",
                    "batch_attention",
                    EmbeddingSize,
                    "batch_projected",
                    "batch_projected_base",
                    "batch_projected_update",
                    EmbeddingSize);
                DispatchBatch(
                    batchResidualRmsKernel!,
                    new(PrefillBatchSize),
                    Arguments(
                        ("hidden", "batch_hidden"),
                        ("residual", "batch_projected"),
                        ("weight", $"ffn_norm_{layer}"),
                        ("control", "control"),
                        ("updated", "batch_hidden_after_attention"),
                        ("output", "batch_normalized")),
                    "batch_hidden_after_attention",
                    "batch_normalized");
                DispatchBatchProjection(
                    $"ffn_up_weight_{layer}",
                    $"ffn_up_{layer}",
                    "batch_normalized",
                    EmbeddingSize,
                    "batch_gate_up",
                    "batch_gate_up_base",
                    "batch_gate_up_update",
                    IntermediateSize * 2);
                DispatchBatch(
                    batchSwiGluKernel!,
                    BatchGrid(PrefillBatchSize * IntermediateSize),
                    Arguments(
                        ("gate_up", "batch_gate_up"),
                        ("control", "control"),
                        ("output", "batch_activated")),
                    "batch_activated");
                DispatchBatchProjection(
                    $"ffn_down_weight_{layer}",
                    $"ffn_down_{layer}",
                    "batch_activated",
                    IntermediateSize,
                    "batch_down",
                    "batch_down_base",
                    "batch_down_update",
                    EmbeddingSize);
                DispatchBatch(
                    batchResidualRmsKernel!,
                    new(PrefillBatchSize),
                    Arguments(
                        ("hidden", "batch_hidden_after_attention"),
                        ("residual", "batch_down"),
                        ("weight", layer == LayerCount - 1
                            ? "output_norm"
                            : $"attn_norm_{layer + 1}"),
                        ("control", "control"),
                        ("updated", "batch_hidden"),
                        ("output", "batch_normalized")),
                    "batch_hidden",
                    "batch_normalized");
            }
            DispatchBatch(
                batchSelectLastKernel!,
                BatchGrid(EmbeddingSize),
                Arguments(
                    ("input", "batch_normalized"),
                    ("control", "control"),
                    ("output", "batch_last")),
                "batch_last");
            var batchVocabularyGroups = GpuMatVecExecution.Groups(VocabularySize);
            DispatchBatch(
                vocabularyKernel,
                new(batchVocabularyGroups.X, batchVocabularyGroups.Y),
                Arguments(
                    ("weight", "token_embedding"),
                    ("input", "batch_last"),
                    ("output", "logits")),
                "logits");
            DispatchBatch(
                argmaxKernel,
                new(1),
                Arguments(("input", "logits"), ("output", "next_token")),
                "next_token");
        }

        var sharedRootParameters = slots.Where(
                slot => slot.Id is not ("control" or "embedding_control" or "fused_embeddings"))
            .Select(slot => new VmParameter(slot.Id, slot.Access, slot.Tensor))
            .ToArray();
        var rootParameters = new List<VmParameter>
        {
            new("token", VmAccess.ReadOnly, I(1)),
            new("position", VmAccess.ReadOnly, I(1)),
        };
        rootParameters.AddRange(sharedRootParameters);
        var decode = new VmDefinition(
            "decode",
            VmDefinitionKind.Orchestration,
            rootParameters,
            nodes);
        definitions.Add(decode);
        var embeddingRootParameters = new List<VmParameter>
        {
            new("input_embedding", VmAccess.ReadOnly, F(EmbeddingSize)),
            new("position", VmAccess.ReadOnly, I(1)),
        };
        embeddingRootParameters.AddRange(sharedRootParameters);
        var embeddingNodes = new List<VmNode>
        {
            new("embedding_copy", new VmDispatch(
                copyKernel,
                Arguments(("input", "input_embedding"), ("output", "hidden")),
                new(48))),
            new("embedding_copy_barrier", new VmBarrier(["hidden"]), ["embedding_copy"]),
        };
        embeddingNodes.AddRange(nodes.Skip(commonStart));
        var decodeEmbedding = new VmDefinition(
            "decode_embedding",
            VmDefinitionKind.Orchestration,
            embeddingRootParameters,
            embeddingNodes);
        definitions.Add(decodeEmbedding);
        VmDefinition? decodeFused = null;
        if (fusedTokenCount > 0)
        {
            var fusedRootParameters = new List<VmParameter>
            {
                new("fused_embeddings", VmAccess.ReadOnly, F(fusedTokenCount, EmbeddingSize)),
                new("position", VmAccess.ReadOnly, I(1)),
            };
            fusedRootParameters.AddRange(sharedRootParameters);
            var fusedNodes = new List<VmNode>
            {
                new("fused_embedding_gather", new VmDispatch(
                    fusedEmbeddingKernel!,
                    Arguments(
                        ("table", "fused_embeddings"),
                        ("index", "position"),
                        ("output", "hidden")),
                    new(48))),
                new("fused_embedding_barrier",
                    new VmBarrier(["hidden"]),
                    ["fused_embedding_gather"]),
            };
            fusedNodes.AddRange(nodes.Skip(commonStart));
            decodeFused = new VmDefinition(
                "decode_fused",
                VmDefinitionKind.Orchestration,
                fusedRootParameters,
                fusedNodes);
            definitions.Add(decodeFused);
        }
        VmDefinition? prefillFused = null;
        if (batchNodes is not null)
        {
            var batchRootParameters = new List<VmParameter>
            {
                new("fused_embeddings", VmAccess.ReadOnly,
                    F(fusedTokenCount, EmbeddingSize)),
                new("control", VmAccess.ReadOnly, I(2)),
            };
            batchRootParameters.AddRange(sharedRootParameters);
            prefillFused = new VmDefinition(
                "prefill_fused_16",
                VmDefinitionKind.Orchestration,
                batchRootParameters,
                batchNodes);
            definitions.Add(prefillFused);
        }
        var entryArguments = new List<VmArgument>
        {
            new("token", "control"),
            new("position", "control", sizeof(int)),
        };
        entryArguments.AddRange(sharedRootParameters
            .Select(parameter => new VmArgument(parameter.Name, parameter.Name)));
        var embeddingEntryArguments = new List<VmArgument>
        {
            new("input_embedding", "embedding_control"),
            new("position", "embedding_control", EmbeddingSize * sizeof(float)),
        };
        embeddingEntryArguments.AddRange(sharedRootParameters
            .Select(parameter => new VmArgument(parameter.Name, parameter.Name)));
        var entries = new List<VmEntry>
        {
            new("decode", decode.Id, entryArguments),
            new("decode_embedding", decodeEmbedding.Id, embeddingEntryArguments),
        };
        if (decodeFused is not null)
        {
            var fusedEntryArguments = new List<VmArgument>
            {
                new("fused_embeddings", "fused_embeddings"),
                new("position", "control", sizeof(int)),
            };
            fusedEntryArguments.AddRange(sharedRootParameters
                .Select(parameter => new VmArgument(parameter.Name, parameter.Name)));
            entries.Add(new("decode_fused", decodeFused.Id, fusedEntryArguments));
        }
        if (prefillFused is not null)
        {
            var batchEntryArguments = new List<VmArgument>
            {
                new("fused_embeddings", "fused_embeddings"),
                new("control", "control"),
            };
            batchEntryArguments.AddRange(sharedRootParameters
                .Select(parameter => new VmArgument(parameter.Name, parameter.Name)));
            entries.Add(new(
                "prefill_fused_16",
                prefillFused.Id,
                batchEntryArguments));
        }
        var program = new VmProgram(
            "phi4-gpu-decode",
            "phi4.decode.fp16",
            VmTarget.Direct3D12,
            slots,
            definitions,
            entries,
            new("phi4-kv-cache", 1, state));
        return new(program, weights, initializers);
    }

    private sealed record BuildResult(
        VmProgram Program,
        IReadOnlyList<(string Slot, GgufModelTensor Tensor)> Weights,
        IReadOnlyList<(string Slot, byte[] Bytes)> Initializers);
}
