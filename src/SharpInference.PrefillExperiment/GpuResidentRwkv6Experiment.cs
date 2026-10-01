using System.Runtime.InteropServices;
using SharpInference;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.Dxc;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.PrefillExperiment;

/// <summary>
/// Standalone RWKV-6 layer-major evaluator. Weights are retained in device memory across
/// calls; activations, including both recurrent components, never leave the device until
/// the final logits and state are copied back.
/// </summary>
internal sealed class GpuResidentRwkv6Experiment : IDisposable
{
    private const int UploadWords = 1 << 20;
    private const int Threads = 64;
    private const string Shader = """
        StructuredBuffer<uint> W : register(t0);
        RWStructuredBuffer<float> A : register(u0);
        RWStructuredBuffer<float> B : register(u1);
        RWStructuredBuffer<float> C : register(u2);
        RWStructuredBuffer<float> D : register(u3);
        RWStructuredBuffer<float> E : register(u4);
        RWStructuredBuffer<float> F : register(u5);
        RWStructuredBuffer<float> G : register(u6);
        RWStructuredBuffer<float> H : register(u7);
        cbuffer Args : register(b0)
        {
            uint P0; uint P1; uint P2; uint P3;
            uint P4; uint P5; uint P6; uint P7;
        };
        float weight(uint n)
        {
            if (P7 == 0) return asfloat(W[n]);
            uint pair = W[n >> 1];
            return f16tof32((pair >> ((n & 1) * 16)) & 65535);
        }
        [numthreads(64,1,1)]
        void Embed(uint i : SV_DispatchThreadID)
        {
            if (i >= P0 * P1) return;
            uint token = asuint(A[i / P1]);
            B[i] = weight(token * P1 + i % P1);
        }
        [numthreads(64,1,1)]
        void Norm(uint t : SV_DispatchThreadID)
        {
            if (t >= P0) return;
            float mean = 0, variance = 0;
            for (uint i = 0; i < P1; i++) mean += A[t * P1 + i];
            mean /= P1;
            for (uint i = 0; i < P1; i++)
            {
                float x = A[t * P1 + i] - mean;
                variance += x * x;
            }
            float scale = rsqrt(variance / P1 + 1e-5f);
            for (uint i = 0; i < P1; i++)
                D[t * P1 + i] = (A[t * P1 + i] - mean) * scale * B[i] + C[i];
        }
        [numthreads(64,1,1)]
        void Mix(uint i : SV_DispatchThreadID)
        {
            if (i >= P0 * P1) return;
            uint t = i / P1, col = i % P1;
            float current = A[i];
            float previous;
            if (t == 0) previous = B[P2 + col];
            else previous = A[i - P1];
            float delta = previous - current;
            C[i] = delta;
            E[i] = current + delta * D[col];
        }
        [numthreads(64,1,1)]
        void SavePrevious(uint col : SV_DispatchThreadID)
        {
            if (col < P1) B[P2 + col] = A[(P0 - 1) * P1 + col];
        }
        groupshared float WeightTile[16][16];
        groupshared float InputTile[16][16];
        [numthreads(16,16,1)]
        void Project(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
        {
            uint row = group.x * 16 + lane.x;
            uint token = group.y * 16 + lane.y;
            float sum = 0;
            for (uint start = 0; start < P0; start += 16)
            {
                uint wc = start + lane.y, ac = start + lane.x;
                WeightTile[lane.x][lane.y] =
                    row < P1 && wc < P0 ? weight(row * P0 + wc) : 0;
                InputTile[lane.y][lane.x] =
                    token < P2 && ac < P0 ? A[token * P0 + ac] : 0;
                GroupMemoryBarrierWithGroupSync();
                [unroll]
                for (uint col = 0; col < 16; col++)
                    sum += WeightTile[lane.x][col] * InputTile[lane.y][col];
                GroupMemoryBarrierWithGroupSync();
            }
            if (row < P1 && token < P2) B[token * P1 + row] = sum;
        }
        [numthreads(64,1,1)]
        void TanhValues(uint i : SV_DispatchThreadID)
        {
            if (i < P0) A[i] = tanh(A[i]);
        }
        [numthreads(64,1,1)]
        void MaaW2(uint i : SV_DispatchThreadID)
        {
            if (i >= P0 * P1 * 5) return;
            uint t = i / (P1 * 5), group = (i / P1) % 5, row = i % P1;
            float sum = 0;
            for (uint k = 0; k < P2; k++)
                sum += weight((group * P1 + row) * P2 + k) *
                    A[(t * 5 + group) * P2 + k];
            B[i] = sum;
        }
        [numthreads(64,1,1)]
        void DynamicMix(uint i : SV_DispatchThreadID)
        {
            if (i >= P0 * P1) return;
            uint token = i / P1, col = i % P1;
            E[i] = A[i] + B[i] * (C[(token * 5 + P2) * P1 + col] + D[col]);
        }
        [numthreads(64,1,1)]
        void Swish(uint i : SV_DispatchThreadID)
        {
            if (i < P0) A[i] = A[i] / (1 + exp(-A[i]));
        }
        [numthreads(64,1,1)]
        void Decay(uint i : SV_DispatchThreadID)
        {
            if (i < P0 * P1) A[i] = exp(-exp(A[i] + B[i % P1]));
        }
        // The affine summary for each state element composes one independent chunk.
        [numthreads(64,1,1)]
        void WkvSummary(uint3 id : SV_DispatchThreadID)
        {
            uint element = id.x, chunk = id.y;
            if (element >= P0 * P1 || chunk >= P5) return;
            uint head = element / (P1 * P1);
            uint key = element / P1 % P1, value = element % P1;
            uint channel = head * P1 + key;
            float a = 1, b = 0;
            for (uint t = chunk * P4; t < min((chunk + 1) * P4, P2); t++)
            {
                uint baseIndex = t * P0;
                b = b * D[baseIndex + channel] +
                    A[baseIndex + channel] * B[baseIndex + head * P1 + value];
                a *= D[baseIndex + channel];
            }
            uint destination = (chunk * P0 * P1 + element) * 2;
            G[destination] = a;
            G[destination + 1] = b;
        }
        [numthreads(64,1,1)]
        void WkvFold(uint element : SV_DispatchThreadID)
        {
            if (element >= P0 * P1) return;
            float state = F[P3 + element];
            for (uint chunk = 0; chunk < P5; chunk++)
            {
                H[chunk * P0 * P1 + element] = state;
                uint source = (chunk * P0 * P1 + element) * 2;
                state = state * G[source] + G[source + 1];
            }
            F[P3 + element] = state;
        }
        [numthreads(64,1,1)]
        void WkvEmit(uint3 id : SV_DispatchThreadID)
        {
            uint channel = id.x, chunk = id.y;
            if (channel >= P0 || chunk >= P5) return;
            uint head = channel / P1, value = channel % P1;
            for (uint t = chunk * P4; t < min((chunk + 1) * P4, P2); t++)
            {
                uint baseIndex = t * P0 + head * P1;
                float v = B[baseIndex + value], sum = 0;
                for (uint k = 0; k < P1; k++)
                {
                    uint stateIndex = chunk * P0 * P1 + (head * P1 + k) * P1 + value;
                    float old = H[stateIndex];
                    float kv = A[baseIndex + k] * v;
                    sum += C[baseIndex + k] * (old + E[head * P1 + k] * kv);
                    H[stateIndex] = old * D[baseIndex + k] + kv;
                }
                G[t * P0 + channel] = sum;
            }
        }
        [numthreads(64,1,1)]
        void GroupGate(uint t : SV_DispatchThreadID)
        {
            if (t >= P0) return;
            for (uint h = 0; h < P2; h++)
            {
                uint start = t * P1 + h * P3;
                float mean = 0, variance = 0;
                for (uint i = 0; i < P3; i++) mean += A[start + i];
                mean /= P3;
                for (uint i = 0; i < P3; i++)
                {
                    float d = A[start + i] - mean;
                    variance += d * d;
                }
                float scale = rsqrt(variance / P3 + 64e-5f);
                for (uint i = 0; i < P3; i++)
                {
                    uint col = h * P3 + i;
                    A[start + i] = ((A[start + i] - mean) * scale * B[col] + C[col]) *
                        D[t * P1 + col];
                }
            }
        }
        [numthreads(64,1,1)]
        void Add(uint i : SV_DispatchThreadID)
        {
            if (i < P0) A[i] += B[i];
        }
        [numthreads(64,1,1)]
        void FfnMix(uint i : SV_DispatchThreadID)
        {
            if (i >= P0 * P1) return;
            uint t = i / P1, col = i % P1;
            float x = A[i];
            float prev;
            if (t == 0) prev = B[P2 + col];
            else prev = A[i - P1];
            float delta = prev - x;
            E[i] = x + delta * C[col];
            F[i] = x + delta * D[col];
        }
        [numthreads(64,1,1)]
        void ReluSquared(uint i : SV_DispatchThreadID)
        {
            if (i < P0) { float v = max(0, A[i]); A[i] = v * v; }
        }
        [numthreads(64,1,1)]
        void FfnAdd(uint i : SV_DispatchThreadID)
        {
            if (i < P0) A[i] += B[i] / (1 + exp(-C[i]));
        }
        """;

    private readonly GgmlModelFile model;
    private readonly ID3D12Device device;
    private readonly ID3D12CommandQueue queue;
    private readonly ID3D12Fence fence;
    private readonly EventWaitHandle completion;
    private readonly ID3D12CommandAllocator allocator;
    private readonly ID3D12GraphicsCommandList list;
    private readonly ID3D12RootSignature root;
    private readonly Dictionary<string, ID3D12PipelineState> pipelines = [];
    private readonly Dictionary<string, (ID3D12Resource Resource, bool Half)> weights = [];
    private readonly ID3D12Resource staging;
    private readonly List<ID3D12Resource> resources = [];
    private readonly int defaultChunkSize;
    private ulong fenceValue;
    private bool disposed;

    public static (float[] Logits, float[] State) Run(
        GgmlModelFile model, int[] tokens, float[] initialState, int chunkSize)
    {
        using var evaluator = new GpuResidentRwkv6Experiment(model);
        return evaluator.Evaluate(tokens, initialState, chunkSize);
    }

    public GpuResidentRwkv6Experiment(GgmlModelFile model, int chunkSize = 64)
    {
        this.model = model ?? throw new ArgumentNullException(nameof(model));
        if (chunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(chunkSize));
        defaultChunkSize = chunkSize;
        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory6>();
        IDXGIAdapter1? adapter = null;
        for (uint i = 0; factory.EnumAdapterByGpuPreference(i, GpuPreference.HighPerformance,
                 out IDXGIAdapter1? candidate).Success; i++)
        {
            if ((candidate!.Description1.Flags & AdapterFlags.Software) == 0 &&
                IsSupported(candidate, FeatureLevel.Level_11_0))
            {
                adapter = candidate;
                break;
            }
            candidate.Dispose();
        }
        if (adapter is null)
            throw new PlatformNotSupportedException("A hardware D3D12 compute adapter is required.");
        using (adapter)
            device = D3D12CreateDevice<ID3D12Device>(adapter, FeatureLevel.Level_11_0);
        queue = device.CreateCommandQueue(CommandListType.Compute);
        fence = device.CreateFence();
        completion = new EventWaitHandle(false, EventResetMode.AutoReset);
        allocator = device.CreateCommandAllocator(CommandListType.Compute);
        list = device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, allocator);
        list.Close();
        var parameters = new List<RootParameter>
        {
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All)
        };
        for (uint i = 0; i < 8; i++)
            parameters.Add(new RootParameter(RootParameterType.UnorderedAccessView,
                new RootDescriptor(i, 0), ShaderVisibility.All));
        parameters.Add(new RootParameter(new RootConstants(0, 0, 8), ShaderVisibility.All));
        root = device.CreateRootSignature(new RootSignatureDescription(
            RootSignatureFlags.None, parameters.ToArray()), RootSignatureVersion.Version10);
        foreach (var entry in new[] { "Embed", "Norm", "Mix", "SavePrevious", "Project", "TanhValues",
                     "MaaW2", "DynamicMix", "Swish", "Decay", "WkvSummary",
                     "WkvFold", "WkvEmit", "GroupGate",
                     "Add", "FfnMix", "ReluSquared", "FfnAdd" })
        {
            using var shader = DxcCompiler.Compile(DxcShaderStage.Compute, Shader, entry,
                new DxcCompilerOptions { ShaderModel = DxcShaderModel.Model6_0 });
            pipelines.Add(entry, device.CreateComputePipelineState(new ComputePipelineStateDescription
            {
                RootSignature = root,
                ComputeShader = shader.GetObjectBytecodeArray()
            }));
        }
        staging = device.CreateCommittedResource(HeapProperties.UploadHeapProperties, HeapFlags.None,
            ResourceDescription.Buffer(Bytes(UploadWords)), ResourceStates.GenericRead);
    }

    private static ulong Bytes(long count) => checked((ulong)count * sizeof(uint));

    private ID3D12Resource Buffer(long words, ResourceFlags flags = ResourceFlags.AllowUnorderedAccess,
        ResourceStates state = ResourceStates.UnorderedAccess)
    {
        var resource = device.CreateCommittedResource(HeapProperties.DefaultHeapProperties,
            HeapFlags.None, ResourceDescription.Buffer(Bytes(Math.Max(words, 1)), flags), state);
        resources.Add(resource);
        return resource;
    }

    private void Begin()
    {
        allocator.Reset();
        list.Reset(allocator);
        list.SetComputeRootSignature(root);
    }

    private void Submit()
    {
        list.Close();
        queue.ExecuteCommandList(list);
        var target = ++fenceValue;
        queue.Signal(fence, target).CheckError();
        if (fence.CompletedValue < target)
        {
            fence.SetEventOnCompletion(target, completion).CheckError();
            completion.WaitOne();
        }
        if (device.DeviceRemovedReason.Failure)
            throw new InvalidOperationException($"D3D12 device removed: {device.DeviceRemovedReason}.");
    }

    private ID3D12Resource Upload(ReadOnlySpan<uint> words,
        ResourceFlags flags = ResourceFlags.None,
        ResourceStates finalState = ResourceStates.NonPixelShaderResource)
    {
        var result = Buffer(words.Length, flags, ResourceStates.CopyDest);
        for (var offset = 0; offset < words.Length;)
        {
            int count = Math.Min(UploadWords, words.Length - offset);
            staging.SetData(words.Slice(offset, count));
            Begin();
            list.CopyBufferRegion(result, Bytes(offset), staging, 0, Bytes(count));
            if (offset + count == words.Length)
                list.ResourceBarrierTransition(result, ResourceStates.CopyDest, finalState);
            Submit();
            offset += count;
        }
        return result;
    }

    private ID3D12Resource UploadFloats(ReadOnlySpan<float> values,
        ResourceFlags flags = ResourceFlags.None,
        ResourceStates state = ResourceStates.NonPixelShaderResource) =>
        Upload(MemoryMarshal.Cast<float, uint>(values), flags, state);

    private (ID3D12Resource Resource, bool Half) Weight(string name)
    {
        if (weights.TryGetValue(name, out var cached)) return cached;
        var tensor = model.GetRequired(name);
        bool half = tensor.DataType == RwkvTensorDataType.Float16;
        if (!half && tensor.DataType != RwkvTensorDataType.Float32)
            throw new NotSupportedException($"Unsupported tensor type for {name}: {tensor.DataType}.");
        bool vector = tensor.Dimensions.Count == 1 ||
            name.EndsWith(".att.time_faaaa", StringComparison.Ordinal) ||
            name.EndsWith(".att.time_decay", StringComparison.Ordinal);
        var flags = vector ? ResourceFlags.AllowUnorderedAccess : ResourceFlags.None;
        var finalState = vector ? ResourceStates.UnorderedAccess : ResourceStates.NonPixelShaderResource;
        if (vector)
        {
            // Small FP16 parameters are expanded on upload: UAV vector indexing is FP32.
            var parameter = UploadFloats(tensor.FloatValues, flags, finalState);
            return weights[name] = (parameter, false);
        }
        ID3D12Resource resource;
        if (half)
        {
            // HalfValues is a mapped span. Never expand matrix weights to FP32.
            var values = MemoryMarshal.Cast<Half, ushort>(tensor.HalfValues);
            var packed = new uint[Math.Min(UploadWords, (values.Length + 1) / 2)];
            resource = Buffer((values.Length + 1L) / 2, flags, ResourceStates.CopyDest);
            int words = (values.Length + 1) / 2;
            for (int offset = 0; offset < words;)
            {
                int count = Math.Min(packed.Length, words - offset);
                for (int i = 0; i < count; i++)
                {
                    int index = (offset + i) * 2;
                    packed[i] = (uint)values[index] |
                        (index + 1 < values.Length ? (uint)values[index + 1] << 16 : 0);
                }
                staging.SetData(packed.AsSpan(0, count));
                Begin();
                list.CopyBufferRegion(resource, Bytes(offset), staging, 0, Bytes(count));
                if (offset + count == words)
                    list.ResourceBarrierTransition(resource, ResourceStates.CopyDest, finalState);
                Submit();
                offset += count;
            }
        }
        else resource = Upload(MemoryMarshal.Cast<float, uint>(tensor.FloatValues), flags, finalState);
        return weights[name] = (resource, half);
    }

    private ID3D12Resource Vector(string name, int count)
    {
        var tensor = model.GetRequired(name);
        if (tensor.Dimensions.Aggregate(1, (a, b) => checked(a * b)) != count)
            throw new InvalidDataException($"Invalid vector '{name}': expected {count} values.");
        return Weight(name).Resource;
    }

    public (float[] Logits, float[] State) Run(int[] tokens, float[] initialState) =>
        Evaluate(tokens, initialState, defaultChunkSize);

    private void Dispatch(string entry, int work, ID3D12Resource? weight = null,
        bool half = false, ID3D12Resource? a = null, ID3D12Resource? b = null,
        ID3D12Resource? c = null, ID3D12Resource? d = null, ID3D12Resource? e = null,
        ID3D12Resource? f = null, ID3D12Resource? g = null, ID3D12Resource? h = null,
        int p0 = 0, int p1 = 0, int p2 = 0, int p3 = 0,
        int p4 = 0, int p5 = 0, int groupsY = 1)
    {
        list.SetPipelineState(pipelines[entry]);
        list.SetComputeRootShaderResourceView(0, (weight ?? staging).GPUVirtualAddress);
        var slots = new[] { a, b, c, d, e, f, g, h };
        for (uint i = 0; i < slots.Length; i++)
            list.SetComputeRootUnorderedAccessView(i + 1,
                (slots[i] ?? a ?? staging).GPUVirtualAddress);
        uint[] args = [(uint)p0, (uint)p1, (uint)p2, (uint)p3,
            (uint)p4, (uint)p5, 0, half ? 1u : 0u];
        for (uint i = 0; i < args.Length; i++)
            list.SetComputeRoot32BitConstant(9, args[i], i);
        list.Dispatch((uint)((work + Threads - 1L) / Threads), (uint)groupsY, 1);
        foreach (var resource in slots)
            if (resource is not null) list.ResourceBarrierUnorderedAccessView(resource);
    }

    private void Project(string name, ID3D12Resource input, ID3D12Resource output,
        int inputWidth, int outputWidth, int tokens)
    {
        var tensor = model.GetRequired(name);
        if (tensor.Dimensions.Count != 2 || tensor.Dimensions[0] != inputWidth ||
            tensor.Dimensions[1] != outputWidth)
            throw new InvalidDataException($"Invalid projection dimensions for '{name}'.");
        var w = Weight(name);
        list.SetPipelineState(pipelines["Project"]);
        list.SetComputeRootShaderResourceView(0, w.Resource.GPUVirtualAddress);
        for (uint i = 0; i < 8; i++)
            list.SetComputeRootUnorderedAccessView(i + 1,
                (i == 0 ? input : output).GPUVirtualAddress);
        uint[] args = [(uint)inputWidth, (uint)outputWidth, (uint)tokens, 0, 0, 0, 0,
            w.Half ? 1u : 0u];
        for (uint i = 0; i < 8; i++) list.SetComputeRoot32BitConstant(9, args[i], i);
        list.Dispatch((uint)((outputWidth + 15L) / 16), (uint)((tokens + 15L) / 16), 1);
        list.ResourceBarrierUnorderedAccessView(output);
    }

    public (float[] Logits, float[] State) Evaluate(int[] tokens, float[] initialState,
        int chunkSize = 64)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(initialState);
        if (tokens.Length == 0) throw new ArgumentException("At least one token is required.", nameof(tokens));
        if (chunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(chunkSize));
        int width = model.EmbeddingSize, length = tokens.Length;
        int vocabulary = model.VocabularySize;
        var first = model.GetRequired("blocks.0.att.time_faaaa");
        if (first.Dimensions.Count != 3 || first.Dimensions[0] != 1 ||
            first.Dimensions[1] <= 0 || first.Dimensions[2] <= 0 ||
            checked(first.Dimensions[1] * first.Dimensions[2]) != width)
            throw new InvalidDataException("Invalid RWKV-6 attention head dimensions.");
        int size = first.Dimensions[1], heads = first.Dimensions[2];
        int stride = checked(2 * width + width * size);
        if (initialState.Length != checked(model.LayerCount * stride))
            throw new ArgumentException("Invalid RWKV-6 state length.", nameof(initialState));
        for (int i = 0; i < length; i++)
            if ((uint)tokens[i] >= (uint)vocabulary)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Invalid token at {i}.");
        // Eagerly upload all weights before recording a single layer-major command list.
        foreach (var name in model.Names)
            Weight(name);
        int transientStart = resources.Count;
        var tokenBuffer = Upload(MemoryMarshal.Cast<int, uint>(tokens.AsSpan()),
            ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
        var state = UploadFloats(initialState, ResourceFlags.AllowUnorderedAccess,
            ResourceStates.UnorderedAccess);
        int full = checked(length * width);
        var x = Buffer(full);
        var next = Buffer(full);
        var norm = Buffer(full);
        var delta = Buffer(full);
        var mixedInput = Buffer(full);
        var maa = Buffer(checked(length * 5 * width));
        var maaHidden = Buffer(checked(length * 5 * width));
        var r = Buffer(full);
        var k = Buffer(full);
        var v = Buffer(full);
        var gate = Buffer(full);
        var decay = Buffer(full);
        var hidden = Buffer(checked(length * 4 * width));
        var att = Buffer(full);
        int chunks = checked((int)((length + (long)chunkSize - 1) / chunkSize));
        int stateElements = checked(width * size);
        var summary = Buffer(checked((long)chunks * stateElements * 2));
        var scratch = Buffer(checked((long)chunks * stateElements));
        var keyInput = Buffer(full);
        var receptanceInput = Buffer(full);
        var ffnR = Buffer(full);
        var logitsBuffer = Buffer(vocabulary);
        var readLogits = device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties,
            HeapFlags.None, ResourceDescription.Buffer(Bytes(vocabulary)), ResourceStates.CopyDest);
        var readState = device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties,
            HeapFlags.None, ResourceDescription.Buffer(Bytes(initialState.Length)), ResourceStates.CopyDest);
        resources.Add(readLogits);
        resources.Add(readState);
        var emb = Weight("emb.weight");
        Begin();
        Dispatch("Embed", full, emb.Resource, emb.Half, tokenBuffer, x,
            p0: length, p1: width);
        Dispatch("Norm", length, a: x, b: Vector("blocks.0.ln0.weight", width),
            c: Vector("blocks.0.ln0.bias", width), d: next, p0: length, p1: width);
        (x, next) = (next, x);
        for (int layer = 0; layer < model.LayerCount; layer++)
        {
            string N(string suffix) => $"blocks.{layer}.{suffix}";
            int offset = checked(layer * stride);
            Dispatch("Norm", length, a: x, b: Vector(N("ln1.weight"), width),
                c: Vector(N("ln1.bias"), width), d: norm, p0: length, p1: width);
            Dispatch("Mix", full, a: norm, b: state, c: delta,
                d: Vector(N("att.time_maa_x"), width), e: mixedInput,
                p0: length, p1: width, p2: offset + width);
            Dispatch("SavePrevious", width, a: norm, b: state,
                p0: length, p1: width, p2: offset + width);
            int maaSize = model.GetRequired(N("att.time_maa_w1")).Dimensions[1] / 5;
            Project(N("att.time_maa_w1"), mixedInput, maaHidden, width, 5 * maaSize, length);
            Dispatch("TanhValues", checked(length * 5 * maaSize), a: maaHidden,
                p0: checked(length * 5 * maaSize));
            var w2Tensor = model.GetRequired(N("att.time_maa_w2"));
            if (w2Tensor.Dimensions.Count != 3 || w2Tensor.Dimensions[0] != maaSize ||
                w2Tensor.Dimensions[1] != width || w2Tensor.Dimensions[2] != 5)
                throw new InvalidDataException($"Invalid MAA W2 dimensions in layer {layer}.");
            var w2 = Weight(N("att.time_maa_w2"));
            Dispatch("MaaW2", checked(length * 5 * width), w2.Resource, w2.Half,
                maaHidden, maa, p0: length, p1: width, p2: maaSize);
            ID3D12Resource Dynamic(int group, string suffix)
            {
                Dispatch("DynamicMix", full, a: norm, b: delta, c: maa,
                    d: Vector(N(suffix), width), e: mixedInput,
                    p0: length, p1: width, p2: group);
                return mixedInput;
            }
            // The shared dynamic input scratch is consumed before producing the next mix.
            Dynamic(3, "att.time_maa_r");
            Project(N("att.receptance.weight"), mixedInput, r, width, width, length);
            Dynamic(1, "att.time_maa_k");
            Project(N("att.key.weight"), mixedInput, k, width, width, length);
            Dynamic(2, "att.time_maa_v");
            Project(N("att.value.weight"), mixedInput, v, width, width, length);
            Dynamic(4, "att.time_maa_g");
            Project(N("att.gate.weight"), mixedInput, gate, width, width, length);
            Dispatch("Swish", full, a: gate, p0: full);
            Dynamic(0, "att.time_maa_w");
            int decaySize = model.GetRequired(N("att.time_decay_w1")).Dimensions[1];
            Project(N("att.time_decay_w1"), mixedInput, hidden, width, decaySize, length);
            Dispatch("TanhValues", checked(length * decaySize), a: hidden,
                p0: checked(length * decaySize));
            Project(N("att.time_decay_w2"), hidden, decay, decaySize, width, length);
            Dispatch("Decay", full, a: decay, b: Vector(N("att.time_decay"), width),
                p0: length, p1: width);
            var timeFirst = Vector(N("att.time_faaaa"), width);
            Dispatch("WkvSummary", stateElements, a: k, b: v, c: r, d: decay,
                e: timeFirst, f: state, g: summary, h: scratch,
                p0: width, p1: size, p2: length, p3: offset + 2 * width,
                p4: chunkSize, p5: chunks, groupsY: chunks);
            Dispatch("WkvFold", stateElements, a: k, b: v, c: r, d: decay,
                e: timeFirst, f: state, g: summary, h: scratch,
                p0: width, p1: size, p2: length, p3: offset + 2 * width,
                p4: chunkSize, p5: chunks);
            Dispatch("WkvEmit", width, a: k, b: v, c: r, d: decay,
                e: timeFirst, f: state, g: att, h: scratch,
                p0: width, p1: size, p2: length, p3: offset + 2 * width,
                p4: chunkSize, p5: chunks, groupsY: chunks);
            Dispatch("GroupGate", length, a: att,
                b: Vector(N("att.ln_x.weight"), width),
                c: Vector(N("att.ln_x.bias"), width), d: gate,
                p0: length, p1: width, p2: heads, p3: size);
            Project(N("att.output.weight"), att, next, width, width, length);
            Dispatch("Add", full, a: x, b: next, p0: full);
            Dispatch("Norm", length, a: x, b: Vector(N("ln2.weight"), width),
                c: Vector(N("ln2.bias"), width), d: norm, p0: length, p1: width);
            Dispatch("FfnMix", full, a: norm, b: state,
                c: Vector(N("ffn.time_maa_k"), width),
                d: Vector(N("ffn.time_maa_r"), width),
                e: keyInput, f: receptanceInput,
                p0: length, p1: width, p2: offset);
            Dispatch("SavePrevious", width, a: norm, b: state,
                p0: length, p1: width, p2: offset);
            Project(N("ffn.receptance.weight"), receptanceInput, ffnR, width, width, length);
            int ffnWidth = model.GetRequired(N("ffn.key.weight")).Dimensions[1];
            Project(N("ffn.key.weight"), keyInput, hidden, width, ffnWidth, length);
            Dispatch("ReluSquared", checked(length * ffnWidth), a: hidden,
                p0: checked(length * ffnWidth));
            Project(N("ffn.value.weight"), hidden, next, ffnWidth, width, length);
            Dispatch("FfnAdd", full, a: x, b: next, c: ffnR, p0: full);
        }
        // Normalize only the final token; avoid projecting preceding tokens into logits.
        // A root UAV points into the last token's subrange of x.
        list.SetPipelineState(pipelines["Norm"]);
        list.SetComputeRootShaderResourceView(0, staging.GPUVirtualAddress);
        var normWeight = Vector("ln_out.weight", width);
        var normBias = Vector("ln_out.bias", width);
        var normSlots = new[] { x.GPUVirtualAddress + Bytes((long)(length - 1) * width),
            normWeight.GPUVirtualAddress, normBias.GPUVirtualAddress, norm.GPUVirtualAddress };
        for (uint i = 0; i < 8; i++)
            list.SetComputeRootUnorderedAccessView(i + 1, normSlots[Math.Min(i, 3)]);
        for (uint i = 0; i < 8; i++)
            list.SetComputeRoot32BitConstant(9, i == 0 ? 1u : i == 1 ? (uint)width : 0u, i);
        list.Dispatch(1, 1, 1);
        list.ResourceBarrierUnorderedAccessView(norm);
        Project("head.weight", norm, logitsBuffer, width, vocabulary, 1);
        list.ResourceBarrierTransition(logitsBuffer, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        list.ResourceBarrierTransition(state, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        list.CopyBufferRegion(readLogits, 0, logitsBuffer, 0, Bytes(vocabulary));
        list.CopyBufferRegion(readState, 0, state, 0, Bytes(initialState.Length));
        list.ResourceBarrierTransition(logitsBuffer, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        list.ResourceBarrierTransition(state, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        Submit();
        var result = new float[vocabulary];
        var finalState = new float[initialState.Length];
        readLogits.GetData(result);
        readState.GetData(finalState);
        for (int i = resources.Count - 1; i >= transientStart; i--)
            resources[i].Dispose();
        resources.RemoveRange(transientStart, resources.Count - transientStart);
        return (result, finalState);
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        foreach (var resource in resources) resource.Dispose();
        foreach (var pipeline in pipelines.Values) pipeline.Dispose();
        staging.Dispose();
        root.Dispose();
        list.Dispose();
        allocator.Dispose();
        completion.Dispose();
        fence.Dispose();
        queue.Dispose();
        device.Dispose();
    }
}
