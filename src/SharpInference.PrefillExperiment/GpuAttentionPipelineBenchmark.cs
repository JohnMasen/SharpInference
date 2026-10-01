using System.Diagnostics;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.Dxc;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.PrefillExperiment;

internal readonly record struct GpuAttentionPipelineResult(
    TimeSpan Sequential, TimeSpan Chunked,
    float SequentialProjectionMaxError, float ChunkedProjectionMaxError,
    float SequentialOutputMaxError, float SequentialStateMaxError,
    float ChunkedOutputMaxError, float ChunkedStateMaxError,
    long AllocatedGpuBytes, string Device);

/// <summary>
/// GPU-resident projection-to-WKV stage probe, not a full RWKV-6 TimeMix layer:
/// callers supply token-major K/V/R activations and precomputed decays. Shared
/// double-normalized first-layer embeddings and static time_decay can serve as
/// smoke-test inputs, but do not reproduce dynamic TimeMix inputs or gates.
/// </summary>
internal static class GpuAttentionPipelineBenchmark
{
    private const int Tile = 16;
    private const int Threads = 64;
    private const long MaxGpuBytes = 768L * 1024 * 1024;
    private const string Shader = """
        StructuredBuffer<uint> Weights : register(t0);
        StructuredBuffer<float> Activations : register(t1);
        StructuredBuffer<float> Decays : register(t2);
        StructuredBuffer<float> TimeFirst : register(t3);
        StructuredBuffer<float> Initial : register(t4);
        RWStructuredBuffer<float> Keys : register(u0);
        RWStructuredBuffer<float> Values : register(u1);
        RWStructuredBuffer<float> Receptance : register(u2);
        RWStructuredBuffer<float> State : register(u3);
        RWStructuredBuffer<float> Output : register(u4);
        RWStructuredBuffer<float> Summary : register(u5);
        RWStructuredBuffer<float> Scratch : register(u6);
        cbuffer Parameters : register(b0)
        {
            uint Width;
            uint Tokens;
            uint HeadSize;
            uint StateCount;
            uint ChunkSize;
            uint ChunkCount;
            uint TokenIndex;
            uint ProjectionIndex;
        };

        float Weight(uint row, uint col)
        {
            uint index = row * Width + col;
            uint pair = Weights[index >> 1];
            return f16tof32((pair >> ((index & 1) * 16)) & 0xffff);
        }

        void StoreProjection(uint index, float value)
        {
            if (ProjectionIndex == 0) Keys[index] = value;
            else if (ProjectionIndex == 1) Values[index] = value;
            else Receptance[index] = value;
        }

        [numthreads(64, 1, 1)]
        void ProjectToken(uint row : SV_DispatchThreadID)
        {
            if (row >= Width) return;
            float sum = 0.0f;
            for (uint col = 0; col < Width; ++col)
                sum += Weight(row, col) * Activations[TokenIndex * Width + col];
            StoreProjection(TokenIndex * Width + row, sum);
        }

        groupshared float WeightTile[16][16];
        groupshared float InputTile[16][16];

        [numthreads(16, 16, 1)]
        void ProjectTile(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
        {
            uint row = group.x * 16 + lane.x;
            uint token = group.y * 16 + lane.y;
            float sum = 0.0f;
            for (uint start = 0; start < Width; start += 16)
            {
                uint weightCol = start + lane.y;
                uint inputCol = start + lane.x;
                WeightTile[lane.x][lane.y] =
                    row < Width && weightCol < Width ? Weight(row, weightCol) : 0.0f;
                InputTile[lane.y][lane.x] =
                    token < Tokens && inputCol < Width ? Activations[token * Width + inputCol] : 0.0f;
                GroupMemoryBarrierWithGroupSync();
                [unroll]
                for (uint col = 0; col < 16; ++col)
                    sum += WeightTile[lane.x][col] * InputTile[lane.y][col];
                GroupMemoryBarrierWithGroupSync();
            }
            if (row < Width && token < Tokens)
                StoreProjection(token * Width + row, sum);
        }

        [numthreads(64, 1, 1)]
        void Sequential(uint id : SV_DispatchThreadID)
        {
            if (id >= Width) return;
            uint head = id / HeadSize, value = id % HeadSize;
            uint input = TokenIndex * Width + head * HeadSize;
            float v = Values[input + value];
            float sum = 0.0f;
            for (uint key = 0; key < HeadSize; ++key)
            {
                uint matrix = (head * HeadSize + key) * HeadSize + value;
                float old = State[matrix];
                float k = Keys[input + key];
                sum += Receptance[input + key] * (old + TimeFirst[head * HeadSize + key] * k * v);
                State[matrix] = old * Decays[input + key] + k * v;
            }
            Output[TokenIndex * Width + id] = sum;
        }

        [numthreads(64, 1, 1)]
        void Summarize(uint3 id : SV_DispatchThreadID)
        {
            uint element = id.x, chunk = id.y;
            if (element >= StateCount || chunk >= ChunkCount) return;
            uint head = element / (HeadSize * HeadSize);
            uint key = element / HeadSize % HeadSize;
            uint value = element % HeadSize;
            uint channel = head * HeadSize + key;
            float a = 1.0f, b = 0.0f;
            uint end = min((chunk + 1) * ChunkSize, Tokens);
            for (uint token = chunk * ChunkSize; token < end; ++token)
            {
                uint input = token * Width;
                b = b * Decays[input + channel] + Keys[input + channel] * Values[input + head * HeadSize + value];
                a *= Decays[input + channel];
            }
            uint offset = (chunk * StateCount + element) * 2;
            Summary[offset] = a;
            Summary[offset + 1] = b;
        }

        [numthreads(64, 1, 1)]
        void Fold(uint element : SV_DispatchThreadID)
        {
            if (element >= StateCount) return;
            float state = Initial[element];
            for (uint chunk = 0; chunk < ChunkCount; ++chunk)
            {
                Scratch[chunk * StateCount + element] = state;
                uint offset = (chunk * StateCount + element) * 2;
                state = state * Summary[offset] + Summary[offset + 1];
            }
            State[element] = state;
        }

        [numthreads(64, 1, 1)]
        void Emit(uint3 id : SV_DispatchThreadID)
        {
            uint channel = id.x, chunk = id.y;
            if (channel >= Width || chunk >= ChunkCount) return;
            uint head = channel / HeadSize, value = channel % HeadSize;
            uint end = min((chunk + 1) * ChunkSize, Tokens);
            for (uint token = chunk * ChunkSize; token < end; ++token)
            {
                uint input = token * Width + head * HeadSize;
                float v = Values[input + value];
                float sum = 0.0f;
                for (uint key = 0; key < HeadSize; ++key)
                {
                    uint element = (head * HeadSize + key) * HeadSize + value;
                    uint scratch = chunk * StateCount + element;
                    float old = Scratch[scratch];
                    float k = Keys[input + key];
                    sum += Receptance[input + key] *
                        (old + TimeFirst[head * HeadSize + key] * k * v);
                    Scratch[scratch] = old * Decays[input + key] + k * v;
                }
                Output[token * Width + channel] = sum;
            }
        }
        """;

    /// <summary>
    /// Projects shared FP32 activations through separate FP16 K/V/R matrices.
    /// This overload is useful for stage-probe smoke tests; it does not generate
    /// the distinct token-dependent activations of a real RWKV-6 TimeMix layer.
    /// </summary>
    internal static GpuAttentionPipelineResult Run(
        int heads, int size, int tokens, int chunkSize,
        Half[] keyWeights, Half[] valueWeights, Half[] receptanceWeights,
        float[] activations, float[] decays, float[] timeFirst,
        float[] initialState, int repeats) =>
        Run(heads, size, tokens, chunkSize, keyWeights, valueWeights, receptanceWeights,
            activations, activations, activations, decays, timeFirst, initialState, repeats);

    /// <summary>
    /// Projects independent token-major FP32 key/value/receptance activations through
    /// separate FP16 K/V/R matrices and
    /// consumes their GPU-resident outputs in WKV6. Weights are [output,input],
    /// each activation and decays are [token,head,channel], timeFirst [head,channel],
    /// and initialState [head,key channel,value channel].
    /// </summary>
    internal static GpuAttentionPipelineResult Run(
        int heads, int size, int tokens, int chunkSize,
        Half[] keyWeights, Half[] valueWeights, Half[] receptanceWeights,
        float[] keyActivations, float[] valueActivations, float[] receptanceActivations,
        float[] decays, float[] timeFirst,
        float[] initialState, int repeats)
    {
        if (heads <= 0 || size <= 0 || tokens <= 0 || chunkSize <= 0 || repeats <= 0)
            throw new ArgumentOutOfRangeException(nameof(heads),
                "Heads, size, tokens, chunk size and repeats must be positive.");
        ArgumentNullException.ThrowIfNull(keyWeights);
        ArgumentNullException.ThrowIfNull(valueWeights);
        ArgumentNullException.ThrowIfNull(receptanceWeights);
        ArgumentNullException.ThrowIfNull(keyActivations);
        ArgumentNullException.ThrowIfNull(valueActivations);
        ArgumentNullException.ThrowIfNull(receptanceActivations);
        ArgumentNullException.ThrowIfNull(decays);
        ArgumentNullException.ThrowIfNull(timeFirst);
        ArgumentNullException.ThrowIfNull(initialState);
        int width = checked(heads * size);
        int weightsCount = checked(width * width);
        int inputCount = checked(tokens * width);
        int stateCount = checked(width * size);
        int chunks = checked(1 + (tokens - 1) / chunkSize);
        int summaryCount = checked(checked(chunks * stateCount) * 2);
        int scratchCount = checked(chunks * stateCount);
        int packedCount = checked(weightsCount / 2 + weightsCount % 2);
        if (keyWeights.Length != weightsCount || valueWeights.Length != weightsCount ||
            receptanceWeights.Length != weightsCount || keyActivations.Length != inputCount ||
            valueActivations.Length != inputCount || receptanceActivations.Length != inputCount ||
            decays.Length != inputCount || timeFirst.Length != width ||
            initialState.Length != stateCount)
            throw new ArgumentException("Expected three FP16 [width,width] weights, three token-major FP32 activation arrays and decays, timeFirst [width], and state [head,key,value].");
        if ((width + (long)Threads - 1) / Threads > 65535 ||
            (stateCount + (long)Threads - 1) / Threads > 65535 ||
            (width + (long)Tile - 1) / Tile > 65535 ||
            (tokens + (long)Tile - 1) / Tile > 65535 || chunks > 65535)
            throw new ArgumentOutOfRangeException(nameof(tokens), "D3D12 dispatch exceeds 65535 groups on an axis.");

        // Count upload, default-heap and readback resources, including the three packed FP16 weights.
        long gpuBytes = checked(4L * (6L * packedCount + 2L * (4L * inputCount + width + stateCount) +
            3L * inputCount + stateCount + inputCount + summaryCount + scratchCount +
            4L * inputCount + stateCount));
        if (gpuBytes > MaxGpuBytes)
            throw new ArgumentOutOfRangeException(nameof(tokens),
                $"Pipeline buffers require {gpuBytes:N0} bytes, exceeding the {MaxGpuBytes:N0}-byte allocation guard.");
        foreach (var (data, name) in new (float[], string)[]
                 { (keyActivations, nameof(keyActivations)),
                   (valueActivations, nameof(valueActivations)),
                   (receptanceActivations, nameof(receptanceActivations)),
                   (decays, nameof(decays)),
                   (timeFirst, nameof(timeFirst)), (initialState, nameof(initialState)) })
            for (int i = 0; i < data.Length; i++)
                if (!float.IsFinite(data[i]))
                    throw new ArgumentException($"{name}[{i}] must be finite.");
        foreach (var (data, name) in new (Half[], string)[]
                 { (keyWeights, nameof(keyWeights)), (valueWeights, nameof(valueWeights)),
                   (receptanceWeights, nameof(receptanceWeights)) })
            for (int i = 0; i < data.Length; i++)
                if (!Half.IsFinite(data[i]))
                    throw new ArgumentException($"{name}[{i}] must be finite.");

        var weightData = new[] { keyWeights, valueWeights, receptanceWeights };
        var activationData = new[] { keyActivations, valueActivations, receptanceActivations };
        var packed = new uint[3][];
        var expectedProjection = new float[3][];
        for (int projection = 0; projection < 3; projection++)
        {
            packed[projection] = new uint[packedCount];
            for (int i = 0; i < weightsCount; i++)
                packed[projection][i >> 1] |= (uint)BitConverter.HalfToUInt16Bits(weightData[projection][i]) << ((i & 1) * 16);
            var result = expectedProjection[projection] = new float[inputCount];
            for (int token = 0; token < tokens; token++)
                for (int row = 0; row < width; row++)
                {
                    float sum = 0;
                    int offset = row * width;
                    for (int col = 0; col < width; col++)
                        sum += (float)weightData[projection][offset + col] *
                            activationData[projection][token * width + col];
                    result[token * width + row] = sum;
                }
        }
        var expectedState = (float[])initialState.Clone();
        var expectedOutput = new float[inputCount];
        for (int token = 0; token < tokens; token++)
            for (int head = 0; head < heads; head++)
                for (int value = 0; value < size; value++)
                {
                    int input = token * width + head * size;
                    float v = expectedProjection[1][input + value];
                    float sum = 0;
                    for (int key = 0; key < size; key++)
                    {
                        int element = (head * size + key) * size + value;
                        float old = expectedState[element];
                        float k = expectedProjection[0][input + key];
                        sum += expectedProjection[2][input + key] *
                            (old + timeFirst[head * size + key] * k * v);
                        expectedState[element] = old * decays[input + key] + k * v;
                    }
                    expectedOutput[token * width + head * size + value] = sum;
                }

        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory6>();
        IDXGIAdapter1? adapter = null;
        for (uint i = 0; factory.EnumAdapterByGpuPreference(
                 i, GpuPreference.HighPerformance, out IDXGIAdapter1? candidate).Success; i++)
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
            throw new PlatformNotSupportedException("No hardware D3D12 compute adapter is available for the attention pipeline.");
        using (adapter)
        {
            string deviceName = adapter.Description1.Description;
            using var device = D3D12CreateDevice<ID3D12Device>(adapter, FeatureLevel.Level_11_0);
            using var queue = device.CreateCommandQueue(CommandListType.Compute);
            using var fence = device.CreateFence();
            using var completion = new EventWaitHandle(false, EventResetMode.AutoReset);
            using var allocator = device.CreateCommandAllocator(CommandListType.Compute);
            using var list = device.CreateCommandList<ID3D12GraphicsCommandList>(
                0, CommandListType.Compute, allocator);
            list.Close();
            var parameters = new List<RootParameter>();
            for (int i = 0; i < 5; i++)
                parameters.Add(new RootParameter(RootParameterType.ShaderResourceView,
                    new RootDescriptor((uint)i, 0), ShaderVisibility.All));
            for (int i = 0; i < 7; i++)
                parameters.Add(new RootParameter(RootParameterType.UnorderedAccessView,
                    new RootDescriptor((uint)i, 0), ShaderVisibility.All));
            parameters.Add(new RootParameter(new RootConstants(0, 0, 8), ShaderVisibility.All));
            using var root = device.CreateRootSignature(new RootSignatureDescription(
                RootSignatureFlags.None, parameters.ToArray()), RootSignatureVersion.Version10);
            using var projectToken = Compile("ProjectToken");
            using var projectTile = Compile("ProjectTile");
            using var sequentialPipeline = Compile("Sequential");
            using var summaryPipeline = Compile("Summarize");
            using var foldPipeline = Compile("Fold");
            using var emitPipeline = Compile("Emit");

            ID3D12PipelineState Compile(string entry)
            {
                using var shader = DxcCompiler.Compile(DxcShaderStage.Compute, Shader, entry,
                    new DxcCompilerOptions { ShaderModel = DxcShaderModel.Model6_0 });
                return device.CreateComputePipelineState(new ComputePipelineStateDescription
                {
                    RootSignature = root,
                    ComputeShader = shader.GetObjectBytecodeArray(),
                });
            }

            using var kUpload = Upload(packed[0]);
            using var vUpload = Upload(packed[1]);
            using var rUpload = Upload(packed[2]);
            using var akUpload = Upload(keyActivations);
            using var avUpload = Upload(valueActivations);
            using var arUpload = Upload(receptanceActivations);
            using var dUpload = Upload(decays);
            using var fUpload = Upload(timeFirst);
            using var sUpload = Upload(initialState);
            using var kWeight = Input(packedCount);
            using var vWeight = Input(packedCount);
            using var rWeight = Input(packedCount);
            using var akBuffer = Input(inputCount);
            using var avBuffer = Input(inputCount);
            using var arBuffer = Input(inputCount);
            using var dBuffer = Input(inputCount);
            using var fBuffer = Input(width);
            using var initialBuffer = Input(stateCount);
            using var keys = Buffer(inputCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var values = Buffer(inputCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var receptance = Buffer(inputCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var state = Buffer(stateCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var output = Buffer(inputCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var summary = Buffer(summaryCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var scratch = Buffer(scratchCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var kRead = Readback(inputCount);
            using var vRead = Readback(inputCount);
            using var rRead = Readback(inputCount);
            using var oRead = Readback(inputCount);
            using var sRead = Readback(stateCount);
            var weightBuffers = new[] { kWeight, vWeight, rWeight };
            var activationBuffers = new[] { akBuffer, avBuffer, arBuffer };
            var projections = new[] { keys, values, receptance };
            var projectionReadbacks = new[] { kRead, vRead, rRead };
            ulong fenceValue = 0;

            void Begin()
            {
                allocator.Reset();
                list.Reset(allocator);
            }

            void Submit()
            {
                list.Close();
                queue.ExecuteCommandList(list);
                ulong target = checked(++fenceValue);
                queue.Signal(fence, target).CheckError();
                if (fence.CompletedValue < target)
                {
                    fence.SetEventOnCompletion(target, completion).CheckError();
                    completion.WaitOne();
                }
                if (device.DeviceRemovedReason.Failure)
                    throw new InvalidOperationException($"D3D12 device was removed: {device.DeviceRemovedReason}.");
            }

            Begin();
            Copy(kWeight, kUpload, packedCount);
            Copy(vWeight, vUpload, packedCount);
            Copy(rWeight, rUpload, packedCount);
            Copy(akBuffer, akUpload, inputCount);
            Copy(avBuffer, avUpload, inputCount);
            Copy(arBuffer, arUpload, inputCount);
            Copy(dBuffer, dUpload, inputCount);
            Copy(fBuffer, fUpload, width);
            Copy(initialBuffer, sUpload, stateCount, ResourceStates.CopySource | ResourceStates.NonPixelShaderResource);
            Submit();

            void Copy(ID3D12Resource destination, ID3D12Resource source, int count,
                ResourceStates finalState = ResourceStates.NonPixelShaderResource)
            {
                list.CopyBufferRegion(destination, 0, source, 0, Bytes(count));
                list.ResourceBarrierTransition(destination, ResourceStates.CopyDest, finalState);
            }

            void Bind()
            {
                list.SetComputeRootSignature(root);
                list.SetComputeRootShaderResourceView(2, dBuffer.GPUVirtualAddress);
                list.SetComputeRootShaderResourceView(3, fBuffer.GPUVirtualAddress);
                list.SetComputeRootShaderResourceView(4, initialBuffer.GPUVirtualAddress);
                for (int i = 0; i < 3; i++)
                    list.SetComputeRootUnorderedAccessView((uint)(5 + i), projections[i].GPUVirtualAddress);
                list.SetComputeRootUnorderedAccessView(8, state.GPUVirtualAddress);
                list.SetComputeRootUnorderedAccessView(9, output.GPUVirtualAddress);
                list.SetComputeRootUnorderedAccessView(10, summary.GPUVirtualAddress);
                list.SetComputeRootUnorderedAccessView(11, scratch.GPUVirtualAddress);
                list.SetComputeRoot32BitConstant(12, (uint)width, 0);
                list.SetComputeRoot32BitConstant(12, (uint)tokens, 1);
                list.SetComputeRoot32BitConstant(12, (uint)size, 2);
                list.SetComputeRoot32BitConstant(12, (uint)stateCount, 3);
                list.SetComputeRoot32BitConstant(12, (uint)chunkSize, 4);
                list.SetComputeRoot32BitConstant(12, (uint)chunks, 5);
            }

            void Project(int projection, bool tiled)
            {
                list.SetComputeRootShaderResourceView(0, weightBuffers[projection].GPUVirtualAddress);
                list.SetComputeRootShaderResourceView(1, activationBuffers[projection].GPUVirtualAddress);
                list.SetComputeRoot32BitConstant(12, (uint)projection, 7);
                list.SetPipelineState(tiled ? projectTile : projectToken);
                if (tiled)
                    list.Dispatch((uint)((width + (long)Tile - 1) / Tile),
                        (uint)((tokens + (long)Tile - 1) / Tile), 1);
                else
                    list.Dispatch((uint)((width + (long)Threads - 1) / Threads), 1, 1);
            }

            void ResetState()
            {
                Begin();
                list.ResourceBarrierTransition(state, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                list.CopyBufferRegion(state, 0, initialBuffer, 0, Bytes(stateCount));
                list.ResourceBarrierTransition(state, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                Submit();
            }

            void Record(bool chunked)
            {
                Begin();
                Bind();
                if (chunked)
                {
                    for (int projection = 0; projection < 3; projection++)
                        Project(projection, true);
                    foreach (var buffer in projections)
                        list.ResourceBarrierUnorderedAccessView(buffer);
                    list.SetPipelineState(summaryPipeline);
                    list.Dispatch((uint)((stateCount + (long)Threads - 1) / Threads), (uint)chunks, 1);
                    list.ResourceBarrierUnorderedAccessView(summary);
                    list.SetPipelineState(foldPipeline);
                    list.Dispatch((uint)((stateCount + (long)Threads - 1) / Threads), 1, 1);
                    list.ResourceBarrierUnorderedAccessView(scratch);
                    list.SetPipelineState(emitPipeline);
                    list.Dispatch((uint)((width + (long)Threads - 1) / Threads), (uint)chunks, 1);
                }
                else
                {
                    for (int token = 0; token < tokens; token++)
                    {
                        list.SetComputeRoot32BitConstant(12, (uint)token, 6);
                        for (int projection = 0; projection < 3; projection++)
                            Project(projection, false);
                        foreach (var buffer in projections)
                            list.ResourceBarrierUnorderedAccessView(buffer);
                        list.SetPipelineState(sequentialPipeline);
                        list.Dispatch((uint)((width + (long)Threads - 1) / Threads), 1, 1);
                        if (token + 1 < tokens)
                            list.ResourceBarrierUnorderedAccessView(state);
                    }
                }
                Submit();
            }

            (TimeSpan Time, float ProjectionError, float OutputError, float StateError) Measure(bool chunked)
            {
                if (!chunked) ResetState();
                Record(chunked);
                long ticks = 0;
                for (int repeat = 0; repeat < repeats; repeat++)
                {
                    if (!chunked) ResetState();
                    long start = Stopwatch.GetTimestamp();
                    Record(chunked);
                    ticks = checked(ticks + Stopwatch.GetTimestamp() - start);
                }
                Begin();
                for (int i = 0; i < 3; i++)
                {
                    list.ResourceBarrierTransition(projections[i], ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    list.CopyBufferRegion(projectionReadbacks[i], 0, projections[i], 0, Bytes(inputCount));
                    list.ResourceBarrierTransition(projections[i], ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                }
                list.ResourceBarrierTransition(output, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                list.ResourceBarrierTransition(state, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                list.CopyBufferRegion(oRead, 0, output, 0, Bytes(inputCount));
                list.CopyBufferRegion(sRead, 0, state, 0, Bytes(stateCount));
                list.ResourceBarrierTransition(output, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                list.ResourceBarrierTransition(state, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                Submit();
                float projectionError = 0;
                for (int i = 0; i < 3; i++)
                {
                    var actual = new float[inputCount];
                    projectionReadbacks[i].GetData(actual);
                    projectionError = MathF.Max(projectionError,
                        Check(expectedProjection[i], actual, $"{(chunked ? "Chunked" : "Sequential")} projection {i}", 0.0001f));
                }
                var actualOutput = new float[inputCount];
                var actualState = new float[stateCount];
                oRead.GetData(actualOutput);
                sRead.GetData(actualState);
                return (TimeSpan.FromSeconds((double)ticks / Stopwatch.Frequency / repeats),
                    projectionError,
                    Check(expectedOutput, actualOutput, chunked ? "Chunked output" : "Sequential output", 0.003f),
                    Check(expectedState, actualState, chunked ? "Chunked state" : "Sequential state", 0.003f));
            }

            var sequential = Measure(false);
            var chunked = Measure(true);
            return new GpuAttentionPipelineResult(sequential.Time, chunked.Time,
                sequential.ProjectionError, chunked.ProjectionError,
                sequential.OutputError, sequential.StateError,
                chunked.OutputError, chunked.StateError, gpuBytes, deviceName);

            ID3D12Resource Upload<T>(T[] data) where T : unmanaged
            {
                var resource = device.CreateCommittedResource(HeapProperties.UploadHeapProperties,
                    HeapFlags.None, ResourceDescription.Buffer(Bytes(data.Length)), ResourceStates.GenericRead);
                resource.SetData(data);
                return resource;
            }

            ID3D12Resource Input(int count) => Buffer(count, ResourceFlags.None, ResourceStates.CopyDest);
            ID3D12Resource Readback(int count) =>
                device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties, HeapFlags.None,
                    ResourceDescription.Buffer(Bytes(count)), ResourceStates.CopyDest);
            ID3D12Resource Buffer(int count, ResourceFlags flags, ResourceStates state) =>
                device.CreateCommittedResource(HeapProperties.DefaultHeapProperties, HeapFlags.None,
                    ResourceDescription.Buffer(Bytes(count), flags), state);
        }
    }

    private static float Check(float[] expected, float[] actual, string label, float tolerance)
    {
        float maximum = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float difference = MathF.Abs(expected[i] - actual[i]);
            if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i]) ||
                difference > tolerance * (1 + MathF.Abs(expected[i])))
                throw new InvalidOperationException(
                    $"{label} differs from CPU at {i}: GPU={actual[i]}, CPU={expected[i]}.");
            maximum = MathF.Max(maximum, difference);
        }
        return maximum;
    }

    private static ulong Bytes(int count) => checked((ulong)count * sizeof(uint));
}
