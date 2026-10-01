using System.Diagnostics;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.Dxc;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.PrefillExperiment;

internal readonly record struct GpuWkv6OutputsResult(
    TimeSpan Sequential, TimeSpan Chunked,
    float SequentialOutputMaxError, float SequentialStateMaxError,
    float ChunkedOutputMaxError, float ChunkedStateMaxError, string Device);

/// <summary>Benchmarks WKV-6 attention for token-major [token, head, channel] inputs.
/// State is [head, key channel, value channel], timeFirst is [head, key channel],
/// and outputs are [token, head, value channel].</summary>
internal static class GpuWkv6OutputsBenchmark
{
    private const int Threads = 64;
    private const long MaxGpuBytes = 512L * 1024 * 1024;
    private const string Shader = """
        StructuredBuffer<float> Keys : register(t0);
        StructuredBuffer<float> Values : register(t1);
        StructuredBuffer<float> Decays : register(t2);
        StructuredBuffer<float> Receptance : register(t3);
        StructuredBuffer<float> TimeFirst : register(t4);
        StructuredBuffer<float> Initial : register(t5);
        RWStructuredBuffer<float> State : register(u0);
        RWStructuredBuffer<float> Output : register(u1);
        RWStructuredBuffer<float> Summary : register(u2);
        RWStructuredBuffer<float> Scratch : register(u3);
        cbuffer Parameters : register(b0)
        {
            uint HeadSize;
            uint StateCount;
            uint Channels;
            uint TokenCount;
            uint ChunkSize;
            uint ChunkCount;
            uint TokenIndex;
        };

        [numthreads(64, 1, 1)]
        void Sequential(uint id : SV_DispatchThreadID)
        {
            if (id >= Channels) return;
            uint head = id / HeadSize, value = id % HeadSize;
            uint input = TokenIndex * Channels + head * HeadSize;
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
            Output[TokenIndex * Channels + id] = sum;
        }

        // Each (chunk, matrix element) independently summarizes its affine transform.
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
            uint end = min((chunk + 1) * ChunkSize, TokenCount);
            for (uint token = chunk * ChunkSize; token < end; ++token)
            {
                uint input = token * Channels;
                b = b * Decays[input + channel] + Keys[input + channel] * Values[input + head * HeadSize + value];
                a *= Decays[input + channel];
            }
            uint summary = (chunk * StateCount + element) * 2;
            Summary[summary] = a;
            Summary[summary + 1] = b;
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

        // Each chunk and value channel owns its scratch matrix column. Chunks
        // advance concurrently; each token is visited just once per matrix element.
        [numthreads(64, 1, 1)]
        void Emit(uint3 id : SV_DispatchThreadID)
        {
            uint channel = id.x, chunk = id.y;
            if (channel >= Channels || chunk >= ChunkCount) return;
            uint head = channel / HeadSize, value = channel % HeadSize;
            uint end = min((chunk + 1) * ChunkSize, TokenCount);
            for (uint token = chunk * ChunkSize; token < end; ++token)
            {
                uint input = token * Channels + head * HeadSize;
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
                Output[token * Channels + channel] = sum;
            }
        }
        """;

    internal static GpuWkv6OutputsResult Run(
        int headCount, int headSize, int tokenCount, int chunkSize,
        float[] keys, float[] values, float[] decays, float[] receptance,
        float[] timeFirst, float[] initialState, int repeats)
    {
        if (headCount <= 0 || headSize <= 0 || tokenCount <= 0 || chunkSize <= 0 || repeats <= 0)
            throw new ArgumentOutOfRangeException(nameof(headCount),
                "Head count, head size, token count, chunk size and repeats must be positive.");
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(decays);
        ArgumentNullException.ThrowIfNull(receptance);
        ArgumentNullException.ThrowIfNull(timeFirst);
        ArgumentNullException.ThrowIfNull(initialState);
        int channels = checked(headCount * headSize);
        int stateCount = checked(channels * headSize);
        int inputCount = checked(tokenCount * channels);
        int outputCount = inputCount;
        int chunkCount = checked(1 + (tokenCount - 1) / chunkSize);
        int summaryCount = checked(checked(chunkCount * stateCount) * 2);
        int scratchCount = checked(chunkCount * stateCount);
        if (keys.Length != inputCount || values.Length != inputCount ||
            decays.Length != inputCount || receptance.Length != inputCount ||
            timeFirst.Length != channels || initialState.Length != stateCount)
            throw new ArgumentException("Expected token-major keys, values, decays and receptance [token, head, channel], timeFirst [head, channel], and initialState [head, key channel, value channel].");
        if (((long)channels + Threads - 1) / Threads > 65535 ||
            ((long)stateCount + Threads - 1) / Threads > 65535 ||
            chunkCount > 65535)
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "D3D12 dispatch exceeds 65535 groups on an axis.");

        long floats = checked(4L * inputCount + channels + 2L * stateCount +
            outputCount + summaryCount + scratchCount);
        // Include uploads and readback buffers as well as default-heap resources.
        long gpuBytes = checked(4L * (floats + 4L * inputCount + channels + stateCount +
            outputCount + stateCount));
        if (gpuBytes > MaxGpuBytes)
            throw new ArgumentOutOfRangeException(nameof(tokenCount),
                $"Benchmark buffers require {gpuBytes:N0} bytes, exceeding the {MaxGpuBytes:N0}-byte GPU allocation guard.");
        foreach (var (data, name) in new (float[], string)[]
                 { (keys, nameof(keys)), (values, nameof(values)), (decays, nameof(decays)),
                   (receptance, nameof(receptance)), (timeFirst, nameof(timeFirst)),
                   (initialState, nameof(initialState)) })
            for (int i = 0; i < data.Length; ++i)
                if (!float.IsFinite(data[i]))
                    throw new ArgumentException($"{name}[{i}] must be finite.");

        var expectedState = (float[])initialState.Clone();
        var expectedOutput = new float[outputCount];
        for (int token = 0; token < tokenCount; ++token)
            for (int head = 0; head < headCount; ++head)
                for (int value = 0; value < headSize; ++value)
                {
                    int baseInput = token * channels + head * headSize;
                    float v = values[baseInput + value];
                    float sum = 0;
                    for (int key = 0; key < headSize; ++key)
                    {
                        int matrix = (head * headSize + key) * headSize + value;
                        float old = expectedState[matrix];
                        float k = keys[baseInput + key];
                        sum += receptance[baseInput + key] *
                               (old + timeFirst[head * headSize + key] * k * v);
                        expectedState[matrix] = old * decays[baseInput + key] + k * v;
                    }
                    expectedOutput[token * channels + head * headSize + value] = sum;
                }

        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory6>();
        IDXGIAdapter1? adapter = null;
        for (uint i = 0; factory.EnumAdapterByGpuPreference(
                 i, GpuPreference.HighPerformance, out IDXGIAdapter1? candidate).Success; ++i)
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
            throw new PlatformNotSupportedException("No hardware D3D12 compute adapter is available for the WKV-6 outputs benchmark.");

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
            using var root = device.CreateRootSignature(new RootSignatureDescription(
                RootSignatureFlags.None,
                [
                    new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(2, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(3, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(4, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(5, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All),
                    new RootParameter(new RootConstants(0, 0, 7), ShaderVisibility.All),
                ]), RootSignatureVersion.Version10);
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

            using var keyUpload = Upload(keys);
            using var valueUpload = Upload(values);
            using var decayUpload = Upload(decays);
            using var receptanceUpload = Upload(receptance);
            using var timeFirstUpload = Upload(timeFirst);
            using var initialUpload = Upload(initialState);
            using var keyBuffer = Input(inputCount);
            using var valueBuffer = Input(inputCount);
            using var decayBuffer = Input(inputCount);
            using var receptanceBuffer = Input(inputCount);
            using var timeFirstBuffer = Input(channels);
            using var initialBuffer = Input(stateCount);
            using var stateBuffer = Buffer(stateCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var outputBuffer = Buffer(outputCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var summaryBuffer = Buffer(summaryCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var scratchBuffer = Buffer(scratchCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var outputReadback = device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties,
                HeapFlags.None, ResourceDescription.Buffer(Bytes(outputCount)), ResourceStates.CopyDest);
            using var stateReadback = device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties,
                HeapFlags.None, ResourceDescription.Buffer(Bytes(stateCount)), ResourceStates.CopyDest);
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
            Copy(keyBuffer, keyUpload, inputCount);
            Copy(valueBuffer, valueUpload, inputCount);
            Copy(decayBuffer, decayUpload, inputCount);
            Copy(receptanceBuffer, receptanceUpload, inputCount);
            Copy(timeFirstBuffer, timeFirstUpload, channels);
            Copy(initialBuffer, initialUpload, stateCount, ResourceStates.CopySource | ResourceStates.NonPixelShaderResource);
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
                list.SetComputeRootShaderResourceView(0, keyBuffer.GPUVirtualAddress);
                list.SetComputeRootShaderResourceView(1, valueBuffer.GPUVirtualAddress);
                list.SetComputeRootShaderResourceView(2, decayBuffer.GPUVirtualAddress);
                list.SetComputeRootShaderResourceView(3, receptanceBuffer.GPUVirtualAddress);
                list.SetComputeRootShaderResourceView(4, timeFirstBuffer.GPUVirtualAddress);
                list.SetComputeRootShaderResourceView(5, initialBuffer.GPUVirtualAddress);
                list.SetComputeRootUnorderedAccessView(6, stateBuffer.GPUVirtualAddress);
                list.SetComputeRootUnorderedAccessView(7, outputBuffer.GPUVirtualAddress);
                list.SetComputeRootUnorderedAccessView(8, summaryBuffer.GPUVirtualAddress);
                list.SetComputeRootUnorderedAccessView(9, scratchBuffer.GPUVirtualAddress);
                list.SetComputeRoot32BitConstant(10, (uint)headSize, 0);
                list.SetComputeRoot32BitConstant(10, (uint)stateCount, 1);
                list.SetComputeRoot32BitConstant(10, (uint)channels, 2);
                list.SetComputeRoot32BitConstant(10, (uint)tokenCount, 3);
                list.SetComputeRoot32BitConstant(10, (uint)chunkSize, 4);
                list.SetComputeRoot32BitConstant(10, (uint)chunkCount, 5);
            }

            (TimeSpan Time, float OutputError, float StateError) Measure(bool chunked)
            {
                long ticks = 0;
                for (int repeat = -1; repeat < repeats; ++repeat)
                {
                    if (!chunked)
                    {
                        Begin();
                        list.ResourceBarrierTransition(stateBuffer, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                        list.CopyBufferRegion(stateBuffer, 0, initialBuffer, 0, Bytes(stateCount));
                        list.ResourceBarrierTransition(stateBuffer, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                        Submit();
                    }
                    long start = Stopwatch.GetTimestamp();
                    Begin();
                    Bind();
                    if (chunked)
                    {
                        list.SetPipelineState(summaryPipeline);
                        list.Dispatch((uint)((stateCount + (long)Threads - 1) / Threads), (uint)chunkCount, 1);
                        list.ResourceBarrierUnorderedAccessView(summaryBuffer);
                        list.SetPipelineState(foldPipeline);
                        list.Dispatch((uint)((stateCount + (long)Threads - 1) / Threads), 1, 1);
                        list.ResourceBarrierUnorderedAccessView(scratchBuffer);
                        list.SetPipelineState(emitPipeline);
                        list.Dispatch((uint)((channels + (long)Threads - 1) / Threads), (uint)chunkCount, 1);
                    }
                    else
                    {
                        list.SetPipelineState(sequentialPipeline);
                        uint groups = (uint)((channels + (long)Threads - 1) / Threads);
                        for (int token = 0; token < tokenCount; ++token)
                        {
                            list.SetComputeRoot32BitConstant(10, (uint)token, 6);
                            list.Dispatch(groups, 1, 1);
                            if (token + 1 < tokenCount)
                                list.ResourceBarrierUnorderedAccessView(stateBuffer);
                        }
                    }
                    Submit();
                    if (repeat >= 0)
                        ticks = checked(ticks + Stopwatch.GetTimestamp() - start);
                }
                Begin();
                list.ResourceBarrierTransition(outputBuffer, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                list.ResourceBarrierTransition(stateBuffer, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                list.CopyBufferRegion(outputReadback, 0, outputBuffer, 0, Bytes(outputCount));
                list.CopyBufferRegion(stateReadback, 0, stateBuffer, 0, Bytes(stateCount));
                list.ResourceBarrierTransition(outputBuffer, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                list.ResourceBarrierTransition(stateBuffer, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                Submit();
                var actualOutput = new float[outputCount];
                var actualState = new float[stateCount];
                outputReadback.GetData(actualOutput);
                stateReadback.GetData(actualState);
                return (TimeSpan.FromSeconds((double)ticks / Stopwatch.Frequency / repeats),
                    Check(expectedOutput, actualOutput, chunked ? "Chunked output" : "Sequential output"),
                    Check(expectedState, actualState, chunked ? "Chunked state" : "Sequential state"));
            }

            var sequential = Measure(false);
            var chunked = Measure(true);
            return new GpuWkv6OutputsResult(sequential.Time, chunked.Time,
                sequential.OutputError, sequential.StateError,
                chunked.OutputError, chunked.StateError, deviceName);

            ID3D12Resource Upload(float[] data)
            {
                var resource = device.CreateCommittedResource(HeapProperties.UploadHeapProperties,
                    HeapFlags.None, ResourceDescription.Buffer(Bytes(data.Length)), ResourceStates.GenericRead);
                resource.SetData(data);
                return resource;
            }

            ID3D12Resource Input(int count) => Buffer(count, ResourceFlags.None, ResourceStates.CopyDest);
            ID3D12Resource Buffer(int count, ResourceFlags flags, ResourceStates state) =>
                device.CreateCommittedResource(HeapProperties.DefaultHeapProperties,
                    HeapFlags.None, ResourceDescription.Buffer(Bytes(count), flags), state);
        }
    }

    private static float Check(float[] expected, float[] actual, string label)
    {
        float maximum = 0;
        for (int i = 0; i < expected.Length; ++i)
        {
            float difference = MathF.Abs(expected[i] - actual[i]);
            if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i]) ||
                difference > 0.003f * (1 + MathF.Abs(expected[i])))
                throw new InvalidOperationException(
                    $"{label} differs from CPU at {i}: GPU={actual[i]}, CPU={expected[i]}.");
            maximum = MathF.Max(maximum, difference);
        }
        return maximum;
    }

    private static ulong Bytes(int count) => checked((ulong)count * sizeof(float));
}
