using System.Diagnostics;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.Dxc;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.PrefillExperiment;

internal readonly record struct GpuWkv6Result(
    TimeSpan Sequential, TimeSpan Chunked, float[] SequentialState, float[] ChunkedState, string Device);

internal static class GpuWkv6Benchmark
{
    private const int Threads = 64;
    private const string Shader = """
        StructuredBuffer<float> Keys : register(t0);
        StructuredBuffer<float> Values : register(t1);
        StructuredBuffer<float> Decays : register(t2);
        RWStructuredBuffer<float> State : register(u0);
        RWStructuredBuffer<float> Summary : register(u1);
        cbuffer Parameters : register(b0)
        {
            uint HeadSize;
            uint StateCount;
            uint TokenCount;
            uint ChunkSize;
            uint ChunkCount;
            uint TokenIndex;
        };

        uint Channel(uint element)
        {
            uint headArea = HeadSize * HeadSize;
            return element / headArea * HeadSize + element % headArea / HeadSize;
        }

        uint ValueChannel(uint element)
        {
            return element / (HeadSize * HeadSize) * HeadSize + element % HeadSize;
        }

        [numthreads(64, 1, 1)]
        void Sequential(uint id : SV_DispatchThreadID)
        {
            if (id >= StateCount) return;
            uint keyIndex = TokenIndex * (StateCount / HeadSize) + Channel(id);
            uint valueIndex = TokenIndex * (StateCount / HeadSize) + ValueChannel(id);
            State[id] = State[id] * Decays[keyIndex] + Keys[keyIndex] * Values[valueIndex];
        }

        [numthreads(64, 1, 1)]
        void Summarize(uint3 id : SV_DispatchThreadID)
        {
            uint element = id.x, chunk = id.y;
            if (element >= StateCount || chunk >= ChunkCount) return;
            uint keyChannel = Channel(element), valueChannel = ValueChannel(element);
            uint channels = StateCount / HeadSize;
            uint end = min((chunk + 1) * ChunkSize, TokenCount);
            float a = 1.0f, b = 0.0f;
            for (uint token = chunk * ChunkSize; token < end; ++token)
            {
                uint keyIndex = token * channels + keyChannel;
                uint valueIndex = token * channels + valueChannel;
                float decay = Decays[keyIndex];
                b = b * decay + Keys[keyIndex] * Values[valueIndex];
                a *= decay;
            }
            uint offset = (chunk * StateCount + element) * 2;
            Summary[offset] = a;
            Summary[offset + 1] = b;
        }

        [numthreads(64, 1, 1)]
        void Fold(uint element : SV_DispatchThreadID)
        {
            if (element >= StateCount) return;
            float state = State[element];
            for (uint chunk = 0; chunk < ChunkCount; ++chunk)
            {
                uint index = (chunk * StateCount + element) * 2;
                state = state * Summary[index] + Summary[index + 1];
            }
            State[element] = state;
        }
        """;

    internal static GpuWkv6Result Run(
        int headCount, int headSize, int tokenCount, int chunkSize,
        float[] keys, float[] values, float[] decays, float[] initialState, int repeats)
    {
        if (headCount <= 0 || headSize <= 0 || tokenCount <= 0 || chunkSize <= 0 || repeats <= 0)
            throw new ArgumentOutOfRangeException(nameof(headCount),
                "Head count, head size, token count, chunk size and repeats must all be positive.");
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(decays);
        ArgumentNullException.ThrowIfNull(initialState);
        int channels = checked(headCount * headSize);
        int stateCount = checked(channels * headSize);
        int inputCount = checked(tokenCount * channels);
        int chunkCount = checked(1 + (tokenCount - 1) / chunkSize);
        int summaryCount = checked(checked(chunkCount * stateCount) * 2);
        if (keys.Length != inputCount || values.Length != inputCount || decays.Length != inputCount ||
            initialState.Length != stateCount)
            throw new ArgumentException(
                "Keys, values and decays must be token-major [token, head, channel]; initial state must be [head, key channel, value channel].");
        if (((long)stateCount + Threads - 1) / Threads > 65535 || chunkCount > 65535)
            throw new ArgumentOutOfRangeException(nameof(chunkSize), "The requested dispatch exceeds the D3D12 group limit.");

        var reference = (float[])initialState.Clone();
        for (int token = 0; token < tokenCount; token++)
            for (int state = 0; state < stateCount; state++)
            {
                int channel = state / (headSize * headSize) * headSize +
                              state % (headSize * headSize) / headSize;
                int valueChannel = state / (headSize * headSize) * headSize + state % headSize;
                int keyIndex = token * channels + channel;
                reference[state] = reference[state] * decays[keyIndex] +
                                   keys[keyIndex] * values[token * channels + valueChannel];
            }

        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory6>();
        IDXGIAdapter1? adapter = null;
        for (uint index = 0; factory.EnumAdapterByGpuPreference(
                 index, GpuPreference.HighPerformance, out IDXGIAdapter1? candidate).Success; index++)
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
            throw new PlatformNotSupportedException("No hardware D3D12 compute adapter is available for the WKV-6 experiment.");
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

            using var root = device.CreateRootSignature(
                new RootSignatureDescription(RootSignatureFlags.None,
                [
                    new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(2, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                    new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
                ]), RootSignatureVersion.Version10);

            using var sequentialPipeline = CompilePipeline("Sequential");
            using var summaryPipeline = CompilePipeline("Summarize");
            using var foldPipeline = CompilePipeline("Fold");
            ID3D12PipelineState CompilePipeline(string entry)
            {
                using var compilation = DxcCompiler.Compile(DxcShaderStage.Compute, Shader, entry,
                    new DxcCompilerOptions { ShaderModel = DxcShaderModel.Model6_0 });
                return device.CreateComputePipelineState(new ComputePipelineStateDescription
                {
                    RootSignature = root,
                    ComputeShader = compilation.GetObjectBytecodeArray(),
                });
            }

            using var keyUpload = Upload(keys);
            using var valueUpload = Upload(values);
            using var decayUpload = Upload(decays);
            using var stateUpload = Upload(initialState);
            using var keyBuffer = DefaultBuffer(Bytes(inputCount), ResourceFlags.None, ResourceStates.CopyDest);
            using var valueBuffer = DefaultBuffer(Bytes(inputCount), ResourceFlags.None, ResourceStates.CopyDest);
            using var decayBuffer = DefaultBuffer(Bytes(inputCount), ResourceFlags.None, ResourceStates.CopyDest);
            using var initialBuffer = DefaultBuffer(Bytes(stateCount), ResourceFlags.None, ResourceStates.CopyDest);
            using var stateBuffer = DefaultBuffer(Bytes(stateCount), ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var summaryBuffer = DefaultBuffer(Bytes(summaryCount), ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            using var readback = device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties,
                HeapFlags.None, ResourceDescription.Buffer(Bytes(stateCount)), ResourceStates.CopyDest);
            ulong fenceValue = 0;

            void Wait()
            {
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

            void Submit()
            {
                list.Close();
                queue.ExecuteCommandList(list);
                Wait();
            }

            void Begin()
            {
                allocator.Reset();
                list.Reset(allocator);
            }

            Begin();
            CopyInput(keyBuffer, keyUpload, inputCount);
            CopyInput(valueBuffer, valueUpload, inputCount);
            CopyInput(decayBuffer, decayUpload, inputCount);
            CopyInput(initialBuffer, stateUpload, stateCount, ResourceStates.CopySource);
            Submit();

            void CopyInput(ID3D12Resource destination, ID3D12Resource source, int count,
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
                list.SetComputeRootUnorderedAccessView(3, stateBuffer.GPUVirtualAddress);
                list.SetComputeRootUnorderedAccessView(4, summaryBuffer.GPUVirtualAddress);
                list.SetComputeRoot32BitConstant(5, checked((uint)headSize), 0);
                list.SetComputeRoot32BitConstant(5, checked((uint)stateCount), 1);
                list.SetComputeRoot32BitConstant(5, checked((uint)tokenCount), 2);
                list.SetComputeRoot32BitConstant(5, checked((uint)chunkSize), 3);
                list.SetComputeRoot32BitConstant(5, checked((uint)chunkCount), 4);
            }

            (TimeSpan Elapsed, float[] State) Measure(bool chunked)
            {
                long ticks = 0;
                for (int repeat = -1; repeat < repeats; repeat++)
                {
                    // Reset via a GPU copy outside the timed dispatch and fence interval.
                    Begin();
                    list.ResourceBarrierTransition(stateBuffer, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                    list.CopyBufferRegion(stateBuffer, 0, initialBuffer, 0, Bytes(stateCount));
                    list.ResourceBarrierTransition(stateBuffer, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                    Submit();

                    long start = Stopwatch.GetTimestamp();
                    Begin();
                    Bind();
                    if (chunked)
                    {
                        list.SetPipelineState(summaryPipeline);
                        list.Dispatch(checked((uint)(((long)stateCount + Threads - 1) / Threads)),
                            checked((uint)chunkCount), 1);
                        list.ResourceBarrierUnorderedAccessView(summaryBuffer);
                        list.SetPipelineState(foldPipeline);
                        list.Dispatch(checked((uint)(((long)stateCount + Threads - 1) / Threads)), 1, 1);
                    }
                    else
                    {
                        list.SetPipelineState(sequentialPipeline);
                        uint groups = checked((uint)(((long)stateCount + Threads - 1) / Threads));
                        for (int token = 0; token < tokenCount; token++)
                        {
                            list.SetComputeRoot32BitConstant(5, checked((uint)token), 5);
                            list.Dispatch(groups, 1, 1);
                            if (token + 1 < tokenCount)
                                list.ResourceBarrierUnorderedAccessView(stateBuffer);
                        }
                    }
                    Submit();
                    if (repeat >= 0)
                        ticks += Stopwatch.GetTimestamp() - start;
                }
                Begin();
                list.ResourceBarrierTransition(stateBuffer, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                list.CopyBufferRegion(readback, 0, stateBuffer, 0, Bytes(stateCount));
                list.ResourceBarrierTransition(stateBuffer, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                Submit();
                var output = new float[stateCount];
                readback.GetData(output);
                for (int i = 0; i < output.Length; i++)
                {
                    float expected = reference[i], actual = output[i];
                    if (!float.IsFinite(actual) || !float.IsFinite(expected) ||
                        MathF.Abs(actual - expected) > 0.002f * (1 + MathF.Abs(expected)))
                        throw new InvalidOperationException(
                            $"{(chunked ? "Chunked" : "Sequential")} GPU WKV-6 state differs from CPU at element {i}: GPU={actual}, CPU={expected}.");
                }
                return (TimeSpan.FromSeconds((double)ticks / Stopwatch.Frequency / repeats), output);
            }

            var sequential = Measure(false);
            var chunked = Measure(true);
            return new GpuWkv6Result(sequential.Elapsed, chunked.Elapsed,
                sequential.State, chunked.State, deviceName);

            ID3D12Resource Upload(float[] data)
            {
                var resource = device.CreateCommittedResource(HeapProperties.UploadHeapProperties,
                    HeapFlags.None, ResourceDescription.Buffer(Bytes(data.Length)), ResourceStates.GenericRead);
                resource.SetData(data);
                return resource;
            }

            ID3D12Resource DefaultBuffer(ulong size, ResourceFlags flags, ResourceStates state) =>
                device.CreateCommittedResource(HeapProperties.DefaultHeapProperties, HeapFlags.None,
                    ResourceDescription.Buffer(size, flags), state);
        }
    }

    private static ulong Bytes(int floatCount) => checked((ulong)floatCount * sizeof(float));
}
