using System.Diagnostics;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.Dxc;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.PrefillExperiment;

internal readonly record struct GpuFp16ProjectionResult(
    TimeSpan Sequential, TimeSpan Batched, float MaxDifference,
    float[] SequentialOutput, float[] BatchedOutput);

internal static class GpuFp16ProjectionBenchmark
{
    private const int Tile = 16;
    private const string Shader = """
        StructuredBuffer<uint> PackedWeights : register(t0);
        StructuredBuffer<float> Activations : register(t1);
        RWStructuredBuffer<float> Output : register(u0);
        cbuffer Parameters : register(b0)
        {
            uint Width;
            uint Tokens;
            uint TokenIndex;
        };

        float Weight(uint row, uint col)
        {
            uint index = row * Width + col;
            uint pair = PackedWeights[index >> 1];
            return f16tof32((pair >> ((index & 1) * 16)) & 0xffff);
        }

        [numthreads(64, 1, 1)]
        void Sequential(uint row : SV_DispatchThreadID)
        {
            if (row >= Width) return;
            float sum = 0.0f;
            for (uint col = 0; col < Width; ++col)
                sum += Weight(row, col) * Activations[TokenIndex * Width + col];
            Output[TokenIndex * Width + row] = sum;
        }

        groupshared float WeightTile[16][16];
        groupshared float InputTile[16][16];

        [numthreads(16, 16, 1)]
        void Batched(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
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
                Output[token * Width + row] = sum;
        }
        """;

    // Packed two-half uint elements give the root SRV a four-byte stride, including for odd widths.
    internal static GpuFp16ProjectionResult Run(
        int tokens = 16, int width = 256, int repeats = 10, Half[]? modelWeights = null)
    {
        if (tokens <= 0 || width <= 0 || repeats <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), "Tokens, width and repeats must be positive.");
        int weightCount = checked(width * width);
        int outputCount = checked(tokens * width);
        if (modelWeights is not null && modelWeights.Length != weightCount)
            throw new ArgumentException("The FP16 model projection must be a square matrix of the selected width.",
                nameof(modelWeights));
        if (((long)width + 63) / 64 > 65535 ||
            ((long)width + Tile - 1) / Tile > 65535 ||
            ((long)tokens + Tile - 1) / Tile > 65535)
            throw new ArgumentOutOfRangeException(nameof(tokens), "The requested dispatch exceeds the D3D12 group limit.");

        var random = new Random(1701);
        var packedWeights = new uint[weightCount / 2 + weightCount % 2];
        for (int i = 0; i < weightCount; i++)
        {
            ushort bits = BitConverter.HalfToUInt16Bits(modelWeights is null
                ? (Half)(random.NextDouble() * 0.02 - 0.01)
                : modelWeights[i]);
            packedWeights[i >> 1] |= (uint)bits << ((i & 1) * 16);
        }
        var activations = new float[outputCount];
        for (int i = 0; i < outputCount; i++)
            activations[i] = (float)(random.NextDouble() * 0.2 - 0.1);

        var reference = new float[outputCount];
        for (int token = 0; token < tokens; token++)
        for (int row = 0; row < width; row++)
        {
            float sum = 0;
            for (int col = 0; col < width; col++)
            {
                int index = row * width + col;
                ushort bits = (ushort)(packedWeights[index >> 1] >> ((index & 1) * 16));
                sum += (float)BitConverter.UInt16BitsToHalf(bits) * activations[token * width + col];
            }
            reference[token * width + row] = sum;
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
            throw new PlatformNotSupportedException("No hardware D3D12 compute adapter is available for the FP16 projection benchmark.");

        using (adapter)
        {
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
                    new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                    new RootParameter(new RootConstants(0, 0, 3), ShaderVisibility.All),
                ]), RootSignatureVersion.Version10);
            using var sequentialPipeline = Compile("Sequential");
            using var batchedPipeline = Compile("Batched");

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

            using var weightUpload = Upload(packedWeights);
            using var inputUpload = Upload(activations);
            using var weightBuffer = DefaultBuffer(Bytes(packedWeights.Length), ResourceFlags.None, ResourceStates.CopyDest);
            using var inputBuffer = DefaultBuffer(Bytes(outputCount), ResourceFlags.None, ResourceStates.CopyDest);
            using var outputBuffer = DefaultBuffer(Bytes(outputCount), ResourceFlags.AllowUnorderedAccess,
                ResourceStates.UnorderedAccess);
            using var readback = device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties,
                HeapFlags.None, ResourceDescription.Buffer(Bytes(outputCount)), ResourceStates.CopyDest);
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

            void Begin()
            {
                allocator.Reset();
                list.Reset(allocator);
            }

            void Submit()
            {
                list.Close();
                queue.ExecuteCommandList(list);
                Wait();
            }

            Begin();
            list.CopyBufferRegion(weightBuffer, 0, weightUpload, 0, Bytes(packedWeights.Length));
            list.CopyBufferRegion(inputBuffer, 0, inputUpload, 0, Bytes(outputCount));
            list.ResourceBarrierTransition(weightBuffer, ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource);
            list.ResourceBarrierTransition(inputBuffer, ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource);
            Submit();

            void Record(bool batched)
            {
                Begin();
                list.SetComputeRootSignature(root);
                list.SetComputeRootShaderResourceView(0, weightBuffer.GPUVirtualAddress);
                list.SetComputeRootShaderResourceView(1, inputBuffer.GPUVirtualAddress);
                list.SetComputeRootUnorderedAccessView(2, outputBuffer.GPUVirtualAddress);
                list.SetComputeRoot32BitConstant(3, checked((uint)width), 0);
                list.SetComputeRoot32BitConstant(3, checked((uint)tokens), 1);
                if (batched)
                {
                    list.SetPipelineState(batchedPipeline);
                    list.Dispatch(checked((uint)(((long)width + Tile - 1) / Tile)),
                        checked((uint)(((long)tokens + Tile - 1) / Tile)), 1);
                }
                else
                {
                    list.SetPipelineState(sequentialPipeline);
                    uint groups = checked((uint)(((long)width + 63) / 64));
                    for (int token = 0; token < tokens; token++)
                    {
                        list.SetComputeRoot32BitConstant(3, checked((uint)token), 2);
                        list.Dispatch(groups, 1, 1);
                    }
                }
                Submit();
            }

            (TimeSpan Elapsed, float[] Data, float Difference) Measure(bool batched)
            {
                Record(batched);
                long ticks = 0;
                for (int i = 0; i < repeats; i++)
                {
                    long start = Stopwatch.GetTimestamp();
                    Record(batched);
                    ticks = checked(ticks + Stopwatch.GetTimestamp() - start);
                }

                Begin();
                list.ResourceBarrierTransition(outputBuffer, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                list.CopyBufferRegion(readback, 0, outputBuffer, 0, Bytes(outputCount));
                list.ResourceBarrierTransition(outputBuffer, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                Submit();
                var actual = new float[outputCount];
                readback.GetData(actual);
                float maximum = 0;
                for (int i = 0; i < actual.Length; i++)
                {
                    float difference = MathF.Abs(actual[i] - reference[i]);
                    if (!float.IsFinite(actual[i]) || !float.IsFinite(reference[i]) ||
                        difference > 0.00002f + MathF.Abs(reference[i]) * 0.0001f)
                        throw new InvalidOperationException(
                            $"{(batched ? "Batched" : "Sequential")} FP16 GPU projection differs from CPU at {i}: GPU={actual[i]}, CPU={reference[i]}.");
                    maximum = MathF.Max(maximum, difference);
                }
                return (TimeSpan.FromSeconds((double)ticks / Stopwatch.Frequency / repeats), actual, maximum);
            }

            var sequential = Measure(false);
            var batched = Measure(true);
            float maxDifference = MathF.Max(sequential.Difference, batched.Difference);
            for (int i = 0; i < outputCount; i++)
                maxDifference = MathF.Max(maxDifference, MathF.Abs(sequential.Data[i] - batched.Data[i]));
            return new GpuFp16ProjectionResult(sequential.Elapsed, batched.Elapsed, maxDifference,
                sequential.Data, batched.Data);

            ID3D12Resource Upload<T>(T[] data) where T : unmanaged
            {
                var resource = device.CreateCommittedResource(HeapProperties.UploadHeapProperties,
                    HeapFlags.None, ResourceDescription.Buffer(Bytes(data.Length)), ResourceStates.GenericRead);
                resource.SetData(data);
                return resource;
            }

            ID3D12Resource DefaultBuffer(ulong bytes, ResourceFlags flags, ResourceStates state) =>
                device.CreateCommittedResource(HeapProperties.DefaultHeapProperties, HeapFlags.None,
                    ResourceDescription.Buffer(bytes, flags), state);
        }
    }

    private static ulong Bytes(int elements) => checked((ulong)elements * sizeof(uint));
}
