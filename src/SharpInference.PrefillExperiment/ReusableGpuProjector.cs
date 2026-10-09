using System.Diagnostics;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.Dxc;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.PrefillExperiment;

internal sealed class ReusableGpuProjector : IDisposable
{
    private const int Tile = 16;
    private const int UploadElements = 1 << 22;
    private const string Shader = """
        StructuredBuffer<uint> PackedWeights : register(t0);
        StructuredBuffer<float> Activations : register(t1);
        RWStructuredBuffer<float> Output : register(u0);
        cbuffer Parameters : register(b0)
        {
            uint InputWidth;
            uint OutputWidth;
            uint Tokens;
            uint IsHalf;
            uint TokenOffset;
        };

        float Weight(uint row, uint col)
        {
            uint index = row * InputWidth + col;
            if (IsHalf != 0)
            {
                uint pair = PackedWeights[index >> 1];
                return f16tof32((pair >> ((index & 1) * 16)) & 0xffff);
            }
            return asfloat(PackedWeights[index]);
        }

        groupshared float WeightTile[16][16];
        groupshared float InputTile[16][16];

        [numthreads(16, 16, 1)]
        void Project(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
        {
            uint row = group.x * 16 + lane.x;
            uint token = group.y * 16 + lane.y;
            float sum = 0.0f;
            for (uint start = 0; start < InputWidth; start += 16)
            {
                uint weightCol = start + lane.y;
                uint inputCol = start + lane.x;
                WeightTile[lane.x][lane.y] =
                    row < OutputWidth && weightCol < InputWidth ? Weight(row, weightCol) : 0.0f;
                InputTile[lane.y][lane.x] =
                    token < Tokens && inputCol < InputWidth
                        ? Activations[token * InputWidth + inputCol] : 0.0f;
                GroupMemoryBarrierWithGroupSync();
                [unroll]
                for (uint col = 0; col < 16; ++col)
                    sum += WeightTile[lane.x][col] * InputTile[lane.y][col];
                GroupMemoryBarrierWithGroupSync();
            }
            if (row < OutputWidth && token < Tokens)
                Output[token * OutputWidth + row] = sum;
        }

        [numthreads(64, 1, 1)]
        void ProjectRows(uint3 id : SV_DispatchThreadID)
        {
            uint row = id.x;
            uint token = TokenOffset + id.y;
            if (row >= OutputWidth || token >= Tokens) return;
            float sum = 0.0f;
            for (uint col = 0; col < InputWidth; ++col)
                sum += Weight(row, col) * Activations[token * InputWidth + col];
            Output[token * OutputWidth + row] = sum;
        }
        """;

    private readonly ID3D12Device device;
    private readonly ID3D12CommandQueue queue;
    private readonly ID3D12Fence fence;
    private readonly EventWaitHandle completion;
    private readonly ID3D12CommandAllocator allocator;
    private readonly ID3D12GraphicsCommandList list;
    private readonly ID3D12RootSignature root;
    private readonly ID3D12PipelineState pipeline;
    private readonly ID3D12PipelineState rowPipeline;
    private readonly ID3D12Resource weightUpload;
    private readonly Dictionary<IModelTensor, ID3D12Resource> weights = new(ReferenceEqualityComparer.Instance);
    private ID3D12Resource? inputUpload;
    private ID3D12Resource? inputBuffer;
    private ID3D12Resource? outputBuffer;
    private ID3D12Resource? readback;
    private int inputCapacity;
    private int outputCapacity;
    private ulong fenceValue;
    private bool disposed;

    internal long ProjectCount { get; private set; }
    internal TimeSpan TotalElapsed { get; private set; }

    internal ReusableGpuProjector()
    {
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
            throw new PlatformNotSupportedException("No hardware D3D12 compute adapter is available for GPU projection.");

        using (adapter)
            device = D3D12CreateDevice<ID3D12Device>(adapter, FeatureLevel.Level_11_0);
        queue = device.CreateCommandQueue(CommandListType.Compute);
        fence = device.CreateFence();
        completion = new EventWaitHandle(false, EventResetMode.AutoReset);
        allocator = device.CreateCommandAllocator(CommandListType.Compute);
        list = device.CreateCommandList<ID3D12GraphicsCommandList>(
            0, CommandListType.Compute, allocator);
        list.Close();
        root = device.CreateRootSignature(new RootSignatureDescription(
            RootSignatureFlags.None,
            [
                new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new RootParameter(new RootConstants(0, 0, 5), ShaderVisibility.All),
            ]), RootSignatureVersion.Version10);
        using var shader = DxcCompiler.Compile(DxcShaderStage.Compute, Shader, "Project",
            new DxcCompilerOptions { ShaderModel = DxcShaderModel.Model6_0 });
        pipeline = device.CreateComputePipelineState(new ComputePipelineStateDescription
        {
            RootSignature = root,
            ComputeShader = shader.GetObjectBytecodeArray(),
        });
        using var rowShader = DxcCompiler.Compile(DxcShaderStage.Compute, Shader, "ProjectRows",
            new DxcCompilerOptions { ShaderModel = DxcShaderModel.Model6_0 });
        rowPipeline = device.CreateComputePipelineState(new ComputePipelineStateDescription
        {
            RootSignature = root,
            ComputeShader = rowShader.GetObjectBytecodeArray(),
        });
        weightUpload = device.CreateCommittedResource(HeapProperties.UploadHeapProperties,
            HeapFlags.None, ResourceDescription.Buffer(Bytes(UploadElements)), ResourceStates.GenericRead);
    }

    internal float[] Project(IModelTensor tensor, float[] tokenMajorInput, int tokenCount, int inputWidth)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentNullException.ThrowIfNull(tokenMajorInput);
        if (tokenCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokenCount));
        if (inputWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputWidth));
        if (tensor.Dimensions.Count != 2 || tensor.Dimensions[0] != inputWidth ||
            tensor.Dimensions[1] <= 0)
            throw new ArgumentException("Projection tensor dimensions must be [inputWidth, outputWidth].", nameof(tensor));
        if (tensor.DataType is not (TensorDataType.Float16 or TensorDataType.Float32))
            throw new NotSupportedException($"Projection tensor type {tensor.DataType} is not supported.");

        int outputWidth = tensor.Dimensions[1];
        int weightCount = checked(inputWidth * outputWidth);
        int inputCount = checked(tokenCount * inputWidth);
        int outputCount = checked(tokenCount * outputWidth);
        if (tokenMajorInput.Length != inputCount)
            throw new ArgumentException("Token-major input length does not match tokenCount * inputWidth.",
                nameof(tokenMajorInput));
        if (((long)outputWidth + Tile - 1) / Tile > 65535 ||
            ((long)tokenCount + Tile - 1) / Tile > 65535)
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "The projection dispatch exceeds the D3D12 group limit.");
        bool half = tensor.DataType == TensorDataType.Float16;
        if (half ? tensor.HalfValues.Length != weightCount : tensor.FloatValues.Length != weightCount)
            throw new ArgumentException("Tensor value count does not match its rectangular dimensions.", nameof(tensor));

        long start = Stopwatch.GetTimestamp();
        if (!weights.TryGetValue(tensor, out ID3D12Resource? weightBuffer))
            weightBuffer = CacheWeight(tensor, weightCount, half);
        EnsureScratch(inputCount, outputCount);

        inputUpload!.SetData(tokenMajorInput);
        Begin();
        list.ResourceBarrierTransition(inputBuffer!, ResourceStates.NonPixelShaderResource, ResourceStates.CopyDest);
        list.CopyBufferRegion(inputBuffer!, 0, inputUpload, 0, Bytes(inputCount));
        list.ResourceBarrierTransition(inputBuffer!, ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource);
        list.SetComputeRootSignature(root);
        list.SetComputeRootShaderResourceView(0, weightBuffer.GPUVirtualAddress);
        list.SetComputeRootShaderResourceView(1, inputBuffer!.GPUVirtualAddress);
        list.SetComputeRootUnorderedAccessView(2, outputBuffer!.GPUVirtualAddress);
        list.SetComputeRoot32BitConstant(3, checked((uint)inputWidth), 0);
        list.SetComputeRoot32BitConstant(3, checked((uint)outputWidth), 1);
        list.SetComputeRoot32BitConstant(3, checked((uint)tokenCount), 2);
        list.SetComputeRoot32BitConstant(3, half ? 1u : 0u, 3);
        list.SetComputeRoot32BitConstant(3, 0u, 4);
        // The row kernel avoids idle token lanes and the tiled shader's large-matrix driver hang.
        if (tokenCount < Tile || inputWidth >= 256 || outputWidth >= 256)
        {
            list.SetPipelineState(rowPipeline);
            for (int tokenOffset = 0; tokenOffset < tokenCount;)
            {
                int batch = Math.Min(65535, tokenCount - tokenOffset);
                list.SetComputeRoot32BitConstant(3, checked((uint)tokenOffset), 4);
                list.Dispatch(checked((uint)(((long)outputWidth + 63) / 64)), checked((uint)batch), 1);
                tokenOffset += batch;
            }
        }
        else
        {
            list.SetPipelineState(pipeline);
            list.Dispatch(checked((uint)(((long)outputWidth + Tile - 1) / Tile)),
                checked((uint)(((long)tokenCount + Tile - 1) / Tile)), 1);
        }
        list.ResourceBarrierTransition(outputBuffer!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        list.CopyBufferRegion(readback!, 0, outputBuffer!, 0, Bytes(outputCount));
        list.ResourceBarrierTransition(outputBuffer!, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        Submit();

        var result = new float[outputCount];
        readback!.GetData(result);
        for (int i = 0; i < result.Length; ++i)
        {
            if (!float.IsFinite(result[i]))
                throw new InvalidOperationException($"GPU projection for '{tensor.Name}' returned non-finite output at {i}: {result[i]}.");
        }
        ProjectCount++;
        TotalElapsed += Stopwatch.GetElapsedTime(start);
        return result;
    }

    private ID3D12Resource CacheWeight(IModelTensor tensor, int weightCount, bool half)
    {
        int packedCount = half ? weightCount / 2 + weightCount % 2 : weightCount;
        var packedChunk = new uint[Math.Min(UploadElements, packedCount)];
        var buffer = DefaultBuffer(Bytes(packedCount), ResourceFlags.None, ResourceStates.CopyDest);
        try
        {
            for (int offset = 0; offset < packedCount;)
            {
                int count = Math.Min(packedChunk.Length, packedCount - offset);
                if (half)
                {
                    ReadOnlySpan<Half> source = tensor.HalfValues;
                    for (int i = 0; i < count; i++)
                    {
                        int position = (offset + i) * 2;
                        uint low = BitConverter.HalfToUInt16Bits(source[position]);
                        uint high = position + 1 < weightCount
                            ? BitConverter.HalfToUInt16Bits(source[position + 1]) : 0u;
                        packedChunk[i] = low | (high << 16);
                    }
                }
                else
                {
                    ReadOnlySpan<float> source = tensor.FloatValues;
                    for (int i = 0; i < count; i++)
                        packedChunk[i] = BitConverter.SingleToUInt32Bits(source[offset + i]);
                }
                weightUpload.SetData(packedChunk.AsSpan(0, count));
                Begin();
                list.CopyBufferRegion(buffer, Bytes(offset), weightUpload, 0, Bytes(count));
                if (count == packedCount - offset)
                    list.ResourceBarrierTransition(buffer, ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource);
                Submit();
                offset += count;
            }
            weights.Add(tensor, buffer);
            return buffer;
        }
        catch
        {
            buffer.Dispose();
            throw;
        }
    }

    private void EnsureScratch(int inputCount, int outputCount)
    {
        if (inputCount > inputCapacity)
        {
            inputUpload?.Dispose();
            inputBuffer?.Dispose();
            inputUpload = device.CreateCommittedResource(HeapProperties.UploadHeapProperties,
                HeapFlags.None, ResourceDescription.Buffer(Bytes(inputCount)), ResourceStates.GenericRead);
            inputBuffer = DefaultBuffer(Bytes(inputCount), ResourceFlags.None, ResourceStates.NonPixelShaderResource);
            inputCapacity = inputCount;
        }
        if (outputCount > outputCapacity)
        {
            outputBuffer?.Dispose();
            readback?.Dispose();
            outputBuffer = DefaultBuffer(Bytes(outputCount), ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            readback = device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties,
                HeapFlags.None, ResourceDescription.Buffer(Bytes(outputCount)), ResourceStates.CopyDest);
            outputCapacity = outputCount;
        }
    }

    private ID3D12Resource DefaultBuffer(ulong bytes, ResourceFlags flags, ResourceStates state) =>
        device.CreateCommittedResource(HeapProperties.DefaultHeapProperties, HeapFlags.None,
            ResourceDescription.Buffer(bytes, flags), state);

    private static ulong Bytes(int elements) => checked((ulong)elements * sizeof(uint));

    private void Begin()
    {
        allocator.Reset();
        list.Reset(allocator);
    }

    private void Submit()
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

    public void Dispose()
    {
        if (disposed)
            return;
        disposed = true;
        foreach (ID3D12Resource buffer in weights.Values)
            buffer.Dispose();
        weights.Clear();
        readback?.Dispose();
        outputBuffer?.Dispose();
        inputBuffer?.Dispose();
        inputUpload?.Dispose();
        weightUpload.Dispose();
        rowPipeline.Dispose();
        pipeline.Dispose();
        root.Dispose();
        list.Dispose();
        allocator.Dispose();
        completion.Dispose();
        fence.Dispose();
        queue.Dispose();
        device.Dispose();
    }
}
