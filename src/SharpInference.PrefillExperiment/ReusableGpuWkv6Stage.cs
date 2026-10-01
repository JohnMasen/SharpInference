using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.Dxc;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.PrefillExperiment;

/// <summary>Reusable GPU WKV-6 attention stage for token-major CPU layer inputs.</summary>
internal sealed class ReusableGpuWkv6Stage : IDisposable
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
        };

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

    private readonly ID3D12Device _device = null!;
    private readonly ID3D12CommandQueue _queue = null!;
    private readonly ID3D12Fence _fence = null!;
    private readonly EventWaitHandle _completion = null!;
    private readonly ID3D12CommandAllocator _allocator = null!;
    private readonly ID3D12GraphicsCommandList _list = null!;
    private readonly ID3D12RootSignature _root = null!;
    private readonly ID3D12PipelineState _summaryPipeline = null!;
    private readonly ID3D12PipelineState _foldPipeline = null!;
    private readonly ID3D12PipelineState _emitPipeline = null!;
    private ulong _fenceValue;
    private bool _disposed;

    public ReusableGpuWkv6Stage()
    {
        try
        {
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
                throw new PlatformNotSupportedException("No hardware D3D12 compute adapter is available for WKV-6.");
            using (adapter)
                _device = D3D12CreateDevice<ID3D12Device>(adapter, FeatureLevel.Level_11_0);

            _queue = _device.CreateCommandQueue(CommandListType.Compute);
            _fence = _device.CreateFence();
            _completion = new EventWaitHandle(false, EventResetMode.AutoReset);
            _allocator = _device.CreateCommandAllocator(CommandListType.Compute);
            _list = _device.CreateCommandList<ID3D12GraphicsCommandList>(
                0, CommandListType.Compute, _allocator);
            _list.Close();
            _root = _device.CreateRootSignature(new RootSignatureDescription(
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
                    new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
                ]), RootSignatureVersion.Version10);
            _summaryPipeline = Compile("Summarize");
            _foldPipeline = Compile("Fold");
            _emitPipeline = Compile("Emit");
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public Wkv6SequenceResult Run(Wkv6Sequence sequence, int chunkSize)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(sequence);
        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSize), "Chunk size must be positive.");
        var inputs = sequence.Inputs;
        ArgumentNullException.ThrowIfNull(inputs);
        int headSize = inputs.HeadSize, tokenCount = inputs.TokenCount;
        if (inputs.HeadCount <= 0 || headSize <= 0 || tokenCount <= 0)
            throw new ArgumentException("Head count, head size and token count must be positive.", nameof(sequence));
        int channels = checked(inputs.HeadCount * headSize);
        int stateCount = checked(channels * headSize);
        int inputCount = checked(tokenCount * channels);
        int chunkCount = 1 + (tokenCount - 1) / chunkSize;
        int chunkStates = checked(chunkCount * stateCount);
        int summaryCount = checked(2 * chunkStates);
        if (inputs.Keys.Length != inputCount || inputs.Values.Length != inputCount ||
            inputs.Decays.Length != inputCount || sequence.Receptances.Length != inputCount ||
            sequence.TimeFirst.Length != channels || inputs.InitialState.Length != stateCount)
            throw new ArgumentException("Expected token-major keys, values, decays and receptances; timeFirst must have one value per channel and initial state one per matrix element.", nameof(sequence));
        if (((long)channels + Threads - 1) / Threads > 65535 ||
            ((long)stateCount + Threads - 1) / Threads > 65535 || chunkCount > 65535)
            throw new ArgumentOutOfRangeException(nameof(chunkSize), "WKV-6 dispatch exceeds the 65535-group D3D12 axis limit.");

        // Includes default-heap buffers, all six uploads and both readbacks.
        long gpuBytes = checked(sizeof(float) * (10L * inputCount + 2L * channels +
            4L * stateCount + 3L * chunkStates));
        if (gpuBytes > MaxGpuBytes)
            throw new ArgumentOutOfRangeException(nameof(chunkSize),
                $"WKV-6 buffers require {gpuBytes:N0} bytes, exceeding the {MaxGpuBytes:N0}-byte allocation guard.");
        CheckFinite(inputs.Keys, "keys");
        CheckFinite(inputs.Values, "values");
        CheckFinite(inputs.Decays, "decays");
        CheckFinite(sequence.Receptances, "receptances");
        CheckFinite(sequence.TimeFirst, "timeFirst");
        CheckFinite(inputs.InitialState, "initialState");

        using var keyUpload = Upload(inputs.Keys);
        using var valueUpload = Upload(inputs.Values);
        using var decayUpload = Upload(inputs.Decays);
        using var receptanceUpload = Upload(sequence.Receptances);
        using var timeFirstUpload = Upload(sequence.TimeFirst);
        using var initialUpload = Upload(inputs.InitialState);
        using var keys = Buffer(inputCount, ResourceFlags.None, ResourceStates.CopyDest);
        using var values = Buffer(inputCount, ResourceFlags.None, ResourceStates.CopyDest);
        using var decays = Buffer(inputCount, ResourceFlags.None, ResourceStates.CopyDest);
        using var receptances = Buffer(inputCount, ResourceFlags.None, ResourceStates.CopyDest);
        using var timeFirst = Buffer(channels, ResourceFlags.None, ResourceStates.CopyDest);
        using var initial = Buffer(stateCount, ResourceFlags.None, ResourceStates.CopyDest);
        using var state = Buffer(stateCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
        using var output = Buffer(inputCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
        using var summary = Buffer(summaryCount, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
        using var scratch = Buffer(chunkStates, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
        using var outputReadback = _device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties,
            HeapFlags.None, ResourceDescription.Buffer(Bytes(inputCount)), ResourceStates.CopyDest);
        using var stateReadback = _device.CreateCommittedResource(HeapProperties.ReadbackHeapProperties,
            HeapFlags.None, ResourceDescription.Buffer(Bytes(stateCount)), ResourceStates.CopyDest);

        _allocator.Reset();
        _list.Reset(_allocator);
        Copy(keys, keyUpload, inputCount);
        Copy(values, valueUpload, inputCount);
        Copy(decays, decayUpload, inputCount);
        Copy(receptances, receptanceUpload, inputCount);
        Copy(timeFirst, timeFirstUpload, channels);
        Copy(initial, initialUpload, stateCount);

        _list.SetComputeRootSignature(_root);
        _list.SetComputeRootShaderResourceView(0, keys.GPUVirtualAddress);
        _list.SetComputeRootShaderResourceView(1, values.GPUVirtualAddress);
        _list.SetComputeRootShaderResourceView(2, decays.GPUVirtualAddress);
        _list.SetComputeRootShaderResourceView(3, receptances.GPUVirtualAddress);
        _list.SetComputeRootShaderResourceView(4, timeFirst.GPUVirtualAddress);
        _list.SetComputeRootShaderResourceView(5, initial.GPUVirtualAddress);
        _list.SetComputeRootUnorderedAccessView(6, state.GPUVirtualAddress);
        _list.SetComputeRootUnorderedAccessView(7, output.GPUVirtualAddress);
        _list.SetComputeRootUnorderedAccessView(8, summary.GPUVirtualAddress);
        _list.SetComputeRootUnorderedAccessView(9, scratch.GPUVirtualAddress);
        _list.SetComputeRoot32BitConstant(10, (uint)headSize, 0);
        _list.SetComputeRoot32BitConstant(10, (uint)stateCount, 1);
        _list.SetComputeRoot32BitConstant(10, (uint)channels, 2);
        _list.SetComputeRoot32BitConstant(10, (uint)tokenCount, 3);
        _list.SetComputeRoot32BitConstant(10, (uint)chunkSize, 4);
        _list.SetComputeRoot32BitConstant(10, (uint)chunkCount, 5);

        _list.SetPipelineState(_summaryPipeline);
        _list.Dispatch((uint)((stateCount + (long)Threads - 1) / Threads), (uint)chunkCount, 1);
        _list.ResourceBarrierUnorderedAccessView(summary);
        _list.SetPipelineState(_foldPipeline);
        _list.Dispatch((uint)((stateCount + (long)Threads - 1) / Threads), 1, 1);
        _list.ResourceBarrierUnorderedAccessView(scratch);
        _list.ResourceBarrierUnorderedAccessView(state);
        _list.SetPipelineState(_emitPipeline);
        _list.Dispatch((uint)((channels + (long)Threads - 1) / Threads), (uint)chunkCount, 1);
        _list.ResourceBarrierTransition(output, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        _list.ResourceBarrierTransition(state, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        _list.CopyBufferRegion(outputReadback, 0, output, 0, Bytes(inputCount));
        _list.CopyBufferRegion(stateReadback, 0, state, 0, Bytes(stateCount));
        _list.Close();
        _queue.ExecuteCommandList(_list);
        ulong target = checked(++_fenceValue);
        _queue.Signal(_fence, target).CheckError();
        if (_fence.CompletedValue < target)
        {
            _fence.SetEventOnCompletion(target, _completion).CheckError();
            _completion.WaitOne();
        }
        if (_device.DeviceRemovedReason.Failure)
            throw new InvalidOperationException($"D3D12 device was removed: {_device.DeviceRemovedReason}.");

        var outputs = new float[inputCount];
        var finalState = new float[stateCount];
        outputReadback.GetData(outputs);
        stateReadback.GetData(finalState);
        CheckFinite(outputs, "WKV-6 output");
        CheckFinite(finalState, "WKV-6 final state");
        return new Wkv6SequenceResult(outputs, finalState);

        void Copy(ID3D12Resource destination, ID3D12Resource source, int count)
        {
            _list.CopyBufferRegion(destination, 0, source, 0, Bytes(count));
            _list.ResourceBarrierTransition(destination, ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource);
        }
    }

    private ID3D12PipelineState Compile(string entry)
    {
        using var shader = DxcCompiler.Compile(DxcShaderStage.Compute, Shader, entry,
            new DxcCompilerOptions { ShaderModel = DxcShaderModel.Model6_0 });
        return _device.CreateComputePipelineState(new ComputePipelineStateDescription
        {
            RootSignature = _root,
            ComputeShader = shader.GetObjectBytecodeArray(),
        });
    }

    private ID3D12Resource Upload(float[] data)
    {
        var resource = _device.CreateCommittedResource(HeapProperties.UploadHeapProperties,
            HeapFlags.None, ResourceDescription.Buffer(Bytes(data.Length)), ResourceStates.GenericRead);
        try
        {
            resource.SetData(data);
            return resource;
        }
        catch
        {
            resource.Dispose();
            throw;
        }
    }

    private ID3D12Resource Buffer(int count, ResourceFlags flags, ResourceStates state) =>
        _device.CreateCommittedResource(HeapProperties.DefaultHeapProperties,
            HeapFlags.None, ResourceDescription.Buffer(Bytes(count), flags), state);

    private static ulong Bytes(int count) => checked((ulong)count * sizeof(float));

    private static void CheckFinite(float[] values, string name)
    {
        for (int i = 0; i < values.Length; ++i)
            if (!float.IsFinite(values[i]))
                throw new InvalidOperationException($"{name}[{i}] is not finite.");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _emitPipeline?.Dispose();
        _foldPipeline?.Dispose();
        _summaryPipeline?.Dispose();
        _root?.Dispose();
        _list?.Dispose();
        _allocator?.Dispose();
        _completion?.Dispose();
        _fence?.Dispose();
        _queue?.Dispose();
        _device?.Dispose();
    }
}
