using System.Runtime.CompilerServices;
using SharpInference.Graphs;
using Vortice.Direct3D12;
using Vortice.Dxc;

namespace SharpInference.Backends.Vortice;

internal sealed class VorticePrimitiveOperatorBackend : IPrimitiveOperatorBackend, IDisposable
{
    private const uint ThreadCount = 64;
    private static readonly string[] OperationNames =
    [
        "Copy", "Add", "Subtract", "Multiply", "Divide", "Maximum",
        "Exp", "Tanh", "Sigmoid", "ReciprocalSquareRoot", "Square", "Relu",
        "ReduceSum", "ReduceMean", "MatVec", "GatherRow",
    ];

    private readonly ID3D12Device device;
    private readonly bool native16BitShaderOpsSupported;
    private readonly ID3D12CommandQueue queue;
    private readonly ID3D12RootSignature rootSignature;
    private readonly ID3D12Fence fence;
    private readonly ID3D12CommandAllocator commandAllocator;
    private readonly ID3D12GraphicsCommandList commandList;
    private readonly EventWaitHandle completionEvent;
    private readonly Dictionary<(string Operation, bool Fp16), ID3D12PipelineState> pipelines = [];
    private readonly object gate = new();
    private ulong fenceValue;
    private bool disposed;

    public VorticePrimitiveOperatorBackend(ID3D12Device device, bool native16BitShaderOpsSupported)
    {
        this.device = device ?? throw new ArgumentNullException(nameof(device));
        this.native16BitShaderOpsSupported = native16BitShaderOpsSupported;
        PrimitiveOperators = PrimitiveGraphOperations.CreateStandardDescriptions(native16BitShaderOpsSupported);
        OperatorImplementations = PrimitiveGraphOperations.CreateStandardImplementations(
            "vortice",
            native16BitShaderOpsSupported,
            new KernelPrecisionProfile(GraphElementType.Float16, GraphElementType.Float16),
            new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32),
            new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32));
        queue = device.CreateCommandQueue(CommandListType.Compute);
        fence = device.CreateFence();
        completionEvent = new EventWaitHandle(false, EventResetMode.AutoReset);
        rootSignature = CreateRootSignature(device);
        commandAllocator = device.CreateCommandAllocator(CommandListType.Compute);
        commandList = device.CreateCommandList<ID3D12GraphicsCommandList>(
            0,
            CommandListType.Compute,
            commandAllocator);
        commandList.Close();
    }

    public IReadOnlyList<PrimitiveOperatorDescription> PrimitiveOperators { get; }
    public IReadOnlyList<OperatorImplementationDescription> OperatorImplementations { get; }

    public void Copy(ReadOnlySpan<float> input, Span<float> output) => Unary("Copy", input, output);
    public void Add(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Binary("Add", left, right, output);
    public void Subtract(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Binary("Subtract", left, right, output);
    public void Multiply(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Binary("Multiply", left, right, output);
    public void Divide(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Binary("Divide", left, right, output);
    public void Maximum(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Binary("Maximum", left, right, output);
    public void Exp(ReadOnlySpan<float> input, Span<float> output) => Unary("Exp", input, output);
    public void Tanh(ReadOnlySpan<float> input, Span<float> output) => Unary("Tanh", input, output);
    public void Sigmoid(ReadOnlySpan<float> input, Span<float> output) => Unary("Sigmoid", input, output);
    public void ReciprocalSquareRoot(ReadOnlySpan<float> input, Span<float> output) => Unary("ReciprocalSquareRoot", input, output);
    public void Square(ReadOnlySpan<float> input, Span<float> output) => Unary("Square", input, output);
    public void Relu(ReadOnlySpan<float> input, Span<float> output) => Unary("Relu", input, output);
    public float ReduceSum(ReadOnlySpan<float> input) => Reduce("ReduceSum", input, allowEmpty: true);
    public float ReduceMean(ReadOnlySpan<float> input) => Reduce("ReduceMean", input, allowEmpty: false);
    public void MatVec(ReadOnlySpan<float> matrix, ReadOnlySpan<float> input, Span<float> output, int rows, int columns) =>
        MatVecCore(matrix, input, output, rows, columns);
    public void GatherRow(ReadOnlySpan<float> table, int rowIndex, int rowLength, Span<float> output) =>
        GatherRowCore(table, rowIndex, rowLength, output);

    public void Copy(ReadOnlySpan<Half> input, Span<Half> output) { RequireFp16(); Unary("Copy", input, output); }
    public void Add(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output) { RequireFp16(); Binary("Add", left, right, output); }
    public void Subtract(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output) { RequireFp16(); Binary("Subtract", left, right, output); }
    public void Multiply(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output) { RequireFp16(); Binary("Multiply", left, right, output); }
    public void Divide(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output) { RequireFp16(); Binary("Divide", left, right, output); }
    public void Maximum(ReadOnlySpan<Half> left, ReadOnlySpan<Half> right, Span<Half> output) { RequireFp16(); Binary("Maximum", left, right, output); }
    public void Exp(ReadOnlySpan<Half> input, Span<Half> output) { RequireFp16(); Unary("Exp", input, output); }
    public void Tanh(ReadOnlySpan<Half> input, Span<Half> output) { RequireFp16(); Unary("Tanh", input, output); }
    public void Sigmoid(ReadOnlySpan<Half> input, Span<Half> output) { RequireFp16(); Unary("Sigmoid", input, output); }
    public void ReciprocalSquareRoot(ReadOnlySpan<Half> input, Span<Half> output) { RequireFp16(); Unary("ReciprocalSquareRoot", input, output); }
    public void Square(ReadOnlySpan<Half> input, Span<Half> output) { RequireFp16(); Unary("Square", input, output); }
    public void Relu(ReadOnlySpan<Half> input, Span<Half> output) { RequireFp16(); Unary("Relu", input, output); }
    public Half ReduceSum(ReadOnlySpan<Half> input) { RequireFp16(); return Reduce("ReduceSum", input, allowEmpty: true); }
    public Half ReduceMean(ReadOnlySpan<Half> input) { RequireFp16(); return Reduce("ReduceMean", input, allowEmpty: false); }
    public void MatVec(ReadOnlySpan<Half> matrix, ReadOnlySpan<Half> input, Span<Half> output, int rows, int columns)
    {
        RequireFp16();
        MatVecCore(matrix, input, output, rows, columns);
    }

    public void GatherRow(ReadOnlySpan<Half> table, int rowIndex, int rowLength, Span<Half> output)
    {
        RequireFp16();
        GatherRowCore(table, rowIndex, rowLength, output);
    }

    private void Unary<T>(string operation, ReadOnlySpan<T> input, Span<T> output)
        where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (input.Length != output.Length)
            throw new ArgumentException("The output length must match the input length.");
        if (input.IsEmpty)
            return;
        Execute(operation, input, default, output, checked((uint)input.Length), 0, 0, 0, DivideRoundUp((uint)input.Length));
    }

    private void Binary<T>(string operation, ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output)
        where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (left.Length != right.Length || left.Length != output.Length)
            throw new ArgumentException("Binary operator input and output lengths must match.");
        if (left.IsEmpty)
            return;
        Execute(operation, left, right, output, checked((uint)left.Length), 0, 0, 0, DivideRoundUp((uint)left.Length));
    }

    private T Reduce<T>(string operation, ReadOnlySpan<T> input, bool allowEmpty)
        where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (input.IsEmpty)
        {
            if (!allowEmpty)
                throw new ArgumentException("The input cannot be empty.", nameof(input));
            return default;
        }

        var result = new T[1];
        Execute(operation, input, default, result, checked((uint)input.Length), 0, 0, 0, 1);
        return result[0];
    }

    private void MatVecCore<T>(
        ReadOnlySpan<T> matrix,
        ReadOnlySpan<T> input,
        Span<T> output,
        int rows,
        int columns)
        where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns <= 0) throw new ArgumentOutOfRangeException(nameof(columns));
        if (matrix.Length != checked(rows * columns) || input.Length != columns || output.Length != rows)
            throw new ArgumentException("The MatVec dimensions do not match the supplied buffers.");
        Execute("MatVec", matrix, input, output, 0, checked((uint)rows), checked((uint)columns), 0, checked((uint)rows));
    }

    private void GatherRowCore<T>(
        ReadOnlySpan<T> table,
        int rowIndex,
        int rowLength,
        Span<T> output)
        where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (rowIndex < 0) throw new ArgumentOutOfRangeException(nameof(rowIndex));
        if (rowLength <= 0) throw new ArgumentOutOfRangeException(nameof(rowLength));
        if (output.Length != rowLength)
            throw new ArgumentException("The output length must match the input length.");
        var offset = checked(rowIndex * rowLength);
        if (offset > table.Length - rowLength)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));
        Execute("GatherRow", table, default, output, 0, 0, checked((uint)rowLength), checked((uint)rowIndex), DivideRoundUp((uint)rowLength));
    }

    private void Execute<T>(
        string operation,
        ReadOnlySpan<T> input0,
        ReadOnlySpan<T> input1,
        Span<T> output,
        uint elementCount,
        uint rowCount,
        uint columnCount,
        uint rowIndex,
        uint dispatchX)
        where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var fp16 = typeof(T) == typeof(Half);
            var pipeline = GetPipeline(operation, fp16);
            using var input0Resource = CreateUploadBuffer(input0);
            using var input1Resource = input1.IsEmpty ? null : CreateUploadBuffer(input1);
            var outputBytes = checked((ulong)output.Length * (uint)Unsafe.SizeOf<T>());
            using var outputResource = device.CreateCommittedResource(
                HeapProperties.DefaultHeapProperties,
                HeapFlags.None,
                ResourceDescription.Buffer(outputBytes, ResourceFlags.AllowUnorderedAccess),
                ResourceStates.UnorderedAccess);
            using var readbackResource = device.CreateCommittedResource(
                HeapProperties.ReadbackHeapProperties,
                HeapFlags.None,
                ResourceDescription.Buffer(outputBytes),
                ResourceStates.CopyDest);

            commandAllocator.Reset();
            commandList.Reset(commandAllocator, pipeline);
            commandList.SetComputeRootSignature(rootSignature);
            commandList.SetPipelineState(pipeline);
            commandList.SetComputeRootShaderResourceView(0, input0Resource.GPUVirtualAddress);
            if (input1Resource is not null)
                commandList.SetComputeRootShaderResourceView(1, input1Resource.GPUVirtualAddress);
            commandList.SetComputeRootUnorderedAccessView(2, outputResource.GPUVirtualAddress);
            commandList.SetComputeRoot32BitConstant(3, elementCount, 0);
            commandList.SetComputeRoot32BitConstant(3, rowCount, 1);
            commandList.SetComputeRoot32BitConstant(3, columnCount, 2);
            commandList.SetComputeRoot32BitConstant(3, rowIndex, 3);
            commandList.Dispatch(dispatchX, 1, 1);
            commandList.ResourceBarrierTransition(outputResource, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            commandList.CopyBufferRegion(readbackResource, 0, outputResource, 0, outputBytes);
            commandList.Close();
            queue.ExecuteCommandList(commandList);
            WaitForCompletion();
            readbackResource.GetData(output);
        }
    }

    private ID3D12Resource CreateUploadBuffer<T>(ReadOnlySpan<T> values)
        where T : unmanaged
    {
        var resource = device.CreateCommittedResource(
            HeapProperties.UploadHeapProperties,
            HeapFlags.None,
            ResourceDescription.Buffer(checked((ulong)values.Length * (uint)Unsafe.SizeOf<T>())),
            ResourceStates.GenericRead);
        resource.SetData(values);
        return resource;
    }

    private ID3D12PipelineState GetPipeline(string operation, bool fp16)
    {
        if (!OperationNames.Contains(operation, StringComparer.Ordinal))
            throw new ArgumentOutOfRangeException(nameof(operation));
        if (pipelines.TryGetValue((operation, fp16), out var pipeline))
            return pipeline;
        var precision = fp16 ? "Fp16" : "Fp32";
        var source = LoadEmbeddedShader($"PrimitiveOperators{precision}.hlsl");
        using var compilation = DxcCompiler.Compile(
            DxcShaderStage.Compute, source, operation,
            new DxcCompilerOptions
            {
                ShaderModel = fp16 ? DxcShaderModel.Model6_2 : DxcShaderModel.Model6_0,
                Enable16bitTypes = fp16,
            });
        pipeline = device.CreateComputePipelineState(new ComputePipelineStateDescription
        {
            RootSignature = rootSignature,
            ComputeShader = compilation.GetObjectBytecodeArray(),
        });
        pipelines.Add((operation, fp16), pipeline);
        return pipeline;
    }

    private static ID3D12RootSignature CreateRootSignature(ID3D12Device device)
    {
        var description = new RootSignatureDescription(
            RootSignatureFlags.None,
            [
                new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new RootParameter(new RootConstants(0, 0, 4), ShaderVisibility.All),
            ]);
        return device.CreateRootSignature(in description, RootSignatureVersion.Version10);
    }

    private static string LoadEmbeddedShader(string fileName)
    {
        var resourceName = $"{typeof(VorticePrimitiveOperatorBackend).Namespace}.Shaders.{fileName}";
        using var stream = typeof(VorticePrimitiveOperatorBackend).Assembly.GetManifestResourceStream(resourceName) ??
            throw new InvalidOperationException($"Embedded Vortice shader resource '{resourceName}' was not found.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private void RequireFp16()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (!native16BitShaderOpsSupported)
            throw new NotSupportedException("The selected adapter does not support Native16BitShaderOps required by Vortice FP16 primitive operators.");
    }

    private void WaitForCompletion()
    {
        var value = checked(++fenceValue);
        queue.Signal(fence, value).CheckError();
        if (fence.CompletedValue >= value)
            return;
        fence.SetEventOnCompletion(value, completionEvent).CheckError();
        completionEvent.WaitOne();
    }

    private static uint DivideRoundUp(uint value) => checked((value + ThreadCount - 1) / ThreadCount);

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed)
                return;
            foreach (var pipeline in pipelines.Values)
                pipeline.Dispose();
            pipelines.Clear();
            commandList.Dispose();
            commandAllocator.Dispose();
            completionEvent.Dispose();
            fence.Dispose();
            rootSignature.Dispose();
            queue.Dispose();
            disposed = true;
        }
    }
}
