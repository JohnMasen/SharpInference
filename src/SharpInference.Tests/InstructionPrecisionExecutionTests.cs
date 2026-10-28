using System.Runtime.InteropServices;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Tests;

public sealed class InstructionPrecisionExecutionTests
{
    [Fact]
    public void HalfStoredMatrixAndBiasUseFloatArithmeticAndExecute()
    {
        IInstructionCollectionProvider[] providers = [new CpuFloat16InstructionCollection()];
        var program = Program(providers, VmTarget.Cpu);
        var artifact = new CpuVmCompiler(providers).Compile(program);
        using var executor = artifact.CreateExecutor();
        var buffers = program.Slots.Select(slot => new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();
        Set("left", [1, 2, 3, 4]);
        Set("right", [5, 6, 7, 8]);
        Set("bias", [10, 20]);
        executor.Execute("forward", buffers);
        var output = program.Slots.ToList().FindIndex(slot => slot.Id == "output");
        Assert.Equal(new Half[] { (Half)29, (Half)42, (Half)53, (Half)70 },
            MemoryMarshal.Cast<byte, Half>(buffers[output]).ToArray());

        void Set(string name, float[] values)
        {
            var index = program.Slots.ToList().FindIndex(slot => slot.Id == name);
            values.Select(value => (Half)value).ToArray().AsSpan()
                .CopyTo(MemoryMarshal.Cast<byte, Half>(buffers[index].AsSpan()));
        }
    }

    [Fact]
    public void HalfStoredGpuMatrixAndBiasCompileWithHalfStores()
    {
        IInstructionCollectionProvider[] providers = [new GpuFloat16InstructionCollection()];
        var program = Program(providers, VmTarget.Direct3D12);
        var compiler = new D3D12VmCompiler(providers);
        var sources = compiler.GenerateSources(program);
        Assert.All(sources.Values, source => Assert.Contains("store16(", source));
        Assert.NotNull(compiler.Compile(program));
    }

    private static VmProgram Program(IInstructionCollectionProvider[] providers, VmTarget target)
    {
        var graph = new LogicalGraph { Id = "half-matrix" };
        var matrix = new TensorDescriptor(ElementType.Float16, [2, 2]);
        var left = graph.Resource("left", matrix, GraphIO.In);
        var right = graph.Resource("right", matrix, GraphIO.In);
        var bias = graph.Resource("bias", new(ElementType.Float16, [2]), GraphIO.In);
        var middle = graph.Resource("middle", matrix);
        var output = graph.Resource("output", matrix, GraphIO.Out);
        var product = graph.Operation("product", PrimitiveOperations.MatrixMultiply);
        product.Bind("left", left, ResourceAccess.Read).Bind("right", right, ResourceAccess.Read)
            .Bind("output", middle, ResourceAccess.Write);
        product.Attributes.Add("transpose_left", "false");
        product.Attributes.Add("transpose_right", "false");
        product.Precision = new(ElementType.Float32, ElementType.Float32);
        var add = graph.Operation("bias-add", PrimitiveOperations.BiasAdd);
        add.Bind("input", middle, ResourceAccess.Read).Bind("bias", bias, ResourceAccess.Read)
            .Bind("output", output, ResourceAccess.Write);
        add.Precision = new(ElementType.Float32, ElementType.Float32);
        var package = new DefaultExecutionGraphConverter(providers).Convert(new(graph),
            new("test", target == VmTarget.Cpu ? "cpu-vm" : "d3d12-vm"));
        return package.Compile(target, providers);
    }
}
