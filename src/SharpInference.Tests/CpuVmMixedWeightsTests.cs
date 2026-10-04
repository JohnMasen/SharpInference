using System.Runtime.InteropServices;
using SharpInference.Backends.CpuVm;
using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class CpuVmMixedWeightsTests
{
    [Theory]
    [InlineData("core.mat-vec", "matrix")]
    [InlineData("core.mat-vec", "weight")]
    [InlineData("core.tensor.batched-mat-vec", "matrix")]
    [InlineData("core.gather-row", "table")]
    public void ExecutesExplicitFp16WeightsWithFp32ActivationsAndAccumulation(string operation, string weightPort)
    {
        var (program, slots, expected) = Example(operation, weightPort);
        var originalWeights = slots[0].ToArray();
        var artifact = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
        Assert.Equal(VmElementType.Float16, artifact.Program.Definitions[0].Parameters[0].Tensor.ElementType);
        Assert.Contains(operation == "core.gather-row" ? "CpuInstructionNumerics.GatherRow" : "CpuInstructionNumerics.MatVec", artifact.Source);
        Assert.DoesNotContain("new float[", artifact.Source);
        using var executor = artifact.CreateExecutor();
        executor.Execute("run", slots);
        Assert.Equal(expected, MemoryMarshal.Cast<byte, float>(slots[^1]).ToArray());
        Assert.Equal(originalWeights, slots[0]);
    }

    [Fact]
    public void ReloadsMixedWeightBinaryWithoutCastingEntireMatrix()
    {
        var (program, slots, expected) = Example("core.mat-vec", "weight");
        var artifact = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
        var directory = Path.Combine(Path.GetTempPath(), "cpuvm-mixed-" + Guid.NewGuid().ToString("N"));
        try
        {
            artifact.Export(directory);
            var loaded = CpuVmCompiledArtifact.Load(directory);
            using var executor = loaded.CreateExecutor();
            executor.Execute("run", slots);
            Assert.Equal(expected, MemoryMarshal.Cast<byte, float>(slots[^1]).ToArray());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Theory]
    [InlineData("core.mat-vec")]
    [InlineData("core.tensor.batched-mat-vec")]
    public void RejectsHalfActivationsWithFloatOutputForMixedWeightVariant(string operation)
    {
        var (program, _, _) = Example(operation, "matrix");
        var original = program.Definitions[0];
        var parameters = original.Parameters.Select((p, i) => i == 1
            ? p with { Tensor = new VmTensor(VmElementType.Float16, p.Tensor.Dimensions) } : p).ToArray();
        var definition = new VmDefinition(original.Id, original.Kind, parameters, original.Nodes);
        var invalid = new VmProgram(program.Name, program.Abi, program.Target,
            program.Slots.Select((s, i) => s with { Tensor = parameters[i].Tensor }), [definition], program.Entries, program.State);
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() =>
            new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).GenerateSource(invalid));
    }

    [Theory]
    [InlineData(3, 2051)]
    [InlineData(64, 1024)]
    public void HandlesBoundedConversionBlocksAndParallelRowsWithFp32Accumulation(int rows, int columns)
    {
        var matrix = Enumerable.Repeat((Half)1, checked(rows * columns)).ToArray();
        var vector = Enumerable.Repeat(.25f, columns).ToArray();
        var output = new float[rows];
        SharpInference.Instructions.Cpu.CpuInstructionNumerics.MatVec(matrix, vector, output, rows, columns);
        Assert.All(output, value => Assert.Equal(columns * .25f, value));
    }

    private static (VmProgram Program, byte[][] Slots, float[] Expected) Example(string operation, string weightPort)
    {
        var batched = operation == "core.tensor.batched-mat-vec";
        var gather = operation == "core.gather-row";
        var weightShape = batched ? new[] { 2, 2, 8 } : new[] { 2, 8 };
        var outputShape = batched ? new[] { 2, 2 } : gather ? new[] { 8 } : new[] { 2 };
        var inputPort = batched ? "vector" : gather ? "index" : "input";
        var inputType = gather ? VmElementType.Int32 : VmElementType.Float32;
        var inputShape = batched ? new[] { 2, 8 } : gather ? new[] { 1 } : new[] { 8 };
        var parameters = new VmParameter[]
        {
            new(weightPort, VmAccess.ReadOnly, new VmTensor(VmElementType.Float16, weightShape)),
            new(inputPort, VmAccess.ReadOnly, new VmTensor(inputType, inputShape)),
            new("output", VmAccess.ReadWrite, new VmTensor(VmElementType.Float32, outputShape)),
        };
        var definition = new VmDefinition("mixed", VmDefinitionKind.Function, parameters,
            [new("mixed", new VmOperator(operation, 1, parameters.Select(p => new VmArgument(p.Name, p.Name))))]);
        var program = new VmProgram("mixed", "cpu", VmTarget.Cpu,
            parameters.Select(p => new VmSlot(p.Name, p.Access == VmAccess.ReadOnly ? VmSlotScope.Global : VmSlotScope.Local,
                p.Access, p.Tensor)), [definition],
            [new("run", "mixed", parameters.Select(p => new VmArgument(p.Name, p.Name)))], new VmState("none", 1, []));
        // Half accumulation loses the unit term next to 2048; FP32 accumulation returns exactly 2.
        Half[] matrix = [(Half)2048, (Half)1, (Half)(-2048), (Half)1, (Half)0, (Half)0, (Half)0, (Half)0,
            (Half)2, (Half).5f, (Half)4, (Half)(-1), (Half)0, (Half)0, (Half)0, (Half)0];
        if (batched) matrix = matrix.Concat(matrix).ToArray();
        var input = gather
            ? MemoryMarshal.AsBytes(new int[] { 1 }.AsSpan()).ToArray()
            : MemoryMarshal.AsBytes((batched
                ? Enumerable.Repeat(1f, 8).Concat(Enumerable.Repeat(2f, 8)).ToArray()
                : Enumerable.Repeat(1f, 8).ToArray()).AsSpan()).ToArray();
        float[] expected = batched ? [2, 5.5f, 4, 11] : gather ? [2, .5f, 4, -1, 0, 0, 0, 0] : [2, 5.5f];
        return (program, [MemoryMarshal.AsBytes(matrix.AsSpan()).ToArray(), input, new byte[(int)parameters[^1].Tensor.ByteLength]], expected);
    }
}
