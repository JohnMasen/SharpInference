using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;

namespace SharpInference.Tests;

public sealed class PortableTransformerInstructionTests
{
    [Fact]
    public void TwentyTwoGpuInstructionsHaveExactlyOneSemanticOwner()
    {
        IInstructionProvider[] providers =
        [
            new GpuTensorInstructionCollection(),
            new GpuNormalizationInstructionCollection(),
            new GpuPhi4ExecutionInstructionCollection(),
        ];
        var registry = new InstructionRegistry(providers);

        Assert.Equal(22, registry.EnumerateInstructions().Count);
        Assert.Equal(22, registry.Rows.Select(row => row.Instruction).Distinct().Count());
        Assert.Equal(4, registry.Rows.Count(row => row.Category == InstructionCategory.Tier0));
        Assert.Equal(7, registry.Rows.Count(row => row.Category == InstructionCategory.Tier1));
        Assert.Equal(11, registry.Rows.Count(row => row.Category == InstructionCategory.Specialized));
        Assert.All(registry.Rows.Where(row => row.Category != InstructionCategory.Specialized),
            row => Assert.Equal("core", row.Domain));
        Assert.All(registry.Rows.Where(row => row.Category == InstructionCategory.Specialized),
            row => Assert.Equal("phi4", row.Domain));
        foreach (var provider in providers)
        {
            var description = Assert.Single(provider.QueryInstructionCollection());
            Assert.NotNull(description.RequiredInstructions);
            Assert.Equal(description.RequiredInstructions!.Order(),
                provider.EnumerateInstructions().Select(instruction => instruction.Name).Order());
        }
        Assert.Equal([ExecutionTarget.Direct3D12], Phi4ExecutionInstructionCollectionIds.SupportedTargets);
    }

    [Fact]
    public void PortableSetsHaveIdenticalCpuGpuSignaturesAndCompleteManifests()
    {
        (IInstructionProvider Cpu, IInstructionProvider Gpu)[] pairs =
        [
            (new CpuTensorInstructionCollection(), new GpuTensorInstructionCollection()),
            (new CpuNormalizationInstructionCollection(), new GpuNormalizationInstructionCollection()),
        ];
        foreach (var (cpu, gpu) in pairs)
        {
            var cpuDescription = Assert.Single(cpu.QueryInstructionCollection());
            var gpuDescription = Assert.Single(gpu.QueryInstructionCollection());
            Assert.Equal(cpuDescription.Id, gpuDescription.Id);
            Assert.Equal(cpuDescription.RequiredInstructions, gpuDescription.RequiredInstructions);
            foreach (var instruction in cpu.EnumerateInstructions())
            {
                var other = Assert.Single(gpu.QueryInstruction(instruction.CollectionId, instruction.Name));
                var expected = Assert.Single(instruction.Signatures);
                var actual = Assert.Single(other.Signatures);
                Assert.Equal(expected.Ports, actual.Ports);
                Assert.Equal(expected.Attributes, actual.Attributes);
                Assert.Equal(expected.Precision, actual.Precision);
                Assert.Equal(instruction.IndexBounds, other.IndexBounds);
            }
        }
        var registry = new InstructionRegistry(pairs.SelectMany(pair =>
            new IInstructionCollectionProvider[] { pair.Cpu, pair.Gpu }));
        Assert.Equal(22, registry.Rows.Count);
    }

    [Fact]
    public void AllPortableInstructionsRecordSourceOnBothArchitectures()
    {
        IInstructionProvider[] providers =
        [
            new CpuTensorInstructionCollection(), new GpuTensorInstructionCollection(),
            new CpuNormalizationInstructionCollection(), new GpuNormalizationInstructionCollection(),
        ];
        foreach (var instruction in providers.SelectMany(provider => provider.EnumerateInstructions()))
        {
            var recorder = new SourceInstructionRecorder(instruction.Target);
            instruction.Invoke(recorder, Parameters(instruction),
                new PrecisionRequirement(ElementType.Float32, ElementType.Float32));
            Assert.False(string.IsNullOrWhiteSpace(Assert.Single(recorder.Recordings).Source));
            if (instruction.Target == ExecutionTarget.Cpu)
            {
                Assert.NotEmpty(recorder.References);
            }
        }
    }

    [Fact]
    public void BothArchitecturesRejectInvalidPortableShapes()
    {
        IInstructionProvider[] providers =
        [
            new CpuTensorInstructionCollection(), new GpuTensorInstructionCollection(),
            new CpuNormalizationInstructionCollection(), new GpuNormalizationInstructionCollection(),
        ];
        foreach (var instruction in providers.SelectMany(provider => provider.EnumerateInstructions()))
        {
            var parameters = Parameters(instruction);
            var outputIndex = Array.FindIndex(parameters, parameter => parameter.Name == "output");
            var output = (InstructionTensorParameter)parameters[outputIndex];
            parameters[outputIndex] = output with
            {
                Tensor = new TensorDescriptor(output.Tensor.ElementType, [3]),
            };
            Assert.Throws<InstructionAdaptationException>(() => instruction.Adapt(parameters,
                new PrecisionRequirement(ElementType.Float32, ElementType.Float32)));
        }
    }

    [Fact]
    public void ArgMaxUsesFirstMaximumAndHandlesNegativeInfinity()
    {
        int[] output = [-1];
        CpuTensorNumerics.ArgMax([-2f, 4f, 4f, -1f], output);
        Assert.Equal(1, output[0]);
        CpuTensorNumerics.ArgMax([float.NegativeInfinity, -float.MaxValue], output);
        Assert.Equal(1, output[0]);
        CpuTensorNumerics.ArgMax([float.NaN, float.NegativeInfinity], output);
        Assert.Equal(1, output[0]);
    }

    [Fact]
    public void GatherConvertsHalfAndCopyPreservesUntouchedRows()
    {
        float[] gathered = new float[4];
        CpuTensorNumerics.Gather([(Half)1, (Half)2, (Half)3, (Half)4],
            [1, 0], gathered, 2);
        Assert.Equal([3f, 4f, 1f, 2f], gathered);
        float[] destination = [-1, -1, -1, -1, -1, -1];
        CpuTensorNumerics.CopyRows(gathered, destination, 2, 2, 1);
        Assert.Equal([-1f, -1f, 3f, 4f, 1f, 2f], destination);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            CpuTensorNumerics.Gather([(Half)1], [-1], new float[1], 1));
    }

    [Fact]
    public void SharedMatVecUsesOnlyValidCountAndLeavesInactiveRowsUntouched()
    {
        Half[] weight = [(Half)1, (Half)2, (Half)3, (Half)4];
        float[] input = [2, 3, 5, 7];
        float[] output = [-1, -1, -1, -1];
        CpuTensorNumerics.SharedMatVec(weight, input, [int.MaxValue, 1], output, 2, 2, 2);
        Assert.Equal([8f, 18f, -1f, -1f], output);
        CpuTensorNumerics.SharedMatVec(weight, input, [int.MinValue, 9], output, 2, 2, 2);
        Assert.Equal([8f, 18f, 19f, 43f], output);
        CpuTensorNumerics.SharedMatVec(weight, input, [0, -1], output, 2, 2, 2);
        Assert.Equal([8f, 18f, 19f, 43f], output);
    }

    [Fact]
    public void NormalizationAndResidualBaselineMatchReference()
    {
        float[] output = new float[2];
        CpuNormalizationNumerics.RmsNorm([3, 4], [2, 1], output, 0.5f);
        var scale = 1f / MathF.Sqrt(13f);
        AssertClose([6 * scale, 4 * scale], output);
        float[] updated = new float[2];
        CpuNormalizationNumerics.ResidualRmsNorm([1, 2], [2, 2], [2, 1], updated, output, 0.5f);
        Assert.Equal([3f, 4f], updated);
        AssertClose([6 * scale, 4 * scale], output);
    }

    [Fact]
    public void SwiGluUsesContiguousHalvesAndScaledAddUsesItsAttribute()
    {
        float[] output = new float[2];
        CpuNormalizationNumerics.SwiGlu([0, 1, 5, 2], output);
        AssertClose([0f, 2f / (1f + MathF.Exp(-1f))], output);
        CpuNormalizationNumerics.ScaledAdd([1, 2], [3, 4], output, 0.5f);
        Assert.Equal([2.5f, 4f], output);
    }

    [Fact]
    public void BatchFusionsUseOnlyValidCountAndPreserveInactiveRows()
    {
        float[] input = [3, 4, 6, 8];
        float[] weight = [2, 1];
        float[] output = [-1, -1, -1, -1];
        var scale = 1f / MathF.Sqrt(13f);
        CpuNormalizationNumerics.BatchRmsNorm(input, weight, [int.MaxValue, 1], output, 2, 2, 0.5f);
        AssertClose([6 * scale, 4 * scale, -1, -1], output);
        float[] updated = [-2, -2, -2, -2];
        CpuNormalizationNumerics.BatchResidualRmsNorm([1, 2, 3, 4], [2, 2, 3, 4],
            weight, [int.MinValue, 1], updated, output, 2, 2, 0.5f);
        Assert.Equal([3f, 4f, -2f, -2f], updated);
        AssertClose([6 * scale, 4 * scale, -1, -1], output);
        CpuNormalizationNumerics.BatchSwiGlu([0, 1, 5, 2, 2, 3, 7, 11],
            [int.MinValue, 1], output, 2, 2);
        AssertClose([0f, 2f / (1 + MathF.Exp(-1)), -1, -1], output);
    }

    [Fact]
    public void InvalidBatchCountsAreClampedConsistentlyWithGpuGuards()
    {
        float[] output = [-1, -1];
        CpuNormalizationNumerics.BatchSwiGlu([0, 2, 1, 3], [0, -1], output, 2, 1);
        Assert.Equal([-1f, -1f], output);
        CpuNormalizationNumerics.BatchSwiGlu([0, 2, 1, 3], [0, 99], output, 2, 1);
        AssertClose([0f, 3f / (1 + MathF.Exp(-1))], output);
    }

    private static InstructionParameter[] Parameters(Instruction instruction)
    {
        var signature = Assert.Single(instruction.Signatures);
        var tensors = signature.Ports.Select(port =>
        {
            int[] dimensions = instruction.Name switch
            {
                "transformer.argmax" => port.Name == "output" ? [1] : [2],
                "transformer.embedding-gather-batch" => port.Name switch
                {
                    "table" or "output" => [2, 2],
                    _ => [2],
                },
                "transformer.copy-embedding-rows" => [2, 2],
                "transformer.batch-shared-matvec" => port.Name == "control" ? [2] : [2, 2],
                "transformer.batch-rms-norm" or "transformer.batch-residual-rms-norm" => port.Name switch
                {
                    "control" or "weight" => [2],
                    _ => [2, 2],
                },
                "transformer.batch-swiglu" => port.Name switch
                {
                    "control" => [2],
                    "gate_up" => [2, 4],
                    _ => [2, 2],
                },
                "transformer.swiglu" => port.Name == "gate_up" ? [4] : [2],
                _ => [2],
            };
            return (InstructionParameter)new InstructionTensorParameter(port.Name,
                new TensorDescriptor(port.ElementType, dimensions), port.Access,
                port.Name, port.Name + "Buffer", "0");
        });
        return tensors.Concat(signature.Attributes.Select(attribute =>
            (InstructionParameter)new InstructionAttributeParameter(attribute, attribute switch
            {
                "rows" => "1",
                "output_start" => "1",
                _ => "0.5",
            }))).ToArray();
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0, 1e-6f);
        }
    }
}
