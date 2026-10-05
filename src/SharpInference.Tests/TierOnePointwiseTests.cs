using System.Runtime.InteropServices;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;
using SharpInference.Runtime;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class TierOnePointwiseTests
{
    public static IEnumerable<object[]> Cases() =>
        from definition in TierOnePointwiseCatalog.Definitions
        from count in new[] { 1, 7, 17, 64, 257, 4096 }
        select new object[] { definition.Name, count };

    [Theory]
    [MemberData(nameof(Cases))]
    public void CpuMatchesCanonicalStagesForFiniteSpecialAndTailInputs(string operation, int count)
    {
        var providers = CpuProviders();
        var reference = VmGraphOptimizer.Optimize(TierOneReferencePrograms.CreateGraph(operation, [count]), VmTarget.Cpu);
        var fused = TierOneReferencePrograms.CreateFused(operation, VmTarget.Cpu, [count]);
        var compiler = new CpuVmCompiler(providers);
        using var baseline = compiler.Compile(reference).LoadExecutable();
        using var optimized = compiler.Compile(fused).LoadExecutable();
        foreach (var special in new[] { false, true })
        {
            var inputs = Inputs(operation, count, special);
            var referenceBuffers = Buffers(reference, inputs);
            var fusedBuffers = Buffers(fused, inputs);
            baseline.Execute("forward", referenceBuffers);
            optimized.Execute("forward", fusedBuffers);
            Equal(Output(reference, referenceBuffers), Output(fused, fusedBuffers));
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void GpuMatchesCanonicalStagesAndOnlyStoresFinalTensor(string operation, int count)
    {
        var providers = GpuProviders();
        var reference = VmGraphOptimizer.Optimize(TierOneReferencePrograms.CreateGraph(operation, [count]), VmTarget.Direct3D12);
        var fused = TierOneReferencePrograms.CreateFused(operation, VmTarget.Direct3D12, [count]);
        var compiler = new D3D12VmCompiler(providers);
        var optimizedArtifact = compiler.Compile(fused);
        Assert.Single(optimizedArtifact.Kernels);
        var source = optimizedArtifact.Kernels[0].Source;
        Assert.Equal(1, source.Split(".Store(", StringSplitOptions.None).Length - 1);
        Assert.Contains("precise float t0", source);
        Assert.DoesNotContain("mad(", source);
        using var baseline = compiler.Compile(reference).CreateExecutor();
        using var optimized = optimizedArtifact.CreateExecutor();
        foreach (var special in new[] { false, true })
        {
            foreach (var (port, bytes) in Inputs(operation, count, special))
            {
                baseline.Upload(port, bytes);
                optimized.Upload(port, bytes);
            }
            baseline.Execute("forward");
            optimized.Execute("forward");
            Equal(MemoryMarshal.Cast<byte, float>(baseline.Readback("output")).ToArray(),
                MemoryMarshal.Cast<byte, float>(optimized.Readback("output")).ToArray());
        }
    }

    [Fact]
    public void MultiplyAddPreservesTheIntermediateFp32Rounding()
    {
        float[] a = [BitConverter.Int32BitsToSingle(0x3f800001), BitConverter.Int32BitsToSingle(0x3f800001)];
        float[] b = [BitConverter.Int32BitsToSingle(0x3f7fffff), BitConverter.Int32BitsToSingle(0x3f7fffff)];
        float[] c = [-1f, -1f];
        var expected = new float[2];
        var temporary = new float[2];
        SharpInference.Backends.Cpu.CpuPrimitiveOperatorBackend.Instance.Multiply(a, b, temporary);
        SharpInference.Backends.Cpu.CpuPrimitiveOperatorBackend.Instance.Add(temporary, c, expected);
        Assert.NotEqual(expected[0], MathF.FusedMultiplyAdd(a[0], b[0], c[0]));
        foreach (var count in new[] { 2, 64 })
        {
            var left = Enumerable.Repeat(a[0], count).ToArray();
            var right = Enumerable.Repeat(b[0], count).ToArray();
            var addend = Enumerable.Repeat(c[0], count).ToArray();
            var output = new float[count];
            CpuTierOneNumerics.MultiplyAdd(left, right, addend, output);
            Assert.All(output, value => Assert.Equal(BitConverter.SingleToInt32Bits(expected[0]), BitConverter.SingleToInt32Bits(value)));
        }
    }

    [Fact]
    public void CapabilitiesAreTargetSpecificImmutableAndRejectIllegalOperands()
    {
        foreach (var provider in new IInstructionOptimizationProvider[]
                 { new CpuTierOneInstructionCollection(), new GpuTierOneInstructionCollection() })
        {
            var capabilities = provider.QueryOptimizationCapabilities();
            Assert.Equal(13, capabilities.Count);
            foreach (var capability in capabilities)
            {
                Assert.Equal(InstructionBenefitKind.Execution, capability.Benefits);
                Assert.Equal(capability.Target, capability.Configuration.Target);
                Assert.Equal(64, capability.ImplementationFingerprint.Length);
                var operands = capability.Definition.Inputs.Select(port => new InstructionOperandDescription(port,
                    new(GraphElementType.Float32, [17]), GraphResourceAccess.Read, port))
                    .Append(new("output", new(GraphElementType.Float32, [17]), GraphResourceAccess.Write, "output")).ToArray();
                var requirement = new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32);
                Assert.True(capability.TryAdapt(operands, requirement, out var diagnostic), diagnostic);
                var aliased = operands.Select(operand => operand.Name == "output" ? operand with { Source = operands[0].Source } : operand).ToArray();
                Assert.False(capability.TryAdapt(aliased, requirement, out diagnostic));
                Assert.Contains("alias", diagnostic);
                var half = operands.Select(operand => operand with { Tensor = new(GraphElementType.Float16, [17]) }).ToArray();
                Assert.False(capability.TryAdapt(half, requirement, out diagnostic));
                Assert.Contains("FP32", diagnostic);
                var shape = operands.Select(operand => operand.Name == "output" ? operand with { Tensor = new(GraphElementType.Float32, [18]) } : operand).ToArray();
                Assert.False(capability.TryAdapt(shape, requirement, out diagnostic));
                Assert.NotEmpty(diagnostic);
            }
        }
    }

    [Fact]
    public void DirectHelpersRejectInplaceAndMismatchedSpans()
    {
        var data = new float[17];
        Assert.Throws<ArgumentException>(() => CpuTierOneNumerics.MultiplyAdd(data, data, data, data));
        Assert.Throws<ArgumentException>(() => CpuTierOneNumerics.AddRsqrt(data, new float[16], new float[17]));
        Assert.Throws<ArgumentException>(() => CpuTierOneNumerics.ReluSquare([], []));
    }

    [Fact]
    public void GpuMultiplyAddPreservesFp32StageRoundingAfterArtifactReload()
    {
        var program = TierOneReferencePrograms.CreateFused("multiply-add", VmTarget.Direct3D12, [64]);
        var artifact = new D3D12VmCompiler(GpuProviders()).Compile(program);
        using var stream = new MemoryStream();
        artifact.Export(stream);
        stream.Position = 0;
        var restored = D3D12VmArtifact.Import(stream);
        Assert.Equal(VmProgramXml.Serialize(artifact.Program), VmProgramXml.Serialize(restored.Program));
        using var executor = restored.CreateExecutor();
        var a = BitConverter.Int32BitsToSingle(0x3f800001);
        var b = BitConverter.Int32BitsToSingle(0x3f7fffff);
        foreach (var (port, value) in new[] { ("a", a), ("b", b), ("c", -1f) })
            executor.Upload(port, MemoryMarshal.AsBytes(Enumerable.Repeat(value, 64).ToArray().AsSpan()).ToArray());
        executor.Execute("forward");
        Assert.All(MemoryMarshal.Cast<byte, float>(executor.Readback("output")).ToArray(),
            value => Assert.Equal(0, BitConverter.SingleToInt32Bits(value)));
    }

    [Fact]
    public void RealCpuTierOneArtifactIncludesItsNumericalDependencyAndReloads()
    {
        var program = TierOneReferencePrograms.CreateFused("multiply-add", VmTarget.Cpu, [17]);
        var artifact = new CpuVmCompiler(CpuProviders()).Compile(program);
        var directory = Path.Combine(Path.GetTempPath(), "t1-kernel-" + Guid.NewGuid().ToString("N"));
        try
        {
            artifact.Export(directory);
            using var executor = CpuVmCompiledArtifact.Load(directory).LoadExecutable();
            var inputs = Inputs("multiply-add", 17);
            var buffers = Buffers(program, inputs);
            executor.Execute("forward", buffers);
            var a = MemoryMarshal.Cast<byte, float>(inputs["a"]).ToArray();
            var b = MemoryMarshal.Cast<byte, float>(inputs["b"]).ToArray();
            var c = MemoryMarshal.Cast<byte, float>(inputs["c"]).ToArray();
            Equal(Enumerable.Range(0, 17).Select(index => a[index] * b[index] + c[index]).ToArray(), Output(program, buffers));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    internal static IInstructionCollectionProvider[] CpuProviders() =>
        [.. DefaultInstructionCollections.Create()];
    internal static IInstructionCollectionProvider[] GpuProviders() =>
        [.. DefaultInstructionCollections.Create()];

    internal static Dictionary<string, byte[]> Inputs(string operation, int count, bool special = false) =>
        TierOnePointwiseCatalog.Get(operation).Inputs.Select((port, index) => (port, index)).ToDictionary(pair => pair.port,
            pair =>
            {
                float[] specials = [0f, -0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, float.Epsilon, -float.Epsilon, float.MaxValue, -float.MaxValue];
                var values = Enumerable.Range(0, count).Select(index => special
                    ? specials[(index + pair.index * 2) % specials.Length]
                    : MathF.Sin(index * 0.13f + pair.index) * 0.75f + 0.5f).ToArray();
                return MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
            }, StringComparer.Ordinal);

    private static byte[][] Buffers(VmProgram program, IReadOnlyDictionary<string, byte[]> inputs) =>
        program.Slots.Select(slot => inputs.TryGetValue(slot.Id, out var bytes) ? bytes.ToArray() :
            new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();

    private static float[] Output(VmProgram program, byte[][] buffers) =>
        MemoryMarshal.Cast<byte, float>(buffers[program.Slots.ToList().FindIndex(slot => slot.Id == "output")]).ToArray();

    internal static void Equal(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
        {
            if (float.IsNaN(expected[index])) Assert.True(float.IsNaN(actual[index]), $"NaN at {index}: {actual[index]}");
            else if (float.IsInfinity(expected[index])) Assert.Equal(expected[index], actual[index]);
            else if (expected[index] == 0) Assert.Equal(BitConverter.SingleToInt32Bits(expected[index]), BitConverter.SingleToInt32Bits(actual[index]));
            else
            {
                Assert.True(float.IsFinite(actual[index]), $"Finite at {index}: {actual[index]}");
                var error = MathF.Abs(expected[index] - actual[index]);
                Assert.True(error <= 2e-6f * MathF.Abs(expected[index]) + 2e-7f,
                    $"Element {index}: expected {expected[index]:R}, actual {actual[index]:R}, error {error:R}.");
            }
        }
    }
}
