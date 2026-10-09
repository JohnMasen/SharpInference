using System.Runtime.InteropServices;
using SharpInference.Gguf;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class TierOneRuntimeTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ExplicitFixtureCostsExerciseRealCpuGpuPrefillDecodeAndState(bool version7, bool gpu)
    {
        var model = version7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6;
        var target = gpu ? VmTarget.Direct3D12 : VmTarget.Cpu;
        var provider = RwkvRuntimeFactory.CreateGraphProvider(version7 ? "rwkv-7" : "rwkv-6");
        using var catalog = TestModelLoader.OpenCatalog(model);
        var logical = provider.Build(catalog);
        var expanded = new GraphOptimizer().Optimize(logical,
            new GraphOptimizationOptions(OptimizationBoundary.Off, DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded));
        var capabilities = new InstructionRegistry(gpu ? SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create() :
            SharpInference.Runtime.Cpu.CpuInstructionCollections.Create()).QueryOptimizationCapabilities();
        var measurements = TierOnePatternMatcher.Find(expanded, capabilities, target).Select(candidate =>
            new TierOneCostMeasurement(candidate.Capability.Name, candidate.Capability.ImplementationFingerprint,
                string.Join("x", expanded.Resources.Single(resource => resource.Id == candidate.Output).Tensor.Dimensions),
                candidate.InputAliases, 64, true, Enumerable.Repeat(20d, 7).ToArray(), Enumerable.Repeat(10d, 7).ToArray()))
            .DistinctBy(measurement => measurement.Key).ToArray();
        Assert.NotEmpty(measurements);
        var profile = new TierOneCostProfile((target == VmTarget.Cpu ? SharpInference.Runtime.Cpu.CpuVmEnvironment.Fingerprint() : SharpInference.Runtime.D3D12.D3D12VmEnvironment.Fingerprint()), DateTimeOffset.UtcNow, measurements);
        var configuration = new VmRuntimeConfig { InferenceInstances = 1, PrefillInstances = 1, TierOneCostProfile = profile };
        using var optimizedBackend = gpu ? SharpInference.Runtime.D3D12.D3D12VmBackendFactory.Create(configuration) : SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create(configuration);
        using var optimized = Processor.LoadGraph(TestModelLoader.GetPath(model), provider, optimizedBackend);
        using var baseline = Processor.LoadGraph(TestModelLoader.GetPath(model), provider, gpu
            ? SharpInference.Runtime.D3D12.D3D12VmBackendFactory.Create(new() { InferenceInstances = 1, PrefillInstances = 1 })
            : SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create(new() { InferenceInstances = 1, PrefillInstances = 1 }));
        Assert.True(optimizedBackend.OptimizationReport!.SelectedCandidates > 0);
        Assert.True(optimizedBackend.OptimizationReport.OptimizedNodeCount < optimizedBackend.OptimizationReport.OriginalNodeCount);
        using var actual = optimized.CreateSession();
        using var expected = baseline.CreateSession();
        Close(expected.Prefill(new int[] { 1, 2, 3 }).Span, actual.Prefill(new int[] { 1, 2, 3 }).Span);
        foreach (var token in new[] { 2, 4, 5 })
            Close(expected.ForwardToken(token).Span, actual.ForwardToken(token).Span);
        using var expectedFile = new MemoryStream();
        using var actualFile = new MemoryStream();
        expected.SaveState(expectedFile);
        actual.SaveState(actualFile);
        expectedFile.Position = 0;
        actualFile.Position = 0;
        var expectedState = GgufStateFile.Read(expectedFile);
        var actualState = GgufStateFile.Read(actualFile);
        foreach (var entry in expectedState.Tensors)
            Close(MemoryMarshal.Cast<byte, float>(entry.Data.Span),
                MemoryMarshal.Cast<byte, float>(actualState.Tensors.Single(candidate => candidate.Name == entry.Name).Data.Span));
        actual.Reset();
        expected.Reset();
        Close(expected.ForwardToken(1).Span, actual.ForwardToken(1).Span);
        var xml = VmProgramXml.Serialize(optimized.InferenceProgram!);
        Assert.Equal(xml, VmProgramXml.Serialize(VmProgramXml.Deserialize(xml)));
    }

    private static void Close(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
            Assert.True(float.IsFinite(actual[index]) && MathF.Abs(expected[index] - actual[index]) <=
                0.0003f + MathF.Abs(expected[index]) * 0.0003f,
                $"Element {index}: expected {expected[index]:R}, actual {actual[index]:R}.");
    }
}
