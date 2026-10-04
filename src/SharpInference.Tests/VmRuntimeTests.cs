using System.Runtime.InteropServices;
using SharpInference.Backends.Cpu;
using SharpInference.Backends.CpuVm;
using SharpInference.Gguf;
using SharpInference.Graphs;
using SharpInference.Runtime;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class VmRuntimeTests
{
    [Fact]
    public void DirectArchitectureCallsPreserveStateWhenTemporarySessionCloses()
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv7Fp16);
        var logical = RwkvRuntimeFactory.CreateGraphProvider("rwkv-7").Build(catalog);
        using var backend = VmBackendFactory.CreateCpu(new() { PrefillInstances = 1, InferenceInstances = 1 });
        var architecture = new PortableGraphArchitecture(backend, backend.Prepare(logical));
        var model = architecture.Bind(catalog);
        var state = Assert.IsType<PortableGraphState>(architecture.CreateState(model));
        var actual = new float[model.Metadata.VocabularySize];
        using var processor = Processor.Load(TestModelLoader.GetPath(TestModel.Rwkv7Fp16));
        using var reference = processor.CreateSession();
        foreach (var token in new[] { 1, 2, 3 })
        {
            architecture.ForwardToken(model, token, state, actual);
            Close(reference.ForwardToken(token).Span, actual);
        }
        var snapshot = new float[state.ElementCount];
        state.CopyTo(snapshot);
        Assert.Contains(snapshot, value => value != 0);
        var fork = state.Clone();
        architecture.ForwardToken(model, 4, fork, actual);
        Close(reference.ForwardToken(4).Span, actual);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task IndependentPrefillProgramRoundTripsThroughRuntimeArtifacts(bool useArtifacts)
    {
        var directory = Path.Combine(Path.GetTempPath(), $"sharp-vm-runtime-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            var path = TestModelLoader.GetPath(TestModel.Rwkv7Fp16);
            using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv7Fp16);
            var provider = RwkvRuntimeFactory.CreateGraphProvider("rwkv-7");
            var logical = provider.Build(catalog);
            var inference = VmGraphOptimizer.Optimize(logical, VmTarget.Cpu, new(PrefillCapacity: 2));
            var prefill = VmGraphOptimizer.Optimize(logical, VmTarget.Cpu, new(PrefillCapacity: 4));
            var inferencePath = Path.Combine(directory, "inference.xml");
            var prefillPath = Path.Combine(directory, "prefill.xml");
            File.WriteAllText(inferencePath, VmProgramXml.Serialize(inference));
            File.WriteAllText(prefillPath, VmProgramXml.Serialize(prefill));
            var inferenceArtifacts = Path.Combine(directory, "inference");
            var prefillArtifacts = Path.Combine(directory, "prefill");
            if (useArtifacts)
            {
                new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(inference).Export(inferenceArtifacts);
                new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(prefill).Export(prefillArtifacts);
            }
            using var processor = Processor.LoadGraph(path, provider, VmBackendFactory.CreateCpu(new()
            {
                PrefillInstances = 1,
                InferenceInstances = 1,
                ProgramPath = useArtifacts ? null : inferencePath,
                PrefillProgramPath = useArtifacts ? null : prefillPath,
                ArtifactDirectory = useArtifacts ? inferenceArtifacts : null,
                PrefillArtifactDirectory = useArtifacts ? prefillArtifacts : null,
            }));
            Assert.DoesNotContain(processor.InferenceProgram!.Entries, entry => entry.Name == "prefill.4");
            Assert.Contains(processor.PrefillProgram!.Entries, entry => entry.Name == "prefill.4");
            var exportedInference = Path.Combine(directory, "exported-inference");
            var exportedPrefill = Path.Combine(directory, "exported-prefill");
            processor.ExportCompiledArtifact(exportedInference);
            processor.ExportCompiledArtifact(exportedPrefill, ProcessorExecutionGraphKind.Prefill);
            using var reloaded = Processor.LoadGraph(path, provider, VmBackendFactory.CreateCpu(new()
            {
                PrefillInstances = 1,
                InferenceInstances = 1,
                ArtifactDirectory = exportedInference,
                PrefillArtifactDirectory = exportedPrefill,
            }));
            using var originalSession = processor.CreateSession();
            using var reloadedSession = reloaded.CreateSession();
            var originalLogits = await originalSession.PrefillAsync(new int[] { 1, 2, 3, 2 });
            var reloadedLogits = await reloadedSession.PrefillAsync(new int[] { 1, 2, 3, 2 });
            Close(originalLogits.Span, reloadedLogits.Span);
            await using var originalGeneration = await originalSession.BeginGenerationAsync();
            await using var reloadedGeneration = await reloadedSession.BeginGenerationAsync();
            Close(originalGeneration.Session.ForwardToken(4).Span, reloadedGeneration.Session.ForwardToken(4).Span);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void ModelGlobalBindingsRejectMismatchedShapeAndTypeBeforeWriting()
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv7Fp16);
        var program = VmGraphOptimizer.Optimize(RwkvRuntimeFactory.CreateGraphProvider("rwkv-7").Build(catalog), VmTarget.Cpu);
        var slot = program.Slots.First(slot => slot.Scope == VmSlotScope.Global && slot.Tensor.ElementType == VmElementType.Float16);
        VmSlot[] invalidSlots =
        [
            slot with { Tensor = new VmTensor(VmElementType.Float32, slot.Tensor.Dimensions) },
            slot with { Tensor = new VmTensor(VmElementType.Float16, [(int)slot.Tensor.ElementCount + 1]) },
        ];
        foreach (var invalid in invalidSlots)
        {
            using var storage = new VmMemoryStorage(checked((int)invalid.Tensor.ByteLength));
            Assert.Throws<InvalidDataException>(() => VmModelBindings.InitializeGlobal(catalog, invalid, storage));
            Assert.All(storage.Buffer, value => Assert.Equal((byte)0, value));
        }
    }

    [Fact]
    [Trait("Category", "LargeModel")]
    public void LargeModelExportsAndReloadsSourceWithoutFullFloatWeightWorkspace()
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv7Large);
        var logical = RwkvRuntimeFactory.CreateGraphProvider("rwkv-7").Build(catalog);
        var program = VmGraphOptimizer.Optimize(logical, VmTarget.Cpu);
        var parsed = VmProgramXml.Deserialize(VmProgramXml.Serialize(program));
        var globals = parsed.Slots.Where(slot => slot.Scope == VmSlotScope.Global).Sum(slot => (long)slot.Tensor.ByteLength);
        var local = parsed.Slots.Where(slot => slot.Scope == VmSlotScope.Local).Sum(slot => (long)slot.Tensor.ByteLength);
        Assert.True(globals > 10_000_000_000);
        Assert.True(local < globals / 100, $"Local workspace {local} must be below 1% of weight storage {globals}.");
        var artifact = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).GenerateSource(parsed);
        Assert.False(artifact.HasBinary);
        Assert.Contains("CpuProgram", artifact.Source);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public Task CompiledCpuMatchesReferencePrefillInferenceAndState(bool rwkv7, bool half) =>
        VerifyBackend(rwkv7, half, gpu: false);

    [Theory]
    [Trait("Category", "Gpu")]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public Task CompiledGpuMatchesReferencePrefillInferenceAndState(bool rwkv7, bool half) =>
        VerifyBackend(rwkv7, half, gpu: true);

    private static async Task VerifyBackend(bool rwkv7, bool half, bool gpu)
    {
        var fixture = !rwkv7 ? TestModel.Rwkv6 : half ? TestModel.Rwkv7Fp16 : TestModel.Rwkv7Fp32;
        var path = TestModelLoader.GetPath(fixture);
        using var catalog = TestModelLoader.OpenCatalog(fixture);
        var logical = RwkvRuntimeFactory.CreateGraphProvider(rwkv7 ? "rwkv-7" : "rwkv-6").Build(catalog);
        var reference = new CpuPrimitiveGraphExecutor(logical, catalog);
        IReadOnlyList<GraphStateValue> expectedState = logical.GraphState.Entries.Select(entry =>
        {
            var tensor = logical.Resources.Single(resource => resource.Id == entry.Resource).Tensor;
            return new GraphStateValue(entry.Name, tensor.Dimensions,
                new float[tensor.Dimensions.Aggregate(1, (count, size) => count * size)]);
        }).ToArray();
        using var processor = gpu
            ? Processor.LoadGraph(path, RwkvRuntimeFactory.CreateGraphProvider(rwkv7 ? "rwkv-7" : "rwkv-6"), VmBackendFactory.CreateD3D12())
            : Processor.Load(path);
        var plan = Assert.IsType<VmCompiledPlan>(processor.PreparedPlan);
        if (half)
        {
            Assert.Contains(plan.Program.Definitions, definition => definition.Parameters.Any(parameter =>
                parameter.Name is "weight" or "matrix" && parameter.Tensor.ElementType == VmElementType.Float16));
            var expanded = VmGraphOptimizer.Optimize(logical, VmTarget.Cpu, new(NativeHalfWeights: false));
            var expandedBytes = expanded.Slots.Where(slot => slot.Scope == VmSlotScope.Local).Sum(slot => (long)slot.Tensor.ByteLength);
            var optimizedBytes = plan.Program.Slots.Where(slot => slot.Scope == VmSlotScope.Local).Sum(slot => (long)slot.Tensor.ByteLength);
            Assert.True(optimizedBytes < expandedBytes, $"Expanded workspace {expandedBytes}, native-half workspace {optimizedBytes}.");
        }
        using var session = processor.CreateSession();
        using var isolated = processor.CreateSession();
        int[] tokens = [1, 2, 3, 2, 4];
        float[] expected = [];
        foreach (var token in tokens)
        {
            var result = reference.Execute(new Dictionary<ResourceId, Array> { [new("token")] = new[] { token } }, expectedState);
            expected = Assert.IsType<float[]>(result.Outputs[new("logits")]);
            expectedState = result.State;
        }
        var actual = (await session.PrefillAsync(tokens.AsMemory(0, 3))).ToArray();
        await using (var generation = await session.BeginGenerationAsync())
        {
            Assert.Throws<InvalidOperationException>(session.Reset);
            actual = generation.Session.ForwardToken(tokens[3]).ToArray();
            actual = generation.Session.ForwardToken(tokens[4]).ToArray();
        }
        Close(expected, actual);
        using var snapshot = new MemoryStream();
        session.SaveState(snapshot);
        snapshot.Position = 0;
        var state = GgufStateFile.Read(snapshot);
        foreach (var entry in expectedState)
            Close(entry.Values, MemoryMarshal.Cast<byte, float>(state.Tensors.Single(tensor => tensor.Name == entry.Name).Data.Span));
        snapshot.Position = 0;
        isolated.LoadState(snapshot);
        using var fork = session.Fork();
        Close(session.ForwardToken(3).Span, isolated.ForwardToken(3).Span);
        Close(session.ForwardToken(4).Span, fork.Prefill(new int[] { 3, 4 }).Span);
        session.Reset();
        isolated.Reset();
        Close(session.ForwardToken(1).Span, isolated.ForwardToken(1).Span);
        using var xml = new MemoryStream();
        processor.ExportExecutionGraph(xml);
        xml.Position = 0;
        using var reader = new StreamReader(xml);
        var program = VmProgramXml.Deserialize(reader.ReadToEnd());
        Assert.Equal(rwkv7 ? "RWKV7_State" : "RWKV6_State", program.State.Schema);
    }

    private static void Close(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
            Assert.True(float.IsFinite(actual[index]) && Math.Abs(expected[index] - actual[index]) <=
                0.0003f + Math.Abs(expected[index]) * 0.0003f,
                $"Index {index}: expected {expected[index]}, actual {actual[index]}.");
    }
}
