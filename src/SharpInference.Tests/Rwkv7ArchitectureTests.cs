using SharpInference.Architectures.Rwkv7;
using SharpInference.Gguf;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class Rwkv7ArchitectureTests
{
    [Fact]
    public void LargeModel_HandlesDifferentProjectionWidths()
    {
        var modelPath = TestModelLoader.GetPath(TestModel.Rwkv7Large);
        using (var catalog = GgmlModelFile.Open(modelPath))
        {
            Assert.NotEqual(
                catalog.GetRequired("blocks.0.att.w1").Dimensions[1],
                catalog.GetRequired("blocks.0.att.g1").Dimensions[1]);
        }

        using var model = Processor.LoadGraph(modelPath, new PortableRwkv7GraphProvider(),
            SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create());
        using var session = model.CreateSession();
        var logits = session.ForwardToken(0).Span;
        Assert.Equal(RwkvModelMetadata.FromModelMetadata(model.Metadata).VocabularySize, logits.Length);
        Assert.True(logits.ToArray().All(float.IsFinite));
    }

    [Theory]
    [InlineData("FP32", 0.0011f)]
    [InlineData("FP16", 0.0061f)]
    public void TinyModel_MatchesRwkvCppExpectedLogits(string format, float maximumDifferenceSum)
    {
        var modelPath = TestModelLoader.GetPath(format switch
        {
            "FP32" => TestModel.Rwkv7Fp32,
            "FP16" => TestModel.Rwkv7Fp16,
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, null),
        });
        var expectedPath = TestModelLoader.GetPath(TestModel.Rwkv7ExpectedLogits);

        using var model = Processor.LoadGraph(modelPath, new PortableRwkv7GraphProvider(),
            SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create());
        using var session = model.CreateSession();

        var logits = session.Prefill(['"', 'i', 'n']).Span;
        var expected = ReadFloats(expectedPath);
        Assert.Equal(expected.Length, logits.Length);
        var differenceSum = 0f;
        for (var i = 0; i < logits.Length; i++)
        {
            Assert.True(float.IsFinite(logits[i]), $"Logit {i} is not finite.");
            differenceSum += MathF.Abs(logits[i] - expected[i]);
        }

        Assert.InRange(differenceSum / logits.Length, 0f, maximumDifferenceSum);
    }

    [Fact]
    public void TinyModel_StateSnapshotRestoresContinuation()
    {
        var modelPath = TestModelLoader.GetPath(TestModel.Rwkv7Fp32);

        using var model = Processor.LoadGraph(modelPath, new PortableRwkv7GraphProvider(),
            SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create());
        using var original = model.CreateSession();
        _ = original.Prefill(['"', 'i']);
        using var snapshot = new MemoryStream();
        original.SaveState(snapshot);
        snapshot.Position = 0;
        var saved = GgufStateFile.Read(snapshot);
        var graph = model.LogicalGraph!;
        Assert.Equal("RWKV7_State", saved.SchemaName);
        Assert.Equal(graph.GraphState.Schema.Name, saved.SchemaName);
        Assert.Equal(graph.GraphState.Slots.Count, saved.Tensors.Count);
        for (var index = 0; index < saved.Tensors.Count; index++)
        {
            var slot = graph.GraphState.Slots[index];
            var resource = Assert.Single(graph.Resources, item => item.Id == slot.Resource);
            Assert.Equal(slot.Name, saved.Tensors[index].Name);
            Assert.Equal(GgufTensorType.Float32, saved.Tensors[index].Type);
            Assert.Equal(resource.Tensor.Dimensions.Select(size => (ulong)size),
                saved.Tensors[index].Dimensions);
        }
        snapshot.Position = 0;
        using var restored = model.CreateSession();
        restored.LoadState(snapshot);
        var expected = original.ForwardToken('n').ToArray();
        var actual = restored.ForwardToken('n').ToArray();

        Assert.Equal(expected, actual);
        var state = StateSnapshotAssertions.Capture(original);
        StateSnapshotAssertions.Equal(state, StateSnapshotAssertions.Capture(restored));
        var metadata = RwkvModelMetadata.FromModelMetadata(model.Metadata);
        Assert.Equal(metadata.LayerCount * metadata.EmbeddingSize * (2 + metadata.HeadSize),
            StateSnapshotAssertions.Values(state).Length);
    }

    [Fact]
    public void TinyModel_IsDetectedAndSelectedByRuntimeFactory()
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv7Fp32);

        Assert.Equal("rwkv-7", RwkvModelArchitectureDetector.Detect(catalog));
        var runtime = new SharpInference.Applications.RwkvApplicationComposition(
            SharpInference.Applications.RwkvApplicationComposition.CreateModelModules()).CreateRuntime(null, catalog);
        Assert.Equal("rwkv-7", runtime.ArchitectureId);
        Assert.IsType<Rwkv7ModelModule>(runtime.Module);
        using var backend = Assert.IsType<VmGraphBackend>(runtime.CreateBackend());
    }

    private static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }
}
