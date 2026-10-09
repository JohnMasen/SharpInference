using System.Security.Cryptography;
using System.Text.Json;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class ModelNeutralityBaselineTests
{
    [Theory]
    [Trait("Category", "ExternalModel")]
    [InlineData("rwkv-6", "FP32")]
    [InlineData("rwkv-6", "FP16")]
    [InlineData("rwkv-7", "FP32")]
    [InlineData("rwkv-7", "FP16")]
    public void TinyModel_PreservesLogitsStateAndContinuation(string architecture, string format)
    {
        var path = architecture == "rwkv-7"
            ? TestModelLoader.GetPath(format == "FP32" ? TestModel.Rwkv7Fp32 : TestModel.Rwkv7Fp16)
            : format == "FP32"
                ? TestModelLoader.GetPath(TestModel.Rwkv6)
                : Path.Combine(Path.GetDirectoryName(TestModelLoader.GetPath(TestModel.Rwkv6))!,
                    "tiny-rwkv-6v0-3m-FP16.bin");
        ILogicalGraphProvider provider = architecture == "rwkv-6"
            ? new PortableRwkv6GraphProvider() : new PortableRwkv7GraphProvider();
        using var model = Processor.LoadGraph(path, provider, SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create());
        using var session = model.CreateSession();
        int[] tokens = ['"', 'i', 'n', ' ', 't'];
        var frames = new List<Frame>();
        foreach (var token in tokens)
        {
            var logits = session.ForwardToken(token).ToArray();
            var state = StateSnapshotAssertions.Capture(session);
            Assert.All(logits, value => Assert.True(float.IsFinite(value)));
            var values = StateSnapshotAssertions.Values(state);
            Assert.All(values, value => Assert.True(float.IsFinite(value)));
            frames.Add(new Frame(token, logits, state.SchemaName,
                state.Tensors.Select(tensor => tensor.Name).ToArray(), values));
        }

        using var snapshot = new MemoryStream();
        session.SaveState(snapshot);
        snapshot.Position = 0;
        using var restored = model.CreateSession();
        restored.LoadState(snapshot);
        Assert.Equal(session.ForwardToken('e').ToArray(), restored.ForwardToken('e').ToArray());
        StateSnapshotAssertions.Equal(StateSnapshotAssertions.Capture(session),
            StateSnapshotAssertions.Capture(restored));

        var directory = Environment.GetEnvironmentVariable("SHARPINFERENCE_BASELINE_DIRECTORY");
        if (directory is null) return;
        Assert.True(Path.IsPathFullyQualified(directory));
        using var source = File.OpenRead(path);
        var baseline = new Baseline(Convert.ToHexString(SHA256.HashData(source)), frames.ToArray());
        var file = Path.Combine(directory, $"{architecture}-{format}.json");
        if (Environment.GetEnvironmentVariable("SHARPINFERENCE_CAPTURE_BASELINE") == "1")
        {
            Directory.CreateDirectory(directory);
            using var destination = new FileStream(file, FileMode.CreateNew, FileAccess.Write);
            JsonSerializer.Serialize(destination, baseline);
            return;
        }

        var expected = JsonSerializer.Deserialize<Baseline>(File.ReadAllText(file))!;
        Assert.Equal(expected.ModelSha256, baseline.ModelSha256);
        Assert.Equal(expected.Frames.Length, baseline.Frames.Length);
        for (var index = 0; index < expected.Frames.Length; index++)
        {
            var actual = baseline.Frames[index];
            var saved = expected.Frames[index];
            Assert.Equal(saved.Token, actual.Token);
            Assert.Equal(saved.StateSchema, actual.StateSchema);
            Assert.Equal(saved.StateNames, actual.StateNames);
            Near(saved.Logits, actual.Logits);
            Near(saved.State, actual.State);
        }
    }

    private static void Near(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
            Assert.True(float.IsFinite(actual[index]) &&
                MathF.Abs(expected[index] - actual[index]) <= 1e-5f + 1e-5f * MathF.Abs(expected[index]),
                $"Value {index}: expected {expected[index]}, actual {actual[index]}.");
    }

    public sealed record Frame(int Token, float[] Logits, string StateSchema, string[] StateNames, float[] State);
    public sealed record Baseline(string ModelSha256, Frame[] Frames);
}
