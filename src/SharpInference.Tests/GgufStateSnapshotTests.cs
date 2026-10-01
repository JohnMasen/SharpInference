using System.Text;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Backends.Cpu;
using SharpInference.Gguf;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class GgufStateSnapshotTests
{
    [Fact]
    public void ProcessorSession_ExportsAndRestoresInitialStateBeforeAnyInference()
    {
        using var processor = LoadPortableModel();
        using var source = processor.CreateSession();
        var initialValues = StateSnapshotAssertions.Capture(source);

        using var stream = new MemoryStream();
        stream.WriteByte(0x7F);
        source.SaveState(stream);
        Assert.True(stream.CanWrite);
        Assert.Equal(0x7F, stream.GetBuffer()[0]);
        Assert.Equal(stream.Length, stream.Position);
        stream.Position = 1;
        var saved = GgufStateFile.Read(stream);
        Assert.Equal(processor.InferenceExecutionGraph!.GraphState.Schema.Name, saved.SchemaName);
        Assert.Equal(processor.InferenceExecutionGraph.GraphState.Slots.Count, saved.Tensors.Count);

        using var restored = processor.CreateSession();
        stream.Position = 1;
        restored.LoadState(stream);
        Assert.True(stream.CanRead);
        Assert.Equal(stream.Length, stream.Position);
        StateSnapshotAssertions.Equal(initialValues, StateSnapshotAssertions.Capture(restored));
        Assert.Equal(source.ForwardToken(1).ToArray(), restored.ForwardToken(1).ToArray());
        StateSnapshotAssertions.Equal(StateSnapshotAssertions.Capture(source), StateSnapshotAssertions.Capture(restored));
    }

    [Fact]
    public void ProcessorSession_ExportsGgufV3StateAndRestoresContinuation()
    {
        using var processor = LoadPortableModel();
        using var source = processor.CreateSession();
        source.Prefill([1, 2, 3]);
        using var stream = new MemoryStream();
        source.SaveState(stream);
        var snapshot = stream.ToArray();

        Assert.Equal("GGUF", Encoding.ASCII.GetString(snapshot, 0, 4));
        Assert.Equal(3u, BitConverter.ToUInt32(snapshot, 4));
        stream.Position = 0;
        var decoded = GgufStateFile.Read(stream);
        Assert.NotEmpty(decoded.SchemaName);
        Assert.NotEmpty(decoded.Tensors);
        Assert.All(decoded.Tensors, tensor => Assert.Equal(GgufTensorType.Float32, tensor.Type));

        using var restored = processor.CreateSession();
        stream.Position = 0;
        restored.LoadState(stream);
        StateSnapshotAssertions.Equal(StateSnapshotAssertions.Capture(source), StateSnapshotAssertions.Capture(restored));
        Assert.Equal(source.ForwardToken(4).ToArray(), restored.ForwardToken(4).ToArray());
    }

    [Fact]
    public void ProcessorSession_RejectsIncompatibleSchemaWithoutChangingState()
    {
        using var processor = LoadPortableModel();
        using var source = processor.CreateSession();
        source.ForwardToken(1);
        using var stream = new MemoryStream();
        source.SaveState(stream);
        stream.Position = 0;
        var decoded = GgufStateFile.Read(stream);
        using var invalid = new MemoryStream();
        GgufStateFile.Write(invalid, new GgufState("unrelated.state.schema", decoded.Tensors));
        using var destination = processor.CreateSession();
        var before = StateSnapshotAssertions.Capture(destination);

        invalid.Position = 0;
        Assert.Throws<InvalidDataException>(() => destination.LoadState(invalid));
        StateSnapshotAssertions.Equal(before, StateSnapshotAssertions.Capture(destination));
    }

    [Fact]
    public void ProcessorSession_RejectsMalformedGgufWithoutChangingState()
    {
        using var processor = LoadPortableModel();
        using var source = processor.CreateSession();
        source.ForwardToken(1);
        using var stream = new MemoryStream();
        source.SaveState(stream);
        var snapshot = stream.ToArray();
        snapshot[0] = (byte)'X';
        using var destination = processor.CreateSession();
        var before = StateSnapshotAssertions.Capture(destination);

        using var invalid = new MemoryStream(snapshot, writable: false);
        Assert.Throws<InvalidDataException>(() => destination.LoadState(invalid));
        StateSnapshotAssertions.Equal(before, StateSnapshotAssertions.Capture(destination));
    }

    [Theory]
    [InlineData("name")]
    [InlineData("shape")]
    [InlineData("type")]
    [InlineData("count")]
    public void ProcessorSession_RejectsIncompatibleTensorDescriptorsWithoutChangingState(string change)
    {
        using var processor = LoadPortableModel();
        using var source = processor.CreateSession();
        source.ForwardToken(1);
        using var stream = new MemoryStream();
        source.SaveState(stream);
        stream.Position = 0;
        var decoded = GgufStateFile.Read(stream);
        var tensor = decoded.Tensors[0];
        var changed = change switch
        {
            "name" => new GgufStateTensor("different", tensor.Type, tensor.Dimensions, tensor.Data),
            "shape" => new GgufStateTensor(tensor.Name, tensor.Type,
                [1UL, .. tensor.Dimensions], tensor.Data),
            "type" => new GgufStateTensor(tensor.Name, GgufTensorType.Int32, tensor.Dimensions, tensor.Data),
            "count" => null,
            _ => throw new ArgumentOutOfRangeException(nameof(change)),
        };
        using var invalid = new MemoryStream();
        GgufStateFile.Write(invalid, new GgufState(decoded.SchemaName,
            change == "count" ? decoded.Tensors.Skip(1) :
            decoded.Tensors.Select((item, index) => index == 0 ? changed! : item)));
        using var destination = processor.CreateSession();
        var before = StateSnapshotAssertions.Capture(destination);

        invalid.Position = 0;
        Assert.Throws<InvalidDataException>(() => destination.LoadState(invalid));
        StateSnapshotAssertions.Equal(before, StateSnapshotAssertions.Capture(destination));
    }

    [Fact]
    public void ProcessorSession_AcceptsEditedTensorPayloadWithoutChecksum()
    {
        using var processor = LoadPortableModel();
        using var source = processor.CreateSession();
        source.ForwardToken(1);
        using var stream = new MemoryStream();
        source.SaveState(stream);
        stream.Position = 0;
        var decoded = GgufStateFile.Read(stream);
        var first = decoded.Tensors[0];
        var editedBytes = first.Data.ToArray();
        BitConverter.GetBytes(42f).CopyTo(editedBytes, 0);
        var tensors = decoded.Tensors.Select((tensor, index) =>
            index == 0 ? new GgufStateTensor(tensor.Name, tensor.Type, tensor.Dimensions, editedBytes) : tensor);
        using var edited = new MemoryStream();
        GgufStateFile.Write(edited, new GgufState(decoded.SchemaName, tensors));

        using var destination = processor.CreateSession();
        edited.Position = 0;
        destination.LoadState(edited);

        Assert.Contains(42f, StateSnapshotAssertions.Values(StateSnapshotAssertions.Capture(destination)));
    }

    private static Processor LoadPortableModel() =>
        Processor.LoadGraph(TestModelLoader.GetPath(TestModel.Rwkv6),
            new PortableRwkv6GraphProvider(), CpuPrimitiveGraphBackend.Instance);
}
