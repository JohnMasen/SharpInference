using System.Text;
using SharpInference.Gguf;

namespace SharpInference.Gguf.Tests;

public sealed class GgufStateFileTests
{
    private static readonly GgufState Sample = new("RWKV6_State",
    [
        new("layer.0.att", GgufTensorType.Float32, [2, 3], Enumerable.Range(0, 24).Select(i => (byte)i).ToArray()),
        new("layer.0.ffn", GgufTensorType.Float16, [3], new byte[] { 1, 2, 3, 4, 5, 6 }),
    ]);

    [Fact]
    public void RoundTrip_PreservesSchemaNamesTypesShapesAndRawBytes()
    {
        using var stream = new MemoryStream();
        GgufStateFile.Write(stream, Sample);
        stream.Position = 0;
        var restored = GgufStateFile.Read(stream);
        Assert.Equal(Sample.SchemaName, restored.SchemaName);
        Assert.Equal(Sample.Tensors.Count, restored.Tensors.Count);
        for (var i = 0; i < Sample.Tensors.Count; i++)
        {
            Assert.Equal(Sample.Tensors[i].Name, restored.Tensors[i].Name);
            Assert.Equal(Sample.Tensors[i].Type, restored.Tensors[i].Type);
            Assert.Equal(Sample.Tensors[i].Dimensions, restored.Tensors[i].Dimensions);
            Assert.Equal(Sample.Tensors[i].Data.ToArray(), restored.Tensors[i].Data.ToArray());
        }
        Assert.True(stream.CanRead);
    }

    [Fact]
    public void RoundTrip_FromNonzeroOffset_PreservesPrefixAndLeavesStreamOpen()
    {
        byte[] prefix = [0xE1, 0xE2, 0xE3, 0xE4, 0xE5];
        var encoded = Encode(Sample);
        using var stream = new MemoryStream();
        stream.Write(prefix);
        GgufStateFile.Write(stream, Sample);

        Assert.True(stream.CanWrite);
        Assert.Equal(prefix.Length + encoded.Length, stream.Position);
        Assert.Equal([.. prefix, .. encoded], stream.ToArray());

        stream.Position = prefix.Length;
        var restored = GgufStateFile.Read(stream);
        Assert.Equal(Sample.SchemaName, restored.SchemaName);
        for (var i = 0; i < Sample.Tensors.Count; i++)
            Assert.Equal(Sample.Tensors[i].Data.ToArray(), restored.Tensors[i].Data.ToArray());
        Assert.Equal(stream.Length, stream.Position);
        Assert.True(stream.CanRead);
    }

    [Fact]
    public void Write_FromNonzeroOffset_PreservesExistingTrailingBytes()
    {
        byte[] prefix = [0x81, 0x82, 0x83];
        byte[] suffix = [0x91, 0x92, 0x93, 0x94];
        var encoded = Encode(Sample);
        using var stream = new MemoryStream();
        stream.Write(prefix);
        stream.Write(new byte[encoded.Length]);
        stream.Write(suffix);
        stream.Position = prefix.Length;

        GgufStateFile.Write(stream, Sample);

        Assert.Equal(prefix.Length + encoded.Length, stream.Position);
        Assert.Equal([.. prefix, .. encoded, .. suffix], stream.ToArray());
        Assert.True(stream.CanWrite);
    }

    [Fact]
    public void Read_FromNonzeroOffset_RejectsExtraSuffix()
    {
        byte[] prefix = [0x82, 0x83, 0x84];
        var encoded = Encode(Sample);
        using var stream = new MemoryStream([.. prefix, .. encoded, 0x42]);
        stream.Position = prefix.Length;

        Assert.Throws<InvalidDataException>(() => GgufStateFile.Read(stream));
        Assert.True(stream.CanRead);
    }

    [Fact]
    public void ReadAndWrite_RejectNonseekableStreams()
    {
        using var readable = new NonSeekableStream(Encode(Sample));
        Assert.Throws<ArgumentException>(() => GgufStateFile.Read(readable));
        Assert.True(readable.CanRead);

        using var writable = new NonSeekableStream();
        Assert.Throws<ArgumentException>(() => GgufStateFile.Write(writable, Sample));
        Assert.True(writable.CanWrite);
    }

    [Fact]
    public void Structure_IsGgufV3WithRequiredArchitectureAndAlignedNativeTensors()
    {
        using var stream = new MemoryStream();
        GgufStateFile.Write(stream, Sample);
        stream.Position = 0;
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        Assert.Equal("GGUF", Encoding.ASCII.GetString(reader.ReadBytes(4)));
        Assert.Equal(3u, reader.ReadUInt32());
        Assert.Equal(2UL, reader.ReadUInt64());
        Assert.Equal(2UL, reader.ReadUInt64());
        Assert.Equal(("general.architecture", 8u, "sharpinference"), ReadMetadata(reader));
        Assert.Equal(("sharpinference.state_schema.name", 8u, "RWKV6_State"), ReadMetadata(reader));

        Assert.Equal("layer.0.att", ReadString(reader));
        Assert.Equal(2u, reader.ReadUInt32());
        Assert.Equal(2UL, reader.ReadUInt64());
        Assert.Equal(3UL, reader.ReadUInt64());
        Assert.Equal(0u, reader.ReadUInt32()); // GGML_TYPE_F32
        Assert.Equal(0UL, reader.ReadUInt64());
        Assert.Equal("layer.0.ffn", ReadString(reader));
        Assert.Equal(1u, reader.ReadUInt32());
        Assert.Equal(3UL, reader.ReadUInt64());
        Assert.Equal(1u, reader.ReadUInt32()); // GGML_TYPE_F16
        Assert.Equal(32UL, reader.ReadUInt64());
        var dataStart = (stream.Position + 31) / 32 * 32;
        Assert.All(reader.ReadBytes((int)(dataStart - stream.Position)), b => Assert.Equal(0, b));
        Assert.Equal(Sample.Tensors[0].Data.ToArray(), reader.ReadBytes(24));
        Assert.Equal(new byte[8], reader.ReadBytes(8));
        Assert.Equal(Sample.Tensors[1].Data.ToArray(), reader.ReadBytes(6));
        Assert.Equal(stream.Length, stream.Position);
    }

    [Fact]
    public void EmptyState_RoundTripsWithoutInventedTensors()
    {
        using var stream = new MemoryStream();
        GgufStateFile.Write(stream, new GgufState("empty", []));
        stream.Position = 0;
        var state = GgufStateFile.Read(stream);
        Assert.Equal("empty", state.SchemaName);
        Assert.Empty(state.Tensors);
    }

    [Fact]
    public void NonRwkvSchemaAndTensor_RoundTripWithoutModelAssumptions()
    {
        var original = new GgufState("AnotherBackend_State/v1", [
            new GgufStateTensor("custom.hidden", GgufTensorType.Float64, [2, 1],
                BitConverter.GetBytes(3.5).Concat(BitConverter.GetBytes(-2.0)).ToArray()),
        ]);
        using var stream = new MemoryStream();
        GgufStateFile.Write(stream, original);
        stream.Position = 0;
        var state = GgufStateFile.Read(stream);
        Assert.Equal(original.SchemaName, state.SchemaName);
        Assert.Equal("custom.hidden", Assert.Single(state.Tensors).Name);
        Assert.Equal(original.Tensors[0].Data.ToArray(), state.Tensors[0].Data.ToArray());
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(4, 2)]
    [InlineData(8, 255)]
    [InlineData(16, 255)]
    public void Read_RejectsMalformedHeader(int index, byte value)
    {
        var bytes = Encode(Sample);
        bytes[index] = value;
        AssertInvalid(bytes);
    }

    [Fact]
    public void Read_RejectsTruncatedAndExtraData()
    {
        var bytes = Encode(Sample);
        AssertInvalid(bytes[..^1]);
        AssertInvalid([.. bytes, 0]);
        AssertInvalid(bytes[..10]);
    }

    [Fact]
    public void Read_RejectsInvalidNativeTypeShapeAndOffsets()
    {
        var bytes = Encode(Sample);
        var location = FirstTensorFields(bytes);
        AssertCorrupt(bytes, location.Dimension, 0); // zero dimension
        AssertCorrupt(bytes, location.Type, 2); // quantized Q4_0 is not raw element data
        AssertCorrupt(bytes, location.Offset, 1); // unaligned offset
        AssertCorrupt(bytes, location.DimensionCount, 5); // exceeds GGML max dimensions
    }

    [Fact]
    public void Read_RejectsMalformedRequiredMetadata()
    {
        var bytes = Encode(Sample);
        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        stream.Position = 24;
        ReadString(reader);
        reader.ReadUInt32();
        var architectureLength = reader.ReadUInt64();
        var architectureValue = (int)stream.Position;
        Assert.Equal((ulong)Encoding.UTF8.GetByteCount("sharpinference"), architectureLength);
        AssertCorrupt(bytes, architectureValue, (byte)'x');
        stream.Position += (long)architectureLength;
        var secondKeyLength = reader.ReadUInt64();
        Assert.Equal((ulong)Encoding.UTF8.GetByteCount("sharpinference.state_schema.name"), secondKeyLength);
        AssertCorrupt(bytes, (int)stream.Position, 0xFF); // invalid UTF-8
        AssertCorrupt(bytes, (int)stream.Position + (int)secondKeyLength, 0); // invalid metadata value type
    }

    [Fact]
    public void Read_RejectsDuplicateNamesAndNonzeroPadding()
    {
        var bytes = Encode(new GgufState("S", [
            new("a", GgufTensorType.Int8, [1], new byte[] { 3 }),
            new("b", GgufTensorType.Int8, [1], new byte[] { 4 }),
        ]));
        using var stream = new MemoryStream(bytes, writable: false);
        using var reader = new BinaryReader(stream);
        stream.Position = 24;
        for (var i = 0; i < 2; i++) { ReadString(reader); reader.ReadUInt32(); ReadString(reader); }
        ReadString(reader);
        reader.ReadUInt32();
        reader.ReadUInt64();
        reader.ReadUInt32();
        reader.ReadUInt64();
        var secondName = (int)stream.Position + 8;
        AssertCorrupt(bytes, secondName, (byte)'a');
        var dataStart = (int)((stream.Position + 8 + 1 + 4 + 8 + 4 + 8 + 31) / 32 * 32);
        AssertCorrupt(bytes, dataStart + 1, 1);
    }

    [Fact]
    public void Write_RejectsInvalidTensorAndDoesNotAddTensors()
    {
        using var stream = new MemoryStream();
        Assert.Throws<ArgumentException>(() => GgufStateFile.Write(stream, new GgufState("S", [
            new("x", GgufTensorType.Float32, [2], new byte[7]),
        ])));
        Assert.Throws<ArgumentException>(() => GgufStateFile.Write(stream, new GgufState("S", [
            new("x", GgufTensorType.Float32, [1], new byte[4]),
            new("x", GgufTensorType.Float32, [1], new byte[4]),
        ])));
        Assert.Throws<ArgumentException>(() => GgufStateFile.Write(stream, new GgufState("S", [
            new("x", (GgufTensorType)2, [1], new byte[4]),
        ])));
    }

    private static byte[] Encode(GgufState state)
    {
        using var stream = new MemoryStream();
        GgufStateFile.Write(stream, state);
        return stream.ToArray();
    }

    private static void AssertInvalid(byte[] bytes)
    {
        using var stream = new MemoryStream(bytes);
        Assert.Throws<InvalidDataException>(() => GgufStateFile.Read(stream));
    }

    private static void AssertCorrupt(byte[] original, int index, byte value)
    {
        var bytes = (byte[])original.Clone();
        bytes[index] = value;
        AssertInvalid(bytes);
    }

    private static (string Key, uint Type, string Value) ReadMetadata(BinaryReader reader) =>
        (ReadString(reader), reader.ReadUInt32(), ReadString(reader));

    private static string ReadString(BinaryReader reader) =>
        Encoding.UTF8.GetString(reader.ReadBytes(checked((int)reader.ReadUInt64())));

    private static (int DimensionCount, int Dimension, int Type, int Offset) FirstTensorFields(byte[] bytes)
    {
        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        stream.Position = 24;
        ReadMetadata(reader);
        ReadMetadata(reader);
        ReadString(reader);
        var dimensionCount = (int)stream.Position;
        reader.ReadUInt32();
        var dimension = (int)stream.Position;
        reader.ReadUInt64();
        reader.ReadUInt64();
        var type = (int)stream.Position;
        reader.ReadUInt32();
        return (dimensionCount, dimension, type, (int)stream.Position);
    }

    private sealed class NonSeekableStream(byte[]? contents = null) : Stream
    {
        private readonly MemoryStream inner = new(contents ?? []);

        public override bool CanRead => inner.CanRead;
        public override bool CanSeek => false;
        public override bool CanWrite => inner.CanWrite;
        public override long Length => throw new NotSupportedException();
        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }
        public override void Flush() => inner.Flush();
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => inner.Write(buffer, offset, count);
        protected override void Dispose(bool disposing)
        {
            if (disposing) inner.Dispose();
            base.Dispose(disposing);
        }
    }
}
