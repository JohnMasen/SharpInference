using System.Text;
using SharpInference.Gguf;

namespace SharpInference.Gguf.Tests;

public sealed class GgufModelFileTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Open_MapsMetadataArraysAndTensorData(bool legacyArrayCount)
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.gguf");
        try
        {
            File.WriteAllBytes(path, CreateModel(legacyArrayCount));
            using var model = GgufModelFile.Open(path);

            Assert.Equal("test", model.GetMetadata<string>("general.architecture"));
            Assert.Equal(new[] { 0.5f, 0.25f, -1f }, model.GetMetadata<float[]>("test.values"));
            var tensor = model.GetTensor("weight");
            Assert.Equal(GgufModelTensorType.Float16, tensor.Type);
            Assert.Equal(new ulong[] { 2 }, tensor.Dimensions);
            Assert.Equal(1f, tensor.GetHalfValue(0));
            Assert.Equal(2f, tensor.GetHalfValue(1));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Open_RejectsTensorOutsideFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.gguf");
        try
        {
            var bytes = CreateModel(false);
            File.WriteAllBytes(path, bytes[..^1]);
            var error = Assert.Throws<InvalidDataException>(() => GgufModelFile.Open(path));
            Assert.Contains("exceeds", error.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static byte[] CreateModel(bool legacyArrayCount)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, new UTF8Encoding(false, true), leaveOpen: true);
        writer.Write(0x46554747u);
        writer.Write(3u);
        writer.Write(1ul);
        writer.Write(2ul);

        WriteString(writer, "general.architecture");
        writer.Write((uint)GgufMetadataType.String);
        WriteString(writer, "test");

        WriteString(writer, "test.values");
        writer.Write((uint)GgufMetadataType.Array);
        writer.Write((uint)GgufMetadataType.Float32);
        if (legacyArrayCount)
            writer.Write(3u);
        else
            writer.Write(3ul);
        writer.Write(0.5f);
        writer.Write(0.25f);
        writer.Write(-1f);

        WriteString(writer, "weight");
        writer.Write(1u);
        writer.Write(2ul);
        writer.Write((uint)GgufModelTensorType.Float16);
        writer.Write(0ul);
        while (stream.Position % 32 != 0)
            writer.Write((byte)0);
        writer.Write(BitConverter.HalfToUInt16Bits((Half)1f));
        writer.Write(BitConverter.HalfToUInt16Bits((Half)2f));
        return stream.ToArray();
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }
}
