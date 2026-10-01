using System.Buffers.Binary;

namespace SharpInference.Tests;

public sealed class GgmlModelFileTests
{
    [Fact]
    public void Open_RejectsInvalidMagic()
    {
        var path = CreateFile(writer =>
        {
            writer.Write(0u);
            writer.Write(101u);
            writer.Write(4u);
            writer.Write(4u);
            writer.Write(1u);
            writer.Write(0u);
        });

        try
        {
            Assert.Throws<InvalidDataException>(() => GgmlModelFile.Open(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Open_ReadsEmptyValidCatalog()
    {
        var path = CreateFile(writer =>
        {
            writer.Write(0x67676d66u);
            writer.Write(101u);
            writer.Write(4u);
            writer.Write(8u);
            writer.Write(2u);
            writer.Write(0u);
        });

        try
        {
            using var model = GgmlModelFile.Open(path);
            Assert.Equal(4, model.VocabularySize);
            Assert.Equal(8, model.EmbeddingSize);
            Assert.Equal(2, model.LayerCount);
            Assert.Empty(model.Names);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Open_RejectsTruncatedTensor()
    {
        var path = CreateFile(writer =>
        {
            writer.Write(0x67676d66u);
            writer.Write(101u);
            writer.Write(4u);
            writer.Write(4u);
            writer.Write(1u);
            writer.Write(0u);
            writer.Write(1u);
            writer.Write(1u);
            writer.Write(0u);
            writer.Write(4u);
            writer.Write((byte)'x');
        });

        try
        {
            Assert.Throws<InvalidDataException>(() => GgmlModelFile.Open(path));
        }

        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Open_ReadsFloat16Tensor()
    {
        var path = CreateFile(writer =>
        {
            writer.Write(0x67676d66u);
            writer.Write(101u);
            writer.Write(4u);
            writer.Write(4u);
            writer.Write(1u);
            writer.Write(1u);
            writer.Write(1u);
            writer.Write(1u);
            writer.Write(1u);
            writer.Write(1u);
            writer.Write((byte)'h');
            writer.Write((ushort)BitConverter.HalfToUInt16Bits((Half)1.5f));
        });

        try
        {
            using var model = GgmlModelFile.Open(path);
            var tensor = model.GetRequired("h");
            Assert.Equal(RwkvTensorDataType.Float16, tensor.DataType);
            Assert.Equal((Half)1.5f, tensor.HalfValues[0]);
            Assert.Equal(1.5f, tensor.FloatValues[0]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string CreateFile(Action<BinaryWriter> write)
    {
        var path = Path.Combine(Path.GetTempPath(), $"sharpinference-{Guid.NewGuid():N}.bin");
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        write(writer);
        return path;
    }
}
