namespace SharpInference.Tests;

public sealed class TestModelLoaderTests
{
    [Fact]
    public void GetPath_RejectsEmptyRelativeAndMissingPaths()
    {
        Assert.Throws<InvalidOperationException>(() => TestModelLoader.GetPath(TestModel.Rwkv6, null));
        Assert.Throws<ArgumentException>(() => TestModelLoader.GetPath(TestModel.Rwkv6, ""));
        Assert.Throws<ArgumentException>(() => TestModelLoader.GetPath(TestModel.Rwkv6, "model.bin"));
        var missing = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.bin");
        Assert.Throws<FileNotFoundException>(() => TestModelLoader.GetPath(TestModel.Rwkv6, missing));
        Assert.Throws<ArgumentException>(() => TestModelLoader.GetPath(TestModel.Rwkv6, null, "relative"));
        Assert.Throws<FileNotFoundException>(() =>
            TestModelLoader.GetPath(TestModel.Rwkv6, null, Path.GetTempPath()));
    }

    [Fact]
    public void GetPath_ResolvesConfiguredModelDirectory()
    {
        var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "tiny-rwkv-6v0-3m-FP32.bin");
        try
        {
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                writer.Write(0x67676d66u);
                writer.Write(101u);
                writer.Write(4u);
                writer.Write(8u);
                writer.Write(2u);
                writer.Write(0u);
            }
            Assert.Equal(path, TestModelLoader.GetPath(TestModel.Rwkv6, null, directory));
        }
        finally
        {
            File.Delete(path);
            Directory.Delete(directory);
        }
    }

    [Fact]
    public void OpenCatalog_PropagatesInvalidModelFormat()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllText(path, "invalid model");
            Assert.Throws<InvalidDataException>(() => TestModelLoader.OpenCatalog(TestModel.Rwkv6, path));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
