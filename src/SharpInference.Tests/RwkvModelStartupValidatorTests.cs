using Microsoft.Extensions.Options;
using SharpInference.WebApi;

namespace SharpInference.Tests;

public sealed class RwkvModelStartupValidatorTests
{
    [Fact]
    public void Validate_RejectsMissingModelFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"missing-rwkv-{Guid.NewGuid():N}.bin");
        var descriptor = CreateDescriptor(path);

        var exception = Assert.Throws<InvalidOperationException>(
            () => RwkvModelStartupValidator.Validate(descriptor));

        Assert.Contains(path, exception.Message);
        Assert.IsType<FileNotFoundException>(exception.InnerException);
    }

    [Fact]
    public void Validate_RejectsInvalidModelHeader()
    {
        var path = CreateModelFile(writer =>
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
            var descriptor = CreateDescriptor(path);

            var exception = Assert.Throws<InvalidOperationException>(
                () => RwkvModelStartupValidator.Validate(descriptor));

            Assert.Contains(path, exception.Message);
            Assert.IsType<InvalidDataException>(exception.InnerException);
            Assert.Contains("Unsupported GGML header", exception.InnerException.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Validate_AcceptsValidModelHeader()
    {
        var path = CreateModelFile(writer =>
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
            RwkvModelStartupValidator.Validate(CreateDescriptor(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static RwkvModelDescriptor CreateDescriptor(string path) =>
        new(Options.Create(new RwkvWebOptions { ModelPath = path }));

    private static string CreateModelFile(Action<BinaryWriter> write)
    {
        var path = Path.Combine(Path.GetTempPath(), $"sharpinference-{Guid.NewGuid():N}.bin");
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        write(writer);
        return path;
    }
}
