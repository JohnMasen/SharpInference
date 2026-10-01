using Microsoft.Extensions.Configuration;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Backends.Cpu;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class RwkvRuntimeFactoryTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void CreateRuntime_DefaultAndExplicitCpuUsePortableGraphForBothModels(
        bool rwkv7, bool explicitCpu)
    {
        using var catalog = TestModelLoader.OpenCatalog(rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6);
        var configuration = CreateConfiguration(("Rwkv:Runtime:Kind", "cpu"));
        var runtime = new RwkvRuntimeFactory().CreateRuntime(
            explicitCpu ? configuration.GetSection("Rwkv:Runtime") : null, catalog);

        Assert.Equal(rwkv7 ? "rwkv-7" : "rwkv-6", runtime.ArchitectureId);
        Assert.NotNull(runtime.Tokenizer);
        if (rwkv7) Assert.IsType<PortableRwkv7GraphProvider>(runtime.Provider);
        else Assert.IsType<PortableRwkv6GraphProvider>(runtime.Provider);
        Assert.Same(CpuPrimitiveGraphBackend.Instance, runtime.CreateBackend());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CreateRuntime_UnknownBackendKindThrowsExplicitError(bool rwkv7)
    {
        using var catalog = TestModelLoader.OpenCatalog(rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6);
        var configuration = CreateConfiguration(("Rwkv:Runtime:Kind", "unknown"));

        var exception = Assert.Throws<InvalidOperationException>(() =>
            new RwkvRuntimeFactory().CreateRuntime(configuration.GetSection("Rwkv:Runtime"), catalog));

        Assert.Contains("unknown", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void CreateRuntime_RejectsGpuReplaySettingOnCpuInsteadOfSilentlyFallingBack()
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv6);
        var configuration = CreateConfiguration(
            ("Rwkv:Runtime:Kind", "cpu"),
            ("Rwkv:Runtime:Vortice:EnableCommandReplay", "true"));

        var exception = Assert.Throws<InvalidOperationException>(() =>
            new RwkvRuntimeFactory().CreateRuntime(configuration.GetSection("Rwkv:Runtime"), catalog));

        Assert.Contains("vortice", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("WeightPrecision", "Fp16")]
    [InlineData("AllowFp16Fallback", "true")]
    public void CreateRuntime_RejectsRemovedVorticeOptions(string option, string value)
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv6);
        var configuration = CreateConfiguration(
            ("Rwkv:Runtime:Kind", "vortice"),
            ($"Rwkv:Runtime:Vortice:{option}", value));

        var exception = Assert.Throws<InvalidOperationException>(() =>
            new RwkvRuntimeFactory().CreateRuntime(configuration.GetSection("Rwkv:Runtime"), catalog));

        Assert.Contains(option, exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    private static IConfigurationRoot CreateConfiguration(params (string Key, string Value)[] values) =>
        new ConfigurationBuilder()
            .AddInMemoryCollection(values.ToDictionary(static item => item.Key, static item => (string?)item.Value))
            .Build();
}
