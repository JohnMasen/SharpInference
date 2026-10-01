using Microsoft.Extensions.Options;
using SharpInference.WebApi;

namespace SharpInference.Tests;

public sealed class RwkvModelDescriptorTests
{
    [Fact]
    public void DerivesModelIdWithoutOpeningModelFile()
    {
        var descriptor = new RwkvModelDescriptor(Options.Create(new RwkvWebOptions
        {
            ModelPath = "C:\\models\\RWKV x060:World (FP16).bin",
        }));

        Assert.Equal("RWKV-x060-World-FP16-.bin", descriptor.ModelId);
        Assert.True(descriptor.MatchesModel("RWKV-x060-World-FP16-.bin"));
        Assert.False(descriptor.MatchesModel("other-model"));
        Assert.Equal(4096, descriptor.DefaultMaxTokens);
        Assert.Equal(12800, descriptor.ContextWindowTokens);
    }

    [Fact]
    public void RejectsNonPositiveContextWindow()
    {
        var exception = Assert.Throws<InvalidOperationException>(() =>
            new RwkvModelDescriptor(Options.Create(new RwkvWebOptions
            {
                ModelPath = "C:\\models\\model.bin",
                ContextWindowTokens = 0,
            })));

        Assert.Contains("ContextWindowTokens must be positive", exception.Message);
    }
}
