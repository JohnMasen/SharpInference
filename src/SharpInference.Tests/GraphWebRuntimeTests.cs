using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using SharpInference.Runtime;
using SharpInference.WebApi;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.Tests;

public sealed class GraphWebRuntimeTests
{
    [Theory]
    [InlineData("cpu")]
    [InlineData("vortice")]
    [InlineData("d3d12")]
    public void CommandLine_OverridesRuntimeKind(string kind)
    {
        var parsed = WebApiCommandLine.Parse(
            ["--model-path", "model.bin", "--runtime-kind", kind]);
        Assert.Equal(kind, parsed.Overrides["Rwkv:Runtime:Kind"]);
    }

    [Fact]
    public void CommandLine_ConfiguresVmPoolsAndRejectsLegacyReplay()
    {
        var parsed = WebApiCommandLine.Parse(
            ["--model-path", "model.bin", "--runtime-kind", "d3d12",
             "--prefill-instances", "3", "--inference-instances", "4",
             "--prefill-queue-capacity", "5", "--inference-queue-capacity", "6"]);
        Assert.Equal("3", parsed.Overrides["Rwkv:Runtime:Vm:PrefillInstances"]);
        Assert.Equal("4", parsed.Overrides["Rwkv:Runtime:Vm:InferenceInstances"]);
        Assert.Equal("5", parsed.Overrides["Rwkv:Runtime:Vm:PrefillQueueCapacity"]);
        Assert.Equal("6", parsed.Overrides["Rwkv:Runtime:Vm:InferenceQueueCapacity"]);
        Assert.Throws<ArgumentException>(() => WebApiCommandLine.Parse(
            ["--model-path", "model.bin", "--enable-command-replay", "true"]));
        Assert.Throws<ArgumentException>(() => WebApiCommandLine.Parse(
            ["--model-path", "model.bin", "--max-in-flight-generation-batches", "4"]));
    }

    [Fact]
    public void ModelHost_RejectsReplayForCpuRuntime()
    {
        var path = TestModelLoader.GetPath(TestModel.Rwkv7Fp32);
        var descriptor = new RwkvModelDescriptor(Options.Create(new RwkvWebOptions { ModelPath = path }));
        var config = new ConfigurationBuilder().AddInMemoryCollection(
            new Dictionary<string, string?>
            {
                ["Rwkv:Runtime:Kind"] = "cpu",
                ["Rwkv:Runtime:Vortice:EnableCommandReplay"] = "true",
            }).Build();
        var error = Assert.Throws<InvalidOperationException>(() => new RwkvModelHost(
            descriptor, config, new RwkvRuntimeFactory(),
            new GpuBatchScheduler(new GpuBatchServiceOptions(), "cpu"),
            new ModelTextTransferResolver([new Rwkv6WorldTextTransfer(), new Rwkv7G1TextTransfer()]),
            new PromptStateManager(new PromptStateManagerOptions()),
            NullLogger<RwkvModelHost>.Instance));
        Assert.Contains("Unsupported RWKV Vortice runtime option 'EnableCommandReplay'", error.Message);
    }

    [Fact]
    public void ModelHost_RejectsLegacyGpuReplay()
    {
        if (!HasGpu()) return;
        var path = TestModelLoader.GetPath(TestModel.Rwkv7Fp32);
        var descriptor = new RwkvModelDescriptor(Options.Create(new RwkvWebOptions { ModelPath = path }));
        var config = new ConfigurationBuilder().AddInMemoryCollection(
            new Dictionary<string, string?>
            {
                ["Rwkv:Runtime:Kind"] = "vortice",
                ["Rwkv:Runtime:Vortice:EnableCommandReplay"] = "true",
            }).Build();
        var error = Assert.Throws<InvalidOperationException>(() => new RwkvModelHost(
            descriptor, config, new RwkvRuntimeFactory(),
            new GpuBatchScheduler(new GpuBatchServiceOptions(), "vortice"),
            new ModelTextTransferResolver([new Rwkv6WorldTextTransfer(), new Rwkv7G1TextTransfer()]),
            new PromptStateManager(new PromptStateManagerOptions()),
            NullLogger<RwkvModelHost>.Instance));
        Assert.Contains("Unsupported RWKV Vortice runtime option 'EnableCommandReplay'", error.Message);
    }

    [Theory]
    [InlineData("cpu", false)]
    [InlineData("cpu", true)]
    [InlineData("vortice", false)]
    [InlineData("vortice", true)]
    public void ModelHost_SelectsGraphBeforeRejectingTinyVocabulary(string kind, bool rwkv7)
    {
        if (kind == "vortice" && !HasGpu()) return;
        var path = TestModelLoader.GetPath(rwkv7 ? TestModel.Rwkv7Fp32 : TestModel.Rwkv6);
        var descriptor = new RwkvModelDescriptor(Options.Create(new RwkvWebOptions { ModelPath = path }));
        var config = new ConfigurationBuilder().AddInMemoryCollection(
            new Dictionary<string, string?> { ["Rwkv:Runtime:Kind"] = kind }).Build();
        var error = Assert.Throws<InvalidOperationException>(() => new RwkvModelHost(
            descriptor, config, new RwkvRuntimeFactory(),
            new GpuBatchScheduler(new GpuBatchServiceOptions(), kind),
            new ModelTextTransferResolver([new Rwkv6WorldTextTransfer(), new Rwkv7G1TextTransfer()]),
            new PromptStateManager(new PromptStateManagerOptions()),
            NullLogger<RwkvModelHost>.Instance));
        Assert.Contains("token IDs outside", error.Message);
    }

    private static bool HasGpu()
    {
        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory6>();
        for (uint index = 0; factory.EnumAdapterByGpuPreference(
                 index, GpuPreference.HighPerformance, out IDXGIAdapter1? adapter).Success; index++)
        {
            using (adapter)
            {
                if ((adapter!.Description1.Flags & AdapterFlags.Software) != 0) continue;
                using var device = D3D12CreateDevice<ID3D12Device>(adapter, FeatureLevel.Level_11_0);
                if (device is not null) return true;
            }
        }
        return false;
    }
}
