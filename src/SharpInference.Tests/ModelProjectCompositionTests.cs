using SharpInference.Architectures.Phi4;
using SharpInference.Architectures.Phi4.D3D12;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Instructions.Phi4;
using SharpInference.Instructions.Phi4.Cpu;
using SharpInference.Instructions.Phi4.D3D12;
using SharpInference.Runtime;

namespace SharpInference.Tests;

public sealed class ModelProjectCompositionTests
{
    [Fact]
    public void RwkvFamilyVersionsAndRuntimeShareOneAssembly()
    {
        var assembly = typeof(RwkvWorldTokenizer).Assembly;
        Assert.Equal("SharpInference.Models.Rwkv", assembly.GetName().Name);
        Assert.Same(assembly, typeof(Rwkv6ModelModule).Assembly);
        Assert.Same(assembly, typeof(Rwkv7ModelModule).Assembly);
        Assert.Same(assembly, typeof(RwkvRuntimeFactory).Assembly);
        Assert.Contains("SharpInference.Resources.rwkv_vocab_v20230424.txt", assembly.GetManifestResourceNames());
        Assert.DoesNotContain(assembly.GetReferencedAssemblies(), reference =>
            reference.Name!.StartsWith("SharpInference.Architectures.", StringComparison.Ordinal));
    }

    [Fact]
    public void Phi4ArchitectureInstructionsAndDeviceComponentsShareOneAssembly()
    {
        var assembly = typeof(Phi4ModelGraphModule).Assembly;
        Assert.Equal("SharpInference.Models.Phi4", assembly.GetName().Name);
        Assert.Same(assembly, typeof(Phi4TextGraphOperations).Assembly);
        Assert.Same(assembly, typeof(Phi4CpuAudioInstructionCollection).Assembly);
        Assert.Same(assembly, typeof(Phi4CpuTextInstructionCollection).Assembly);
        Assert.Same(assembly, typeof(Phi4D3D12AudioInstructionCollection).Assembly);
        Assert.Same(assembly, typeof(Phi4D3D12TextInstructionCollection).Assembly);
        Assert.Same(assembly, typeof(Phi4D3D12VisionInstructionCollection).Assembly);
        Assert.Same(assembly, typeof(Phi4D3D12MultimodalSession).Assembly);
        Assert.DoesNotContain(assembly.GetReferencedAssemblies(), reference =>
            reference.Name!.StartsWith("SharpInference.Architectures.", StringComparison.Ordinal) ||
            reference.Name.StartsWith("SharpInference.Instructions.Phi4", StringComparison.Ordinal));
    }
}
