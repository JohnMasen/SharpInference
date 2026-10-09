using Microsoft.Extensions.Configuration;
using SharpInference.Applications;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;
using SharpInference.Runtime;
using SharpInference.Runtime.Cpu;
using SharpInference.Runtime.D3D12;
using SharpInference.Vm;

namespace SharpInference.Tests;

[Collection(VmGraphGpuCollection.Name)]
public sealed class ApplicationCompositionTests
{
    [Fact]
    public void ModelRuntimeHasNoBackendOrConfigurationDependencies()
    {
        var assembly = typeof(RwkvRuntimeFactory).Assembly;
        Assert.DoesNotContain(assembly.GetReferencedAssemblies(), reference =>
            reference.Name!.StartsWith("SharpInference.Backends.", StringComparison.Ordinal) ||
            reference.Name is "SharpInference.Runtime.Cpu" or "SharpInference.Runtime.D3D12" or
                "SharpInference.Runtime.Compatibility" ||
            reference.Name.StartsWith("Microsoft.Extensions.Configuration", StringComparison.Ordinal));
        Assert.All(typeof(RwkvRuntimeFactory).GetMethods().Where(method => method.Name == "Load"), method =>
            Assert.Contains(method.GetParameters(), parameter => parameter.ParameterType == typeof(VmGraphBackend)));
    }

    [Fact]
    public void ApplicationsOwnConcreteModelAndBackendRegistrationWithoutCompatibility()
    {
        var assembly = typeof(RwkvApplicationComposition).Assembly;
        var references = assembly.GetReferencedAssemblies();
        Assert.Contains(references, reference => reference.Name == "SharpInference.Runtime.Cpu");
        Assert.Contains(references, reference => reference.Name == "SharpInference.Runtime.D3D12");
        Assert.DoesNotContain(references, reference => reference.Name == "SharpInference.Runtime.Compatibility");
        var modules = RwkvApplicationComposition.CreateModelModules();
        Assert.IsType<Rwkv6ModelModule>(modules.GetRequired("rwkv-6"));
        Assert.IsType<Rwkv7ModelModule>(modules.GetRequired("rwkv-7"));
    }

    [Fact]
    public void EmptyRegistryDoesNotAutomaticallyRegisterModels()
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv6);
        Assert.Throws<NotSupportedException>(() =>
            new RwkvApplicationComposition(new ModelGraphModuleRegistry()).CreateRuntime(null, catalog));
    }

    [Theory]
    [InlineData("cpu", VmTarget.Cpu)]
    [InlineData("CPU", VmTarget.Cpu)]
    [InlineData("d3d12", VmTarget.Direct3D12)]
    [InlineData("vortice", VmTarget.Direct3D12)]
    public void AliasesSelectExplicitAdapterAndPreserveConfiguredConcurrency(string kind, VmTarget expected)
    {
        using var catalog = TestModelLoader.OpenCatalog(TestModel.Rwkv6);
        var configuration = new ConfigurationBuilder().AddInMemoryCollection(new Dictionary<string, string?>
        {
            ["Runtime:Kind"] = kind,
            ["Runtime:Vm:InferenceInstances"] = "3",
            ["Runtime:Vm:PrefillInstances"] = "1",
            ["Runtime:Vm:PrefillCapacity"] = "8",
            ["Runtime:Vm:ThreadsPerGroup"] = "32",
            ["Runtime:Vm:ReuseLocalStorage"] = "false",
            ["Runtime:Vm:NativeHalfWeights"] = "false",
            ["Runtime:Vm:WeightViews"] = "false",
            ["Runtime:Vortice:AdapterIndex"] = "0",
        }).Build();
        var selection = new RwkvApplicationComposition(RwkvApplicationComposition.CreateModelModules())
            .CreateRuntime(configuration.GetSection("Runtime"), catalog);
        using var backend = selection.CreateBackend();
        Assert.Equal(3, backend.ExecutionCapabilities.MaximumConcurrentSessions);
        Assert.Equal(expected == VmTarget.Direct3D12, backend.ExecutionCapabilities.RequiresResidentSessionAdmission);
        Assert.Equal(expected == VmTarget.Cpu ? "vm.cpu.host" : "vm.d3d12.device",
            Assert.Single(backend.ExecutionCapabilities.StorageDomains));
        var program = backend.Prepare(selection.Module.Build(catalog)).Program;
        Assert.Equal(expected, program.Target);
    }

    [Fact]
    public void StaticCpuArtifactAdvertisesItsConfiguredExecutionCapacity()
    {
        var program = SharpInference.Vm.Optimization.TierOneReferencePrograms.CreateFused("multiply-add", VmTarget.Cpu, [4]);
        var artifact = new SharpInference.Backends.CpuVm.CpuVmCompiler(CpuInstructionCollections.Create()).GenerateSource(program);
        using var backend = CpuVmBackendFactory.CreateArtifact(artifact, new VmEngineOptions(16, 16, InferenceInstances: 3));
        Assert.Equal(3, backend.ExecutionCapabilities.MaximumConcurrentSessions);
        Assert.Equal("vm.cpu.host", Assert.Single(backend.ExecutionCapabilities.StorageDomains));
    }

    [Fact]
    public void BackendInstructionCatalogsContainOnlyTheirSelectedTarget()
    {
        Assert.All(CpuInstructionCollections.Create().SelectMany(provider => provider.QueryInstructionCollection()),
            collection => Assert.Equal(SharpInference.Instructions.InstructionTarget.Cpu, collection.Architecture));
        Assert.All(D3D12InstructionCollections.Create().SelectMany(provider => provider.QueryInstructionCollection()),
            collection => Assert.Equal(SharpInference.Instructions.InstructionTarget.Direct3D12, collection.Architecture));
    }
}
