namespace SharpInference.Tests;

public sealed class FoundationDependencyTests
{
    [Fact]
    public void RuntimeDoesNotReferenceModelsOrBackendImplementations()
    {
        var assembly = typeof(SharpInference.Runtime.Processor).Assembly;
        Assert.DoesNotContain(assembly.GetReferencedAssemblies(), reference =>
            IsModelAssembly(reference.Name!) ||
            reference.Name!.StartsWith("SharpInference.Backends.", StringComparison.Ordinal) ||
            reference.Name is "SharpInference.Instructions.Cpu" or "SharpInference.Instructions.D3D12" ||
            reference.Name.StartsWith("SharpInference.Runtime.", StringComparison.Ordinal));
        Assert.DoesNotContain(assembly.GetExportedTypes(), type =>
            type.Name.Contains("Rwkv", StringComparison.OrdinalIgnoreCase) ||
            type.Name.Contains("D3D12", StringComparison.OrdinalIgnoreCase));
        Assert.Equal(typeof(ModelMetadata), typeof(SharpInference.Runtime.Processor).GetProperty("Metadata")!.PropertyType);
        var framework = Assert.Single(assembly.GetCustomAttributes(typeof(System.Runtime.Versioning.TargetFrameworkAttribute), false));
        Assert.Equal(".NETCoreApp,Version=v10.0", ((System.Runtime.Versioning.TargetFrameworkAttribute)framework).FrameworkName);
    }

    [Fact]
    public void CpuRuntimeAdapterDoesNotReferenceGpuOrModels()
    {
        var assembly = typeof(SharpInference.Runtime.Cpu.CpuVmBackendFactory).Assembly;
        Assert.DoesNotContain(assembly.GetReferencedAssemblies(), reference =>
            reference.Name!.Contains("D3D12", StringComparison.OrdinalIgnoreCase) ||
            IsModelAssembly(reference.Name!) ||
            reference.Name.StartsWith("Vortice.", StringComparison.Ordinal));
    }

    [Fact]
    public void AbstractionsDoesNotExposeRwkvContracts()
    {
        Assert.DoesNotContain(typeof(IModel).Assembly.GetExportedTypes(),
            type => type.Name.Contains("Rwkv", StringComparison.OrdinalIgnoreCase));
        Assert.Equal(typeof(TensorDataType), typeof(IModelTensor).GetProperty("DataType")!.PropertyType);
        Assert.Equal(["Names"], typeof(IModelTensorCatalog).GetProperties().Select(property => property.Name));
    }

    [Fact]
    public void GraphContractsDoNotReferenceModelsOrBackends()
    {
        var assembly = typeof(SharpInference.Graphs.PortableGraphModel).Assembly;
        Assert.DoesNotContain(assembly.GetReferencedAssemblies(), reference =>
            IsModelAssembly(reference.Name!) ||
            reference.Name!.StartsWith("SharpInference.Backends.", StringComparison.Ordinal));
        Assert.DoesNotContain(assembly.GetExportedTypes(),
            type => type.Name.Contains("Rwkv", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void AbstractionsDoesNotReferenceModelsOrBackends()
    {
        var references = typeof(IModelTensorCatalog).Assembly.GetReferencedAssemblies();
        Assert.DoesNotContain(references, assembly =>
            IsModelAssembly(assembly.Name!) ||
            assembly.Name!.StartsWith("SharpInference.Backends.", StringComparison.Ordinal));
    }

    [Fact]
    public void CoreDoesNotReferenceModelsOrBackends()
    {
        var assembly = typeof(OwnedModelTensorCatalog).Assembly;
        var references = assembly.GetReferencedAssemblies();
        Assert.DoesNotContain(references, assembly =>
            assembly.Name!.StartsWith("SharpInference.Backends.", StringComparison.Ordinal) ||
            IsModelAssembly(assembly.Name!));
        Assert.DoesNotContain(assembly.GetExportedTypes(),
            type => type.Name.Contains("Rwkv", StringComparison.OrdinalIgnoreCase));
        Assert.Empty(assembly.GetManifestResourceNames());
    }

    [Fact]
    public void AllSharedAssembliesRejectConcreteModulesAndModelNamedPublicMembers()
    {
        var assemblies = new[]
        {
            typeof(IModel).Assembly, typeof(OwnedModelTensorCatalog).Assembly,
            typeof(SharpInference.Graphs.LogicalGraph).Assembly, typeof(TextGenerator).Assembly,
            typeof(SharpInference.Instructions.Instruction).Assembly, typeof(SharpInference.Vm.VmProgram).Assembly,
            typeof(SharpInference.Vm.Optimization.VmGraphOptimizer).Assembly, typeof(SharpInference.Runtime.Processor).Assembly,
        };
        foreach (var assembly in assemblies)
        {
            Assert.DoesNotContain(assembly.GetReferencedAssemblies(), reference =>
                IsModelAssembly(reference.Name!) ||
                reference.Name!.StartsWith("SharpInference.Backends.", StringComparison.Ordinal) ||
                reference.Name.StartsWith("SharpInference.Runtime.", StringComparison.Ordinal) ||
                reference.Name.StartsWith("SharpInference.Instructions.", StringComparison.Ordinal) ||
                reference.Name == "SharpInference.Applications" || reference.Name.StartsWith("Vortice.", StringComparison.Ordinal));
            foreach (var type in assembly.GetExportedTypes())
            {
                Assert.False(ModelName(type.Name), $"Model-specific shared type: {type.FullName}");
                Assert.DoesNotContain(type.GetMembers(), member => ModelName(member.Name));
            }
        }
        static bool ModelName(string name) => name.Contains("Rwkv", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("Phi4", StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void GenerationDoesNotReferenceModelsOrBackends()
    {
        var assembly = typeof(TextGenerator).Assembly;
        Assert.DoesNotContain(assembly.GetReferencedAssemblies(), reference =>
            IsModelAssembly(reference.Name!) ||
            reference.Name!.StartsWith("SharpInference.Backends.", StringComparison.Ordinal));
        Assert.DoesNotContain(assembly.GetExportedTypes(),
            type => type.Name.Contains("Rwkv", StringComparison.OrdinalIgnoreCase));
    }

    private static bool IsModelAssembly(string name) =>
        name.StartsWith("SharpInference.Architectures.", StringComparison.Ordinal) ||
        name is "SharpInference.Models.Rwkv" or "SharpInference.Models.Phi4";
}
