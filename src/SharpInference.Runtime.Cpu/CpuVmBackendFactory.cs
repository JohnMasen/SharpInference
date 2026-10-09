using SharpInference.Backends.CpuVm;
using SharpInference.Instructions;
using SharpInference.Vm;

namespace SharpInference.Runtime.Cpu;

public static class CpuVmBackendFactory
{
    public static VmGraphBackend Create(VmRuntimeConfig? configuration = null,
        IEnumerable<IInstructionCollectionProvider>? instructionCollections = null,
        IEnumerable<IInstructionCollectionProvider>? generatorCollections = null)
    {
        configuration ??= new();
        var optimization = configuration.OptimizationOptions(configuration.TierOneCostProfile is null ? null :
            CpuVmEnvironment.Fingerprint(), VmTarget.Cpu);
        var collections = instructionCollections?.ToArray() ?? CpuInstructionCollections.Create();
        var compiler = new CpuVmCompiler(collections);
        var options = configuration.EngineOptions();
        CpuVmCompiledArtifact? compiled = configuration.ArtifactDirectory is { } directory
            ? CpuVmCompiledArtifact.Load(directory) : null;
        var loaded = compiled;
        var program = loaded?.Program ?? (configuration.ProgramPath is { } path
            ? VmProgramXml.Deserialize(File.ReadAllText(path)) : null);
        CpuVmCompiledArtifact? prefillCompiled = configuration.PrefillArtifactDirectory is { } prefillDirectory
            ? CpuVmCompiledArtifact.Load(prefillDirectory) : null;
        var prefillLoaded = prefillCompiled;
        var prefillProgram = prefillLoaded?.Program ?? (configuration.PrefillProgramPath is { } prefillPath
            ? VmProgramXml.Deserialize(File.ReadAllText(prefillPath)) : null);
        return new VmGraphBackend(VmTarget.Cpu, candidate =>
        {
            var image = loaded ?? compiler.Compile(candidate);
            compiled = image;
            return () => image.CreateExecutor();
        }, options, optimization, suppliedProgram: program,
            exportArtifact: output => (compiled ?? throw new InvalidOperationException("The VM has not been compiled.")).Export(output),
            suppliedPrefillProgram: prefillProgram, compilePrefill: candidate =>
            {
                var image = prefillLoaded ?? compiler.Compile(candidate);
                prefillCompiled = image;
                return () => image.CreateExecutor();
            }, exportPrefillArtifact: output => (prefillCompiled ??
                throw new InvalidOperationException("The prefill VM has not been compiled.")).Export(output),
            generatorCollections: generatorCollections ?? collections.Select(provider =>
                new TargetInstructionCollection(provider, InstructionTarget.Cpu)),
            executionCapabilities: new(options.InferenceInstances, 1, new HashSet<string> { "vm.cpu.host" }));
    }

    public static VmGraphBackend CreateArtifact(CpuVmCompiledArtifact artifact, VmEngineOptions? options = null,
        ICpuVmCode? staticCode = null)
    {
        ArgumentNullException.ThrowIfNull(artifact);
        options ??= new VmEngineOptions(16, 16);
        options.Validate();
        return new VmGraphBackend(VmTarget.Cpu, _ =>
            staticCode is null ? () => artifact.CreateExecutor() : () => artifact.CreateExecutor(staticCode),
            options, suppliedProgram: artifact.Program, exportArtifact: directory => artifact.Export(directory, artifact.HasBinary),
            executionCapabilities: new(options.InferenceInstances, 1, new HashSet<string> { "vm.cpu.host" }));
    }
}
