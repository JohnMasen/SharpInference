using SharpInference.Backends.CpuVm;
using SharpInference.Instructions;
using SharpInference.Graphs;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime.Cpu;

/// <summary>Creates CPU graph backends from runtime configuration or compiled VM artifacts.</summary>
public static class CpuVmBackendFactory
{
    /// <summary>Creates a CPU backend with configured queues, optimization, instruction collections, and optional precompiled programs.</summary>
    /// <param name="configuration">The runtime configuration; null selects default settings.</param>
    /// <param name="instructionCollections">The optional instruction providers used by the backend compiler.</param>
    /// <param name="generatorCollections">The optional instruction providers used for target execution-plan generation.</param>
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
        var executionPlan = configuration.ExecutionPlanPath is { } planPath
            ? ExecutionPlanXml.DeserializeTarget(File.ReadAllText(planPath)) : null;
        CpuVmCompiledArtifact? compiled = configuration.ArtifactDirectory is { } directory
            ? CpuVmCompiledArtifact.Load(directory) : null;
        var loaded = compiled;
        if (loaded is { } artifact && configuration.ArtifactDirectory is { } artifactDirectory &&
            File.Exists(Path.Combine(artifactDirectory, "execution-plan.xml")))
        {
            executionPlan = ExecutionPlanXml.DeserializeTarget(File.ReadAllText(Path.Combine(artifactDirectory, "execution-plan.xml")));
            ExecutionPlanAssembler.ValidateArtifact(executionPlan, artifact.Program, new(collections));
        }
        var program = executionPlan is not null ? null : loaded?.Program ?? (configuration.ProgramPath is { } path
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
            executionCapabilities: new(options.InferenceInstances, 1, new HashSet<string> { "vm.cpu.host" }),
            suppliedExecutionPlan: executionPlan);
    }

    /// <summary>Creates a CPU backend from an existing artifact and optional statically supplied VM code.</summary>
    /// <param name="artifact">The existing compiled CPU VM artifact.</param>
    /// <param name="options">The optional VM engine queue and instance settings.</param>
    /// <param name="staticCode">The optional static CPU VM code implementation used instead of artifact-generated code.</param>
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
