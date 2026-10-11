using SharpInference.Backends.D3D12Vm;
using SharpInference.Instructions;
using SharpInference.Graphs;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime.D3D12;

/// <summary>Creates Direct3D 12 graph backends with device resource pooling and compiled artifact support.</summary>
public static class D3D12VmBackendFactory
{
    /// <summary>Creates a Direct3D 12 backend on the selected adapter with configured compilation, optimization, queues, and pooled storage.</summary>
    /// <param name="configuration">The runtime configuration; null selects default settings.</param>
    /// <param name="adapterIndex">The zero-based Direct3D 12 adapter index.</param>
    /// <param name="instructionCollections">The optional instruction providers used by the backend compiler.</param>
    /// <param name="generatorCollections">The optional instruction providers used for target execution-plan generation.</param>
    public static VmGraphBackend Create(VmRuntimeConfig? configuration = null, int adapterIndex = 0,
        IEnumerable<IInstructionCollectionProvider>? instructionCollections = null,
        IEnumerable<IInstructionCollectionProvider>? generatorCollections = null)
    {
        configuration ??= new();
        var optimization = configuration.OptimizationOptions(configuration.TierOneCostProfile is null && configuration.GpuMatVecCostProfile is null ? null :
            D3D12VmEnvironment.Fingerprint(adapterIndex), VmTarget.Direct3D12);
        var collections = instructionCollections?.ToArray() ?? D3D12InstructionCollections.Create();
        var compiler = new D3D12VmCompiler(collections);
        var options = configuration.EngineOptions();
        var executionPlan = configuration.ExecutionPlanPath is { } planPath
            ? ExecutionPlanXml.DeserializeTarget(File.ReadAllText(planPath)) : null;
        D3D12VmArtifact? compiled = configuration.ArtifactDirectory is { } directory ? Load(directory) : null;
        var loaded = compiled;
        if (loaded is { } artifact && configuration.ArtifactDirectory is { } artifactDirectory &&
            File.Exists(Path.Combine(artifactDirectory, "execution-plan.xml")))
        {
            executionPlan = ExecutionPlanXml.DeserializeTarget(File.ReadAllText(Path.Combine(artifactDirectory, "execution-plan.xml")));
            ExecutionPlanAssembler.ValidateArtifact(executionPlan, artifact.Program, new(collections));
        }
        var program = executionPlan is not null ? null : loaded?.Program ?? (configuration.ProgramPath is { } path
            ? VmProgramXml.Deserialize(File.ReadAllText(path)) : null);
        D3D12VmArtifact? prefillCompiled = configuration.PrefillArtifactDirectory is { } prefillDirectory ? Load(prefillDirectory) : null;
        var prefillLoaded = prefillCompiled;
        var prefillProgram = prefillLoaded?.Program ?? (configuration.PrefillProgramPath is { } prefillPath
            ? VmProgramXml.Deserialize(File.ReadAllText(prefillPath)) : null);
        var (device, deviceName) = D3D12VmDeviceFactory.Create(adapterIndex);
        D3D12VmResourcePool? pool = null;
        try
        {
            pool = new D3D12VmResourcePool(device);
            var resources = pool;
            return new VmGraphBackend(VmTarget.Direct3D12, candidate =>
            {
                var image = loaded ?? compiler.Compile(candidate);
                compiled = image;
                return () => image.CreateExecutor(resources);
            }, options, optimization, disposeCompiler: Release,
                suppliedProgram: program, exportArtifact: output =>
                    Save(compiled ?? throw new InvalidOperationException("The VM has not been compiled."), output),
                suppliedPrefillProgram: prefillProgram, compilePrefill: candidate =>
                {
                    var image = prefillLoaded ?? compiler.Compile(candidate);
                    prefillCompiled = image;
                    return () => image.CreateExecutor(resources);
                }, exportPrefillArtifact: output =>
                    Save(prefillCompiled ?? throw new InvalidOperationException("The prefill VM has not been compiled."), output),
                allocate: resources.Allocate, deviceName: deviceName,
                generatorCollections: generatorCollections ?? collections.Select(provider =>
                    new TargetInstructionCollection(provider, InstructionTarget.Direct3D12)),
                diagnostics: new D3D12VmDiagnostics(() => resources.TaskStatistics),
                executionCapabilities: new(options.InferenceInstances, 1, new HashSet<string> { "vm.d3d12.device" },
                    requiresResidentSessionAdmission: true), suppliedExecutionPlan: executionPlan);
        }
        catch (Exception error)
        {
            try { Release(); }
            catch (Exception cleanup) { throw new AggregateException("GPU backend initialization and cleanup failed.", error, cleanup); }
            throw;
        }

        void Release()
        {
            List<Exception>? errors = null;
            try { pool?.Dispose(); }
            catch (Exception error) { (errors ??= []).Add(error); }
            try { device.Dispose(); }
            catch (Exception error) { (errors ??= []).Add(error); }
            if (errors is not null) throw new AggregateException("GPU resource pool shutdown failed.", errors);
        }
    }

    private static D3D12VmArtifact Load(string directory)
    {
        using var source = File.OpenRead(Path.Combine(directory, "program.vm.zip"));
        return D3D12VmArtifact.Import(source);
    }

    private static void Save(D3D12VmArtifact artifact, string directory)
    {
        Directory.CreateDirectory(directory);
        using var destination = File.Create(Path.Combine(directory, "program.vm.zip"));
        artifact.Export(destination);
    }
}
