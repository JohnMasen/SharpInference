using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Vm;
using SharpInference.Instructions;

namespace SharpInference.Runtime;

public static class VmBackendFactory
{
    public static VmGraphBackend CreateD3D12(VmRuntimeConfig? configuration = null, int adapterIndex = 0,
        IEnumerable<IInstructionCollectionProvider>? instructionCollections = null,
        IEnumerable<IInstructionCollectionProvider>? generatorCollections = null)
    {
        configuration ??= new();
        var collections = instructionCollections?.ToArray() ?? DefaultInstructionCollections.Create();
        var compiler = new D3D12VmCompiler(collections);
        var options = configuration.EngineOptions();
        D3D12VmArtifact? compiled = configuration.ArtifactDirectory is { } directory ? Load(directory) : null;
        var loaded = compiled;
        var program = loaded?.Program ?? (configuration.ProgramPath is { } path
            ? VmProgramXml.Deserialize(File.ReadAllText(path)) : null);
        D3D12VmArtifact? prefillCompiled = configuration.PrefillArtifactDirectory is { } prefillDirectory ? Load(prefillDirectory) : null;
        var prefillLoaded = prefillCompiled;
        var prefillProgram = prefillLoaded?.Program ?? (configuration.PrefillProgramPath is { } prefillPath
            ? VmProgramXml.Deserialize(File.ReadAllText(prefillPath)) : null);
        var (device, deviceName) = D3D12VmDeviceFactory.Create(adapterIndex);
        D3D12VmResourcePool pool;
        try { pool = new D3D12VmResourcePool(device); }
        catch (Exception error)
        {
            try { device.Dispose(); }
            catch (Exception cleanup) { throw new AggregateException("GPU pool initialization and cleanup failed.", error, cleanup); }
            throw;
        }
        return new VmGraphBackend(VmTarget.Direct3D12, candidate =>
        {
            var image = loaded ?? compiler.Compile(candidate);
            compiled = image;
            return () => image.CreateExecutor(pool);
        }, options, configuration.OptimizationOptions(), disposeCompiler: () =>
        {
            List<Exception>? errors = null;
            try { pool.Dispose(); }
            catch (Exception error) { (errors ??= []).Add(error); }
            try { device.Dispose(); }
            catch (Exception error) { (errors ??= []).Add(error); }
            if (errors is not null) throw new AggregateException("GPU resource pool shutdown failed.", errors);
        },
            suppliedProgram: program, exportArtifact: output =>
                Save(compiled ?? throw new InvalidOperationException("The VM has not been compiled."), output),
            suppliedPrefillProgram: prefillProgram, compilePrefill: candidate =>
            {
                var image = prefillLoaded ?? compiler.Compile(candidate);
                prefillCompiled = image;
                return () => image.CreateExecutor(pool);
            }, exportPrefillArtifact: output =>
                Save(prefillCompiled ?? throw new InvalidOperationException("The prefill VM has not been compiled."), output),
            allocate: pool.Allocate, deviceName: deviceName, generatorCollections: generatorCollections ??
                DefaultInstructionCollections.Create().Select(provider => new TargetInstructionCollection(provider, InstructionTarget.Direct3D12)));

        static D3D12VmArtifact Load(string directory)
        {
            using var source = File.OpenRead(Path.Combine(directory, "program.vm.zip"));
            return D3D12VmArtifact.Import(source);
        }
        static void Save(D3D12VmArtifact artifact, string directory)
        {
            Directory.CreateDirectory(directory);
            using var destination = File.Create(Path.Combine(directory, "program.vm.zip"));
            artifact.Export(destination);
        }
    }

    public static VmGraphBackend CreateCpu(VmRuntimeConfig? configuration = null,
        IEnumerable<IInstructionCollectionProvider>? instructionCollections = null,
        IEnumerable<IInstructionCollectionProvider>? generatorCollections = null)
    {
        configuration ??= new();
        var collections = instructionCollections?.ToArray() ?? DefaultInstructionCollections.Create();
        var compiler = new CpuVmCompiler(collections);
        var options = configuration.EngineOptions();
        CpuVmCompiledArtifact? compiled = configuration.ArtifactDirectory is { } directory
            ? CpuVmCompiledArtifact.Load(directory) : null;
        var loaded = compiled;
        VmProgram? program = compiled?.Program ?? (configuration.ProgramPath is { } path
            ? VmProgramXml.Deserialize(File.ReadAllText(path)) : null);
        CpuVmCompiledArtifact? prefillCompiled = configuration.PrefillArtifactDirectory is { } prefillDirectory
            ? CpuVmCompiledArtifact.Load(prefillDirectory) : null;
        var prefillLoaded = prefillCompiled;
        var prefillProgram = prefillCompiled?.Program ?? (configuration.PrefillProgramPath is { } prefillPath
            ? VmProgramXml.Deserialize(File.ReadAllText(prefillPath)) : null);
        return new VmGraphBackend(VmTarget.Cpu, candidate =>
        {
            var image = loaded ?? compiler.Compile(candidate);
            compiled = image;
            return () => image.CreateExecutor();
        }, options, configuration.OptimizationOptions(), suppliedProgram: program,
            exportArtifact: output => (compiled ?? throw new InvalidOperationException("The VM has not been compiled.")).Export(output),
            suppliedPrefillProgram: prefillProgram, compilePrefill: candidate =>
            {
                var image = prefillLoaded ?? compiler.Compile(candidate);
                prefillCompiled = image;
                return () => image.CreateExecutor();
            }, exportPrefillArtifact: output => (prefillCompiled ??
                throw new InvalidOperationException("The prefill VM has not been compiled.")).Export(output),
            generatorCollections: generatorCollections ??
                DefaultInstructionCollections.Create().Select(provider => new TargetInstructionCollection(provider, InstructionTarget.Cpu)));
    }

    public static VmGraphBackend CreateCpuArtifact(CpuVmCompiledArtifact artifact, VmEngineOptions? options = null,
        ICpuVmCode? staticCode = null)
    {
        ArgumentNullException.ThrowIfNull(artifact);
        return new VmGraphBackend(VmTarget.Cpu, _ =>
            staticCode is null ? () => artifact.CreateExecutor() : () => artifact.CreateExecutor(staticCode),
            options, suppliedProgram: artifact.Program, exportArtifact: directory => artifact.Export(directory, artifact.HasBinary));
    }
}
