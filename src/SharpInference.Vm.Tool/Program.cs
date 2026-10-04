using System.Runtime.InteropServices;
using System.Text.Json;
using SharpInference;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Backends.Cpu;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

if (args.Length == 0 || args[0] is "--help" or "-h")
{
    Console.WriteLine("""
        build <cpu|d3d12> <weights.bin> <artifact-directory> [program.xml] [--source-only]
        tune <cpu|d3d12> <weights.bin> <report.json> <hardware-id> <driver-id> <compiler-id> <prefill|inference> <candidate.xml>...
        Tuning validates logits and all registered State entries against the portable CPU reference.
        Tuning tokens are 1,2,3,2; each measured run restores zero initial State.
        """);
    return;
}

try
{
    if (args.Length < 4 || args[1] is not ("cpu" or "d3d12"))
        throw new ArgumentException("Select a supported target ('cpu'/'d3d12') and supply the required arguments.");
    var target = args[1] == "cpu" ? VmTarget.Cpu : VmTarget.Direct3D12;
    using var catalog = GgmlModelFile.Open(args[2]);
    var architecture = RwkvModelArchitectureDetector.Detect(catalog);
    ILogicalGraphProvider provider = architecture switch
    {
        "rwkv-6" => new PortableRwkv6GraphProvider(),
        "rwkv-7" => new PortableRwkv7GraphProvider(),
        _ => throw new NotSupportedException($"Unsupported model architecture '{architecture}'."),
    };
    var logical = provider.Build(catalog);
    if (args[0] == "build")
    {
        var sourceOnly = args[^1] == "--source-only";
        var count = args.Length - (sourceOnly ? 1 : 0);
        if (count is < 4 or > 5) throw new ArgumentException("Invalid build arguments.");
        var program = count == 5 ? ReadProgram(args[4]) : VmGraphOptimizer.Optimize(logical, target);
        if (target == VmTarget.Cpu)
        {
            var compiler = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create());
            var artifact = sourceOnly ? compiler.GenerateSource(program) : compiler.Compile(program);
            artifact.Export(args[3], includeBinary: !sourceOnly);
        }
        else
        {
            if (sourceOnly) throw new ArgumentException("DXIL publication requires a compiled package; --source-only is CPU-only.");
            var artifact = new D3D12VmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
            Directory.CreateDirectory(args[3]);
            using var destination = File.Create(Path.Combine(args[3], "program.vm.zip"));
            artifact.Export(destination);
            File.WriteAllText(Path.Combine(args[3], "program.xml"), VmProgramXml.Serialize(program));
        }
        Console.WriteLine($"Exported {program.Target} VM: {Path.GetFullPath(args[3])}");
        var xml = VmProgramXml.Serialize(program);
        Console.WriteLine($"Slots: {program.Slots.Count}; definitions: {program.Definitions.Count}; XML characters: {xml.Length}; " +
            $"elements: {System.Xml.Linq.XDocument.Parse(xml).Descendants().Count()}; " +
            $"local bytes: {program.Slots.Where(slot => slot.Scope == VmSlotScope.Local).Sum(slot => (long)slot.Tensor.ByteLength)}.");
    }
    else if (args[0] == "tune")
    {
        if (args.Length < 9) throw new ArgumentException("Tuning requires target fingerprints, workload and candidates.");
        var workload = args[7] switch
        {
            "prefill" => VmTuningWorkload.Prefill,
            "inference" => VmTuningWorkload.Inference,
            _ => throw new ArgumentException("Workload must be 'prefill' or 'inference'."),
        };
        var candidates = args.Skip(8).Select(path => new VmTuningCandidate(Path.GetFileNameWithoutExtension(path), ReadProgram(path))).ToArray();
        int[] tokens = [1, 2, 3, 2];
        IReadOnlyList<GraphStateValue> state = logical.GraphState.Entries.Select(entry =>
        {
            var tensor = logical.Resources.Single(resource => resource.Id == entry.Resource).Tensor;
            return new GraphStateValue(entry.Name, tensor.Dimensions,
                new float[tensor.Dimensions.Aggregate(1, (count, size) => checked(count * size))]);
        }).ToArray();
        var initial = Bytes(state);
        var referenceExecutor = new CpuPrimitiveGraphExecutor(logical, catalog);
        using var device = target == VmTarget.Direct3D12 ? D3D12VmDeviceFactory.Create().Device : null;
        using var pool = device is not null ? new D3D12VmResourcePool(device) : null;
        float[] logits = [];
        foreach (var token in tokens)
        {
            var result = referenceExecutor.Execute(new Dictionary<ResourceId, Array> { [new("token")] = new[] { token } }, state);
            logits = (float[])result.Outputs[new("logits")];
            state = result.State;
        }
        var resultReport = await VmProgramTuner.TuneAsync(candidates, "token", "logits", tokens, initial,
            new(logits, Bytes(state)), (program, cancellation) =>
            {
                cancellation.ThrowIfCancellationRequested();
                if (target == VmTarget.Cpu)
                {
                    var artifact = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
                    return ValueTask.FromResult<Func<IVmExecutable>>(() => artifact.CreateExecutor());
                }
                var gpuArtifact = new D3D12VmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
                return ValueTask.FromResult<Func<IVmExecutable>>(() => gpuArtifact.CreateExecutor(
                    pool ?? throw new InvalidOperationException("The tuning GPU pool is not initialized.")));
            }, Initialize, new(args[4], args[5], args[1], args[6]),
            new(Workload: workload), allocate: pool is null ? null : pool.Allocate);
        File.WriteAllText(args[3], JsonSerializer.Serialize(resultReport, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine(resultReport.Winner is { } winner ? $"Winner: {winner}" : "No numerically valid candidate.");
        if (resultReport.Winner is null) Environment.ExitCode = 1;
    }
    else throw new ArgumentException($"Unknown command '{args[0]}'.");

    VmProgram ReadProgram(string path)
    {
        var program = VmProgramXml.Deserialize(File.ReadAllText(path));
        if (program.Target != target || program.Abi != $"vm:{logical.Model.StateAbiId}")
            throw new InvalidDataException($"Program '{path}' has an incompatible target/model ABI.");
        return program;
    }

    void Initialize(VmSlot slot, IVmStorage storage)
    {
        VmModelBindings.InitializeGlobal(catalog, slot, storage);
    }
}
catch (Exception error) when (error is IOException or ArgumentException or InvalidOperationException or NotSupportedException)
{
    Console.Error.WriteLine(error);
    Environment.ExitCode = 1;
}

static IReadOnlyDictionary<string, byte[]> Bytes(IReadOnlyList<GraphStateValue> values) =>
    values.ToDictionary(value => value.Name, value => MemoryMarshal.AsBytes(value.Values.AsSpan()).ToArray(), StringComparer.Ordinal);
