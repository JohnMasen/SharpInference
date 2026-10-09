using System.Reflection;
using System.Runtime.InteropServices;
using SharpInference;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;
#if GPU_VM
using SharpInference.Backends.D3D12Vm;
#else
using SharpInference.Backends.CpuVm;
#endif
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

if (args.Length is < 2 or > 4)
{
    Console.Error.WriteLine("Usage: inference-vm <weights.bin> <prompt> [maximum-tokens]");
    Console.Error.WriteLine("       inference-vm <weights.bin> --tokens <comma-separated-token-ids> [logits.bin]");
    Environment.ExitCode = 2;
    return;
}

try
{
    var tokenMode = args[1] == "--tokens";
    if (tokenMode && args.Length < 3 || !tokenMode && args.Length > 3)
        throw new ArgumentException("Invalid template arguments.");
    var maximum = !tokenMode && args.Length == 3 ? int.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture) : 32;
    var files = new Dictionary<string, byte[]>(StringComparer.Ordinal);
    var assembly = Assembly.GetExecutingAssembly();
#if GPU_VM
    using var package = assembly.GetManifestResourceStream("VmPackage.program.vm.zip") ??
        throw new InvalidDataException("Missing embedded DXIL VM package.");
    var artifact = D3D12VmArtifact.Import(package);
    var (device, adapter) = D3D12VmDeviceFactory.Create();
    using var deviceOwner = device;
    using var pool = new D3D12VmResourcePool(device);
    Func<IVmExecutable> factory = () => artifact.CreateExecutor(pool);
    Console.Error.WriteLine($"Embedded DXIL VM on {adapter}; no source compiler is invoked.");
#else
    foreach (var name in assembly.GetManifestResourceNames().Where(name => name.StartsWith("VmPackage.", StringComparison.Ordinal)))
    {
        using var source = assembly.GetManifestResourceStream(name) ??
            throw new InvalidDataException($"Missing embedded VM resource '{name}'.");
        using var destination = new MemoryStream();
        source.CopyTo(destination);
        files.Add(name["VmPackage.".Length..], destination.ToArray());
    }
    var artifact = CpuVmCompiledArtifact.LoadFromResources(files);
    Console.Error.WriteLine($"Embedded CPU VM; dynamic code supported: {System.Runtime.CompilerServices.RuntimeFeature.IsDynamicCodeSupported}.");
    Func<IVmExecutable> factory = () => artifact.CreateExecutor(new SharpInference.Generated.CpuProgram());
#endif
    using var catalog = GgmlModelFile.Open(args[0]);
    var modules = new ModelGraphModuleRegistry();
    modules.Register(new Rwkv6ModelModule());
    modules.Register(new Rwkv7ModelModule());
    var architecture = modules.Resolve(catalog).ArchitectureId;
    if (artifact.Program.Abi != $"vm:{architecture}.state.fp32@1")
        throw new InvalidDataException("The embedded VM and weight model have incompatible State ABIs.");
    await using var engine = await VmInferenceEngine.CreateAsync(artifact.Program, "token", "logits",
        (slot, storage) => VmModelBindings.InitializeGlobal(catalog, slot, storage),
        factory, new(16, 16)
#if GPU_VM
        , allocate: pool.Allocate
#endif
    );
#if GPU_VM
    using (var process = System.Diagnostics.Process.GetCurrentProcess())
    {
        var dxcLoaded = process.Modules.Cast<System.Diagnostics.ProcessModule>().Any(module =>
            string.Equals(module.ModuleName, "dxcompiler.dll", StringComparison.OrdinalIgnoreCase));
        if (dxcLoaded) throw new InvalidOperationException("Static DXIL deployment unexpectedly loaded DXC.");
        Console.Error.WriteLine("DXC loaded: False.");
    }
#endif
    await using var session = engine.CreateSession();
    if (tokenMode)
    {
        var tokens = args[2].Split(',').Select(value => int.Parse(value, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
        var logits = await session.PrefillAsync(tokens);
        if (args.Length == 4)
            File.WriteAllBytes(args[3], MemoryMarshal.AsBytes(logits.AsSpan()).ToArray());
        Console.WriteLine("logits: " + string.Join(",", logits.Take(8).Select(value =>
            value.ToString("R", System.Globalization.CultureInfo.InvariantCulture))));
        return;
    }
    var tokenizer = RwkvTokenizerResolver.LoadForArchitecture(architecture);
    if (tokenizer.TokenIds.Any(token => token >= engine.VocabularySize))
        throw new InvalidDataException("This model does not cover the bundled tokenizer; use raw-token mode or a full-vocabulary model.");
    await foreach (var text in RwkvTextGenerator.GenerateAsync(new GenerationSession(session), tokenizer, args[1],
        new RwkvGenerationOptions { MaxTokens = maximum, TopK = 1 }))
        Console.Write(text);
    Console.WriteLine();
}
catch (Exception error) when (error is IOException or ArgumentException or InvalidOperationException or NotSupportedException or FormatException or OverflowException)
{
    Console.Error.WriteLine(error);
    Environment.ExitCode = 1;
}

internal sealed class GenerationSession(VmInferenceSession session) : IAsyncTokenPrefillSession, IScopedTokenGenerationSession
{
    public ReadOnlyMemory<float> ForwardToken(int token) =>
        throw new InvalidOperationException("Use the whole-generation VM lease.");
    public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens) =>
        throw new InvalidOperationException("Use asynchronous prefill.");
    public async ValueTask<ReadOnlyMemory<float>> PrefillAsync(ReadOnlyMemory<int> tokens, CancellationToken cancellationToken = default) =>
        await session.PrefillAsync(tokens, cancellationToken).ConfigureAwait(false);
    public async ValueTask<ITokenGenerationScope> BeginGenerationAsync(CancellationToken cancellationToken = default) =>
        new GenerationScope(await session.BeginGenerationAsync(cancellationToken).ConfigureAwait(false));
    private sealed class GenerationScope(VmGenerationLease lease) : ITokenGenerationScope, IAsyncTokenGenerationSession
    {
        public ITokenGenerationSession Session => this;
        public ReadOnlyMemory<float> ForwardToken(int token) => lease.ForwardToken(token);
        public async ValueTask<ReadOnlyMemory<float>> ForwardTokenAsync(int token,
            CancellationToken cancellationToken = default) =>
            await lease.ForwardTokenAsync(token, cancellationToken).ConfigureAwait(false);
        public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokens) =>
            throw new InvalidOperationException("Prefill cannot run inside a generation lease.");
        public ValueTask DisposeAsync() => lease.DisposeAsync();
    }
}
