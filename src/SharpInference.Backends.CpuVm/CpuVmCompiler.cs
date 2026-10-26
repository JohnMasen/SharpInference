using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;
using SharpInference.Instructions;
using SharpInference.Vm;

namespace SharpInference.Backends.CpuVm;

/// <summary>Defines the entry-point contract implemented by generated or statically published CPU VM code.</summary>
public interface ICpuVmCode
{
    /// <summary>Gets the program-and-source-ABI hash used to verify compatibility with a compiled artifact.</summary>
    string ProgramHash { get; }
    /// <summary>Invokes a compiled entry using the slot buffers and active call frames of the execution context.</summary>
    /// <param name="entry">The name of the compiled entry to invoke or inspect.</param>
    /// <param name="context">The execution context containing the bound slot buffers.</param>
    void Invoke(string entry, CpuVmContext context);
}

/// <summary>Configures CPU code optimization and numerical precision for VM compilation.</summary>
/// <param name="Optimize">Whether Roslyn emits optimized release code instead of debug code.</param>
public sealed record CpuVmCompilerOptions(bool Optimize = true)
{
    /// <summary>Gets the arithmetic precision requested for compilation; currently only Float32 is supported.</summary>
    public string Arithmetic { get; init; } = "Float32";
    /// <summary>Gets the accumulation precision requested for compilation; currently only Float32 is supported.</summary>
    public string Accumulator { get; init; } = "Float32";
}

/// <summary>Reports errors emitted by Roslyn while compiling generated CPU VM source.</summary>
/// <param name="diagnostics">The compilation error messages reported by Roslyn.</param>
public sealed class CpuVmCompilationException(IEnumerable<string> diagnostics) :
    InvalidOperationException(string.Join(Environment.NewLine, diagnostics))
{
    /// <summary>Gets the compiler error messages that prevented generation of an executable assembly.</summary>
    public IReadOnlyList<string> Diagnostics { get; } = diagnostics.ToArray();
}

/// <summary>Compiles definitions once. Execution contains direct typed calls, never node dispatch.</summary>
public sealed class CpuVmCompiler
{
    private readonly InstructionRegistry registry;

    /// <summary>Creates a compiler with explicitly supplied CPU instruction collection providers.</summary>
    /// <param name="providers">The instruction collection providers used to resolve program operators.</param>
    public CpuVmCompiler(IEnumerable<IInstructionCollectionProvider> providers) =>
        registry = new InstructionRegistry(providers);
    /// <summary>Gets the registry of instruction collections supplied to this compiler.</summary>
    public IInstructionCollectionProvider InstructionCollections => registry;

    /// <summary>Validates a CPU VM program and generates C# source and binding metadata without compiling an assembly.</summary>
    /// <param name="program">The VM program whose definitions, slots, and entries are compiled or bound.</param>
    /// <param name="options">Compilation settings, or null to use the default settings.</param>
    public CpuVmCompiledArtifact GenerateSource(VmProgram program, CpuVmCompilerOptions? options = null)
    {
        options ??= new();
        if (options.Arithmetic != "Float32" || options.Accumulator != "Float32")
        {
            throw new NotSupportedException("CPU VM supports Float32 arithmetic and accumulation only.");
        }

        var contracts = VmInstructionContracts.Bind(program, registry);
        var generator = new CpuVmSourceGenerator(program, registry, contracts);
        var source = generator.Generate();
        return new CpuVmCompiledArtifact(program, options, source, null, null, generator.Accesses, generator.Frames,
            generator.References, contracts);
    }

    /// <summary>Generates and compiles CPU VM source into an in-memory assembly with portable debug symbols.</summary>
    /// <param name="program">The VM program whose definitions, slots, and entries are compiled or bound.</param>
    /// <param name="options">Compilation settings, or null to use the default settings.</param>
    [RequiresDynamicCode("Runtime compilation is unavailable under Native AOT. Publish generated source instead.")]
    [RequiresAssemblyFiles("Roslyn requires on-disk reference assemblies. Use static source publication in single-file applications.")]
    public CpuVmCompiledArtifact Compile(VmProgram program, CpuVmCompilerOptions? options = null)
    {
        if (!RuntimeFeature.IsDynamicCodeSupported)
        {
            throw new PlatformNotSupportedException("Use GenerateSource and statically register ICpuVmCode under Native AOT.");
        }

        var artifact = GenerateSource(program, options);
        var paths = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES"))?.Split(Path.PathSeparator)
            ?? throw new PlatformNotSupportedException("Runtime reference assemblies are unavailable.");
        var assemblyPaths = paths.Concat(artifact.References.Select(assembly => assembly.Location))
            .Concat([typeof(CpuVmContext).Assembly.Location, typeof(VmProgram).Assembly.Location])
            .Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        if (assemblyPaths.Any(string.IsNullOrEmpty))
        {
            throw new PlatformNotSupportedException("Compilation requires on-disk assemblies. Publish generated source for single-file applications.");
        }

        var references = assemblyPaths.Select(path => MetadataReference.CreateFromFile(path));
        var syntax = CSharpSyntaxTree.ParseText(artifact.Source, path: "CpuProgram.g.cs", encoding: Encoding.UTF8);
        var compilation = CSharpCompilation.Create("CpuVm_" + Guid.NewGuid().ToString("N"), [syntax], references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary,
                optimizationLevel: artifact.Options.Optimize ? OptimizationLevel.Release : OptimizationLevel.Debug,
                deterministic: true, nullableContextOptions: NullableContextOptions.Enable));
        using var dll = new MemoryStream();
        using var pdb = new MemoryStream();
        var result = compilation.Emit(dll, pdb, options: new EmitOptions(debugInformationFormat: DebugInformationFormat.PortablePdb));
        if (!result.Success)
        {
            throw new CpuVmCompilationException(result.Diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).Select(d => d.ToString()));
        }

        return artifact.WithBinary(dll.ToArray(), pdb.ToArray());
    }
}

/// <summary>Contains a CPU VM program, generated source, binding metadata, and optional compiled assembly.</summary>
public sealed class CpuVmCompiledArtifact
{
    /// <summary>Identifies the generated CPU code contract included in program compatibility hashes.</summary>
    public const string SourceAbi = "cpu-vm-ic-precision-v4";
    internal IReadOnlyList<System.Reflection.Assembly> References { get; }
    private readonly VmProgram contracts;
    private readonly byte[]? binary;
    private readonly byte[]? symbols;
    private readonly Dictionary<string, List<CpuVmAccess>> accesses;
    private readonly Dictionary<string, CpuVmCallFrame> frames;
    internal CpuVmCompiledArtifact(VmProgram program, CpuVmCompilerOptions options, string source,
        byte[]? binary, byte[]? symbols, Dictionary<string, List<CpuVmAccess>> accesses,
        Dictionary<string, CpuVmCallFrame> frames, IReadOnlyList<System.Reflection.Assembly>? references = null,
        VmProgram? contracts = null)
    {
        Program = program;
        Options = options;
        Source = source;
        this.binary = binary;
        this.symbols = symbols;
        this.accesses = accesses;
        this.frames = frames;
        References = references ?? [];
        this.contracts = contracts ?? program;
    }

    /// <summary>Gets the VM program represented by this compilation artifact.</summary>
    public VmProgram Program { get; }
    /// <summary>Gets the compilation options used to produce this artifact.</summary>
    public CpuVmCompilerOptions Options { get; }
    /// <summary>Gets the generated C# source implementing the CPU VM entry points.</summary>
    public string Source { get; }
    /// <summary>Gets the hash of the serialized program and generated-source ABI.</summary>
    public string ProgramHash => ComputeProgramHash(Program);
    /// <summary>Gets whether this artifact contains a compiled assembly.</summary>
    public bool HasBinary => binary is not null;
    /// <summary>Gets a defensive copy of the compiled assembly bytes, or empty memory for a source-only artifact.</summary>
    public ReadOnlyMemory<byte> Binary => binary is null ? ReadOnlyMemory<byte>.Empty : binary.ToArray();
    /// <summary>Gets a defensive copy of the portable debug symbols, or empty memory when no symbols are present.</summary>
    public ReadOnlyMemory<byte> PortablePdb => symbols is null ? ReadOnlyMemory<byte>.Empty : symbols.ToArray();

    internal CpuVmCompiledArtifact WithBinary(byte[] dll, byte[] pdb) =>
        new(Program, Options, Source, dll, pdb, accesses, frames, References, contracts);

    /// <summary>Register published code directly; this path works without reflection or dynamic code.</summary>
    public CpuVmExecutable CreateExecutable(ICpuVmCode code)
    {
        ArgumentNullException.ThrowIfNull(code);
        if (code.ProgramHash != ProgramHash)
        {
            throw new InvalidDataException("Static code has a different program or source ABI. Regenerate source and rebuild.");
        }

        return new CpuVmExecutable(this, code, accesses, frames, null);
    }

    /// <summary>Creates an executable by registering static CPU code and verifying its program-and-source-ABI hash.</summary>
    /// <param name="code">Statically registered CPU code whose program and source ABI must match this artifact.</param>
    public CpuVmExecutable CreateExecutor(ICpuVmCode code) => CreateExecutable(code);

    /// <summary>Loads the artifact's compiled assembly into a collectible context and creates an executable.</summary>
    [RequiresDynamicCode("Binary loading is unavailable under Native AOT.")]
    [RequiresUnreferencedCode("Use static source registration for trimmed applications.")]
    public CpuVmExecutable CreateExecutor() => LoadExecutable();

    /// <summary>Loads the compiled assembly into a collectible load context and verifies its program hash.</summary>
    [RequiresDynamicCode("Binary loading is unavailable under Native AOT.")]
    [RequiresUnreferencedCode("The generated entry class is resolved by name. Use static code registration for trimmed applications.")]
    public CpuVmExecutable LoadExecutable()
    {
        if (!RuntimeFeature.IsDynamicCodeSupported)
        {
            throw new PlatformNotSupportedException("Use static source registration under Native AOT.");
        }

        if (binary is null)
        {
            throw new InvalidOperationException("This is a source-only artifact.");
        }

        var loader = new AssemblyLoadContext("CpuVm_" + Guid.NewGuid().ToString("N"), isCollectible: true);
        try
        {
            using var dll = new MemoryStream(binary, writable: false);
            using var pdb = new MemoryStream(symbols!, writable: false);
            var assembly = loader.LoadFromStream(dll, pdb);
            var type = assembly.GetType("SharpInference.Generated.CpuProgram", throwOnError: true)!;
            var code = (ICpuVmCode)(Activator.CreateInstance(type) ?? throw new InvalidDataException("Missing generated CPU entry."));
            if (code.ProgramHash != ProgramHash)
            {
                throw new InvalidDataException("Binary program hash mismatch.");
            }

            return new CpuVmExecutable(this, code, accesses, frames, loader);
        }
        catch { loader.Unload(); throw; }
    }

    /// <summary>Exports actual C# and optionally assembly/PDB, plus XML, options and SHA256 integrity manifest.</summary>
    public void Export(string directory, bool includeBinary = true)
    {
        if (includeBinary && binary is null)
        {
            throw new InvalidOperationException("Compile before exporting a binary package.");
        }

        VmInstructionContracts.ValidateResolved(contracts);
        Directory.CreateDirectory(directory);
        var files = new Dictionary<string, byte[]>
        {
            ["program.xml"] = Encoding.UTF8.GetBytes(VmProgramXml.Serialize(Program)),
            ["contracts.xml"] = Encoding.UTF8.GetBytes(VmProgramXml.Serialize(contracts)),
            ["options.json"] = JsonSerializer.SerializeToUtf8Bytes(Options, CpuVmJsonContext.Default.CpuVmCompilerOptions),
            ["CpuProgram.g.cs"] = Encoding.UTF8.GetBytes(Source),
        };
        if (includeBinary)
        {
            files.Add("CpuProgram.dll", binary!);
            files.Add("CpuProgram.pdb", symbols!);
        }
        foreach (var pair in files)
        {
            File.WriteAllBytes(Path.Combine(directory, pair.Key), pair.Value);
        }

        File.WriteAllText(Path.Combine(directory, "manifest.json"),
            JsonSerializer.Serialize(new CpuVmPackageManifest(3, files.ToDictionary(pair => pair.Key, pair => Hash(pair.Value))),
                CpuVmJsonContext.Default.CpuVmPackageManifest));
    }

    /// <summary>Loads and validates package contents. Never invokes Roslyn, including for source-only packages.</summary>
    public static CpuVmCompiledArtifact Load(string directory) =>
        LoadPackage(name => File.ReadAllBytes(Path.Combine(directory, name)));

    /// <summary>Loads embedded package bytes without filesystem access or Roslyn. Keys are the exported filenames.</summary>
    public static CpuVmCompiledArtifact LoadFromResources(IReadOnlyDictionary<string, byte[]> files)
    {
        ArgumentNullException.ThrowIfNull(files);
        return LoadPackage(name =>
        {
            if (!files.TryGetValue(name, out var bytes) || bytes is null)
            {
                throw new InvalidDataException($"Missing CPU package resource: {name}.");
            }

            return bytes.ToArray();
        });
    }

    private static CpuVmCompiledArtifact LoadPackage(Func<string, byte[]> readFile)
    {
        var manifest = JsonSerializer.Deserialize(readFile("manifest.json"),
            CpuVmJsonContext.Default.CpuVmPackageManifest)
            ?? throw new InvalidDataException("Missing CPU package manifest.");
        var required = new[] { "program.xml", "contracts.xml", "options.json", "CpuProgram.g.cs" };
        if (manifest.Version != 3 || manifest.Files is null ||
            required.Any(name => !manifest.Files.ContainsKey(name)) ||
            manifest.Files.Keys.Any(name => !required.Contains(name) && name is not ("CpuProgram.dll" or "CpuProgram.pdb")) ||
            manifest.Files.ContainsKey("CpuProgram.dll") != manifest.Files.ContainsKey("CpuProgram.pdb"))
        {
            throw new InvalidDataException("Invalid CPU package manifest.");
        }

        var files = manifest.Files.ToDictionary(pair => pair.Key, pair =>
        {
            var bytes = readFile(pair.Key);
            if (!string.Equals(Hash(bytes), pair.Value, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"CPU package integrity failure: {pair.Key}.");
            }

            return bytes;
        });
        var program = VmProgramXml.Deserialize(Encoding.UTF8.GetString(files["program.xml"]));
        var options = JsonSerializer.Deserialize(files["options.json"], CpuVmJsonContext.Default.CpuVmCompilerOptions)
            ?? throw new InvalidDataException("Invalid CPU compiler options.");
        var source = Encoding.UTF8.GetString(files["CpuProgram.g.cs"]);
        var contracts = VmProgramXml.Deserialize(Encoding.UTF8.GetString(files["contracts.xml"]));
        VmInstructionContracts.ValidateResolved(contracts);
        if (VmProgramXml.Serialize(VmInstructionContracts.WithoutContracts(program)) !=
            VmProgramXml.Serialize(VmInstructionContracts.WithoutContracts(contracts)) ||
            contracts.Definitions.SelectMany(definition => definition.Nodes).Select(node => node.Instruction)
                .OfType<VmOperator>().Any(operation => operation.ParameterAccesses.Count == 0))
        {
            throw new InvalidDataException("CPU package instruction contracts differ from its program.");
        }

        if (options.Arithmetic != "Float32" || options.Accumulator != "Float32" ||
            !source.Contains($"public string ProgramHash => \"{ComputeProgramHash(program)}\";", StringComparison.Ordinal))
        {
            throw new InvalidDataException("CPU source ABI, program identity or compiler options mismatch.");
        }

        var bindings = new CpuVmSourceGenerator(program, new InstructionRegistry([]), contracts);
        bindings.PrepareBindings();
        return new CpuVmCompiledArtifact(program, options, source,
            files.GetValueOrDefault("CpuProgram.dll"), files.GetValueOrDefault("CpuProgram.pdb"), bindings.Accesses, bindings.Frames,
            contracts: contracts);
    }

    internal static string Hash(byte[] value) => Convert.ToHexString(SHA256.HashData(value));
    internal static string ComputeProgramHash(VmProgram program) =>
        Hash(Encoding.UTF8.GetBytes(SourceAbi + "\n" + VmProgramXml.Serialize(program)));
}

internal sealed record CpuVmPackageManifest(int Version, Dictionary<string, string> Files);

[JsonSerializable(typeof(CpuVmPackageManifest))]
[JsonSerializable(typeof(CpuVmCompilerOptions))]
[JsonSerializable(typeof(string))]
internal partial class CpuVmJsonContext : JsonSerializerContext;

/// <summary>Executes compiled CPU VM entries with validated slot access and call-frame bindings.</summary>
public sealed class CpuVmExecutable : IVmTaskExecutable
{
    private readonly CpuVmCompiledArtifact artifact;
    private ICpuVmCode? code;
    private readonly Dictionary<string, List<CpuVmAccess>> accesses;
    private readonly Dictionary<string, CpuVmCallFrame> frames;
    private readonly AssemblyLoadContext? loader;

    internal CpuVmExecutable(CpuVmCompiledArtifact artifact, ICpuVmCode code,
        Dictionary<string, List<CpuVmAccess>> accesses, Dictionary<string, CpuVmCallFrame> frames,
        AssemblyLoadContext? loader)
    {
        this.artifact = artifact;
        this.code = code;
        this.accesses = accesses;
        this.frames = frames;
        this.loader = loader;
    }

    /// <inheritdoc/>
    public VmProgram Program => artifact.Program;
    /// <summary>Creates an execution context over caller-owned slot buffers without allocating tensor storage.</summary>
    /// <param name="buffers">Caller-owned byte arrays in program slot order, each matching its slot&apos;s exact capacity.</param>
    /// <param name="initializedLocalSlots">Names of local slots supplied with initialized contents; omitted local slots must be written before reading.</param>
    public CpuVmContext Prepare(byte[][] buffers, IEnumerable<string>? initializedLocalSlots = null)
    {
        return new(Program, buffers, initializedLocalSlots);
    }

    /// <summary>Read-only local slots are supplied inputs; writable local scratch must initialize in the entry.</summary>
    public void Execute(string entryName, byte[][] slots)
    {
        Invoke(entryName, Prepare(slots, LocalInputs()));
    }

    /// <inheritdoc/>
    public void Execute(string entryName, VmExecutionLease execution)
    {
        ArgumentNullException.ThrowIfNull(execution);
        Invoke(entryName, new CpuVmContext(execution.Program, execution.GetBuffers(), LocalInputs()));
    }

    /// <inheritdoc/>
    public void ExecuteTask(string entryName, VmExecutionLease execution, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        Execute(entryName, execution);
        cancellation.ThrowIfCancellationRequested();
    }

    private IEnumerable<string> LocalInputs() => Program.Slots
        .Where(slot => slot.Scope == VmSlotScope.Local && slot.Access == VmAccess.ReadOnly).Select(slot => slot.Id);

    /// <summary>Runs a compiled entry after validating context compatibility, access permissions, and local initialization.</summary>
    /// <param name="entry">The name of the compiled entry to invoke or inspect.</param>
    /// <param name="context">The execution context containing the bound slot buffers.</param>
    public void Invoke(string entry, CpuVmContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        var current = code ?? throw new ObjectDisposedException(nameof(CpuVmExecutable));
        if (!accesses.TryGetValue(entry, out var plan))
        {
            throw new ArgumentException($"Unknown entry '{entry}'.", nameof(entry));
        }

        context.Run(Program, plan, frames, () => current.Invoke(entry, context));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref code, null) is not null)
        {
            loader?.Unload();
        }
    }
}
