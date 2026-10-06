using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using SharpInference.Vm;
using Vortice.Direct3D12;

namespace SharpInference.Backends.D3D12Vm;

public sealed class D3D12VmKernel
{
    private readonly byte[] dxil;
    public D3D12VmKernel(string definition, string source, byte[] dxil)
    {
        Definition = definition;
        Source = source;
        this.dxil = (byte[])dxil.Clone();
    }
    public string Definition { get; }
    public string Source { get; }
    public byte[] GetDxil() => (byte[])dxil.Clone();
    internal ReadOnlyMemory<byte> Bytecode => dxil;
}

/// <summary>
/// ZIP artifact containing program XML, generated HLSL, DXIL, target/options and SHA256 integrity.
/// Import validates hashes, kernel membership and index contracts without IC providers or DXC.
/// Checksums detect corruption; they are not a signature or a trust boundary.
/// </summary>
public sealed class D3D12VmArtifact
{
    public const string Abi = "SharpInference.D3D12Vm.raw-uav.ic-precision.v3";
    internal D3D12VmArtifact(VmProgram program, D3D12VmCompilerOptions options, IEnumerable<D3D12VmKernel> kernels,
        VmProgram? contracts = null)
    {
        Program = program;
        Options = options;
        Kernels = Array.AsReadOnly(kernels.ToArray());
        Contracts = contracts ?? program;
    }
    public VmProgram Program { get; }
    public D3D12VmCompilerOptions Options { get; }
    public IReadOnlyList<D3D12VmKernel> Kernels { get; }
    internal VmProgram Contracts { get; }

    /// <summary>Creates an independent VM and an owned hardware device. No source compilation occurs.</summary>
    public D3D12VmExecutor CreateExecutor() => CreateExecutor(0);

    public D3D12VmExecutor CreateExecutor(int adapterIndex)
    {
        var (device, _) = D3D12VmDeviceFactory.Create(adapterIndex);
        return D3D12VmExecutor.CreateOwned(device, this);
    }

    /// <summary>
    /// Creates an independent VM whose read-only global slots must all be explicitly uploaded before use.
    /// This avoids zero-initializing large resident model weights.
    /// </summary>
    public D3D12VmExecutor CreateExecutorForExplicitGlobalUploads(int adapterIndex = 0)
    {
        var (device, _) = D3D12VmDeviceFactory.Create(adapterIndex);
        return D3D12VmExecutor.CreateOwned(device, this, skipReadOnlyGlobalInitialization: true);
    }

    /// <summary>
    /// Creates an independent VM without initializing any slot. The caller must upload every
    /// control/global value before it is read, and the program must overwrite scratch/state before reading it.
    /// </summary>
    public D3D12VmExecutor CreateExecutorForExplicitInitialization(int adapterIndex = 0)
    {
        var (device, _) = D3D12VmDeviceFactory.Create(adapterIndex);
        return D3D12VmExecutor.CreateOwned(
            device, this, skipReadOnlyGlobalInitialization: true, skipAllInitialization: true);
    }

    /// <summary>
    /// Creates an independent VM on a caller-owned device. Sharing the device does not share
    /// GPU weight allocations; each VM owns all of its slot resources.
    /// </summary>
    public D3D12VmExecutor CreateExecutor(ID3D12Device device) => new(device, this);

    /// <summary>Creates a VM binding allocator-owned global/session/local GPU resources from the shared pool.</summary>
    public D3D12VmExecutor CreateExecutor(D3D12VmResourcePool pool) => new(pool, this);

    private sealed record Module(string Definition, string SourcePath, string DxilPath, string SourceHash, string DxilHash);
    private sealed record Manifest(int Version, string BackendAbi, string Target,
        D3D12VmCompilerOptions Options, string ProgramHash, Module[] Modules, string ContractsHash);
    private sealed record Envelope(string Manifest, string Sha256);

    public void Export(Stream destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        VmInstructionContracts.ValidateResolved(Contracts);
        using var zip = new ZipArchive(destination, ZipArchiveMode.Create, leaveOpen: true);
        var xml = Encoding.UTF8.GetBytes(VmProgramXml.Serialize(Program));
        Write(zip, "program.xml", xml);
        var contracts = Encoding.UTF8.GetBytes(VmProgramXml.Serialize(Contracts));
        Write(zip, "contracts.xml", contracts);
        var modules = Kernels.Select((k, i) =>
        {
            var source = Encoding.UTF8.GetBytes(k.Source);
            var dxil = k.GetDxil();
            var s = $"kernels/{i}.hlsl";
            var d = $"kernels/{i}.dxil";
            Write(zip, s, source);
            Write(zip, d, dxil);
            return new Module(k.Definition, s, d, Hash(source), Hash(dxil));
        }).ToArray();
        var manifest = JsonSerializer.Serialize(new Manifest(3, Abi, "Direct3D12", Options, Hash(xml), modules, Hash(contracts)));
        Write(zip, "manifest.json", JsonSerializer.SerializeToUtf8Bytes(new Envelope(manifest, Hash(Encoding.UTF8.GetBytes(manifest)))));
    }

    public static D3D12VmArtifact Import(Stream source)
    {
        ArgumentNullException.ThrowIfNull(source);
        using var zip = new ZipArchive(source, ZipArchiveMode.Read, leaveOpen: true);
        if (zip.Entries.Count > 10000 || zip.Entries.Sum(e => e.Length) > 512L * 1024 * 1024 ||
            zip.Entries.Any(e => e.Length > 64L * 1024 * 1024) ||
            zip.Entries.Select(e => e.FullName).Distinct(StringComparer.Ordinal).Count() != zip.Entries.Count)
            throw new InvalidDataException("Invalid or oversized D3D12 VM archive.");
        var envelope = JsonSerializer.Deserialize<Envelope>(Read(zip, "manifest.json")) ??
            throw new InvalidDataException("Missing artifact manifest.");
        if (envelope.Manifest is null || envelope.Sha256 != Hash(Encoding.UTF8.GetBytes(envelope.Manifest)))
            throw new InvalidDataException("Manifest integrity check failed.");
        var manifest = JsonSerializer.Deserialize<Manifest>(envelope.Manifest) ??
            throw new InvalidDataException("Invalid artifact manifest.");
        if (manifest.Version != 3 || manifest.BackendAbi != Abi || manifest.Target != "Direct3D12" ||
            manifest.Options is null || manifest.Modules is null)
            throw new InvalidDataException("Unsupported D3D12 VM artifact target/version/ABI.");
        D3D12VmCompiler.ValidateOptions(manifest.Options);
        var xml = Read(zip, "program.xml");
        Verify(xml, manifest.ProgramHash);
        var program = VmProgramXml.Deserialize(Encoding.UTF8.GetString(xml));
        var contractsXml = Read(zip, "contracts.xml");
        Verify(contractsXml, manifest.ContractsHash);
        var contracts = VmProgramXml.Deserialize(Encoding.UTF8.GetString(contractsXml));
        VmInstructionContracts.ValidateResolved(contracts);
        if (VmProgramXml.Serialize(VmInstructionContracts.WithoutContracts(program)) !=
            VmProgramXml.Serialize(VmInstructionContracts.WithoutContracts(contracts)))
            throw new InvalidDataException("Artifact contracts differ from the execution program.");
        var kernelsInProgram = program.Definitions.Where(definition => definition.Kind == VmDefinitionKind.Kernel)
            .Select(definition => definition.Id).ToHashSet(StringComparer.Ordinal);
        if (manifest.Modules.Length != kernelsInProgram.Count ||
            !kernelsInProgram.SetEquals(manifest.Modules.Select(module => module.Definition)))
            throw new InvalidDataException("Artifact kernel set differs from the program.");
        var paths = new HashSet<string>(StringComparer.Ordinal) { "manifest.json", "program.xml", "contracts.xml" };
        var kernels = new List<D3D12VmKernel>();
        foreach (var m in manifest.Modules)
        {
            if (!paths.Add(m.SourcePath) || !paths.Add(m.DxilPath))
                throw new InvalidDataException("Duplicate artifact module path.");
            var text = Read(zip, m.SourcePath);
            var dxil = Read(zip, m.DxilPath);
            Verify(text, m.SourceHash);
            Verify(dxil, m.DxilHash);
            var hlsl = Encoding.UTF8.GetString(text);
            if (string.IsNullOrWhiteSpace(hlsl) || dxil.Length < 4 || !dxil.AsSpan(0, 4).SequenceEqual("DXBC"u8))
                throw new InvalidDataException("Artifact source or DXIL container does not match its program.");
            kernels.Add(new(m.Definition, hlsl, dxil));
        }
        if (zip.Entries.Any(e => !paths.Contains(e.FullName)))
            throw new InvalidDataException("Unexpected artifact entries.");
        return new(program, manifest.Options, kernels, contracts);
    }


    private static string Hash(byte[] data) => Convert.ToHexString(SHA256.HashData(data));
    private static void Verify(byte[] data, string expected)
    {
        if (Hash(data) != expected) throw new InvalidDataException("Artifact integrity check failed.");
    }
    private static void Write(ZipArchive zip, string path, byte[] data)
    {
        using var stream = zip.CreateEntry(path, CompressionLevel.Optimal).Open();
        stream.Write(data);
    }
    private static byte[] Read(ZipArchive zip, string path)
    {
        var entry = zip.GetEntry(path) ?? throw new InvalidDataException($"Artifact entry '{path}' is missing.");
        using var stream = entry.Open();
        var data = new byte[checked((int)entry.Length)];
        stream.ReadExactly(data);
        return data;
    }
}
