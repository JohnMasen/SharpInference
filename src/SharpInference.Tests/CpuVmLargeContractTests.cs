using System.Diagnostics;
using System.Xml.Linq;

namespace SharpInference.Tests;

public sealed class CpuVmLargeContractTests
{
    [Fact]
    public async Task ExecutesLargeFixedSlotPrefillOnDefaultThreadPoolStackInChildProcess()
    {
        var directory = Path.Combine(Path.GetTempPath(), "cpuvm-large-contract-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var paths = ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator)
                .Where(path => Path.GetFileNameWithoutExtension(path) is "SharpInference.Backends.CpuVm" or
                    "SharpInference.Backends.Cpu" or "SharpInference.Vm" or "SharpInference.Graphs" or
                    "SharpInference.Instructions" or "SharpInference.Instructions.Cpu" or "SharpInference.Instructions.TierZero" or
                    "SharpInference.Abstractions" or "System.Numerics.Tensors" or
                    "Microsoft.CodeAnalysis" or "Microsoft.CodeAnalysis.CSharp");
            var project = new XElement("Project", new XAttribute("Sdk", "Microsoft.NET.Sdk"),
                new XElement("PropertyGroup", new XElement("TargetFramework", "net10.0"),
                    new XElement("OutputType", "Exe"), new XElement("ImplicitUsings", "enable")),
                new XElement("ItemGroup", paths.Select(path => new XElement("Reference",
                    new XAttribute("Include", Path.GetFileNameWithoutExtension(path)), new XElement("HintPath", path)))));
            var projectPath = Path.Combine(directory, "LargeContract.csproj");
            File.WriteAllText(projectPath, project.ToString());
            File.WriteAllText(Path.Combine(directory, "Program.cs"), """
                using System.Runtime.InteropServices;
                using System.Reflection;
                using SharpInference.Backends.CpuVm;
                using SharpInference.Vm;
                var scalar = new VmTensor(VmElementType.Float32, [1]);
                var token = new VmTensor(VmElementType.Int32, [1]);
                var slots = new List<VmSlot> {
                    new("tokens", VmSlotScope.Local, VmAccess.ReadOnly, new VmTensor(VmElementType.Int32, [64])),
                    new("table", VmSlotScope.Global, VmAccess.ReadOnly, new VmTensor(VmElementType.Float32, [64, 1])),
                    new("output", VmSlotScope.Local, VmAccess.ReadWrite, scalar) };
                for (int i = slots.Count; i < 1200; i++)
                    slots.Add(new("fixed." + i, VmSlotScope.Global, VmAccess.ReadOnly, scalar));
                var leaf = new VmDefinition("gather", VmDefinitionKind.Function,
                    [new("table", VmAccess.ReadOnly, slots[1].Tensor), new("index", VmAccess.ReadOnly, token),
                     new("output", VmAccess.ReadWrite, scalar)],
                    [new("gather", new VmOperator("core.gather-row", 1,
                        [new("table", "table"), new("index", "index"), new("output", "output")]))]);
                var forward = new VmDefinition("forward", VmDefinitionKind.Orchestration,
                    slots.Select(s => new VmParameter(s.Id, s.Access, s.Id == "tokens" ? token : s.Tensor)),
                    [new("gather", new VmCall("gather",
                        [new("table", "table"), new("index", "tokens"), new("output", "output")]))]);
                var prefill = new VmDefinition("prefill.64", VmDefinitionKind.Orchestration,
                    slots.Select(s => new VmParameter(s.Id, s.Access, s.Tensor)),
                    Enumerable.Range(0, 64).Select(i => new VmNode("token." + i,
                        new VmCall("forward", slots.Select(s => new VmArgument(s.Id, s.Id,
                            s.Id == "tokens" ? (ulong)(i * sizeof(int)) : 0))),
                        i == 0 ? [] : ["token." + (i - 1)])));
                var program = new VmProgram("large-contract", "cpu", VmTarget.Cpu, slots,
                    [leaf, forward, prefill],
                    [new("forward", "forward", slots.Select(s => new VmArgument(s.Id, s.Id))),
                     new("prefill.64", "prefill.64", slots.Select(s => new VmArgument(s.Id, s.Id)))],
                    new VmState("none", 1, []));
                var artifact = new CpuVmCompiler([new SharpInference.Instructions.Cpu.CpuFloat32InstructionCollection()]).Compile(program);
                var generated = Assembly.Load(artifact.Binary.ToArray()).GetType("SharpInference.Generated.CpuProgram", true);
                foreach (var method in generated.GetMethods(BindingFlags.NonPublic | BindingFlags.Static))
                {
                    var parameters = method.GetParameters();
                    if (parameters.Length != 2 || parameters[0].ParameterType != typeof(CpuVmContext) ||
                        parameters[1].ParameterType != typeof(CpuVmCallFrame))
                        throw new Exception("Generated method arguments scale with slot count: " + method.Name);
                }
                var buffers = slots.Select(s => new byte[(int)s.Tensor.ByteLength]).ToArray();
                for (int i = 0; i < 64; i++) {
                    MemoryMarshal.Cast<byte, int>(buffers[0].AsSpan())[i] = i;
                    MemoryMarshal.Cast<byte, float>(buffers[1].AsSpan())[i] = i + 1;
                }
                using var executor = artifact.CreateExecutor();
                Task.Run(() => executor.Execute("prefill.64", buffers)).GetAwaiter().GetResult();
                if (MemoryMarshal.Cast<byte, float>(buffers[2])[0] != 64) throw new Exception("Incorrect prefill offset.");
                Task.Run(() => executor.Execute("forward", buffers)).GetAwaiter().GetResult();
                if (MemoryMarshal.Cast<byte, float>(buffers[2])[0] != 1) throw new Exception("Incorrect forward entry.");
                var package = Path.Combine(Environment.CurrentDirectory, "package");
                artifact.Export(package);
                using var reloaded = CpuVmCompiledArtifact.Load(package).CreateExecutor();
                Task.Run(() => reloaded.Execute("prefill.64", buffers)).GetAwaiter().GetResult();
                if (MemoryMarshal.Cast<byte, float>(buffers[2])[0] != 64) throw new Exception("Incorrect binary reload.");
                Console.WriteLine("1200 parameters; prefill.64; default ThreadPool stack; correct.");
                """);
            var start = new ProcessStartInfo("dotnet")
            {
                WorkingDirectory = directory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            start.Environment["DOTNET_TieredCompilation"] = "0";
            foreach (var argument in new[] { "run", "--project", projectPath, "--configuration", "Release", "--no-launch-profile" })
                start.ArgumentList.Add(argument);
            using var process = Process.Start(start) ?? throw new InvalidOperationException("Could not launch dotnet SDK.");
            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();
            using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(2));
            try { await process.WaitForExitAsync(timeout.Token); }
            catch (OperationCanceledException)
            {
                process.Kill(entireProcessTree: true);
                await process.WaitForExitAsync();
                throw new TimeoutException("Large-contract child validation timed out.");
            }
            var output = await stdout;
            var errors = await stderr;
            if (errors.Length > 4096) errors = errors[..2048] + "\n...\n" + errors[^2048..];
            Assert.True(process.ExitCode == 0, $"Child exit code: {process.ExitCode}\n{output}\n{errors}");
            Assert.Contains("1200 parameters; prefill.64; default ThreadPool stack; correct.", output);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }
}
