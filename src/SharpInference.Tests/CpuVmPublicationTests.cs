using SharpInference.Graphs;
using System.Diagnostics;
using System.Xml.Linq;
using SharpInference.Backends.CpuVm;
using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class CpuVmPublicationTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task BuildsExportedSourceWithSdkAndLoadsSourcePackageWithoutRoslyn(bool embedded)
    {
        var tensor = new VmTensor(VmElementType.Float32, [2]);
        var parameters = new[] { new VmParameter("left", VmAccess.ReadOnly, tensor),
            new VmParameter("right", VmAccess.ReadOnly, tensor), new VmParameter("output", VmAccess.ReadWrite, tensor) };
        var definition = new VmDefinition("add", VmDefinitionKind.Function, parameters,
            [new VmNode("add", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), "core.add", 1, parameters.Select(p => new VmArgument(p.Name, p.Name))))]);
        var program = new VmProgram("static-add", "cpu", VmTarget.Cpu,
            parameters.Select(p => new VmSlot(p.Name, p.Access == VmAccess.ReadOnly ? VmSlotScope.Global : VmSlotScope.Local, p.Access, p.Tensor)),
            [definition], [new("run", "add", parameters.Select(p => new VmArgument(p.Name, p.Name)))], new VmState("none", 1, []));
        var directory = Path.Combine(Path.GetTempPath(), "cpuvm-source-" + Guid.NewGuid().ToString("N"));
        try
        {
            new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).GenerateSource(program).Export(directory, includeBinary: false);
            var paths = ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator)
                .Where(path => Path.GetFileNameWithoutExtension(path) is "SharpInference.Backends.CpuVm" or
                    "SharpInference.Backends.Cpu" or "SharpInference.Vm" or "SharpInference.Graphs" or
                    "SharpInference.Instructions" or "SharpInference.Instructions.Cpu" or
                    "SharpInference.Abstractions" or "System.Numerics.Tensors").ToArray();
            var project = new XElement("Project", new XAttribute("Sdk", "Microsoft.NET.Sdk"),
                new XElement("PropertyGroup", new XElement("TargetFramework", "net10.0"),
                    new XElement("OutputType", "Exe"), new XElement("ImplicitUsings", "enable")),
                new XElement("ItemGroup", paths.Select(path => new XElement("Reference",
                    new XAttribute("Include", Path.GetFileNameWithoutExtension(path)), new XElement("HintPath", path)))));
            if (embedded)
                project.Add(new XElement("ItemGroup",
                    new[] { "manifest.json", "program.xml", "contracts.xml", "options.json", "CpuProgram.g.cs" }.Select(name =>
                        new XElement("EmbeddedResource", new XAttribute("Include", name), new XElement("LogicalName", name)))));
            var projectPath = Path.Combine(directory, "Published.csproj");
            File.WriteAllText(projectPath, project.ToString());
            var load = embedded ? """
                var files = new Dictionary<string, byte[]>();
                foreach (var name in new[] { "manifest.json", "program.xml", "contracts.xml", "options.json", "CpuProgram.g.cs" })
                {
                    using var stream = typeof(SharpInference.Generated.CpuProgram).Assembly.GetManifestResourceStream(name)
                        ?? throw new Exception("Missing embedded resource: " + name);
                    using var buffer = new MemoryStream();
                    stream.CopyTo(buffer);
                    files.Add(name, buffer.ToArray());
                }
                var artifact = CpuVmCompiledArtifact.LoadFromResources(files);
                foreach (var name in new[] { "program.xml", "options.json", "CpuProgram.g.cs" })
                {
                    var original = files[name];
                    files[name] = original.Append((byte)0).ToArray();
                    try
                    {
                        CpuVmCompiledArtifact.LoadFromResources(files);
                        throw new Exception("Accepted tampered embedded resource: " + name);
                    }
                    catch (InvalidDataException) { }
                    files[name] = original;
                }
                files.Remove("options.json");
                try
                {
                    CpuVmCompiledArtifact.LoadFromResources(files);
                    throw new Exception("Accepted missing embedded resource.");
                }
                catch (InvalidDataException) { }
                """ : "var artifact = CpuVmCompiledArtifact.Load(args[0]);";
            File.WriteAllText(Path.Combine(directory, "Program.cs"), """
                using System.Runtime.InteropServices;
                using SharpInference.Backends.CpuVm;
                __LOAD__
                using var executable = artifact.CreateExecutor(new SharpInference.Generated.CpuProgram());
                byte[][] buffers = [
                    MemoryMarshal.AsBytes(new float[] { 1, 2 }.AsSpan()).ToArray(),
                    MemoryMarshal.AsBytes(new float[] { 3, 4 }.AsSpan()).ToArray(),
                    new byte[8]];
                executable.Invoke("run", executable.Prepare(buffers));
                var result = MemoryMarshal.Cast<byte, float>(buffers[2]);
                if (result[0] != 4 || result[1] != 6) throw new Exception("Incorrect static result.");
                if (AppDomain.CurrentDomain.GetAssemblies().Any(a => a.GetName().Name.StartsWith("Microsoft.CodeAnalysis")))
                    throw new Exception("Source package loading used Roslyn.");
                Console.WriteLine("Published source result: [4,6]; Roslyn not loaded.");
                """.Replace("__LOAD__", load, StringComparison.Ordinal));
            var start = new ProcessStartInfo("dotnet")
            {
                WorkingDirectory = directory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            foreach (var argument in new[] { "run", "--project", projectPath, "--configuration", "Release", "--no-launch-profile", "--", directory })
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
                throw new TimeoutException("Published source SDK validation timed out.");
            }
            var output = await stdout;
            var errors = await stderr;
            Assert.True(process.ExitCode == 0, output + Environment.NewLine + errors);
            Assert.Contains("Published source result: [4,6]; Roslyn not loaded.", output);
        }
        finally
        {
            if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
        }
    }
}
