using System.Runtime.InteropServices;
using Microsoft.Win32;
using SharpInference.Backends.CpuVm;
using SharpInference.Instructions.Cpu;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime.Cpu;

/// <summary>Identifies CPU hardware and runtime binaries for optimization profile compatibility.</summary>
public static class CpuVmEnvironment
{
    /// <summary>Returns the CPU model, process architecture, and logical processor count, or throws if the model cannot be identified.</summary>
    public static string HardwareIdentity()
    {
        string? name = null;
        if (OperatingSystem.IsWindows())
        {
            using var processor = Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
            name = processor?.GetValue("ProcessorNameString") as string;
        }
        else if (OperatingSystem.IsLinux())
        {
            var line = File.ReadLines("/proc/cpuinfo").FirstOrDefault(value => value.StartsWith("model name", StringComparison.Ordinal) ||
                value.StartsWith("Hardware", StringComparison.Ordinal));
            if (line is not null && line.IndexOf(':') is var separator && separator >= 0)
                name = line[(separator + 1)..];
        }
        if (string.IsNullOrWhiteSpace(name))
            throw new NotSupportedException("Cannot identify the CPU model for an offline cost profile on this platform.");
        return $"{name.Trim()}|{RuntimeInformation.ProcessArchitecture}|{Environment.ProcessorCount}";
    }

    /// <summary>Returns a profile fingerprint combining CPU hardware identity with runtime, compiler, and instruction assembly identities.</summary>
    public static string Fingerprint() => TierOneEnvironmentFingerprint.Create(string.Join("|", HardwareIdentity(),
        typeof(VmGraphBackend).Assembly.ManifestModule.ModuleVersionId,
        typeof(CpuVmCompiler).Assembly.ManifestModule.ModuleVersionId,
        typeof(CpuFloat32InstructionCollection).Assembly.ManifestModule.ModuleVersionId));
}
