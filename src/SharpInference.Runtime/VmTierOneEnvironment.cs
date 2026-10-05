using Microsoft.Win32;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime;

public static class VmTierOneEnvironment
{
    public static string HardwareIdentity(VmTarget target, int adapterIndex = 0)
    {
        using var processor = Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            ?? throw new InvalidOperationException("Cannot identify the CPU for an offline cost profile.");
        var name = processor.GetValue("ProcessorNameString") as string;
        if (string.IsNullOrWhiteSpace(name))
            throw new InvalidOperationException("Cannot identify the CPU model for an offline cost profile.");
        var identity = $"{name.Trim()}|{Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER")}";
        return target == VmTarget.Cpu ? identity : identity + "|" + D3D12VmDeviceFactory.HardwareIdentity(adapterIndex);
    }

    public static string Fingerprint(VmTarget target, int adapterIndex = 0) =>
        TierOneEnvironmentFingerprint.Create(string.Join("|", HardwareIdentity(target, adapterIndex),
            typeof(VmGraphBackend).Assembly.ManifestModule.ModuleVersionId,
            typeof(CpuVmCompiler).Assembly.ManifestModule.ModuleVersionId,
            typeof(D3D12VmCompiler).Assembly.ManifestModule.ModuleVersionId,
            typeof(SharpInference.Instructions.D3D12.GpuFloat32InstructionCollection).Assembly.ManifestModule.ModuleVersionId));
}
