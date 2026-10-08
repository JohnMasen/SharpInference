using Microsoft.Win32;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime;

/// <summary>Builds hardware and assembly identities for VM tier-one cost profiles.</summary>
public static class VmTierOneEnvironment
{
    /// <summary>Gets a CPU or CPU-plus-GPU hardware identity for the selected target.</summary>
    /// <param name="target">The VM target whose hardware is identified.</param>
    /// <param name="adapterIndex">The Direct3D12 adapter index when applicable.</param>
    /// <returns>A hardware identity string.</returns>
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

    /// <summary>Creates a tier-one fingerprint from hardware identity and relevant assembly module IDs.</summary>
    /// <param name="target">The VM target whose environment is fingerprinted.</param>
    /// <param name="adapterIndex">The Direct3D12 adapter index when applicable.</param>
    /// <returns>The environment fingerprint.</returns>
    public static string Fingerprint(VmTarget target, int adapterIndex = 0) =>
        TierOneEnvironmentFingerprint.Create(string.Join("|", HardwareIdentity(target, adapterIndex),
            typeof(VmGraphBackend).Assembly.ManifestModule.ModuleVersionId,
            typeof(CpuVmCompiler).Assembly.ManifestModule.ModuleVersionId,
            typeof(D3D12VmCompiler).Assembly.ManifestModule.ModuleVersionId,
            typeof(SharpInference.Instructions.D3D12.GpuFloat32InstructionCollection).Assembly.ManifestModule.ModuleVersionId));
}
