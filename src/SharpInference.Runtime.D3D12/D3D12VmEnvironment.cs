using SharpInference.Backends.D3D12Vm;
using SharpInference.Instructions.D3D12;
using SharpInference.Runtime.Cpu;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime.D3D12;

/// <summary>Identifies CPU and GPU hardware and Direct3D 12 runtime binaries for cost-profile compatibility.</summary>
public static class D3D12VmEnvironment
{
    /// <summary>Returns the combined CPU and selected Direct3D 12 adapter hardware identity.</summary>
    /// <param name="adapterIndex">The zero-based Direct3D 12 adapter index.</param>
    public static string HardwareIdentity(int adapterIndex = 0) =>
        CpuVmEnvironment.HardwareIdentity() + "|" + D3D12VmDeviceFactory.HardwareIdentity(adapterIndex);

    /// <summary>Returns a profile fingerprint combining hardware identity with runtime, compiler, and GPU instruction assembly identities.</summary>
    /// <param name="adapterIndex">The zero-based Direct3D 12 adapter index.</param>
    public static string Fingerprint(int adapterIndex = 0) =>
        TierOneEnvironmentFingerprint.Create(string.Join("|", HardwareIdentity(adapterIndex),
            typeof(VmGraphBackend).Assembly.ManifestModule.ModuleVersionId,
            typeof(D3D12VmCompiler).Assembly.ManifestModule.ModuleVersionId,
            typeof(GpuFloat32InstructionCollection).Assembly.ManifestModule.ModuleVersionId));
}
