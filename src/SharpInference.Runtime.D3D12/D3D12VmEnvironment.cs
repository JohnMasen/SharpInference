using SharpInference.Backends.D3D12Vm;
using SharpInference.Instructions.D3D12;
using SharpInference.Runtime.Cpu;
using SharpInference.Vm.Optimization;

namespace SharpInference.Runtime.D3D12;

public static class D3D12VmEnvironment
{
    public static string HardwareIdentity(int adapterIndex = 0) =>
        CpuVmEnvironment.HardwareIdentity() + "|" + D3D12VmDeviceFactory.HardwareIdentity(adapterIndex);

    public static string Fingerprint(int adapterIndex = 0) =>
        TierOneEnvironmentFingerprint.Create(string.Join("|", HardwareIdentity(adapterIndex),
            typeof(VmGraphBackend).Assembly.ManifestModule.ModuleVersionId,
            typeof(D3D12VmCompiler).Assembly.ManifestModule.ModuleVersionId,
            typeof(GpuFloat32InstructionCollection).Assembly.ManifestModule.ModuleVersionId));
}
