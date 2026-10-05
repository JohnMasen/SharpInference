using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.Backends.D3D12Vm;

public static class D3D12VmDeviceFactory
{
    /// <summary>Creates a caller-owned hardware device; never falls back to WARP.</summary>
    public static (ID3D12Device Device, string Name) Create(int adapterIndex = 0)
    {
        return WithAdapter(adapterIndex, adapter =>
            (D3D12CreateDevice<ID3D12Device>(adapter, FeatureLevel.Level_11_0), adapter.Description1.Description));
    }

    public static string HardwareIdentity(int adapterIndex = 0)
    {
        return WithAdapter(adapterIndex, adapter =>
        {
            adapter.CheckInterfaceSupport(typeof(IDXGIDevice).GUID, out var driverVersion).CheckError();
            var description = adapter.Description1;
            return $"{description.Description}|{description.VendorId:X8}|{description.DeviceId:X8}|" +
                $"{description.SubsystemId:X8}|{description.Revision:X8}|driver:{driverVersion:X16}";
        });
    }

    private static T WithAdapter<T>(int adapterIndex, Func<IDXGIAdapter1, T> action)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(adapterIndex);
        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory6>();
        var supported = 0;
        for (uint i = 0; factory.EnumAdapterByGpuPreference(i, GpuPreference.HighPerformance,
                 out IDXGIAdapter1? adapter).Success; i++)
        {
        {
            using (adapter)
                if ((adapter!.Description1.Flags & AdapterFlags.Software) == 0 &&
                    IsSupported(adapter, FeatureLevel.Level_11_0) && supported++ == adapterIndex)
                    return action(adapter);
        }
        }
        throw new NotSupportedException($"No supported D3D12 hardware adapter at index {adapterIndex}.");
    }
}
