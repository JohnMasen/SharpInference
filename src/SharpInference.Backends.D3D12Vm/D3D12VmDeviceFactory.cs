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
        ArgumentOutOfRangeException.ThrowIfNegative(adapterIndex);
        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory6>();
        var supported = 0;
        for (uint i = 0; factory.EnumAdapterByGpuPreference(i, GpuPreference.HighPerformance,
                 out IDXGIAdapter1? adapter).Success; i++)
        {
            using (adapter)
                if ((adapter!.Description1.Flags & AdapterFlags.Software) == 0 &&
                    IsSupported(adapter, FeatureLevel.Level_11_0) && supported++ == adapterIndex)
                    return (D3D12CreateDevice<ID3D12Device>(adapter, FeatureLevel.Level_11_0),
                        adapter.Description1.Description);
        }
        throw new NotSupportedException($"No supported D3D12 hardware adapter at index {adapterIndex}.");
    }
}
