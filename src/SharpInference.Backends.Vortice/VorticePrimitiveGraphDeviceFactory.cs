using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;

namespace SharpInference.Backends.Vortice;

internal static class VorticePrimitiveGraphDeviceFactory
{
    public static (ID3D12Device Device, string Name) Create(int? adapterIndex)
    {
        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory6>();
        var hardware = new List<IDXGIAdapter1>();
        try
        {
            for (uint index = 0; factory.EnumAdapterByGpuPreference(
                     index, GpuPreference.HighPerformance, out IDXGIAdapter1? adapter).Success; index++)
            {
                if ((adapter!.Description1.Flags & AdapterFlags.Software) == 0 &&
                    IsSupported(adapter, FeatureLevel.Level_11_0))
                    hardware.Add(adapter);
                else
                    adapter.Dispose();
            }
            var selected = adapterIndex ?? 0;
            if ((uint)selected >= (uint)hardware.Count)
                throw new ArgumentOutOfRangeException(nameof(adapterIndex),
                    $"No D3D12 hardware adapter exists at supported index {selected}.");
            var device = D3D12CreateDevice<ID3D12Device>(hardware[selected], FeatureLevel.Level_11_0);
            return (device, hardware[selected].Description1.Description);
        }
        finally
        {
            foreach (var adapter in hardware) adapter.Dispose();
        }
    }
}
