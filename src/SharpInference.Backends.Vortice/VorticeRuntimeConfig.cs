namespace SharpInference.Backends.Vortice;

/// <summary>Configures hardware adapter selection and optional replay for the Vortice primitive graph backend.</summary>
public sealed class VorticeRuntimeConfig : IRuntimeConfig
{
    /// <summary>Gets the supported hardware adapter index in high-performance order, or null to select the default adapter.</summary>
    public int? AdapterIndex { get; init; }

    /// <summary>Opt into per-session replay for supported scalar-token portable graphs.</summary>
    public bool EnableCommandReplay { get; init; }
}
