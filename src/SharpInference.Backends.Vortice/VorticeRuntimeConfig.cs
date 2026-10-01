namespace SharpInference.Backends.Vortice;

public sealed class VorticeRuntimeConfig : IRuntimeConfig
{
    public int? AdapterIndex { get; init; }

    /// <summary>Opt into per-session replay for supported scalar-token portable graphs.</summary>
    public bool EnableCommandReplay { get; init; }
}
