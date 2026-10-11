namespace SharpInference.Runtime;

/// <summary>Provides named numeric measurements of VM backend activity.</summary>
public interface IVmBackendDiagnostics
{
    /// <summary>Returns a snapshot of backend counters and timings indexed by metric name.</summary>
    IReadOnlyDictionary<string, double> ReadMetrics();
}
