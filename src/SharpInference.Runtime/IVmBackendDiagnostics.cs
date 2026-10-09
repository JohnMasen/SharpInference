namespace SharpInference.Runtime;

public interface IVmBackendDiagnostics
{
    IReadOnlyDictionary<string, double> ReadMetrics();
}
