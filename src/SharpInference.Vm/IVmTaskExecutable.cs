namespace SharpInference.Vm;

/// <summary>Executes a named VM entry point using a leased execution context.</summary>
public interface IVmTaskExecutable : IVmExecutable
{
    /// <summary>Executes the specified entry point with the supplied lease.</summary>
    void ExecuteTask(string entryName, VmExecutionLease lease, CancellationToken cancellation);
}
