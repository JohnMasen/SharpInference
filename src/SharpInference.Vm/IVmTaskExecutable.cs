namespace SharpInference.Vm;

public interface IVmTaskExecutable : IVmExecutable
{
    void ExecuteTask(string entryName, VmExecutionLease lease, CancellationToken cancellation);
}
