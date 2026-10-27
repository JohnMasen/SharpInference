using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Core;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization;

public static class ExecutionPackageCompilation
{
    internal static ExecutionStorage DefaultStorage(Resource resource) => new(resource.Scope == ResourceScope.Global
        ? ResourceStorageKind.SharedBinding
        : resource.Scope == ResourceScope.Session || resource.GraphIO != GraphIO.Internal || resource.BindingKey is not null
            ? ResourceStorageKind.ExternalArgument : ResourceStorageKind.LocalBuffer);

    internal static void Prepare(
        ExecutionGraphPackage source, string backend, ExecutionImplementationSelector selector)
    {
        var diagnostics = source.Validate();
        GraphValidation.ThrowIfErrors(diagnostics);
        if (source.GetTarget() is { } requested &&
            ExecutionImplementationSelector.Target(requested) != ExecutionImplementationSelector.Target(backend))
        {
            throw new InvalidDataException($"Package target '{requested}' does not match '{backend}'.");
        }

        var analyses = source.Analyze();
        foreach (var graph in source)
        {
            var analysis = analyses[graph.Id];
            GraphValidation.ThrowIfErrors(analysis.Diagnostics);
            graph.GraphFlow.Clear();
            graph.GraphFlow.AddRange(analysis.Dependencies);
            foreach (var resource in graph.Resources)
            {
                graph.Storage.TryAdd(resource.Id, DefaultStorage(resource));
            }
        }
        selector.Select(source, backend);
        GraphValidation.ThrowIfErrors(ExecutionPlanValidation.ValidatePlan(source));
    }

    public static VmProgram Compile(
        this ExecutionGraphPackage package, VmTarget target,
        IEnumerable<IInstructionCollectionProvider> providers, VmExecutionGraphEmitterOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(package);
        ArgumentNullException.ThrowIfNull(providers);
        var collections = providers.ToArray();
        var source = package.Snapshot();
        var backend = target switch
        {
            VmTarget.Cpu => "cpu-vm",
            VmTarget.Direct3D12 => "d3d12-vm",
            _ => throw new NotSupportedException($"Unsupported VM target '{target}'."),
        };
        Prepare(source, backend, new(collections));
        return new VmExecutionGraphEmitter(target, collections).Emit(source, options);
    }
}
