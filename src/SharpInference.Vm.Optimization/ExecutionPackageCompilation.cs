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

    internal static string StorageSlot(ExecutionGraph graph, string id)
    {
        var visited = new HashSet<string>(StringComparer.Ordinal);
        while (true)
        {
            if (!visited.Add(id))
            {
                throw new InvalidDataException($"Storage aliases contain a cycle at '{graph.Id}/{id}'.");
            }

            if (!graph.Storage.TryGetValue(id, out var storage))
            {
                throw new InvalidDataException($"Resource '{graph.Id}/{id}' has no selected storage.");
            }

            if (storage.Slot is not { } next || next == id)
            {
                return id;
            }

            id = next;
        }
    }

    internal static void ValidatePlan(GraphPackage<ExecutionGraph> package)
    {
        var diagnostics = package.Validate();
        GraphValidation.ThrowIfErrors(diagnostics);
        var analyses = new Dictionary<string, GraphAnalysis>(StringComparer.Ordinal);
        foreach (var graph in package)
        {
            var analysis = GraphDependencyAnalyzer.AnalyzeValidated(package, graph, diagnostics, analyses);
            GraphValidation.ThrowIfErrors(analysis.Diagnostics);
            var structure = graph.DependencyGraph();
            var reach = structure.CreateReachabilityIndex(structure.Nodes, structure.Nodes);
            bool Before(string first, string second) =>
                first != second && reach.IsReachable(new GraphNodeId(first), new GraphNodeId(second));
            if (analysis.Dependencies.Any(flow => !Before(flow.Source.Action, flow.Target.Action)))
            {
                throw new InvalidDataException($"Execution graph '{graph.Id}' is missing derived dependencies. Compile the package before emission.");
            }

            foreach (var group in graph.Resources.GroupBy(resource => StorageSlot(graph, resource.Id))
                .Where(group => group.Count() > 1))
            {
                var resources = group.ToArray();
                foreach (var resource in resources)
                {
                    if (resource.Scope != ResourceScope.Call || resource.GraphIO != GraphIO.Internal ||
                        resource.IsState || resource.BindingKey is not null ||
                        resource.Usage is ResourceUsage.Weight or ResourceUsage.Constant ||
                        !GraphValidation.SameTensor(resource.Tensor, resources[0].Tensor) ||
                        resource.Permission != resources[0].Permission)
                    {
                        throw new InvalidDataException($"Storage alias '{graph.Id}/{resource.Id}' has an incompatible resource contract.");
                    }

                    if (package.Any(caller => caller.Actions.SelectMany(action => action.Bindings).Any(binding =>
                        binding.Target.TargetProperty == resource.Id &&
                        (binding.Target.TargetKind == ResourceTarget.Graph && binding.Target.Target == graph.Id && caller != graph ||
                         binding.Target.TargetKind == ResourceTarget.Call && caller.Actions.OfType<CallAction>().Any(call =>
                             call.Id == binding.Target.Target && call.Target == graph.Id)))))
                    {
                        throw new InvalidDataException($"Externally referenced resource '{graph.Id}/{resource.Id}' cannot share temporary storage.");
                    }
                }
                for (var left = 0; left < resources.Length; left++)
                {
                    for (var right = left + 1; right < resources.Length; right++)
                    {
                        var first = graph.Readers(resources[left]).Concat(graph.Writers(resources[left])).Distinct().ToArray();
                        var second = graph.Readers(resources[right]).Concat(graph.Writers(resources[right])).Distinct().ToArray();
                        if (first.Length == 0 || second.Length == 0)
                        {
                            continue;
                        }

                        if (!first.All(a => second.All(b => Before(a.Id, b.Id))) &&
                            !second.All(a => first.All(b => Before(a.Id, b.Id))))
                        {
                            throw new InvalidDataException($"Storage alias '{graph.Id}/{group.Key}' has overlapping or unproven lifetimes.");
                        }
                    }
                }
            }
        }
    }

    internal static void Prepare(
        GraphPackage<ExecutionGraph> source, string backend, ExecutionImplementationSelector selector)
    {
        var diagnostics = source.Validate();
        GraphValidation.ThrowIfErrors(diagnostics);
        if (source.GetTarget() is { } requested &&
            ExecutionImplementationSelector.Target(requested) != ExecutionImplementationSelector.Target(backend))
        {
            throw new InvalidDataException($"Package target '{requested}' does not match '{backend}'.");
        }

        var analyses = new Dictionary<string, GraphAnalysis>(StringComparer.Ordinal);
        foreach (var graph in source)
        {
            var analysis = GraphDependencyAnalyzer.AnalyzeValidated(source, graph, diagnostics, analyses);
            GraphValidation.ThrowIfErrors(analysis.Diagnostics);
            graph.GraphFlow.Clear();
            graph.GraphFlow.AddRange(analysis.Dependencies);
            foreach (var resource in graph.Resources)
            {
                graph.Storage.TryAdd(resource.Id, DefaultStorage(resource));
            }
        }
        selector.Select(source, backend);
        ValidatePlan(source);
    }

    public static VmProgram Compile(
        this GraphPackage<ExecutionGraph> package, VmTarget target,
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
