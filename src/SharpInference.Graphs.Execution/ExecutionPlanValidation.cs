using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Execution;

/// <summary>Checks an existing execution plan without selecting or changing it.</summary>
public static class ExecutionPlanValidation
{
    public static string StorageSlot(ExecutionGraph graph, string id)
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

    public static IReadOnlyList<GraphDiagnostic> ValidatePlan(ExecutionGraphPackage package)
    {
        var diagnostics = package.Validate();
        if (diagnostics.Any(item => item.Severity == GraphDiagnosticSeverity.Error))
        {
            return diagnostics;
        }
        var findings = diagnostics.ToList();
        var analyses = package.Analyze();
        foreach (var graph in package)
        {
            foreach (var operation in graph.Actions.OfType<ExecutionOperation>())
            {
                if (operation.SelectedImplementation is null)
                {
                    findings.Add(new(GraphDiagnosticSeverity.Error, "E0006",
                        "An execution plan requires a selected implementation.", operation.Id, Graph: graph.Id));
                }
            }
            var analysis = analyses[graph.Id];
            var structure = graph.DependencyGraph();
            var reach = structure.CreateReachabilityIndex(structure.Nodes, structure.Nodes);
            bool Before(string first, string second) =>
                first != second && reach.IsReachable(new GraphNodeId(first), new GraphNodeId(second));
            if (analysis.Dependencies.Any(flow => !Before(flow.Source.Action, flow.Target.Action)))
            {
                findings.Add(new(GraphDiagnosticSeverity.Error, "E0001", $"Execution graph '{graph.Id}' is missing derived dependencies.", Graph: graph.Id));
            }

            var slots = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var resource in graph.Resources)
            {
                try
                {
                    slots.Add(resource.Id, StorageSlot(graph, resource.Id));
                }
                catch (InvalidDataException error)
                {
                    findings.Add(new(GraphDiagnosticSeverity.Error, "E0002", error.Message, Resource: resource.Id, Graph: graph.Id));
                }
            }
            if (slots.Count != graph.Resources.Count)
            {
                continue;
            }
            foreach (var group in graph.Resources.GroupBy(resource => slots[resource.Id])
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
                        findings.Add(new(GraphDiagnosticSeverity.Error, "E0003", $"Storage alias '{graph.Id}/{resource.Id}' has an incompatible resource contract.", Resource: resource.Id, Graph: graph.Id));
                    }

                    if (package.Any(caller => caller.Actions.SelectMany(action => action.Bindings).Any(binding =>
                        binding.Target.TargetProperty == resource.Id &&
                        (binding.Target.TargetKind == ResourceTarget.Graph && binding.Target.Target == graph.Id && caller != graph ||
                         binding.Target.TargetKind == ResourceTarget.Call && caller.Actions.OfType<CallAction>().Any(call =>
                             call.Id == binding.Target.Target && call.Target == graph.Id)))))
                    {
                        findings.Add(new(GraphDiagnosticSeverity.Error, "E0004", $"Externally referenced resource '{graph.Id}/{resource.Id}' cannot share temporary storage.", Resource: resource.Id, Graph: graph.Id));
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
                            findings.Add(new(GraphDiagnosticSeverity.Error, "E0005", $"Storage alias '{graph.Id}/{group.Key}' has overlapping or unproven lifetimes.", Graph: graph.Id));
                        }
                    }
                }
            }
        }
        return findings.Distinct().ToArray();
    }

}
