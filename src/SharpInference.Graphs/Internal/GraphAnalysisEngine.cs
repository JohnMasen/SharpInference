using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using ComputationGraph = SharpInference.Graphs.Graph;

namespace SharpInference.Graphs;

public sealed record GraphAnalysis(
    IReadOnlyList<GraphAction> Order,
    IReadOnlyList<GraphFlow> Dependencies,
    IReadOnlyList<GraphDiagnostic> Diagnostics)
{
    internal IReadOnlyList<GraphResourceEffect> Effects { get; init; } = [];
}

internal readonly record struct ResourceInstance(
    string Graph, string? Call, string Resource, ResourceScope? SharedScope = null);
internal sealed record GraphResourceEffect(ResourceInstance Identity, Resource Resource, ResourceAccess Access);

/// <summary>Derives value dependencies from explicit flow, entry values and resolved resource accesses.</summary>
internal static class GraphAnalysisEngine
{
    internal static GraphAnalysis AnalyzeValidated<TGraph>(
        GraphPackage<TGraph> package, TGraph graph, IReadOnlyList<GraphDiagnostic> packageDiagnostics,
        Dictionary<string, GraphAnalysis>? cache = null)
        where TGraph : ComputationGraph
    {
        cache ??= new(StringComparer.Ordinal);
        if (cache.TryGetValue(graph.Id, out var cached))
        {
            return cached;
        }

        var diagnostics = packageDiagnostics.ToList();
        if (diagnostics.Any(item => item.Severity == GraphDiagnosticSeverity.Error))
        {
            return new([], [], diagnostics);
        }

        var definitions = package.ToDictionary(item => item.Id, item => (ComputationGraph)item, StringComparer.Ordinal);
        var accesses = new Dictionary<ResourceInstance, List<Access>>();
        var resources = new Dictionary<ResourceInstance, Resource>();
        var flow = graph.GraphFlow.Distinct().ToList();
        void Error(string code, string message, GraphAction? action = null, Resource? resource = null)
        {
            diagnostics.Add(new(GraphDiagnosticSeverity.Error, code, message, action?.Id,
                resource?.Id, Graph: graph.Id));
        }

        foreach (var action in graph.Actions)
        {
            foreach (var binding in action.Bindings)
            {
                if (Resolve(graph, binding.Target, [], out var identity, out var resource))
                {
                    Register(identity, resource, action, binding.Access);
                }
            }
            if (action is not CallAction call)
            {
                continue;
            }

            var target = definitions[call.Target];
            var nested = AnalyzeValidated(package, package.Single(item => item.Id == target.Id), packageDiagnostics, cache);
            diagnostics.AddRange(nested.Diagnostics);
            foreach (var effect in nested.Effects)
            {
                var resource = effect.Resource;
                var binding = effect.Identity.Call is null && resource.Graph.Id == target.Id
                    ? call.Bindings.SingleOrDefault(binding => binding.SourceProperty == resource.Id) : null;
                if (binding is not null)
                {
                    if (Resolve(graph, binding.Target, [], out var actual, out var declaration))
                    {
                        Register(actual, declaration, call, effect.Access);
                    }
                }
                else
                {
                    var identity = effect.Identity;
                    if (resource.Scope == ResourceScope.Call &&
                        (identity.Graph == target.Id || identity.Call is not null))
                    {
                        identity = identity with
                        {
                            Call = InstancePath(call.Id, identity.Call)
                        };
                    }

                    Register(identity, resource, call, effect.Access);
                }
            }
        }

        SharpInference.Graphs.Core.Graph<ActionNode, FlowEdge> Structure() => new(
            graph.Actions.Select(action => new ActionNode(action)),
            flow.Select((edge, index) => new FlowEdge(new(index.ToString()),
                new(edge.Source.Action), new(edge.Target.Action))));

        ReachabilityIndex<ActionNode>? reach = null;
        bool Before(GraphAction first, GraphAction second) =>
            first != second && reach!.IsReachable(new GraphNodeId(first.Id), new GraphNodeId(second.Id));

        void Add(GraphAction before, GraphAction after)
        {
            var edge = new GraphFlow(new(before), new(after));
            if (!flow.Contains(edge))
            {
                flow.Add(edge);
            }
        }

        bool changed;
        do
        {
            var structure = Structure();
            IReadOnlyList<ActionNode> order;
            try
            {
                order = structure.TopologicalOrder();
            }
            catch (InvalidDataException error)
            {
                Error("A0001", $"Inferred dependencies contain a cycle: {error.Message}");
                return new([], flow, diagnostics);
            }
            reach = structure.CreateReachabilityIndex(order, order);
            var count = flow.Count;
            foreach (var (identity, uses) in accesses)
            {
                var resource = resources[identity];
                if (EntryValue(identity, resource))
                {
                    continue;
                }

                foreach (var read in uses.Where(use => use.Kind != ResourceAccess.Write))
                {
                    var writers = uses.Where(use => use.Action != read.Action && use.Kind != ResourceAccess.Read).ToArray();
                    if (writers.Any(writer => Before(writer.Action, read.Action)))
                    {
                        continue;
                    }

                    var producers = writers.Where(writer => !Before(read.Action, writer.Action)).ToArray();
                    var last = producers.Where(writer =>
                        !producers.Any(other => Before(writer.Action, other.Action))).ToArray();
                    if (last.Length == 1)
                    {
                        Add(last[0].Action, read.Action);
                    }
                }
            }
            changed = count != flow.Count;
        } while (changed);

        foreach (var (identity, uses) in accesses)
        {
            var resource = resources[identity];
            for (var first = 0; first < uses.Count; first++)
            {
                var left = uses[first];
                if (left.Kind != ResourceAccess.Write && !EntryValue(identity, resource) &&
                    !uses.Any(other => other.Kind != ResourceAccess.Read && Before(other.Action, left.Action)))
                {
                    Error("A0002", $"Resource '{resource.Id}' is read without an available value.", left.Action, resource);
                }

                for (var second = first + 1; second < uses.Count; second++)
                {
                    var right = uses[second];
                    if ((left.Kind != ResourceAccess.Read || right.Kind != ResourceAccess.Read) &&
                        !Before(left.Action, right.Action) && !Before(right.Action, left.Action))
                    {
                        Error("A0003", $"Unordered accesses to '{resource.Id}' by '{left.Action.Id}' and '{right.Action.Id}'.",
                            left.Action, resource);
                    }
                }
            }
        }
        foreach (var output in graph.Outputs)
        {
            if (!HasEntryValue(output) && !accesses.Any(pair => ReferenceEquals(resources[pair.Key], output) &&
                pair.Value.Any(use => use.Kind != ResourceAccess.Read)))
            {
                Error("A0004", $"Output '{output.Id}' has no available value.", resource: output);
            }
        }

        var result = new GraphAnalysis(Structure().TopologicalOrder().Select(node => node.Action).ToArray(), flow,
            diagnostics.Distinct().ToArray())
        {
            Effects = accesses.Select(pair =>
            {
                var writes = pair.Value.Any(use => use.Kind != ResourceAccess.Read);
                var readsEntry = pair.Value.Any(use => use.Kind != ResourceAccess.Write &&
                    !pair.Value.Any(writer => writer.Kind != ResourceAccess.Read && Before(writer.Action, use.Action)));
                return new GraphResourceEffect(pair.Key, resources[pair.Key], writes
                    ? readsEntry ? ResourceAccess.ReadWrite : ResourceAccess.Write : ResourceAccess.Read);
            }).ToArray(),
        };
        cache.Add(graph.Id, result);
        return result;

        bool EntryValue(ResourceInstance identity, Resource resource) =>
            HasEntryValue(resource) || resource.Scope == ResourceScope.Call && identity.Call is null &&
                resource.Graph.Id != graph.Id && Calls(resource.Graph.Id, graph.Id, []);

        bool Calls(string caller, string target, HashSet<string> visited) =>
            visited.Add(caller) && definitions[caller].Actions.OfType<CallAction>()
                .Any(call => call.Target == target || Calls(call.Target, target, visited));

        void Register(ResourceInstance identity, Resource resource, GraphAction action, ResourceAccess kind)
        {
            resources.TryAdd(identity, resource);
            if (!accesses.TryGetValue(identity, out var uses))
            {
                accesses.Add(identity, uses = []);
            }

            var index = uses.FindIndex(use => ReferenceEquals(use.Action, action));
            if (index < 0)
            {
                uses.Add(new(action, kind));
            }
            else if (uses[index].Kind != kind)
            {
                uses[index] = new(action, ResourceAccess.ReadWrite);
            }
        }

        IEnumerable<(ResourceInstance Identity, Resource Resource)> Instances(
            ComputationGraph context, string target, string property,
            HashSet<(string Graph, ResourceRef Reference)> visited)
        {
            if (context.Id == target)
            {
                var declared = context.Resources.Single(resource => resource.Id == property);
                yield return (new(context.Id, null, property), declared);
                yield break;
            }
            foreach (var call in context.Actions.OfType<CallAction>())
            {
                foreach (var found in Instances(definitions[call.Target], target, property, visited))
                {
                    var binding = found.Identity.Call is null && found.Resource.Graph.Id == call.Target
                        ? call.Bindings.SingleOrDefault(binding => binding.SourceProperty == found.Resource.Id) : null;
                    if (binding is not null)
                    {
                        if (Resolve(context, binding.Target, new(visited), out var actual, out var declared))
                        {
                            yield return (actual, declared);
                        }
                    }
                    else
                    {
                        var identity = found.Identity;
                        if (found.Resource.Scope == ResourceScope.Call &&
                            (identity.Graph == call.Target || identity.Call is not null))
                        {
                            identity = identity with
                            {
                                Call = InstancePath(call.Id, identity.Call)
                            };
                        }

                        yield return (identity, found.Resource);
                    }
                }
            }
        }

        bool Resolve(ComputationGraph context, ResourceRef reference,
            HashSet<(string Graph, ResourceRef Reference)> visited,
            out ResourceInstance identity, out Resource resource)
        {
            if (!GraphValidation.TryResolve(context, reference, definitions, out resource))
            {
                identity = default;
                Error("A0005", $"Cannot resolve '{reference}'.");
                return false;
            }
            if (!visited.Add((context.Id, reference)))
            {
                identity = default;
                Error("A0005", $"Resource alias cycle at '{reference}'.");
                return false;
            }
            var declared = resource;
            if (reference.TargetKind == ResourceTarget.Call)
            {
                var owner = context.Actions.OfType<CallAction>().Single(call => call.Id == reference.Target);
                var actual = owner.Bindings.SingleOrDefault(binding => binding.SourceProperty == declared.Id);
                if (actual is not null)
                {
                    return Resolve(context, actual.Target, visited, out identity, out resource);
                }

                identity = new(resource.Graph.Id, resource.Scope == ResourceScope.Call ? InstancePath(owner.Id, null) : null, resource.Id);
                identity = SharedIdentity(identity, resource);
                return true;
            }
            if (resource.Graph.Id != context.Id && resource.Scope == ResourceScope.Call)
            {
                if (Calls(resource.Graph.Id, context.Id, []))
                {
                    identity = new(resource.Graph.Id, null, resource.Id);
                    return true;
                }
                var candidates = Instances(context, resource.Graph.Id, resource.Id, visited)
                    .DistinctBy(candidate => candidate.Identity).Take(2).ToArray();
                if (candidates.Length != 1)
                {
                    identity = default;
                    Error("A0006", $"Call resource '{resource.Graph.Id}/{resource.Id}' requires a unique active instance.", resource: resource);
                    return false;
                }
                identity = candidates[0].Identity;
                resource = candidates[0].Resource;
                return true;
            }
            identity = new(resource.Graph.Id, null, resource.Id);
            identity = SharedIdentity(identity, resource);
            return true;
        }
    }

    private static bool HasEntryValue(Resource resource) =>
        resource.GraphIO is GraphIO.In or GraphIO.InOut ||
        resource.Scope == ResourceScope.Global || resource.BindingKey is not null;

    private static ResourceInstance SharedIdentity(ResourceInstance identity, Resource resource) =>
        resource.Scope != ResourceScope.Call && resource.BindingKey is { } key
            ? new("", null, key, resource.Scope) : identity;

    private static string InstancePath(string call, string? nested) =>
        Uri.EscapeDataString(call) + (nested is null ? "" : "/" + nested);
    private sealed record Access(GraphAction Action, ResourceAccess Kind);
}
