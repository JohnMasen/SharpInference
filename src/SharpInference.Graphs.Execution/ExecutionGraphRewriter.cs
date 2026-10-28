using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Execution;

namespace SharpInference.Graphs.Execution;

public static class ExecutionGraphRewriter
{
    public static ExecutionGraphPackage ReplaceGraphCallWithOperation(
        ExecutionGraphPackage package, string graphId, string callId, OperationId operation,
        IEnumerable<Binding>? bindings = null, ExecutionImplementation? implementation = null)
    {
        var result = CopyPackage(package);
        var graph = result.Single(graph => graph.Id == graphId);
        var call = graph.Actions.OfType<ExecutionCall>().Single(action => action.Id == callId);
        var index = graph.Actions.IndexOf(call);
        var replacement = graph.Operation(callId, operation);
        graph.Actions.RemoveAt(graph.Actions.Count - 1);
        replacement.Bindings.AddRange(bindings ?? call.Bindings);
        replacement.SelectedImplementation = implementation;
        replacement.Sources.AddRange(call.Sources);
        replacement.Category = call.Category == "call" ? "operation" : call.Category;
        CopyMetadata(call, replacement);
        graph.Actions[index] = replacement;
        foreach (var action in graph.Actions)
        {
            for (var bindingIndex = 0; bindingIndex < action.Bindings.Count; bindingIndex++)
            {
                var binding = action.Bindings[bindingIndex];
                if (binding.Target.TargetKind != ResourceTarget.Call || binding.Target.Target != callId)
                {
                    continue;
                }

                var actual = call.Bindings.SingleOrDefault(item => item.SourceProperty == binding.Target.TargetProperty)
                    ?? throw new InvalidDataException($"Replacing '{callId}' requires an explicit mapping for its private resource '{binding.Target.TargetProperty}'.");
                action.Bindings[bindingIndex] = binding with
                {
                    Target = actual.Target
                };
            }
        }

        GraphValidation.ThrowIfErrors(result.ValidateStructure());
        return result;
    }

    public static ExecutionGraphPackage FuseNodesToOperation(
        ExecutionGraphPackage package, string graphId, IEnumerable<string> nodeIds,
        string fusedId, OperationId operation, IEnumerable<Binding> bindings,
        ExecutionImplementation? implementation = null)
    {
        ArgumentNullException.ThrowIfNull(nodeIds);
        ArgumentNullException.ThrowIfNull(bindings);
        var result = CopyPackage(package);
        var graph = result.Single(graph => graph.Id == graphId);
        var selected = SelectActions(graph, nodeIds);
        var ids = selected.Select(action => action.Id).ToHashSet(StringComparer.Ordinal);
        RequireAvailableId(graph, fusedId, ids);
        var supplied = bindings.ToArray();
        var references = supplied.Select(binding => Canonical(graph, binding.Target)).ToHashSet();
        var boundary = graph.Actions.Except(selected).SelectMany(action => action.Bindings)
            .Select(binding => Canonical(graph, binding.Target))
            .Concat(graph.GraphIOs.Select(resource => Canonical(graph, new ResourceRef(resource)))).ToHashSet();
        foreach (var reference in selected.SelectMany(action => action.Bindings).Select(binding => Canonical(graph, binding.Target)))
        {
            if (boundary.Contains(reference) && !references.Contains(reference))
            {
                throw new InvalidDataException($"Fusion must preserve boundary resource '{reference}'.");
            }
        }

        var structure = graph.DependencyGraph();
        var reach = structure.CreateReachabilityIndex(structure.Nodes, structure.Nodes);
        foreach (var outside in graph.Actions.Except(selected))
        {
            if (selected.Any(action => reach.IsReachable(new GraphNodeId(action.Id), new GraphNodeId(outside.Id))) &&
                selected.Any(action => reach.IsReachable(new GraphNodeId(outside.Id), new GraphNodeId(action.Id))))
            {
                throw new InvalidDataException($"Fusion would cross intervening action '{outside.Id}'.");
            }
        }

        var position = graph.Actions.IndexOf(selected[0]);
        var fused = graph.Operation(fusedId, operation);
        graph.Actions.RemoveAt(graph.Actions.Count - 1);
        fused.Bindings.AddRange(supplied);
        fused.SelectedImplementation = implementation;
        fused.Sources.AddRange(selected.SelectMany(Sources).Distinct());
        fused.Metadata.Add("transformation", "fuse");
        graph.Actions.RemoveAll(action => ids.Contains(action.Id));
        graph.Actions.Insert(Math.Min(position, graph.Actions.Count), fused);
        ReplaceFlow(graph, ids, fusedId);
        GraphValidation.ThrowIfErrors(result.ValidateStructure());
        return result;
    }

    public static ExecutionGraphPackage ExtractNodesToGraph(
        ExecutionGraphPackage package, string graphId, IEnumerable<string> nodeIds,
        string extractedId, string callId)
    {
        var result = CopyPackage(package);
        if (result.Any(graph => graph.Id == extractedId))
        {
            throw new InvalidDataException($"Graph '{extractedId}' is already declared.");
        }

        var graph = result.Single(graph => graph.Id == graphId);
        var selected = SelectActions(graph, nodeIds);
        var ids = selected.Select(action => action.Id).ToHashSet(StringComparer.Ordinal);
        RequireAvailableId(graph, callId, ids);
        var child = new ExecutionGraph { Id = extractedId };
        var definitions = result.ToDictionary(item => item.Id, item => (SharpInference.Graphs.Graph)item, StringComparer.Ordinal);
        var refs = selected.SelectMany(action => action.Bindings).Select(binding => Canonical(graph, binding.Target)).Distinct().ToArray();
        var map = new Dictionary<ResourceRef, ResourceRef>();
        var callBindings = new List<Binding>();
        var moved = new HashSet<string>(StringComparer.Ordinal);
        var structure = graph.DependencyGraph();
        var reach = structure.CreateReachabilityIndex(structure.Nodes, structure.Nodes);
        foreach (var reference in refs)
        {
            if (!GraphValidation.TryResolve(graph, reference, definitions, out var resource))
            {
                throw new InvalidDataException($"Cannot resolve extraction resource '{reference}'.");
            }

            var uses = selected.SelectMany(action => action.Bindings.Where(binding =>
                Canonical(graph, binding.Target) == reference).Select(binding => (Action: action, binding.Access))).ToArray();
            var privateLocal = resource.Graph == graph && resource.Scope == ResourceScope.Call &&
                resource.GraphIO == GraphIO.Internal && !resource.IsState && resource.BindingKey is null &&
                !result.SelectMany(caller => caller.Actions.SelectMany(action =>
                    (caller == graph && ids.Contains(action.Id) ? [] : action.Bindings)
                        .Select(binding => Canonical(caller, binding.Target)))).Contains(reference);
            var name = resource.Id;
            for (var suffix = 1; child.Resources.Any(item => item.Id == name); suffix++)
            {
                name = resource.Id + "." + suffix;
            }

            var writes = uses.Any(use => use.Access != ResourceAccess.Read);
            var readsEntry = uses.Any(use => use.Access != ResourceAccess.Write &&
                !uses.Any(writer => writer.Action != use.Action && writer.Access != ResourceAccess.Read &&
                    reach.IsReachable(new GraphNodeId(writer.Action.Id), new GraphNodeId(use.Action.Id))));
            var io = privateLocal ? GraphIO.Internal : !writes ? GraphIO.In : readsEntry ? GraphIO.InOut : GraphIO.Out;
            var declared = child.Resource(name, resource.Tensor, io,
                privateLocal ? resource.Scope : ResourceScope.Call, resource.Usage, resource.IsState, resource.Permission);
            declared.Category = resource.Category;
            child.Storage.Add(name, new(privateLocal ? ResourceStorageKind.LocalBuffer : ResourceStorageKind.ExternalArgument));
            map.Add(reference, new ResourceRef(declared));
            if (privateLocal)
            {
                moved.Add(resource.Id);
            }
            else
            {
                callBindings.Add(new(name, reference, io switch
                {
                    GraphIO.In => ResourceAccess.Read,
                    GraphIO.Out => ResourceAccess.Write,
                    _ => ResourceAccess.ReadWrite,
                }));
            }
        }
        foreach (var action in selected)
        {
            CloneAction(action, child, binding => map[Canonical(graph, binding.Target)]);
        }

        child.GraphFlow.AddRange(graph.GraphFlow.Where(flow =>
            ids.Contains(flow.Source.Action) && ids.Contains(flow.Target.Action)).Select(flow =>
                new GraphFlow(new(child.Id, flow.Source.Action), new(child.Id, flow.Target.Action))));
        var position = graph.Actions.IndexOf(selected[0]);
        var call = graph.Call(callId, child);
        graph.Actions.RemoveAt(graph.Actions.Count - 1);
        call.Bindings.AddRange(callBindings);
        call.Sources.AddRange(selected.SelectMany(Sources).Distinct());
        call.Metadata.Add("transformation", "extract");
        graph.Actions.RemoveAll(action => ids.Contains(action.Id));
        graph.Actions.Insert(Math.Min(position, graph.Actions.Count), call);
        graph.Resources.RemoveAll(resource => moved.Contains(resource.Id));
        foreach (var id in moved)
        {
            graph.Storage.Remove(id);
        }

        ReplaceFlow(graph, ids, callId);
        result.Add(child);
        GraphValidation.ThrowIfErrors(result.ValidateStructure());
        return result;
    }

    public static ExecutionGraphPackage InlineGraphCall(
        ExecutionGraphPackage package, string graphId, string callId)
    {
        var result = CopyPackage(package);
        var graph = result.Single(graph => graph.Id == graphId);
        var call = graph.Actions.OfType<ExecutionCall>().Single(action => action.Id == callId);
        var target = result.Single(graph => graph.Id == call.Target);
        var resources = new Dictionary<string, ResourceRef>(StringComparer.Ordinal);
        foreach (var resource in target.Resources)
        {
            var binding = call.Bindings.SingleOrDefault(binding => binding.SourceProperty == resource.Id);
            if (binding is not null)
            {
                resources.Add(resource.Id, binding.Target);
            }
            else if (resource.Scope != ResourceScope.Call)
            {
                resources.Add(resource.Id, new ResourceRef(resource));
            }
            else
            {
                var name = call.Id + "/" + resource.Id;
                for (var suffix = 1; graph.Resources.Any(resource => resource.Id == name); suffix++)
                {
                    name = call.Id + "/" + resource.Id + "." + suffix;
                }

                var declared = graph.Resource(name, resource.Tensor, GraphIO.Internal, resource.Scope,
                    resource.Usage, resource.IsState, resource.Permission, resource.BindingKey);
                declared.Category = resource.Category;
                graph.Storage.Add(name, target.Storage.GetValueOrDefault(resource.Id) ?? new(ResourceStorageKind.LocalBuffer));
                resources.Add(resource.Id, new ResourceRef(declared));
            }
        }
        foreach (var resource in target.Resources.Where(resource => resource.Scope == ResourceScope.Call &&
            !call.Bindings.Any(binding => binding.SourceProperty == resource.Id)))
        {
            if (target.Storage.TryGetValue(resource.Id, out var storage) && storage.Slot is { } slot)
            {
                graph.Storage[resources[resource.Id].TargetProperty] = storage with
                {
                    Slot = resources[slot].TargetProperty
                };
            }
        }

        var actionIds = graph.Actions.Select(action => action.Id).ToHashSet(StringComparer.Ordinal);
        var names = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var action in target.Actions)
        {
            var name = call.Id + "/" + action.Id;
            for (var suffix = 1; !actionIds.Add(name); suffix++)
            {
                name = call.Id + "/" + action.Id + "." + suffix;
            }

            names.Add(action.Id, name);
        }
        ResourceRef Rebind(Binding binding)
        {
            var reference = binding.Target;
            if (reference.TargetKind == ResourceTarget.Graph && (reference.Target is null || reference.Target == target.Id))
            {
                return resources[reference.TargetProperty];
            }

            if (reference.TargetKind == ResourceTarget.Call && reference.Target is { } owner)
            {
                return ResourceRef.InCall(names[owner], reference.TargetProperty);
            }

            return reference;
        }
        var originalActions = graph.Actions.Count;
        foreach (var action in target.Actions)
        {
            CloneAction(action, graph, Rebind, names[action.Id]);
        }

        var inlined = graph.Actions.Skip(originalActions).ToArray();
        foreach (var action in inlined)
        {
            if (action is ExecutionOperation operation)
            {
                var origins = operation.Sources.Concat(call.Sources).Distinct().ToArray();
                operation.Sources.Clear();
                operation.Sources.AddRange(origins);
            }
            else if (action is ExecutionCall nested)
            {
                var origins = nested.Sources.Concat(call.Sources).Distinct().ToArray();
                nested.Sources.Clear();
                nested.Sources.AddRange(origins);
            }
        }

        graph.Actions.RemoveRange(originalActions, inlined.Length);
        var position = graph.Actions.IndexOf(call);
        graph.Actions.RemoveAt(position);
        graph.Actions.InsertRange(position, inlined);
        foreach (var action in graph.Actions)
        {
            for (var index = 0; index < action.Bindings.Count; index++)
            {
                var binding = action.Bindings[index];
                if (binding.Target.TargetKind == ResourceTarget.Call && binding.Target.Target == callId)
                {
                    action.Bindings[index] = binding with
                    {
                        Target = resources[binding.Target.TargetProperty]
                    };
                }
            }
        }

        var internalFlow = target.GraphFlow.Select(flow => new GraphFlow(
            new(graph.Id, names[flow.Source.Action]), new(graph.Id, names[flow.Target.Action]))).ToArray();
        var entries = inlined.Where(action => !internalFlow.Any(flow => flow.Target.Action == action.Id)).ToArray();
        var exits = inlined.Where(action => !internalFlow.Any(flow => flow.Source.Action == action.Id)).ToArray();
        var incoming = graph.GraphFlow.Where(flow => flow.Target.Action == callId).ToArray();
        var outgoing = graph.GraphFlow.Where(flow => flow.Source.Action == callId).ToArray();
        graph.GraphFlow.RemoveAll(flow => flow.Source.Action == callId || flow.Target.Action == callId);
        graph.GraphFlow.AddRange(internalFlow);
        if (inlined.Length == 0)
        {
            graph.GraphFlow.AddRange(incoming.SelectMany(before => outgoing.Select(after => new GraphFlow(before.Source, after.Target))));
        }
        else
        {
            graph.GraphFlow.AddRange(incoming.SelectMany(before => entries.Select(entry => new GraphFlow(before.Source, new(entry)))));
            graph.GraphFlow.AddRange(outgoing.SelectMany(after => exits.Select(exit => new GraphFlow(new(exit), after.Target))));
        }
        GraphValidation.ThrowIfErrors(result.ValidateStructure());
        return result;
    }

    private static ExecutionGraphPackage CopyPackage(ExecutionGraphPackage package)
    {
        ArgumentNullException.ThrowIfNull(package);
        GraphValidation.ThrowIfErrors(package.ValidateStructure());
        return package.Snapshot();
    }

    private static GraphAction[] SelectActions(ExecutionGraph graph, IEnumerable<string> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var requested = ids.ToHashSet(StringComparer.Ordinal);
        var selected = graph.Actions.Where(action => requested.Contains(action.Id)).ToArray();
        if (selected.Length == 0 || selected.Length != requested.Count)
        {
            throw new InvalidDataException("The rewrite selection must contain existing actions.");
        }

        return selected;
    }

    private static void RequireAvailableId(ExecutionGraph graph, string id, HashSet<string> replaced)
    {
        if (string.IsNullOrWhiteSpace(id) || graph.Actions.Any(action => action.Id == id && !replaced.Contains(id)))
        {
            throw new InvalidDataException($"Action identifier '{id}' is unavailable.");
        }
    }

    private static ResourceRef Canonical(SharpInference.Graphs.Graph graph, ResourceRef reference) =>
        reference.TargetKind == ResourceTarget.Graph && reference.Target is null
            ? reference with
            {
                Target = graph.Id
            } : reference;

    private static IEnumerable<SourceOrigin> Sources(GraphAction action) => action switch
    {
        ExecutionOperation operation => operation.Sources,
        ExecutionCall call => call.Sources,
        _ => [],
    };

    private static void CopyMetadata(GraphAction source, GraphAction target)
    {
        foreach (var pair in source.Metadata)
        {
            target.Metadata.Add(pair.Key, pair.Value);
        }
    }

    private static void ReplaceFlow(ExecutionGraph graph, HashSet<string> replaced, string id)
    {
        var flow = graph.GraphFlow.Select(edge => new GraphFlow(
            new(graph.Id, replaced.Contains(edge.Source.Action) ? id : edge.Source.Action),
            new(graph.Id, replaced.Contains(edge.Target.Action) ? id : edge.Target.Action)))
            .Where(edge => edge.Source != edge.Target).Distinct().ToArray();
        graph.GraphFlow.Clear();
        graph.GraphFlow.AddRange(flow);
    }

    private static void CloneAction(GraphAction action, ExecutionGraph target, Func<Binding, ResourceRef> rebind, string? id = null)
    {
        GraphAction copy;
        if (action is ExecutionOperation operation)
        {
            var replacement = target.Operation(id ?? action.Id, operation.Operation);
            replacement.Precision = operation.Precision;
            replacement.SelectedImplementation = operation.SelectedImplementation;
            replacement.Sources.AddRange(operation.Sources);
            foreach (var pair in operation.Attributes)
            {
                replacement.Attributes.Add(pair.Key, pair.Value);
            }

            copy = replacement;
        }
        else if (action is ExecutionCall call)
        {
            var replacement = target.Call(id ?? action.Id, call.Target);
            replacement.Sources.AddRange(call.Sources);
            copy = replacement;
        }
        else
        {
            throw new NotSupportedException($"Unsupported execution action '{action.GetType().Name}'.");
        }

        copy.Category = action.Category;
        CopyMetadata(action, copy);
        copy.Bindings.AddRange(action.Bindings.Select(binding => binding with { Target = rebind(binding) }));
    }
}
