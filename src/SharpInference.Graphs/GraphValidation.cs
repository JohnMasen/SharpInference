namespace SharpInference.Graphs;

/// <summary>Read-only source validation. Value and execution analysis run at compilation boundaries.</summary>
public static class GraphValidation
{
    public static IReadOnlyList<GraphDiagnostic> Validate(Graph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var diagnostics = new List<GraphDiagnostic>();
        void Error(string code, string message, string? node = null, Resource? resource = null, string? port = null)
        {
            diagnostics.Add(new(GraphDiagnosticSeverity.Error, code, message, node,
                resource?.Id, port, graph.Id));
        }

        if (string.IsNullOrWhiteSpace(graph.Id))
        {
            Error("G0001", "A graph identifier is required.");
        }

        CheckIdentities(graph.Resources.Select(resource => resource.Id), "resource",
            (code, message) => Error(code, message));
        CheckIdentities(graph.Actions.Select(action => action.Id), "action",
            (code, message) => Error(code, message));
        if (diagnostics.Any(diagnostic => diagnostic.Code is "G0001" or "G0002"))
        {
            return diagnostics;
        }
        var resourceIndex = graph.Resources.ToLookup(resource => resource.Id, StringComparer.Ordinal);
        var actionIds = graph.Actions.Select(action => action.Id).ToHashSet(StringComparer.Ordinal);
        foreach (var resource in graph.Resources)
        {
            if (!ReferenceEquals(resource.Graph, graph))
            {
                Error("G0004", "The resource belongs to a different graph.", resource: resource);
            }

            if (!Enum.IsDefined(resource.GraphIO) || !Enum.IsDefined(resource.Scope) ||
                !Enum.IsDefined(resource.Permission) || resource.Usage is { } usage && !Enum.IsDefined(usage) ||
                !Enum.IsDefined(resource.Tensor.ElementType))
            {
                Error("G0005", "Unknown resource attribute value.", resource: resource);
            }

            if (string.IsNullOrWhiteSpace(resource.Category) || resource.Category is "operation" or "call")
            {
                Error("G0006", "The resource category is invalid.", resource: resource);
            }

            if (resource.Usage is ResourceUsage.Weight or ResourceUsage.Constant &&
                resource.Permission != ResourcePermission.ReadOnly)
            {
                Error("G0007", "Weight and constant resources require read-only permission.", resource: resource);
            }

            if (resource.BindingKey is not null && string.IsNullOrWhiteSpace(resource.BindingKey))
            {
                Error("G0020", "An external binding key must be nonblank.", resource: resource);
            }
        }

        foreach (var action in graph.Actions)
        {
            if (!ReferenceEquals(action.Graph, graph))
            {
                Error("G0004", "The action belongs to a different graph.", action.Id);
            }

            if (string.IsNullOrWhiteSpace(action.Category) || action.Category == "resource" ||
                action is OperationAction && action.Category == "call" ||
                action is CallAction && action.Category == "operation")
            {
                Error("G0006", "The action category is invalid.", action.Id);
            }

            switch (action)
            {
                case OperationAction operation:
                    if (string.IsNullOrWhiteSpace(operation.Operation.Name) || operation.Operation.Version <= 0)
                    {
                        Error("G0008", "An operation requires a name and positive contract version.", action.Id);
                    }

                    break;
                case CallAction call:
                    if (string.IsNullOrWhiteSpace(call.Target))
                    {
                        Error("G0009", "A call requires a target graph.", action.Id);
                    }

                    break;
                default:
                    Error("G0010", $"Unsupported action type '{action.GetType().Name}'.", action.Id);
                    break;
            }
            CheckIdentities(action.Bindings.Select(binding => binding.SourceProperty), "port",
                (code, message) => Error(code, message, action.Id));
            foreach (var binding in action.Bindings)
            {
                if (!Enum.IsDefined(binding.Access) || !Enum.IsDefined(binding.Target.TargetKind))
                {
                    Error("G0005", "Unknown binding attribute value.", action.Id, port: binding.SourceProperty);
                }

                if (string.IsNullOrWhiteSpace(binding.Target.TargetProperty))
                {
                    Error("G0011", "A binding requires a resource identifier.", action.Id, port: binding.SourceProperty);
                }

                if (binding.Target.TargetKind == ResourceTarget.Graph &&
                    (binding.Target.Target is null || binding.Target.Target == graph.Id))
                {
                    var resources = resourceIndex[binding.Target.TargetProperty].ToArray();
                    if (resources.Length == 0)
                    {
                        Error("G0012", $"Resource '{binding.Target.TargetProperty}' is not declared.",
                            action.Id, port: binding.SourceProperty);
                    }

                    if (resources.Length == 1)
                    {
                        CheckAccess(resources[0], binding, (code, message) =>
                            Error(code, message, action.Id, resources[0], binding.SourceProperty));
                    }
                }
                else if (binding.Target.TargetKind == ResourceTarget.Call &&
                    !graph.Actions.OfType<CallAction>().Any(call => call.Id == binding.Target.Target))
                {
                    Error("G0013", $"Call instance '{binding.Target.Target}' is not declared.",
                        action.Id, port: binding.SourceProperty);
                }
            }
        }

        foreach (var flow in graph.GraphFlow)
        {
            if (flow.Source.Graph != graph.Id || flow.Target.Graph != graph.Id ||
                !actionIds.Contains(flow.Source.Action) || !actionIds.Contains(flow.Target.Action))
            {
                Error("G0014", $"GraphFlow '{flow.Source.Action}' -> '{flow.Target.Action}' has an invalid endpoint.");
            }
        }
        if (HasCycle(graph.Actions.Select(action => action.Id),
            graph.GraphFlow.Select(flow => (flow.Source.Action, flow.Target.Action))))
        {
            Error("G0015", "GraphFlow contains a cycle.");
        }

        return diagnostics;
    }

    public static IReadOnlyList<GraphDiagnostic> ValidatePackage<TGraph>(
        IEnumerable<TGraph> graphs, string? mainGraphId) where TGraph : Graph
    {
        ArgumentNullException.ThrowIfNull(graphs);
        var definitions = graphs.ToArray();
        var diagnostics = definitions.SelectMany(graph => graph.Validate()).ToList();
        void Error(string code, string message, string? graph = null, string? node = null, string? port = null)
        {
            diagnostics.Add(new(GraphDiagnosticSeverity.Error, code, message, node, Port: port, Graph: graph));
        }
        CheckIdentities(definitions.Select(graph => graph.Id), "graph",
            (code, message) => Error(code, message));
        if (string.IsNullOrWhiteSpace(mainGraphId) || !definitions.Any(graph => graph.Id == mainGraphId))
        {
            Error("P0001", $"Main graph '{mainGraphId}' is not declared.");
        }

        if (definitions.Any(graph => string.IsNullOrWhiteSpace(graph.Id)) ||
            definitions.Select(graph => graph.Id).Distinct(StringComparer.Ordinal).Count() != definitions.Length)
        {
            return diagnostics;
        }

        var index = definitions.ToDictionary(graph => graph.Id, graph => (Graph)graph, StringComparer.Ordinal);
        foreach (var shared in definitions.SelectMany(graph => graph.Resources)
            .Where(resource => resource.Scope != ResourceScope.Call && resource.BindingKey is not null)
            .GroupBy(resource => (resource.Scope, resource.BindingKey)))
        {
            var first = shared.First();
            if (shared.Any(resource => !SameTensor(first.Tensor, resource.Tensor) || first.Permission != resource.Permission))
            {
                Error("P0011", $"Shared binding '{shared.Key}' has incompatible tensor or permission declarations.");
            }
        }
        var used = new HashSet<string>(StringComparer.Ordinal);
        var pending = new Queue<string>();
        if (mainGraphId is not null && index.ContainsKey(mainGraphId))
        {
            pending.Enqueue(mainGraphId);
        }

        while (pending.TryDequeue(out var id))
        {
            if (!used.Add(id))
            {
                continue;
            }

            foreach (var target in ReferencedGraphs(index[id]))
            {
                if (index.ContainsKey(target))
                {
                    pending.Enqueue(target);
                }
            }
        }

        foreach (var graph in definitions)
        {
            if (!used.Contains(graph.Id))
            {
                diagnostics.Add(new(GraphDiagnosticSeverity.Warning, "P0002", "The graph is unused.", Graph: graph.Id));
            }

            foreach (var call in graph.Actions.OfType<CallAction>())
            {
                if (call.Target == mainGraphId)
                {
                    Error("P0003", "The main graph cannot be called.", graph.Id, call.Id);
                }

                if (!index.TryGetValue(call.Target, out var target))
                {
                    Error("P0004", $"Target graph '{call.Target}' is not declared.", graph.Id, call.Id);
                    continue;
                }
                foreach (var formal in target.GraphIOs)
                {
                    var bindings = call.Bindings.Where(binding => binding.SourceProperty == formal.Id).ToArray();
                    if (bindings.Length == 0)
                    {
                        Error("P0005", $"Call port '{formal.Id}' is missing.", graph.Id, call.Id, formal.Id);
                    }

                    if (bindings.Length != 1)
                    {
                        continue;
                    }

                    var binding = bindings[0];
                    var expected = formal.GraphIO switch
                    {
                        GraphIO.In => ResourceAccess.Read,
                        GraphIO.Out => ResourceAccess.Write,
                        _ => ResourceAccess.ReadWrite,
                    };
                    if (binding.Access != expected)
                    {
                        Error("P0006", $"Port '{formal.Id}' requires {expected} access.", graph.Id, call.Id, formal.Id);
                    }

                    if (TryResolve(graph, binding.Target, index, out var actual) && !SameTensor(actual.Tensor, formal.Tensor))
                    {
                        Error("P0007", $"Port '{formal.Id}' has an incompatible tensor.", graph.Id, call.Id, formal.Id);
                    }
                }
                foreach (var binding in call.Bindings)
                {
                    if (!target.GraphIOs.Any(formal => formal.Id == binding.SourceProperty))
                    {
                        Error("P0008", $"Unknown call port '{binding.SourceProperty}'.", graph.Id, call.Id, binding.SourceProperty);
                    }
                }
            }
            foreach (var action in graph.Actions)
            {
                foreach (var binding in action.Bindings)
                {
                    if (!TryResolve(graph, binding.Target, index, out var resource))
                    {
                        Error("P0009", $"Cannot resolve resource '{binding.Target}'.", graph.Id, action.Id, binding.SourceProperty);
                        continue;
                    }
                    CheckAccess(resource, binding, (code, message) =>
                        Error(code, message, graph.Id, action.Id, binding.SourceProperty));
                }
            }
        }
        if (HasCycle(index.Keys, definitions.SelectMany(graph =>
            graph.Actions.OfType<CallAction>().Select(call => (graph.Id, call.Target)))))
        {
            Error("P0010", "Graph calls contain a recursion cycle.");
        }

        return diagnostics.Distinct().ToArray();
    }

    public static bool TryResolve(
        Graph context, ResourceRef reference, IReadOnlyDictionary<string, Graph> graphs, out Resource resource)
    {
        Graph? target = null;
        if (reference.TargetKind == ResourceTarget.Graph)
        {
            graphs.TryGetValue(reference.Target ?? context.Id, out target);
        }
        else
        {
            var calls = context.Actions.OfType<CallAction>().Where(call => call.Id == reference.Target).ToArray();
            if (calls.Length == 1)
            {
                graphs.TryGetValue(calls[0].Target, out target);
            }
        }
        var candidates = target?.Resources.Where(value => value.Id == reference.TargetProperty).ToArray() ?? [];
        resource = candidates.Length == 1 ? candidates[0] : null!;
        return candidates.Length == 1;
    }

    public static bool SameTensor(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType && left.Layout == right.Layout &&
        left.Dimensions.SequenceEqual(right.Dimensions);

    public static void ThrowIfErrors(IEnumerable<GraphDiagnostic> diagnostics)
    {
        var errors = diagnostics.Where(item => item.Severity == GraphDiagnosticSeverity.Error).ToArray();
        if (errors.Length != 0)
        {
            throw new InvalidDataException(string.Join(Environment.NewLine, errors.Select(error =>
                $"{error.Code} [{error.Graph}/{error.Node}/{error.Port}]: {error.Message}")));
        }
    }

    private static void CheckAccess(Resource resource, Binding binding, Action<string, string> error)
    {
        if (binding.Access != ResourceAccess.Read &&
            (resource.Permission == ResourcePermission.ReadOnly || resource.GraphIO == GraphIO.In))
        {
            error("G0016", $"Resource '{resource.Id}' is read-only.");
        }
    }

    private static void CheckIdentities(IEnumerable<string> ids, string kind, Action<string, string> error)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var id in ids)
        {
            if (string.IsNullOrWhiteSpace(id))
            {
                error("G0001", $"A {kind} identifier is required.");
            }
            else if (!seen.Add(id))
            {
                error("G0002", $"Duplicate {kind} identifier '{id}'.");
            }
        }
    }

    private static IEnumerable<string> ReferencedGraphs(Graph graph) =>
        graph.Actions.OfType<CallAction>().Select(call => call.Target).Concat(
            graph.Actions.SelectMany(action => action.Bindings)
                .Where(binding => binding.Target.TargetKind == ResourceTarget.Graph && binding.Target.Target is not null)
                .Select(binding => binding.Target.Target!));

    private static bool HasCycle(IEnumerable<string> vertices, IEnumerable<(string From, string To)> edges)
    {
        var ids = vertices.Distinct(StringComparer.Ordinal).ToArray();
        var degree = ids.ToDictionary(id => id, _ => 0, StringComparer.Ordinal);
        var next = ids.ToDictionary(id => id, _ => new List<string>(), StringComparer.Ordinal);
        foreach (var (from, to) in edges)
        {
            if (!next.TryGetValue(from, out var successors) || !degree.ContainsKey(to))
            {
                continue;
            }

            successors.Add(to);
            degree[to]++;
        }
        var ready = new Queue<string>(ids.Where(id => degree[id] == 0));
        var count = 0;
        while (ready.TryDequeue(out var id))
        {
            count++;
            foreach (var successor in next[id])
            {
                if (--degree[successor] == 0)
                {
                    ready.Enqueue(successor);
                }
            }
        }
        return count != ids.Length;
    }
}
