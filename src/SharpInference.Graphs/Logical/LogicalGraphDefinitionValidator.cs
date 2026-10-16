using SharpInference.Graphs.Core;

namespace SharpInference.Graphs.Logical;

/// <summary>Validates logical graph relationships and derives dependency completion information.</summary>
internal static class LogicalGraphDefinitionValidator
{

    /// <summary>Validates a logical graph and derives completion sinks and terminal resource writers.</summary>
    /// <param name="graph">The definition whose nodes, bindings, resources, and dependencies are analyzed.</param>
    /// <returns>Completion information and validation diagnostics without modifying the definition.</returns>
    /// <remarks>
    /// Multiple-writer comparisons reuse one lazily constructed dependency reachability index.
    /// Reports missing definition identifiers (LG0001), invalid boundaries (LG0002), unknown
    /// prerequisites (LG0010), empty or duplicate ports (LG0012), writes to weights or constants
    /// (LG0013), unknown child definitions (LG0014), dependency cycles (LG0015), unknown binding
    /// resources (LG0016), unwritten public outputs or input/output resources (LG0020), and child
    /// port access, tensor, or missing-binding errors (LG0024, LG0025, LG0026).
    /// Writes in another call's instance scope produce LGW003. Multiple dependency-maximal
    /// writers of a parent output or input/output resource produce LGW002, not an error.
    /// Completion sinks use dependency edges only, as do terminal-writer reachability comparisons.
    /// Structural and binding reconstruction failures may throw before diagnostics can be returned.
    /// </remarks>
    /// <exception cref="InvalidDataException">Resource-access edges cannot be reconstructed into consistent bindings.</exception>
    /// <exception cref="ArgumentException">Operation, child-definition, or binding identities are duplicated in indexed collections.</exception>
    /// <exception cref="InvalidOperationException">A required single resource or call-port binding is not unique.</exception>
    internal static LogicalGraphAnalysis Analyze(LogicalGraphDefinition graph)
    {
        var diagnostics = new List<GraphDiagnostic>();
        if (string.IsNullOrWhiteSpace(graph.Id.Value))
        {
            diagnostics.Add(Error("LG0001", "A graph definition identifier is required."));
        }

        if (!Enum.IsDefined(graph.CompileBoundary))
        {
            diagnostics.Add(Error(
                "LG0002", $"Graph '{graph.Id}' has an invalid compile boundary."));
        }

        var nodes = graph.Nodes.ToDictionary(node => node.Id);
        var definitions = graph.Definitions.ToDictionary(definition => definition.Id);
        var calls = graph.Structure.Filter(node: node => node is GraphCallNode)
            .Nodes.Cast<GraphCallNode>().ToDictionary(call => call.Id);
        foreach (var resourceNode in graph.Structure.Nodes.OfType<LogicalResourceNode>())
        {
            var resource = resourceNode.Resource;
            if (resourceNode.ResourceNodeId.Owner.IsParent)
            {
                if (resource.Owner != graph.Id)
                    diagnostics.Add(Error("LG0028",
                        $"Parent resource '{resource.Id}' is declared by another graph '{resource.Owner}'.",
                        resource: resource.Id));
                continue;
            }
            var call = calls[resourceNode.ResourceNodeId.Owner.Call!.Value];
            if (!definitions.TryGetValue(call.Definition, out var definition))
                continue;
            var declared = definition.Resources.SingleOrDefault(value => value.Id == resource.Id);
            if (declared is null || declared.Owner != resource.Owner ||
                declared.Kind != resource.Kind || !SameTensor(declared.Tensor, resource.Tensor))
                diagnostics.Add(Error("LG0029",
                    $"Resource '{resourceNode.ResourceNodeId}' does not match its called definition.",
                    call.Id, resource.Id));
        }
        foreach (var node in graph.Nodes)
        {
            var ports = new HashSet<string>(StringComparer.Ordinal);
            foreach (var binding in graph.Bindings(node))
            {
                if (string.IsNullOrWhiteSpace(binding.Port) || !ports.Add(binding.Port))
                {
                    diagnostics.Add(Error(
                        "LG0012",
                        $"Node '{node.Id}' has a duplicate or empty port '{binding.Port}'.",
                        node.Id));
                }

                var resourceNode = graph.Structure.GetNode(binding.Resource.GraphId)
                    as LogicalResourceNode;
                if (resourceNode is null)
                {
                    diagnostics.Add(Error(
                        "LG0016",
                        $"Node '{node.Id}' references unknown resource '{binding.Resource}'.",
                        node.Id,
                        binding.Resource.Resource));
                    continue;
                }
                if (!binding.Resource.Owner.IsParent &&
                    binding.Access != GraphResourceAccess.Read)
                {
                    diagnostics.Add(new(
                        GraphDiagnosticSeverity.Warning,
                        "LGW003",
                        $"Node '{node.Id}' writes resource '{binding.Resource.Resource}' owned by " +
                        $"graph instance '{binding.Resource.Owner}'. This may cause unexpected " +
                        "cross-instance behavior.",
                        node.Id.Value,
                        binding.Resource.Resource));
                }

                if (resourceNode.Kind is GraphDefinitionResourceKind.Weight
                        or GraphDefinitionResourceKind.Constant &&
                    binding.Access != GraphResourceAccess.Read)
                {
                    diagnostics.Add(Error(
                        "LG0013",
                        $"Node '{node.Id}' writes read-only resource '{binding.Resource.Resource}'.",
                        node.Id,
                        binding.Resource.Resource));
                }
            }

            foreach (var dependency in graph.Dependencies(node))
            {
                if (!nodes.ContainsKey(dependency))
                {
                    diagnostics.Add(Error(
                        "LG0010", $"Node '{node.Id}' depends on an unknown node.", node.Id));
                }
            }

            if (node is not GraphCallNode call)
            {
                continue;
            }

            if (!definitions.TryGetValue(call.Definition, out var called))
            {
                diagnostics.Add(Error(
                    "LG0014",
                    $"Call '{call.Id}' references unknown definition '{call.Definition}'.",
                    call.Id));
                continue;
            }
            var bindings = graph.Bindings(call);
            var publicPorts = called.Resources.Where(resource =>
                resource.Kind is GraphDefinitionResourceKind.Input
                    or GraphDefinitionResourceKind.Output
                    or GraphDefinitionResourceKind.InOut).ToArray();
            foreach (var binding in bindings.Where(binding =>
                !publicPorts.Any(port => port.Id.Value == binding.Port)))
                diagnostics.Add(new(GraphDiagnosticSeverity.Error, "LG0027",
                    $"Call '{call.Id}' has unknown port '{binding.Port}'.",
                    call.Id.Value, binding.Resource.Resource, binding.Port));
            foreach (var port in publicPorts)
            {
                var binding = bindings.SingleOrDefault(value => value.Port == port.Id.Value);
                if (binding is null)
                {
                    diagnostics.Add(Error(
                        "LG0026", $"Call '{call.Id}' is missing port '{port.Id}'.",
                        call.Id, port.Id));
                    continue;
                }
                var expected = port.Kind switch
                {
                    GraphDefinitionResourceKind.Input => GraphResourceAccess.Read,
                    GraphDefinitionResourceKind.Output => GraphResourceAccess.Write,
                    _ => GraphResourceAccess.ReadWrite,
                };
                if (binding.Access != expected)
                {
                    diagnostics.Add(Error(
                        "LG0024",
                        $"Call '{call.Id}' port '{binding.Port}' requires {expected} access.",
                        call.Id,
                        binding.Resource.Resource));
                }

                var actual = (LogicalResourceNode)graph.Structure.GetNode(
                    binding.Resource.GraphId);
                if (!SameTensor(port.Tensor, actual.Tensor))
                {
                    diagnostics.Add(Error(
                        "LG0025",
                        $"Call '{call.Id}' port '{binding.Port}' has an incompatible tensor resource.",
                        call.Id,
                        binding.Resource.Resource));
                }
            }
        }

        try
        {
            _ = graph.Structure.DependencyView().TopologicalOrder();
        }
        catch (InvalidDataException exception)
        {
            diagnostics.Add(Error("LG0015", exception.Message));
        }

        var sinks = graph.Nodes
            .Where(node => !graph.Structure.Successors(node).Any())
            .Select(node => node.Id)
            .ToArray();
        var terminal = new Dictionary<ResourceId, GraphDefinitionNodeId[]>();
        ReachabilityIndex<LogicalGraphNode>? reachability = null;
        foreach (var output in graph.Resources.Where(resource =>
            resource.Kind is GraphDefinitionResourceKind.Output
                or GraphDefinitionResourceKind.InOut))
        {
            var resourceNode = graph.Structure.NodesOfType<LogicalResourceNode>()
                .Single(node => node.ResourceNodeId ==
                    new ModelResourceNodeId(GraphBindingOwner.Parent, output.Id));
            var writers = graph.Structure.Writers(resourceNode)
                .Cast<GraphDefinitionNode>()
                .Select(node => node.Id)
                .Distinct()
                .ToArray();
            if (writers.Length == 0)
            {
                diagnostics.Add(Error(
                    "LG0020",
                    $"Graph output '{output.Id}' is never written.",
                    resource: output.Id));
                terminal.Add(output.Id, []);
                continue;
            }
            var maximal = writers;
            if (writers.Length > 1)
            {
                reachability ??= graph.Structure.DependencyView()
                    .CreateReachabilityIndex(graph.Nodes, graph.Nodes);
                maximal = writers.Where(writer => !writers.Any(other =>
                    other != writer &&
                    reachability.IsReachable(
                        GraphDefinitionNode.NodeGraphId(writer),
                        GraphDefinitionNode.NodeGraphId(other)))).ToArray();
            }
            terminal.Add(output.Id, maximal);
            if (maximal.Length > 1)
            {
                diagnostics.Add(new(
                    GraphDiagnosticSeverity.Warning,
                    "LGW002",
                    $"Graph output '{output.Id}' has unordered terminal writers: " +
                    string.Join(", ", maximal.Select(writer => $"'{writer}'")) + ".",
                    Resource: output.Id));
            }
        }
        return new LogicalGraphAnalysis(sinks, terminal, diagnostics);
    }

    /// <summary>Creates an error finding with optional operation and resource context.</summary>
    /// <param name="code">The diagnostic code.</param>
    /// <param name="message">The human-readable failure description.</param>
    /// <param name="node">The affected operation identity, if applicable.</param>
    /// <param name="resource">The affected resource identity, if applicable.</param>
    /// <returns>An error-severity diagnostic.</returns>
    private static GraphDiagnostic Error(
        string code,
        string message,
        GraphDefinitionNodeId? node = null,
        ResourceId? resource = null) =>
        new(GraphDiagnosticSeverity.Error, code, message, node?.Value, resource);

    /// <summary>Compares the complete tensor contract used for child-port compatibility.</summary>
    /// <param name="left">The declared child-port descriptor.</param>
    /// <param name="right">The bound resource descriptor.</param>
    /// <returns>True when element type, layout, and ordered dimensions match.</returns>
    private static bool SameTensor(TensorDescriptor left, TensorDescriptor right) =>
        left.ElementType == right.ElementType &&
        left.Layout == right.Layout &&
        left.Dimensions.SequenceEqual(right.Dimensions);
}
