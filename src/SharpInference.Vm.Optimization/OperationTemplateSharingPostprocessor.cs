using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using SharpInference.Graphs;
using SharpInference.Graphs.Core;

namespace SharpInference.Vm.Optimization;

/// <summary>Extracts repeated operation contracts into parameterized atomic graphs without eliminating executions.</summary>
public sealed class OperationTemplateSharingPostprocessor : IExecutionGraphPostprocessor
{
    public string Name => "share-execution-operation-templates";

    public ExecutionGraphProgram Process(
        ExecutionGraphProgram program, ExecutionGraphPostprocessContext context)
    {
        ArgumentNullException.ThrowIfNull(program);
        ArgumentNullException.ThrowIfNull(context);
        var graphIds = program.Graphs.Select(graph => graph.Id).ToHashSet();
        var result = new List<ExecutionGraph>();
        var templates = new List<ExecutionGraph>();
        foreach (var graph in program.Graphs)
        {
            var groups = new Dictionary<string, List<ExecutionOperationNode>>(StringComparer.Ordinal);
            var bindings = new Dictionary<ExecutionNodeId, IReadOnlyList<ModelResourceBinding>>();
            foreach (var operation in graph.Nodes.OfType<ExecutionOperationNode>())
            {
                var ports = graph.Bindings(operation);
                // Alias-bearing and instance-scoped contracts remain untouched.
                if (ports.Count == 0 || ports.Any(port => !port.Resource.Owner.IsParent) ||
                    ports.Select(port => port.Resource).Distinct().Count() != ports.Count)
                    continue;
                bindings.Add(operation.Id, ports);
                var key = Signature(graph, operation, ports);
                if (!groups.TryGetValue(key, out var group))
                    groups.Add(key, group = []);
                group.Add(operation);
            }

            var calls = new Dictionary<ExecutionNodeId, ExecutionGraphCallNode>();
            foreach (var (key, operations) in groups.Where(pair => pair.Value.Count > 1))
            {
                var idBase = "shared-operation." + Convert.ToHexStringLower(
                    SHA256.HashData(Encoding.UTF8.GetBytes(graph.Id.Value + "\n" + key)));
                var id = new ExecutionGraphId(idBase);
                for (var suffix = 1; !graphIds.Add(id); suffix++)
                    id = new(idBase + "." + suffix);
                var exemplar = operations[0];
                var ports = bindings[exemplar.Id];
                var builder = new ExecutionGraphBuilder(id, GraphCompileBoundary.Atomic);
                foreach (var port in ports)
                {
                    var resource = (ExecutionResourceNode)graph.Structure.GetNode(port.Resource.GraphId);
                    builder.AddResource(new(new(port.Port), port.Port, port.Access switch
                    {
                        GraphResourceAccess.Read => GraphDefinitionResourceKind.Input,
                        GraphResourceAccess.Write => GraphDefinitionResourceKind.Output,
                        _ => GraphDefinitionResourceKind.InOut,
                    }, resource.Tensor));
                }
                builder.AddOperation(exemplar with
                {
                    Id = new("body"),
                    Source = new(operations.SelectMany(operation => operation.Source.LogicalNodes)
                        .Distinct().ToArray(), Name),
                    Architecture = null,
                }, ports.Select(port => port with
                {
                    Resource = new(GraphBindingOwner.Parent, new(port.Port)),
                    BindingId = $"body:{port.Port}",
                }));
                templates.Add(builder.Build());
                foreach (var operation in operations)
                    calls.Add(operation.Id, new(operation.Id, id, GraphCompileBoundary.Atomic,
                        operation.Source with { Transformation = Name }, operation.Architecture));
            }
            result.Add(calls.Count == 0 ? graph : ExecutionDependencyAnalysis.WithStructure(graph,
                new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(
                    graph.Structure.Nodes.Select(node =>
                        node is ExecutionOperationNode operation &&
                        calls.TryGetValue(operation.Id, out var call) ? call : node),
                    graph.Structure.Edges)));
        }
        return new(result.Concat(templates));
    }

    private static string Signature(
        ExecutionGraph graph, ExecutionOperationNode operation,
        IReadOnlyList<ModelResourceBinding> ports) =>
        JsonSerializer.Serialize(new
        {
            operation.Operation,
            operation.Requirements,
            Attributes = operation.Attributes.OrderBy(pair => pair.Key, StringComparer.Ordinal).ToArray(),
            Ports = ports.Select(port => new
            {
                port.Port,
                port.Access,
                Tensor = ((ExecutionResourceNode)graph.Structure.GetNode(port.Resource.GraphId)).Tensor,
            }).ToArray(),
        });
}
