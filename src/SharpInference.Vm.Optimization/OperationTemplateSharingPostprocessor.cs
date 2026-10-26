using SharpInference.Instructions;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Execution;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Vm.Optimization;

/// <summary>Shares repeated operation definitions while retaining each call and its bindings.</summary>
public sealed class OperationTemplateSharingPostprocessor : IExecutionGraphPostprocessor
{
    public string Name => "share-execution-operation-templates";

    public GraphPackage<ExecutionGraph> Process(GraphPackage<ExecutionGraph> package, ExecutionPassContext context)
    {
        ArgumentNullException.ThrowIfNull(package);
        ArgumentNullException.ThrowIfNull(context);
        GraphValidation.ThrowIfErrors(package.Validate());
        var result = package.Snapshot();
        var definitions = result.ToDictionary(graph => graph.Id, graph => (SharpInference.Graphs.Graph)graph, StringComparer.Ordinal);
        var ids = result.Select(graph => graph.Id).ToHashSet(StringComparer.Ordinal);
        var templates = new List<ExecutionGraph>();
        foreach (var graph in result)
        {
            var candidates = graph.Actions.OfType<ExecutionOperation>().Where(operation =>
                operation.Bindings.Count > 0 && operation.Bindings.All(binding =>
                    binding.Target.TargetKind == ResourceTarget.Graph &&
                    (binding.Target.Target is null || binding.Target.Target == graph.Id)) &&
                operation.Bindings.Select(binding => binding.Target.TargetProperty).Distinct(StringComparer.Ordinal).Count() ==
                    operation.Bindings.Count).ToArray();
            string Key(ExecutionOperation operation) => JsonSerializer.Serialize(new
            {
                operation.Category,
                operation.Operation,
                operation.Precision,
                operation.SelectedImplementation,
                Attributes = operation.Attributes.OrderBy(pair => pair.Key, StringComparer.Ordinal),
                Ports = operation.Bindings.OrderBy(binding => binding.SourceProperty, StringComparer.Ordinal).Select(binding =>
                {
                    if (!GraphValidation.TryResolve(graph, binding.Target, definitions, out var resource))
                    {
                        throw new InvalidDataException($"Cannot resolve template port '{binding.SourceProperty}'.");
                    }

                    return new { binding.SourceProperty, binding.Access, resource.Tensor };
                }),
            });
            foreach (var group in candidates.GroupBy(Key).Where(group => group.Count() > 1))
            {
                var name = "shared-operation." + Convert.ToHexStringLower(
                    SHA256.HashData(Encoding.UTF8.GetBytes(graph.Id + "\n" + group.Key)));
                var id = name;
                for (var suffix = 1; !ids.Add(id); suffix++)
                {
                    id = name + "." + suffix;
                }

                var template = new ExecutionGraph { Id = id };
                var exemplar = group.First();
                var body = template.Operation("body", exemplar.Operation);
                body.Category = exemplar.Category;
                body.Precision = exemplar.Precision;
                body.SelectedImplementation = exemplar.SelectedImplementation;
                foreach (var pair in exemplar.Attributes)
                {
                    body.Attributes.Add(pair.Key, pair.Value);
                }

                body.Sources.AddRange(group.SelectMany(operation => operation.Sources).Distinct());
                foreach (var binding in exemplar.Bindings)
                {
                    GraphValidation.TryResolve(graph, binding.Target, definitions, out var actual);
                    var resource = template.Resource(binding.SourceProperty, actual.Tensor, binding.Access switch
                    {
                        ResourceAccess.Read => GraphIO.In,
                        ResourceAccess.Write => GraphIO.Out,
                        _ => GraphIO.InOut,
                    });
                    template.Storage.Add(resource.Id, new(ResourceStorageKind.ExternalArgument));
                    body.Bind(binding.SourceProperty, resource, binding.Access);
                }
                foreach (var operation in group)
                {
                    var index = graph.Actions.IndexOf(operation);
                    var call = graph.Call(operation.Id, template);
                    graph.Actions.RemoveAt(graph.Actions.Count - 1);
                    call.Category = operation.Category == "operation" ? "call" : operation.Category;
                    call.Metadata.Add("transformation", Name);
                    call.Bindings.AddRange(operation.Bindings);
                    call.Sources.AddRange(operation.Sources);
                    foreach (var pair in operation.Metadata)
                    {
                        call.Metadata.TryAdd(pair.Key, pair.Value);
                    }

                    graph.Actions[index] = call;
                }
                templates.Add(template);
            }
        }
        foreach (var template in templates)
        {
            result.Add(template);
        }

        return result;
    }

}
