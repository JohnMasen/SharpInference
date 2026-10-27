using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;
using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Analyzes a stable logical package and selects concrete backend implementations.</summary>
public sealed class DefaultExecutionGraphConverter : IExecutionGraphConverter
{
    private readonly IReadOnlyList<IInstructionCollectionProvider> providers;

    public DefaultExecutionGraphConverter(IEnumerable<IInstructionCollectionProvider> providers)
    {
        ArgumentNullException.ThrowIfNull(providers);
        this.providers = providers.ToArray();
    }

    public ExecutionGraphPackage Convert(
        LogicalGraphPackage package, ExecutionGraphCompilationContext context)
    {
        ArgumentNullException.ThrowIfNull(package);
        ArgumentNullException.ThrowIfNull(context);
        var source = package.Snapshot();
        var diagnostics = source.Validate();
        GraphValidation.ThrowIfErrors(diagnostics);
        var result = new ExecutionGraphPackage { MainGraphId = source.MainGraphId };
        var analyses = source.Analyze();
        foreach (var pair in source.Metadata)
        {
            result.Metadata.Add(pair.Key, pair.Value);
        }

        foreach (var definition in source)
        {
            var analysis = analyses[definition.Id];
            GraphValidation.ThrowIfErrors(analysis.Diagnostics);
            var graph = new ExecutionGraph { Id = definition.Id, Name = definition.Name, Description = definition.Description };
            foreach (var pair in definition.Metadata)
            {
                graph.Metadata.Add(pair.Key, pair.Value);
            }

            foreach (var resource in definition.Resources)
            {
                graph.Resource(resource.Id, resource.Tensor, resource.GraphIO, resource.Scope, resource.Usage,
                    resource.IsState, resource.Permission, resource.BindingKey).Category = resource.Category;
                graph.Storage.Add(resource.Id, ExecutionPackageCompilation.DefaultStorage(resource));
            }
            foreach (var action in definition.Actions)
            {
                GraphAction converted;
                switch (action)
                {
                    case OperationAction operation:
                        var execution = graph.Operation(action.Id, operation.Operation);
                        execution.Precision = operation.Precision;
                        foreach (var pair in operation.Attributes)
                        {
                            execution.Attributes.Add(pair.Key, pair.Value);
                        }

                        execution.Source(source.Metadata.GetValueOrDefault("document") ?? context.ModelType, definition.Id, action.Id);
                        converted = execution;
                        break;
                    case CallAction call:
                        converted = graph.Call(call.Id, call.Target)
                            .Source(source.Metadata.GetValueOrDefault("document") ?? context.ModelType, definition.Id, action.Id);
                        break;
                    default:
                        throw new NotSupportedException($"Unsupported action '{definition.Id}/{action.Id}'.");
                }
                converted.Category = action.Category;
                converted.Bindings.AddRange(action.Bindings);
                foreach (var pair in action.Metadata)
                {
                    converted.Metadata.Add(pair.Key, pair.Value);
                }
            }
            graph.GraphFlow.AddRange(analysis.Dependencies);
            result.Add(graph);
        }
        new ExecutionImplementationSelector(providers).Select(result, context.Backend);
        GraphValidation.ThrowIfErrors(result.Validate());
        return result;
    }

}
