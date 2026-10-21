using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Graphs.Logical;
using SharpInference.Instructions;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Compiles stable logical sources through ordered execution transformations.</summary>
public sealed class ExecutionGraphCompiler
{
    private readonly IExecutionGraphConverter converter;
    private readonly IReadOnlyList<IExecutionGraphPostprocessor> postprocessors;
    private readonly ExecutionImplementationSelector selector;

    public ExecutionGraphCompiler(
        IEnumerable<IInstructionCollectionProvider> providers,
        IEnumerable<IExecutionGraphPostprocessor>? postprocessors = null,
        IExecutionGraphConverter? converter = null)
    {
        ArgumentNullException.ThrowIfNull(providers);
        var registry = new InstructionRegistry(providers);
        this.converter = converter ?? new DefaultExecutionGraphConverter([registry]);
        selector = new([registry]);
        this.postprocessors = postprocessors?.ToArray() ?? [];
        if (this.postprocessors.Any(processor => processor is null || string.IsNullOrWhiteSpace(processor.Name)))
            throw new ArgumentException("Every postprocessor requires an instance and name.", nameof(postprocessors));
    }

    public GraphPackage<ExecutionGraph> Compile(
        GraphPackage<LogicalGraph> package, ExecutionGraphCompilationContext context)
    {
        ArgumentNullException.ThrowIfNull(package);
        ArgumentNullException.ThrowIfNull(context);
        var source = package.Snapshot();
        var diagnostics = source.Validate();
        GraphValidation.ThrowIfErrors(diagnostics);
        var analyses = new Dictionary<string, GraphAnalysis>(StringComparer.Ordinal);
        foreach (var graph in source)
            GraphValidation.ThrowIfErrors(GraphDependencyAnalyzer.AnalyzeValidated(source, graph, diagnostics, analyses).Diagnostics);
        var result = (converter.Convert(source, context)
            ?? throw new InvalidOperationException("The converter returned null.")).Snapshot();
        ExecutionPackageCompilation.Prepare(result, context.Backend, selector);
        for (var index = 0; index < postprocessors.Count; index++)
        {
            var processor = postprocessors[index];
            result = (processor.Process(result, new(source, context, index, processor.Name))
                ?? throw new InvalidOperationException($"Processor '{processor.Name}' returned null.")).Snapshot();
            ExecutionPackageCompilation.Prepare(result, context.Backend, selector);
        }
        return result;
    }
}
