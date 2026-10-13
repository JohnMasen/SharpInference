using System.Collections.ObjectModel;
using SharpInference.Graphs.Core;

namespace SharpInference.Graphs;

public interface IExecutionGraphConverter
{
    ExecutionGraphProgram Convert(
        LogicalGraphDefinition graph,
        ExecutionGraphCompilationContext context);
}

public interface IExecutionGraphPostprocessor
{
    string Name { get; }
    ExecutionGraphProgram Process(
        ExecutionGraphProgram program,
        ExecutionGraphPostprocessContext context);
}

public sealed record ExecutionGraphPostprocessContext(
    LogicalGraphDefinition SourceGraph,
    ExecutionGraphCompilationContext Compilation,
    int ProcessorIndex,
    string ProcessorName);

public sealed class ExecutionGraphCompiler
{
    private readonly IExecutionGraphConverter converter;
    private readonly IReadOnlyList<IExecutionGraphPostprocessor> postprocessors;

    public ExecutionGraphCompiler(
        IEnumerable<IExecutionGraphPostprocessor>? postprocessors = null)
        : this(new DefaultExecutionGraphConverter(), postprocessors) { }

    public ExecutionGraphCompiler(
        IExecutionGraphConverter converter,
        IEnumerable<IExecutionGraphPostprocessor>? postprocessors = null)
    {
        this.converter = converter ?? throw new ArgumentNullException(nameof(converter));
        this.postprocessors = Array.AsReadOnly(postprocessors?.ToArray() ?? []);
        if (this.postprocessors.Any(processor =>
            processor is null || string.IsNullOrWhiteSpace(processor.Name)))
        {
            throw new ArgumentException("Every postprocessor requires an instance and name.", nameof(postprocessors));
        }
    }

    public ExecutionGraphProgram Compile(
        LogicalGraphDefinition graph,
        ExecutionGraphCompilationContext context)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(context);
        var program = converter.Convert(graph, context)
            ?? throw new InvalidOperationException("The execution graph converter returned null.");
        for (var index = 0; index < postprocessors.Count; index++)
        {
            var processor = postprocessors[index];
            program = processor.Process(
                program,
                new ExecutionGraphPostprocessContext(graph, context, index, processor.Name))
                ?? throw new InvalidOperationException(
                    $"Execution graph postprocessor '{processor.Name}' returned null.");
            program = new ExecutionGraphProgram(program.Graphs);
        }
        return program;
    }
}

/// <summary>Copies logical definitions and calls into an unoptimized execution graph program.</summary>
public sealed class DefaultExecutionGraphConverter : IExecutionGraphConverter
{
    public ExecutionGraphProgram Convert(
        LogicalGraphDefinition graph,
        ExecutionGraphCompilationContext context)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(context);
        var definitions = Collect(graph);
        var graphs = definitions.Values.Select(ConvertDefinition).ToArray();
        return new ExecutionGraphProgram(graphs);
    }

    private static IReadOnlyDictionary<GraphDefinitionId, LogicalGraphDefinition> Collect(
        LogicalGraphDefinition root)
    {
        var result = new Dictionary<GraphDefinitionId, LogicalGraphDefinition>();
        Visit(root);
        return result;

        void Visit(LogicalGraphDefinition definition)
        {
            if (result.TryGetValue(definition.Id, out var existing))
            {
                if (!ReferenceEquals(existing, definition) &&
                    GraphXml.Serialize(existing) != GraphXml.Serialize(definition))
                {
                    throw new InvalidDataException(
                        $"Conflicting logical definitions share identifier '{definition.Id}'.");
                }

                return;
            }
            result.Add(definition.Id, definition);
            foreach (var child in definition.Definitions)
            {
                Visit(child);
            }
        }
    }

    private static ExecutionGraph ConvertDefinition(LogicalGraphDefinition definition)
    {
        var nodes = definition.Structure.Nodes.Select<LogicalGraphNode, ExecutionGraphStructureNode>(
            node => node switch
        {
            LogicalResourceNode resource => new ExecutionResourceNode(
                resource.ResourceNodeId,
                new ExecutionGraphResource(
                    resource.Resource.Id,
                    resource.Resource.Name,
                    resource.Resource.Kind,
                    resource.Resource.Tensor,
                    resource.Resource.BindingKey)),
            GraphOperationNode operation => new ExecutionOperationNode(
                new ExecutionNodeId(operation.Id.Value),
                operation.Operation,
                new ReadOnlyDictionary<string, string>(
                    new Dictionary<string, string>(operation.Attributes, StringComparer.Ordinal)),
                operation.Requirements,
                new ExecutionSourceMap([operation.Id]),
                ConvertArchitecture(operation.Annotation)),
            GraphCallNode call => new ExecutionGraphCallNode(
                new ExecutionNodeId(call.Id.Value),
                new ExecutionGraphId(call.Definition.Value),
                definition.Definitions.Single(child => child.Id == call.Definition).CompileBoundary,
                new ExecutionSourceMap([call.Id]),
                ConvertArchitecture(call.Annotation)),
            _ => throw new InvalidDataException(
                $"Unsupported logical node type '{node.GetType().Name}'."),
        }).ToArray();
        var edges = definition.Structure.Edges.Select<LogicalGraphEdge, ExecutionGraphEdge>(
            edge => edge switch
            {
                LogicalDependencyEdge dependency => new ExecutionDependencyEdge(
                    dependency.Id,
                    dependency.Source,
                    dependency.Target),
                LogicalResourceReadEdge read => new ExecutionResourceReadEdge(
                    read.Id,
                    read.Source,
                    read.Target,
                    read.Port,
                    read.BindingId),
                LogicalResourceWriteEdge write => new ExecutionResourceWriteEdge(
                    write.Id,
                    write.Source,
                    write.Target,
                    write.Port,
                    write.BindingId),
                _ => throw new InvalidDataException(
                    $"Unsupported logical edge type '{edge.GetType().Name}'."),
            }).ToArray();
        return new ExecutionGraph(
            new ExecutionGraphId(definition.Id.Value),
            new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(nodes, edges),
            definition.CompileBoundary,
            ConvertArchitecture(definition.Annotation),
            new ExecutionGraphSource(definition.Id));
    }

    private static GraphArchitecture? ConvertArchitecture(GraphAnnotation? annotation) =>
        annotation is null
            ? null
            : new GraphArchitecture(
                annotation.Name,
                annotation.Role,
                annotation.Description,
                annotation.Attributes);
}
