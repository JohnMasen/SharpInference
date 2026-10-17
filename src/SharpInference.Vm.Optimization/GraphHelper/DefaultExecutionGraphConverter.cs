using System.Collections.ObjectModel;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Logical;

namespace SharpInference.Vm.Optimization.GraphHelper;

/// <summary>Copies logical definitions and calls into an unoptimized execution graph program.</summary>
/// <remarks>Nested definitions are collected by identity; operations and calls retain their local
/// identifiers, resource scopes, bindings, dependencies, and compilation-boundary metadata.
/// No boundary value triggers inlining or backend-specific lowering, and the context is only null-checked.</remarks>
public sealed class DefaultExecutionGraphConverter : IExecutionGraphConverter
{
    /// <summary>Copies the root and recursively declared definitions into a validated execution program.</summary>
    /// <param name="graph">The logical root and its nested definition declarations.</param>
    /// <param name="context">The required compilation context; its settings are not otherwise inspected.</param>
    /// <returns>A program whose call structure determines its unique root.</returns>
    /// <exception cref="ArgumentNullException">The graph or context is null.</exception>
    /// <exception cref="InvalidDataException">Definitions sharing an identity serialize differently,
    /// a node or edge type is unsupported, or the resulting execution graph or program is invalid.</exception>
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

    /// <summary>Collects recursively declared definitions, accepting repeated identities only for equivalent definitions.</summary>
    /// <param name="root">The definition at which recursive collection starts.</param>
    /// <returns>The first encountered definition for each identity.</returns>
    /// <exception cref="InvalidDataException">Distinct objects sharing an identity have different logical XML.</exception>
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
                    LogicalGraphXml.Serialize(existing) != LogicalGraphXml.Serialize(definition))
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

    /// <summary>Copies one definition's typed structure, boundaries, architecture, and logical provenance.</summary>
    /// <remarks>Operation attributes are copied with ordinal keys. Calls retain the target definition identity
    /// and use that directly declared definition's boundary; architecture attribute dictionaries are retained.</remarks>
    /// <param name="definition">The definition to convert without recursively converting its children here.</param>
    /// <returns>The corresponding validated execution graph.</returns>
    /// <exception cref="InvalidDataException">A node or edge type is unsupported or the copied graph fails validation.</exception>
    /// <exception cref="InvalidOperationException">A call target does not resolve to exactly one directly declared definition.</exception>
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

    /// <summary>Transfers optional logical annotation values without copying the attribute dictionary.</summary>
    /// <param name="annotation">The logical annotation, or null.</param>
    /// <returns>The corresponding architecture metadata, or null when no annotation exists.</returns>
    private static GraphArchitecture? ConvertArchitecture(GraphAnnotation? annotation) =>
        annotation is null
            ? null
            : new GraphArchitecture(
                annotation.Name,
                annotation.Role,
                annotation.Description,
                annotation.Attributes);
}
