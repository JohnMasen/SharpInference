using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

namespace SharpInference.Architectures;

internal static class RwkvPortableTensorOperations
{
    internal static readonly GraphOperationId Fill = new("core.tensor.fill");
    internal static readonly GraphOperationId CastFp16ToFp32 = new("core.tensor.cast-f16-f32");
    internal static readonly GraphOperationId Reshape = new("core.tensor.reshape");
    internal static readonly GraphOperationId Slice = new("core.tensor.slice");
    internal static readonly GraphOperationId Broadcast = new("core.tensor.broadcast");
    internal static readonly GraphOperationId BatchedMatVec = new("core.tensor.batched-mat-vec");
    internal static readonly GraphOperationId ReduceLastSum = new("core.tensor.reduce-last-sum");
    internal static readonly GraphOperationId ReduceLastMean = new("core.tensor.reduce-last-mean");
    internal static readonly GraphOperationId HeadOuter = new("core.tensor.head-outer");

    internal static IReadOnlyDictionary<string, string> FillAttributes(float value)
    {
        if (!float.IsFinite(value))
        {
            throw new ArgumentOutOfRangeException(nameof(value));
        }

        return new Dictionary<string, string>
        {
            ["value"] = value.ToString("R", CultureInfo.InvariantCulture),
        };
    }

    internal static IReadOnlyDictionary<string, string> SliceAttributes(
        int axis,
        int start,
        int length)
    {
        if (axis < 0 || start < 0 || length <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }

        return new Dictionary<string, string>
        {
            ["axis"] = axis.ToString(CultureInfo.InvariantCulture),
            ["start"] = start.ToString(CultureInfo.InvariantCulture),
            ["length"] = length.ToString(CultureInfo.InvariantCulture),
        };
    }
}

internal sealed record RwkvArchitectureAnnotation(
    string? Role = null,
    string? Description = null,
    string? Formula = null);

internal sealed class RwkvArchitectureScope
{
    private readonly PortableRwkvGraphDefinitionComposer composer;

    internal RwkvArchitectureScope(PortableRwkvGraphDefinitionComposer composer) =>
        this.composer = composer;

    internal void Child(
        string id,
        string name,
        Action<RwkvArchitectureScope> build,
        RwkvArchitectureAnnotation? architecture = null) =>
        composer.WithArchitecture(id, name, architecture, () => build(this));

    internal T Value<T>(
        string id,
        string name,
        Func<RwkvArchitectureScope, T> build,
        RwkvArchitectureAnnotation? architecture = null) =>
        composer.WithArchitecture(id, name, architecture, () => build(this));
}

internal sealed class PortableRwkvGraphDefinitionComposer
{
    private sealed record ArchitectureFrame(
        string Id,
        string Name,
        RwkvArchitectureAnnotation? Annotation);

    private readonly GraphDefinitionBuilder builder;
    private readonly HashSet<string> states = new(StringComparer.Ordinal);
    private readonly List<ArchitectureFrame> architecture = [];
    private GraphDefinitionNodeId? previous;

    private PortableRwkvGraphDefinitionComposer(GraphDefinitionBuilder builder)
    {
        this.builder = builder;
    }

    internal static LogicalGraphDefinition Define(
        string id,
        int vocabularySize,
        Action<PortableRwkvGraphDefinitionComposer> build,
        GraphAnnotation annotation)
    {
        return LogicalGraphDefinition.Define(
            id,
            graph =>
            {
                graph.AddInput("token", new TensorDescriptor(GraphElementType.Int32, [1]));
                graph.AddOutput("logits", new TensorDescriptor(GraphElementType.Float32, [vocabularySize]));
                var composer = new PortableRwkvGraphDefinitionComposer(graph);
                build(composer);
            },
            GraphCompileBoundary.Atomic,
            annotation);
    }

    internal RwkvArchitectureScope Root(
        string id,
        string name,
        RwkvArchitectureAnnotation? annotation = null)
    {
        architecture.Add(new(id, name, annotation));
        return new(this);
    }

    internal void AddWeight(string id, string name, TensorDescriptor tensor, string bindingKey) =>
        builder.AddResource(id, name, tensor, GraphDefinitionResourceKind.Weight, bindingKey);

    internal void AddState(string id, TensorDescriptor tensor)
    {
        builder.AddResource(id, id, tensor, GraphDefinitionResourceKind.State, id);
        states.Add(id);
    }

    internal void AddTemporary(string id, TensorDescriptor tensor) =>
        builder.AddResource(id, id, tensor, GraphDefinitionResourceKind.Temporary);

    internal void AddNode(
        string id,
        GraphOperationId operation,
        IEnumerable<(string Port, string Resource)> inputs,
        string output,
        IReadOnlyDictionary<string, string>? attributes,
        PrecisionRequirement requirements)
    {
        builder.AddNode(id, operation, node =>
        {
            foreach (var input in inputs)
            {
                var resource = new ResourceId(input.Resource);
                if (states.Contains(input.Resource))
                {
                    node.AddInOut(input.Port, resource);
                }
                else
                {
                    node.AddInput(input.Port, resource);
                }
            }
            node.AddOutput("output", new ResourceId(output));
            if (attributes is not null)
            {
                foreach (var attribute in attributes)
                {
                    node.AddAttribute(attribute.Key, attribute.Value);
                }
            }

            node.RequirePrecision(requirements);
            if (previous is { } dependency)
            {
                node.DependsOn(dependency);
            }
        }, CurrentAnnotation());
        previous = new(id);
    }

    internal T WithArchitecture<T>(
        string id,
        string name,
        RwkvArchitectureAnnotation? annotation,
        Func<T> build)
    {
        architecture.Add(new(id, name, annotation));
        try
        {
            return build();
        }
        finally
        {
            architecture.RemoveAt(architecture.Count - 1);
        }
    }

    internal void WithArchitecture(
        string id,
        string name,
        RwkvArchitectureAnnotation? annotation,
        Action build) =>
        WithArchitecture(id, name, annotation, () =>
        {
            build();
            return true;
        });

    private GraphAnnotation CurrentAnnotation()
    {
        var leaf = architecture[^1];
        var attributes = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["architecture.path"] = string.Join("/", architecture.Select(frame => frame.Id)),
            ["architecture.name"] = leaf.Name,
        };
        if (leaf.Annotation?.Formula is { Length: > 0 } formula)
        {
            attributes.Add("architecture.formula", formula);
        }

        return new(leaf.Name, leaf.Annotation?.Role, leaf.Annotation?.Description, attributes);
    }
}
