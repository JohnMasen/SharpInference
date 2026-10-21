using SharpInference.Instructions;
using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

namespace SharpInference.Architectures;

internal static class RwkvPortableTensorOperations
{
    internal static readonly OperationId Fill = new("core.tensor.fill");
    internal static readonly OperationId CastFp16ToFp32 = new("core.tensor.cast-f16-f32");
    internal static readonly OperationId Reshape = new("core.tensor.reshape");
    internal static readonly OperationId Slice = new("core.tensor.slice");
    internal static readonly OperationId Broadcast = new("core.tensor.broadcast");
    internal static readonly OperationId BatchedMatVec = new("core.tensor.batched-mat-vec");
    internal static readonly OperationId ReduceLastSum = new("core.tensor.reduce-last-sum");
    internal static readonly OperationId ReduceLastMean = new("core.tensor.reduce-last-mean");
    internal static readonly OperationId HeadOuter = new("core.tensor.head-outer");

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
    private readonly PortableRwkvGraphComposer composer;

    internal RwkvArchitectureScope(PortableRwkvGraphComposer composer) =>
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

internal sealed class PortableRwkvGraphComposer
{
    private sealed record ArchitectureFrame(
        string Id,
        string Name,
        RwkvArchitectureAnnotation? Annotation);

    private readonly LogicalGraph graph;
    private readonly HashSet<string> states = new(StringComparer.Ordinal);
    private readonly List<ArchitectureFrame> architecture = [];
    private OperationAction? previous;

    private PortableRwkvGraphComposer(LogicalGraph graph)
    {
        this.graph = graph;
    }

    internal static GraphPackage<LogicalGraph> Define(
        string id,
        int vocabularySize,
        Action<PortableRwkvGraphComposer> build,
        string name,
        string description)
    {
        var graph = InferenceGraphBuilder.Builder<LogicalGraph>(id, graph =>
        {
            graph.CompileBoundary = GraphCompileBoundary.Atomic;
            graph.Name = name;
            graph.Description = description;
            graph.Metadata["role"] = "forward-token";
            graph.Resource("token", new TensorDescriptor(ElementType.Int32, [1]), GraphIO.In);
            graph.Resource("logits", new TensorDescriptor(ElementType.Float32, [vocabularySize]), GraphIO.Out);
            build(new PortableRwkvGraphComposer(graph));
        });
        return new(graph);
    }

    internal RwkvArchitectureScope Root(
        string id,
        string name,
        RwkvArchitectureAnnotation? annotation = null)
    {
        architecture.Add(new(id, name, annotation));
        return new(this);
    }

    internal void AddWeight(string id, string name, TensorDescriptor tensor, string bindingKey)
    {
        graph.Resource(id, tensor, scope: ResourceScope.Global, usage: ResourceUsage.Weight,
            permission: ResourcePermission.ReadOnly, bindingKey: bindingKey);
        graph.Metadata[$"resource.{id}.name"] = name;
    }

    internal void AddState(string id, TensorDescriptor tensor)
    {
        graph.Resource(id, tensor, GraphIO.InOut, ResourceScope.Session, isState: true, bindingKey: id);
        states.Add(id);
    }

    internal void AddTemporary(string id, TensorDescriptor tensor) =>
        graph.Resource(id, tensor);

    internal void AddNode(
        string id,
        OperationId operation,
        IEnumerable<(string Port, string Resource)> inputs,
        string output,
        IReadOnlyDictionary<string, string>? attributes,
        PrecisionRequirement requirements)
    {
        var node = graph.Operation(id, operation);
        foreach (var input in inputs)
        {
            node.Bind(input.Port, input.Resource,
                states.Contains(input.Resource) ? ResourceAccess.ReadWrite : ResourceAccess.Read);
        }
        node.Bind("output", output, ResourceAccess.Write);
        node.Precision = requirements;
        if (attributes is not null)
        {
            foreach (var attribute in attributes)
            {
                node.Attributes.Add(attribute.Key, attribute.Value);
            }
        }
        Annotate(node);

        previous?.Then(node);
        previous = node;
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

    private void Annotate(OperationAction node)
    {
        var leaf = architecture[^1];
        node.Metadata["name"] = leaf.Name;
        node.Metadata["architecture.path"] = string.Join("/", architecture.Select(frame => frame.Id));
        node.Metadata["architecture.name"] = leaf.Name;
        if (leaf.Annotation?.Role is { } role)
            node.Metadata["role"] = role;
        if (leaf.Annotation?.Description is { } description)
            node.Metadata["description"] = description;
        if (leaf.Annotation?.Formula is { Length: > 0 } formula)
        {
            node.Metadata["architecture.formula"] = formula;
        }
    }
}
