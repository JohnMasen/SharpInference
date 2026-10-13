using System.Threading;
using System.Globalization;
using SharpInference.Graphs;

namespace SharpInference.Architectures;

internal sealed class RwkvTensor;

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

internal sealed record RwkvGraphPorts(
    GraphInput<RwkvTensor> Token,
    GraphOutput<RwkvTensor> Logits) : IGraphPortContract<RwkvGraphPorts>
{
    private static readonly AsyncLocal<(TensorDescriptor Token, TensorDescriptor Logits)?> descriptors = new();

    public static RwkvGraphPorts Create(IGraphPortFactory ports)
    {
        var value = descriptors.Value ??
            throw new InvalidOperationException("RWKV graph port descriptors are not configured.");
        return new(
            ports.AddInput<RwkvTensor>("token", value.Token),
            ports.AddOutput<RwkvTensor>("logits", value.Logits));
    }

    internal static IDisposable Use(TensorDescriptor token, TensorDescriptor logits)
    {
        var previous = descriptors.Value;
        descriptors.Value = (token, logits);
        return new DescriptorScope(previous);
    }

    private sealed class DescriptorScope((TensorDescriptor Token, TensorDescriptor Logits)? previous) : IDisposable
    {
        public void Dispose() => descriptors.Value = previous;
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

    private readonly GraphDefinitionBuilder<RwkvGraphPorts> builder;
    private readonly Dictionary<string, GraphResource<RwkvTensor>> resources =
        new(StringComparer.Ordinal);
    private readonly List<ArchitectureFrame> architecture = [];
    private GraphDefinitionNodeId? previous;

    private PortableRwkvGraphDefinitionComposer(GraphDefinitionBuilder<RwkvGraphPorts> builder)
    {
        this.builder = builder;
        resources.Add("token", builder.Ports.Token);
        resources.Add("logits", builder.Ports.Logits);
    }

    internal static LogicalGraphDefinition Define(
        string id,
        int vocabularySize,
        Action<PortableRwkvGraphDefinitionComposer> build,
        GraphAnnotation annotation)
    {
        using var descriptors = RwkvGraphPorts.Use(
            new TensorDescriptor(GraphElementType.Int32, [1]),
            new TensorDescriptor(GraphElementType.Float32, [vocabularySize]));
        return LogicalGraphDefinition.Define<RwkvGraphPorts>(
            id,
            graph =>
            {
                var composer = new PortableRwkvGraphDefinitionComposer(graph);
                build(composer);
            },
            GraphCompileBoundary.Atomic,
            annotation).Definition;
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
        resources.Add(id, builder.AddResource<RwkvTensor>(
            id, name, tensor, GraphDefinitionResourceKind.Weight, bindingKey));

    internal void AddState(string id, TensorDescriptor tensor) =>
        resources.Add(id, builder.AddResource<RwkvTensor>(
            id, id, tensor, GraphDefinitionResourceKind.State, id));

    internal void AddTemporary(string id, TensorDescriptor tensor) =>
        resources.Add(id, builder.AddResource<RwkvTensor>(
            id, id, tensor, GraphDefinitionResourceKind.Temporary));

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
                var resource = Get(input.Resource);
                if (resource.Resource.Kind == GraphDefinitionResourceKind.State)
                {
                    node.AddInOut(input.Port, resource);
                }
                else
                {
                    node.AddInput(input.Port, resource);
                }
            }
            node.AddOutput("output", Get(output));
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

    private GraphResource<RwkvTensor> Get(string id) =>
        resources.TryGetValue(id, out var resource)
            ? resource
            : throw new InvalidDataException($"Unknown RWKV graph resource '{id}'.");

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
