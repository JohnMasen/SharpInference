using SharpInference.Graphs;

namespace SharpInference.Architectures.Phi4;

internal sealed record GraphId(string Architecture, int Version, string EntryPoint);
internal sealed record GraphModelSignature(
    string Architecture,
    string StateAbiId,
    IReadOnlyDictionary<string, int> Dimensions);
internal sealed record GraphStateSchema(string Id);
internal enum GraphResourceKind { Input, Output, Temporary, Weight, SessionState, Constant }
internal enum GraphResourceLifetime { External, Invocation, Model, Session }
internal enum GraphArchitectureView { Architecture, Computation }
internal enum GraphArchitectureStep { Token }
internal static class GraphSectionTypes
{
    internal const string Graph = "graph";
    internal const string Stage = "stage";
    internal const string Architecture = "architecture";
    internal const string Layer = "layer";
}
internal sealed record GraphSectionAnnotation(
    string? Role = null,
    string? Description = null,
    string? Formula = null,
    bool? DefaultCollapsed = null,
    bool? RepeatGroup = null,
    GraphArchitectureView? DefaultView = null,
    GraphArchitectureStep? Step = null);
internal sealed record Phi4TensorView(ulong Offset, TensorDescriptor Tensor);
internal sealed record NodeResourceBinding(
    string Port,
    ResourceId Resource,
    GraphResourceAccess Access,
    bool InitializedBeforeRead = false,
    Phi4TensorView? View = null);
internal static class GraphBindings
{
    internal static NodeResourceBinding Read(string port, string resource) =>
        new(port, new(resource), GraphResourceAccess.Read);
    internal static NodeResourceBinding Write(string port, string resource) =>
        new(port, new(resource), GraphResourceAccess.Write);
    internal static NodeResourceBinding ReadWrite(string port, string resource) =>
        new(port, new(resource), GraphResourceAccess.ReadWrite);
}
internal static class PortableTensorOperationContracts
{
    internal static readonly GraphOperationId CastFp16ToFp32 = new("core.tensor.cast-fp16-fp32", 1);
    internal static readonly GraphOperationId Fill = new("core.tensor.fill", 1);
}
internal sealed record TensorFillValue(float Value)
{
    internal IReadOnlyDictionary<string, string> ToAttributes() =>
        new Dictionary<string, string>
        {
            ["value"] = Value.ToString(System.Globalization.CultureInfo.InvariantCulture),
        };
}

/// <summary>Creates a reusable Phi-4 logical graph definition.</summary>
public interface IPhi4GraphModule
{
    string ArchitectureId { get; }
    bool CanLoad(IModelTensorCatalog tensors);
    ModelMetadata ReadMetadata(IModelTensorCatalog tensors);
    LogicalGraphDefinition Build(IModelTensorCatalog tensors);
}

/// <summary>
/// Adapts Phi-4's architecture-oriented construction code to the definition IR.
/// Sections are retained as node annotations because the definition IR has no region tree.
/// </summary>
internal sealed class Phi4GraphDefinitionBuilder
{
    private readonly string id;
    private readonly GraphAnnotation annotation;
    private readonly Dictionary<string, ResourceSpec> resources = new(StringComparer.Ordinal);
    private readonly List<StepSpec> steps = [];

    internal Phi4GraphDefinitionBuilder(string id, GraphAnnotation annotation)
    {
        this.id = id;
        this.annotation = annotation;
    }

    internal Phi4GraphDefinitionBuilder(GraphId id, GraphModelSignature model)
        : this(id.Architecture, new GraphAnnotation(
            id.EntryPoint,
            "model",
            Attributes: model.Dimensions.ToDictionary(
                pair => "dimension." + pair.Key,
                pair => pair.Value.ToString(System.Globalization.CultureInfo.InvariantCulture),
                StringComparer.Ordinal)))
    {
    }

    internal Phi4DefinitionScope Root(string id, string type, string name, string? role = null, GraphSectionAnnotation? architecture = null) =>
        new(this, id, new GraphAnnotation(name, role, architecture?.Description,
            architecture is null ? null : AnnotationAttributes(type, architecture)));

    internal Phi4GraphDefinitionBuilder AddResource(
        string id,
        string name,
        GraphResourceKind kind,
        GraphResourceLifetime lifetime,
        TensorDescriptor tensor,
        string? bindingKey = null,
        bool graphInput = false,
        bool graphOutput = false,
        ResourceId? declaredScope = null)
    {
        var definitionKind = graphInput ? GraphDefinitionResourceKind.Input :
            graphOutput ? GraphDefinitionResourceKind.Output :
            kind switch
            {
                GraphResourceKind.Input => GraphDefinitionResourceKind.Input,
                GraphResourceKind.Output => GraphDefinitionResourceKind.Output,
                GraphResourceKind.Weight => GraphDefinitionResourceKind.Weight,
                GraphResourceKind.SessionState => GraphDefinitionResourceKind.State,
                GraphResourceKind.Constant => GraphDefinitionResourceKind.Constant,
                _ => GraphDefinitionResourceKind.Temporary,
            };
        resources.Add(id, new(id, name, definitionKind, tensor, bindingKey));
        return this;
    }

    internal void AddNode(
        string id,
        GraphOperationId operation,
        IEnumerable<NodeResourceBinding> bindings,
        PrecisionRequirement? requirements = null,
        IReadOnlyDictionary<string, string>? attributes = null,
        GraphAnnotation? annotation = null)
    {
        steps.Add(new NodeSpec(id, operation, bindings.ToArray(), requirements, attributes, annotation));
    }

    internal void AppendDefinition(
        LogicalGraphDefinition definition,
        string tag,
        IReadOnlyDictionary<string, string> aliases,
        bool exposeOutputs = false)
    {
        string Map(ResourceId value) =>
            aliases.TryGetValue(value.Value, out var alias) ? alias : $"{tag}.{value.Value}";

        foreach (var resource in definition.Resources.Where(resource =>
                     resource.Kind is GraphDefinitionResourceKind.Input
                         or GraphDefinitionResourceKind.Output
                         or GraphDefinitionResourceKind.InOut))
        {
            var mapped = Map(resource.Id);
            if (resources.TryGetValue(mapped, out var existing))
            {
                if (existing.Tensor.ElementType != resource.Tensor.ElementType ||
                    existing.Tensor.Layout != resource.Tensor.Layout ||
                    !existing.Tensor.Dimensions.SequenceEqual(resource.Tensor.Dimensions))
                    throw new InvalidDataException($"Phi4 component '{tag}' has an incompatible resource '{resource.Id}'.");
                continue;
            }
            var kind = resource.Kind == GraphDefinitionResourceKind.Output && !exposeOutputs
                ? GraphDefinitionResourceKind.Temporary
                : resource.Kind;
            resources.Add(mapped, new(mapped, resource.Name, kind, resource.Tensor, resource.BindingKey));
        }
        steps.Add(new CallSpec(
            tag,
            definition,
            definition.Resources.Where(resource =>
                    resource.Kind is GraphDefinitionResourceKind.Input
                        or GraphDefinitionResourceKind.Output
                        or GraphDefinitionResourceKind.InOut)
                .Select(resource => new CallBindingSpec(
                    resource.Id.Value,
                    Map(resource.Id),
                    resource.Kind == GraphDefinitionResourceKind.Input
                        ? GraphResourceAccess.Read
                        : resource.Kind == GraphDefinitionResourceKind.Output
                            ? GraphResourceAccess.Write
                            : GraphResourceAccess.ReadWrite))
                .ToArray(),
            definition.Annotation));
    }

    private static IReadOnlyDictionary<string, string> AnnotationAttributes(
        string type,
        GraphSectionAnnotation architecture)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal) { ["section.type"] = type };
        if (architecture.Formula is { } formula) values["formula"] = formula;
        if (architecture.DefaultView is { } view) values["default_view"] = view.ToString();
        if (architecture.Step is { } step) values["step"] = step.ToString();
        return values;
    }

    internal LogicalGraphDefinition Build()
    {
        DynamicPorts.Specs.Value = resources.Values
            .Where(value => value.Kind is GraphDefinitionResourceKind.Input or GraphDefinitionResourceKind.Output or GraphDefinitionResourceKind.InOut)
            .ToArray();
        try
        {
            return LogicalGraphDefinition.Define<DynamicPorts>(
                id,
                builder =>
                {
                    var handles = new Dictionary<string, GraphResource<object>>(builder.Ports.Resources, StringComparer.Ordinal);
                    foreach (var resource in resources.Values.Where(value =>
                                 value.Kind is not (GraphDefinitionResourceKind.Input or GraphDefinitionResourceKind.Output or GraphDefinitionResourceKind.InOut)))
                        handles.Add(resource.Id, builder.AddResource<object>(
                            resource.Id, resource.Name, resource.Tensor, resource.Kind, resource.BindingKey));

                    GraphDefinitionNodeId? previous = null;
                    foreach (var step in steps)
                    {
                        if (step is CallSpec call)
                        {
                            previous = builder.AddCall(
                                call.Id,
                                call.Definition,
                                call.Bindings.Select(binding => (
                                    binding.Port,
                                    handles[binding.Resource],
                                    binding.Access)),
                                previous is { } callDependency ? [callDependency] : [],
                                call.Annotation);
                            continue;
                        }
                        var node = (NodeSpec)step;
                        builder.AddNode(node.Id, node.Operation, operation =>
                        {
                            foreach (var binding in node.Bindings)
                            {
                                if (!handles.TryGetValue(binding.Resource.Value, out var resource))
                                    throw new InvalidDataException($"Phi4 node '{node.Id}' references unknown resource '{binding.Resource}'.");
                                if (binding.Access == GraphResourceAccess.Read)
                                    operation.AddInput(binding.Port, resource);
                                else if (binding.Access == GraphResourceAccess.Write)
                                    operation.AddOutput(binding.Port, resource);
                                else
                                    operation.AddInOut(binding.Port, resource);
                            }

                            if (previous is { } dependency) operation.DependsOn(dependency);
                            if (node.Requirements is { } requirements)
                                operation.RequirePrecision(requirements);
                            if (node.Attributes is not null)
                                foreach (var (name, value) in node.Attributes) operation.AddAttribute(name, value);
                            foreach (var binding in node.Bindings.Where(binding => binding.View is not null))
                            {
                                var view = binding.View!;
                                operation.AddAttribute($"view.{binding.Port}.offset", view.Offset.ToString());
                                operation.AddAttribute($"view.{binding.Port}.element_type", view.Tensor.ElementType.ToString());
                                operation.AddAttribute($"view.{binding.Port}.dimensions",
                                    string.Join(",", view.Tensor.Dimensions));
                                operation.AddAttribute($"view.{binding.Port}.layout", view.Tensor.Layout);
                            }
                        }, node.Annotation);
                        previous = new(node.Id);
                    }
                },
                GraphCompileBoundary.Atomic,
                annotation).Definition;
        }
        finally
        {
            DynamicPorts.Specs.Value = null;
        }
    }

    internal sealed class Phi4DefinitionScope
    {
        private readonly Phi4GraphDefinitionBuilder builder;
        private readonly string path;
        private readonly GraphAnnotation annotation;

        internal Phi4DefinitionScope(Phi4GraphDefinitionBuilder builder, string path, GraphAnnotation annotation)
        {
            this.builder = builder;
            this.path = path;
            this.annotation = annotation;
        }

        internal Phi4DefinitionScope AddResource(string id, string name, GraphResourceKind kind,
            GraphResourceLifetime lifetime, TensorDescriptor tensor, string? bindingKey = null,
            bool graphInput = false, bool graphOutput = false, ResourceId? declaredScope = null)
        {
            builder.AddResource(id, name, kind, lifetime, tensor, bindingKey, graphInput, graphOutput, declaredScope);
            return this;
        }

        internal Phi4DefinitionScope SetStateSchema(GraphStateSchema schema) => this;

        internal Phi4DefinitionScope AddStateSlot(string name, string resource) => this;

        internal void AppendDefinition(LogicalGraphDefinition definition, string tag,
            IReadOnlyDictionary<string, string> aliases, bool exposeOutputs = false) =>
            builder.AppendDefinition(definition, tag, aliases, exposeOutputs);

        internal void Section(string id, string type, string name, Action<Phi4DefinitionScope> build,
            string? role = null, GraphSectionAnnotation? architecture = null) =>
            build(new(builder, id, new GraphAnnotation(name, role, architecture?.Description,
                architecture is null ? null : AnnotationAttributes(type, architecture))));

        internal T SectionValue<T>(string id, string type, string name, Func<Phi4DefinitionScope, T> build,
            string? role = null, GraphSectionAnnotation? architecture = null) =>
            build(new(builder, id, new GraphAnnotation(name, role, architecture?.Description,
                architecture is null ? null : AnnotationAttributes(type, architecture))));

        internal void AddNode(string id, GraphOperationId operation, IEnumerable<NodeResourceBinding> bindings,
            PrecisionRequirement? requirements = null, IReadOnlyDictionary<string, string>? attributes = null) =>
            builder.AddNode(id, operation, bindings, requirements, attributes, annotation with
            {
                Attributes = new Dictionary<string, string>(
                    annotation.Attributes ?? new Dictionary<string, string>()) { ["section.id"] = path },
            });

        internal Phi4DefinitionScope MatVec(string id, string matrix, string vector, string output)
        {
            AddNode(id, PrimitiveGraphOperations.MatVec,
                [GraphBindings.Read("matrix", matrix), GraphBindings.Read("vector", vector), GraphBindings.Write("output", output)]);
            return this;
        }

        internal Phi4DefinitionScope Add(string id, string left, string right, string output)
        {
            AddNode(id, PrimitiveGraphOperations.Add,
                [GraphBindings.Read("left", left), GraphBindings.Read("right", right), GraphBindings.Write("output", output)]);
            return this;
        }

        internal Phi4DefinitionScope MatrixMultiply(string id, string left, string right, string output,
            bool transposeLeft = false, bool transposeRight = false)
        {
            AddNode(id, PrimitiveGraphOperations.MatrixMultiply,
                [GraphBindings.Read("left", left), GraphBindings.Read("right", right), GraphBindings.Write("output", output)],
                attributes: new Dictionary<string, string>
                {
                    ["transpose_left"] = transposeLeft.ToString(),
                    ["transpose_right"] = transposeRight.ToString(),
                });
            return this;
        }

        internal Phi4DefinitionScope Affine(string id, string input, string weight, string bias, string output,
            bool transposeRight = false)
        {
            AddNode(id, PrimitiveGraphOperations.Affine,
                [GraphBindings.Read("input", input), GraphBindings.Read("weight", weight),
                 GraphBindings.Read("bias", bias), GraphBindings.Write("output", output)],
                attributes: new Dictionary<string, string> { ["transpose_right"] = transposeRight.ToString() });
            return this;
        }

        internal LogicalGraphDefinition Build() => builder.Build();
    }

    private sealed record ResourceSpec(
        string Id,
        string Name,
        GraphDefinitionResourceKind Kind,
        TensorDescriptor Tensor,
        string? BindingKey);

    private abstract record StepSpec;

    private sealed record NodeSpec(
        string Id,
        GraphOperationId Operation,
        NodeResourceBinding[] Bindings,
        PrecisionRequirement? Requirements,
        IReadOnlyDictionary<string, string>? Attributes,
        GraphAnnotation? Annotation) : StepSpec;

    private sealed record CallSpec(
        string Id,
        LogicalGraphDefinition Definition,
        CallBindingSpec[] Bindings,
        GraphAnnotation? Annotation) : StepSpec;

    private sealed record CallBindingSpec(
        string Port,
        string Resource,
        GraphResourceAccess Access);

    private sealed class DynamicPorts : IGraphPortContract<DynamicPorts>
    {
        internal static readonly AsyncLocal<ResourceSpec[]?> Specs = new();

        private DynamicPorts(IReadOnlyDictionary<string, GraphResource<object>> resources) =>
            Resources = resources;

        internal IReadOnlyDictionary<string, GraphResource<object>> Resources { get; }

        public static DynamicPorts Create(IGraphPortFactory ports)
        {
            var values = new Dictionary<string, GraphResource<object>>(StringComparer.Ordinal);
            foreach (var spec in Specs.Value ?? throw new InvalidOperationException("Phi4 graph ports were not configured."))
                values.Add(spec.Id, spec.Kind switch
                {
                    GraphDefinitionResourceKind.Input => ports.AddInput<object>(spec.Id, spec.Tensor),
                    GraphDefinitionResourceKind.Output => ports.AddOutput<object>(spec.Id, spec.Tensor),
                    GraphDefinitionResourceKind.InOut => ports.AddInOut<object>(spec.Id, spec.Tensor),
                    _ => throw new InvalidOperationException($"Invalid Phi4 port kind '{spec.Kind}'."),
                });
            return new(values);
        }
    }
}
