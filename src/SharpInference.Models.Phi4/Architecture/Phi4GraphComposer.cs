using SharpInference.Instructions;
using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

namespace SharpInference.Architectures.Phi4;

internal sealed record GraphId(string Architecture, int Version, string EntryPoint);
internal sealed record GraphModelSignature(
    string Architecture,
    string StateAbiId,
    IReadOnlyDictionary<string, int> Dimensions);
internal sealed record GraphStateSchema(string Id);
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
    string Resource,
    ResourceAccess Access,
    bool InitializedBeforeRead = false,
    Phi4TensorView? View = null);
internal static class GraphBindings
{
    internal static NodeResourceBinding Read(string port, string resource) =>
        new(port, resource, ResourceAccess.Read);
    internal static NodeResourceBinding Write(string port, string resource) =>
        new(port, resource, ResourceAccess.Write);
    internal static NodeResourceBinding ReadWrite(string port, string resource) =>
        new(port, resource, ResourceAccess.ReadWrite);
}
internal static class PortableTensorOperationContracts
{
    internal static readonly OperationId CastFp16ToFp32 = new("core.tensor.cast-fp16-fp32", 1);
    internal static readonly OperationId Fill = new("core.tensor.fill", 1);
}
internal sealed record TensorFillValue(float Value)
{
    internal IReadOnlyDictionary<string, string> ToAttributes() =>
        new Dictionary<string, string>
        {
            ["value"] = Value.ToString(System.Globalization.CultureInfo.InvariantCulture),
        };
}

/// <summary>Creates an editable Phi-4 graph package. Callers explicitly validate before compilation.</summary>
public interface IPhi4GraphModule
{
    string ArchitectureId { get; }
    bool CanLoad(IModelTensorCatalog tensors);
    ModelMetadata ReadMetadata(IModelTensorCatalog tensors);
    GraphPackage<LogicalGraph> Build(IModelTensorCatalog tensors);
}

/// <summary>Composes Phi-4 architecture sections and component calls into native logical graphs.</summary>
internal sealed class Phi4GraphComposer
{
    private readonly LogicalGraph graph;
    private readonly Dictionary<string, Resource> resources = new(StringComparer.Ordinal);
    private readonly Dictionary<string, LogicalGraph> definitions = new(StringComparer.Ordinal);
    private GraphAction? previous;

    internal Phi4GraphComposer(GraphId id, GraphModelSignature model)
    {
        graph = InferenceGraphBuilder.Builder<LogicalGraph>(id.Architecture, graph =>
        {
            graph.CompileBoundary = GraphCompileBoundary.Atomic;
            graph.Name = id.EntryPoint;
            graph.Metadata["role"] = "model";
            graph.Metadata["architecture"] = model.Architecture;
            graph.Metadata["version"] = id.Version.ToString(System.Globalization.CultureInfo.InvariantCulture);
            graph.Metadata["state.abi"] = model.StateAbiId;
            foreach (var (name, value) in model.Dimensions)
                graph.Metadata["dimension." + name] = value.ToString(System.Globalization.CultureInfo.InvariantCulture);
        });
    }

    internal Scope Root(string id, string type, string name, string? role = null,
        GraphSectionAnnotation? architecture = null)
    {
        var metadata = SectionMetadata(id, type, name, role, architecture);
        graph.Name = name;
        graph.Description = architecture?.Description;
        foreach (var pair in metadata)
            graph.Metadata[pair.Key] = pair.Value;
        return new(this, id, metadata);
    }

    internal Phi4GraphComposer Resource(
        string id, TensorDescriptor tensor, GraphIO graphIO = GraphIO.Internal,
        ResourceScope scope = ResourceScope.Call, ResourceUsage? usage = null,
        bool isState = false, ResourcePermission permission = ResourcePermission.ReadWrite,
        string? bindingKey = null, string? name = null)
    {
        var resource = graph.Resource(id, tensor, graphIO, scope, usage, isState, permission, bindingKey);
        resources.Add(id, resource);
        if (name is not null)
            graph.Metadata[$"resource.{id}.name"] = name;
        return this;
    }

    internal void AddNode(
        string id, OperationId operation, IEnumerable<NodeResourceBinding> bindings,
        PrecisionRequirement? requirements, IReadOnlyDictionary<string, string>? attributes,
        IReadOnlyDictionary<string, string> metadata)
    {
        var node = graph.Operation(id, operation);
        foreach (var binding in bindings)
        {
            if (!resources.TryGetValue(binding.Resource, out var resource))
                throw new InvalidDataException($"Phi4 node '{id}' references unknown resource '{binding.Resource}'.");
            node.Bind(binding.Port, resource, binding.Access);
            if (binding.View is { } view)
            {
                node.Attributes[$"view.{binding.Port}.offset"] = view.Offset.ToString(System.Globalization.CultureInfo.InvariantCulture);
                node.Attributes[$"view.{binding.Port}.element_type"] = view.Tensor.ElementType.ToString();
                node.Attributes[$"view.{binding.Port}.dimensions"] = string.Join(",", view.Tensor.Dimensions);
                node.Attributes[$"view.{binding.Port}.layout"] = view.Tensor.Layout;
            }
            if (binding.InitializedBeforeRead)
                node.Metadata[$"binding.{binding.Port}.initialized_before_read"] = bool.TrueString;
        }
        node.Precision = requirements;
        if (attributes is not null)
            foreach (var pair in attributes)
                node.Attributes[pair.Key] = pair.Value;
        foreach (var pair in metadata)
            node.Metadata[pair.Key] = pair.Value;

        previous?.Then(node);
        previous = node;
    }

    internal void AppendDefinition(
        GraphPackage<LogicalGraph> package, string tag,
        IReadOnlyDictionary<string, string> aliases, bool exposeOutputs = false)
    {
        var definition = package.MainGraph
            ?? throw new InvalidDataException($"Phi4 component '{tag}' requires an explicit main graph.");
        foreach (var component in package)
        {
            if (component.Id == graph.Id)
                throw new InvalidDataException($"Phi4 component '{tag}' conflicts with its parent graph.");
            if (definitions.TryGetValue(component.Id, out var existing))
            {
                if (!ReferenceEquals(existing, component))
                    throw new InvalidDataException($"Phi4 component '{tag}' duplicates graph '{component.Id}'.");
            }
            else
                definitions.Add(component.Id, component);
        }

        var call = graph.Call(tag, definition);
        foreach (var resource in definition.GraphIOs)
        {
            var mapped = aliases.TryGetValue(resource.Id, out var alias) ? alias : $"{tag}.{resource.Id}";
            if (resources.TryGetValue(mapped, out var existing))
            {
                if (existing.Tensor.ElementType != resource.Tensor.ElementType ||
                    existing.Tensor.Layout != resource.Tensor.Layout ||
                    !existing.Tensor.Dimensions.SequenceEqual(resource.Tensor.Dimensions))
                    throw new InvalidDataException($"Phi4 component '{tag}' has an incompatible resource '{resource.Id}'.");
            }
            else
            {
                var graphIO = resource.GraphIO == GraphIO.Out && !exposeOutputs
                    ? GraphIO.Internal : resource.GraphIO;
                Resource(mapped, resource.Tensor, graphIO, resource.Scope, resource.Usage,
                    resource.IsState, resource.Permission, resource.BindingKey);
            }
            call.Bind(resource.Id, resources[mapped], resource.GraphIO switch
            {
                GraphIO.In => ResourceAccess.Read,
                GraphIO.Out => ResourceAccess.Write,
                _ => ResourceAccess.ReadWrite,
            });
        }
        if (definition.Name is { } name)
            call.Metadata["name"] = name;
        if (definition.Description is { } description)
            call.Metadata["description"] = description;
        foreach (var pair in definition.Metadata)
            call.Metadata[pair.Key] = pair.Value;

        previous?.Then(call);
        previous = call;
    }

    private static Dictionary<string, string> SectionMetadata(
        string path, string type, string name, string? role, GraphSectionAnnotation? architecture)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["name"] = name,
            ["section.id"] = path,
            ["section.type"] = type,
            ["architecture.path"] = path,
            ["architecture.name"] = name,
        };
        if ((role ?? architecture?.Role) is { } sectionRole)
            values["role"] = sectionRole;
        if (architecture?.Description is { } description)
            values["description"] = description;
        if (architecture?.Formula is { } formula)
        {
            values["formula"] = formula;
            values["architecture.formula"] = formula;
        }
        if (architecture?.DefaultCollapsed is { } collapsed)
            values["default_collapsed"] = collapsed.ToString();
        if (architecture?.RepeatGroup is { } repeat)
            values["repeat_group"] = repeat.ToString();
        if (architecture?.DefaultView is { } view)
            values["default_view"] = view.ToString();
        if (architecture?.Step is { } step)
            values["step"] = step.ToString();
        return values;
    }

    internal GraphPackage<LogicalGraph> Build() => new(graph, definitions.Values.ToArray());

    internal sealed class Scope
    {
        private readonly Phi4GraphComposer composer;
        private readonly string path;
        private readonly IReadOnlyDictionary<string, string> metadata;

        internal Scope(Phi4GraphComposer composer, string path, IReadOnlyDictionary<string, string> metadata)
        {
            this.composer = composer;
            this.path = path;
            this.metadata = metadata;
        }

        internal Scope Resource(
            string id, TensorDescriptor tensor, GraphIO graphIO = GraphIO.Internal,
            ResourceScope scope = ResourceScope.Call, ResourceUsage? usage = null,
            bool isState = false, ResourcePermission permission = ResourcePermission.ReadWrite,
            string? bindingKey = null, string? name = null)
        {
            composer.Resource(id, tensor, graphIO, scope, usage, isState, permission, bindingKey, name);
            return this;
        }

        internal Scope SetStateSchema(GraphStateSchema schema)
        {
            composer.graph.Metadata["state.abi"] = schema.Id;
            return this;
        }

        internal Scope AddStateSlot(string name, string resource)
        {
            composer.graph.Metadata[$"state.slot.{name}"] = resource;
            return this;
        }

        internal void AppendDefinition(GraphPackage<LogicalGraph> definition, string tag,
            IReadOnlyDictionary<string, string> aliases, bool exposeOutputs = false) =>
            composer.AppendDefinition(definition, tag, aliases, exposeOutputs);

        private Scope Child(string id, string type, string name, string? role, GraphSectionAnnotation? architecture)
        {
            var childPath = $"{path}/{id}";
            return new(composer, childPath, SectionMetadata(childPath, type, name, role, architecture));
        }

        internal void Section(string id, string type, string name, Action<Scope> build,
            string? role = null, GraphSectionAnnotation? architecture = null) =>
            build(Child(id, type, name, role, architecture));

        internal T SectionValue<T>(string id, string type, string name, Func<Scope, T> build,
            string? role = null, GraphSectionAnnotation? architecture = null) =>
            build(Child(id, type, name, role, architecture));

        internal void AddNode(string id, OperationId operation, IEnumerable<NodeResourceBinding> bindings,
            PrecisionRequirement? requirements = null, IReadOnlyDictionary<string, string>? attributes = null) =>
            composer.AddNode(id, operation, bindings, requirements, attributes, metadata);

        internal Scope MatVec(string id, string matrix, string vector, string output)
        {
            AddNode(id, PrimitiveOperations.MatVec,
                [GraphBindings.Read("matrix", matrix), GraphBindings.Read("vector", vector), GraphBindings.Write("output", output)]);
            return this;
        }

        internal Scope Add(string id, string left, string right, string output)
        {
            AddNode(id, PrimitiveOperations.Add,
                [GraphBindings.Read("left", left), GraphBindings.Read("right", right), GraphBindings.Write("output", output)]);
            return this;
        }

        internal Scope MatrixMultiply(string id, string left, string right, string output,
            bool transposeLeft = false, bool transposeRight = false)
        {
            AddNode(id, PrimitiveOperations.MatrixMultiply,
                [GraphBindings.Read("left", left), GraphBindings.Read("right", right), GraphBindings.Write("output", output)],
                attributes: new Dictionary<string, string>
                {
                    ["transpose_left"] = transposeLeft.ToString(),
                    ["transpose_right"] = transposeRight.ToString(),
                });
            return this;
        }

        internal Scope Affine(string id, string input, string weight, string bias, string output,
            bool transposeRight = false)
        {
            AddNode(id, PrimitiveOperations.Affine,
                [GraphBindings.Read("input", input), GraphBindings.Read("weight", weight),
                 GraphBindings.Read("bias", bias), GraphBindings.Write("output", output)],
                attributes: new Dictionary<string, string> { ["transpose_right"] = transposeRight.ToString() });
            return this;
        }

        internal GraphPackage<LogicalGraph> Build() => composer.Build();
    }
}
