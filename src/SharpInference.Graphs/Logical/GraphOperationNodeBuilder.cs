using System.Collections.ObjectModel;

namespace SharpInference.Graphs.Logical;

/// <summary>Records resource names without requiring their declarations to exist yet.</summary>
public sealed class GraphOperationNodeBuilder
{
    private readonly string id;
    private readonly List<ModelResourceBinding> bindings = [];
    private readonly List<GraphDefinitionNodeId> dependencies = [];
    private readonly Dictionary<string, string> attributes = new(StringComparer.Ordinal);
    private PrecisionRequirement requirements = new(GraphElementType.Float32, GraphElementType.Float32);

    internal GraphOperationNodeBuilder(string id) => this.id = id;

    internal IReadOnlyList<ModelResourceBinding> Bindings => bindings;
    internal IReadOnlyList<GraphDefinitionNodeId> Dependencies => dependencies;

    public void AddInput(string port, ResourceId resource) =>
        AddInput(port, new ModelResourceNodeId(GraphBindingOwner.Parent, resource));

    public void AddInput(string port, ModelResourceNodeId resource) =>
        Bind(port, resource, GraphResourceAccess.Read);

    public ResourceId AddOutput(string port, ResourceId resource)
    {
        AddOutput(port, new ModelResourceNodeId(GraphBindingOwner.Parent, resource));
        return resource;
    }

    public ModelResourceNodeId AddOutput(string port, ModelResourceNodeId resource)
    {
        Bind(port, resource, GraphResourceAccess.Write);
        return resource;
    }

    public ResourceId AddInOut(string port, ResourceId resource)
    {
        AddInOut(port, new ModelResourceNodeId(GraphBindingOwner.Parent, resource));
        return resource;
    }

    public ModelResourceNodeId AddInOut(string port, ModelResourceNodeId resource)
    {
        Bind(port, resource, GraphResourceAccess.ReadWrite);
        return resource;
    }

    public void DependsOn(GraphDefinitionNodeId dependency) => dependencies.Add(dependency);

    public void AddAttribute(string name, string value) =>
        attributes.Add(LogicalGraphDefinition.RequireId(name, nameof(name)),
            value ?? throw new ArgumentNullException(nameof(value)));

    public void RequirePrecision(PrecisionRequirement value) =>
        requirements = value ?? throw new ArgumentNullException(nameof(value));

    internal GraphOperationNode Build(GraphOperationId operation, GraphAnnotation? annotation) =>
        new(new(id), operation,
            new ReadOnlyDictionary<string, string>(new Dictionary<string, string>(attributes, StringComparer.Ordinal)),
            requirements, annotation);

    private void Bind(string port, ModelResourceNodeId resource, GraphResourceAccess access)
    {
        LogicalGraphDefinition.RequireId(port, nameof(port));
        if (bindings.Any(binding => binding.Port == port))
            throw new InvalidOperationException($"Node '{id}' already binds port '{port}'.");
        bindings.Add(new(port, resource, access, $"{id}:{port}"));
    }
}
