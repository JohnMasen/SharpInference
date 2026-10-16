using System.Collections.ObjectModel;

namespace SharpInference.Graphs.Logical;

/// <summary>Collects one operation's parent-owned resource bindings, dependencies, attributes, and precision.</summary>
public sealed class GraphOperationNodeBuilder
{
    /// <summary>The containing definition identity required of directly bound resources.</summary>
    private readonly GraphDefinitionId owner;
    /// <summary>The operation identity to assign when building.</summary>
    private readonly GraphDefinitionNodeId id;
    /// <summary>The ordered port bindings with distinct port names.</summary>
    private readonly List<PendingBinding> bindings = [];
    /// <summary>Explicit prerequisites, deduplicated later by the definition builder.</summary>
    private readonly List<GraphDefinitionNodeId> dependencies = [];
    /// <summary>Operation attributes keyed by ordinal name.</summary>
    private readonly Dictionary<string, string> attributes = new(StringComparer.Ordinal);
    /// <summary>The first resource declaration encountered for each identifier.</summary>
    private readonly Dictionary<ResourceId, GraphDefinitionResource> resources = [];
    /// <summary>The most recently specified precision requirement, or null for the default.</summary>
    private PrecisionRequirement? requirements;

    /// <summary>Creates an empty configuration for an operation in a definition.</summary>
    /// <param name="owner">The containing graph identity.</param>
    /// <param name="id">The operation identity.</param>
    internal GraphOperationNodeBuilder(GraphDefinitionId owner, GraphDefinitionNodeId id)
    {
        this.owner = owner;
        this.id = id;
    }

    /// <summary>Gets the referenced parent resource declarations.</summary>
    internal IReadOnlyCollection<GraphDefinitionResource> Resources => resources.Values;
    /// <summary>Gets bindings in configuration order.</summary>
    internal IReadOnlyList<PendingBinding> Bindings => bindings;
    /// <summary>Gets explicitly declared prerequisites, including any duplicates.</summary>
    internal IReadOnlyList<GraphDefinitionNodeId> Dependencies => dependencies;

    /// <summary>Adds a read-only binding to a parent-owned resource.</summary>
    /// <typeparam name="T">The resource handle's type marker.</typeparam>
    /// <param name="port">The nonblank, unique operation port name.</param>
    /// <param name="resource">The resource to read.</param>
    /// <exception cref="ArgumentNullException">The resource is null.</exception>
    /// <exception cref="ArgumentException">The port is blank or the resource belongs to another graph.</exception>
    /// <exception cref="InvalidOperationException">The port is already bound.</exception>
    public void AddInput<T>(string port, GraphResource<T> resource) =>
        AddBinding(port, resource, GraphResourceAccess.Read);

    /// <summary>Adds a write-only binding to a parent-owned resource.</summary>
    /// <typeparam name="T">The resource handle's type marker.</typeparam>
    /// <param name="port">The nonblank, unique operation port name.</param>
    /// <param name="resource">The resource to write.</param>
    /// <returns>The supplied resource handle.</returns>
    /// <exception cref="ArgumentNullException">The resource is null.</exception>
    /// <exception cref="ArgumentException">The port is blank or the resource belongs to another graph.</exception>
    /// <exception cref="InvalidOperationException">The port is already bound.</exception>
    public GraphResource<T> AddOutput<T>(string port, GraphResource<T> resource)
    {
        AddBinding(port, resource, GraphResourceAccess.Write);
        return resource;
    }

    /// <summary>Adds paired read/write access to a parent-owned resource.</summary>
    /// <typeparam name="T">The resource handle's type marker.</typeparam>
    /// <param name="port">The nonblank, unique operation port name.</param>
    /// <param name="resource">The resource to read and write.</param>
    /// <returns>The supplied resource handle.</returns>
    /// <exception cref="ArgumentNullException">The resource is null.</exception>
    /// <exception cref="ArgumentException">The port is blank or the resource belongs to another graph.</exception>
    /// <exception cref="InvalidOperationException">The port is already bound.</exception>
    public GraphResource<T> AddInOut<T>(string port, GraphResource<T> resource)
    {
        AddBinding(port, resource, GraphResourceAccess.ReadWrite);
        return resource;
    }

    /// <summary>Records an explicit prerequisite without checking existence or cycles at this stage.</summary>
    /// <param name="dependency">The prerequisite operation identity in the containing definition.</param>
    public void DependsOn(GraphDefinitionNodeId dependency) => dependencies.Add(dependency);

    /// <summary>Adds an operation attribute with a unique ordinal name.</summary>
    /// <param name="name">The nonblank attribute name.</param>
    /// <param name="value">The nonnull attribute value.</param>
    /// <exception cref="ArgumentException">The name is blank or already present.</exception>
    /// <exception cref="ArgumentNullException">The value is null.</exception>
    public void AddAttribute(string name, string value) =>
        attributes.Add(
            LogicalGraphDefinition.RequireId(name, nameof(name)),
            value ?? throw new ArgumentNullException(nameof(value)));

    /// <summary>Replaces this operation's minimum arithmetic and accumulator precision requirement.</summary>
    /// <param name="value">The nonnull requirement to use.</param>
    /// <exception cref="ArgumentNullException">The requirement is null.</exception>
    public void RequirePrecision(PrecisionRequirement value) =>
        requirements = value ?? throw new ArgumentNullException(nameof(value));

    /// <summary>Creates the operation, defaulting both precision minima to Float32 when unspecified.</summary>
    /// <param name="operation">The operation name and version.</param>
    /// <param name="annotation">Optional descriptive metadata.</param>
    /// <returns>The configured operation with a read-only view of its attribute dictionary.</returns>
    internal GraphOperationNode Build(GraphOperationId operation, GraphAnnotation? annotation) =>
        new(
            id,
            operation,
            new ReadOnlyDictionary<string, string>(attributes),
            requirements ?? new PrecisionRequirement(
                GraphElementType.Float32, GraphElementType.Float32),
            annotation);

    /// <summary>Checks a direct parent-owned access and appends a uniquely named port binding.</summary>
    /// <typeparam name="T">The resource handle's type marker.</typeparam>
    /// <param name="port">The nonblank operation port name.</param>
    /// <param name="resource">The resource whose declaration owner must match the containing graph.</param>
    /// <param name="access">The access represented by the eventual read and write edges.</param>
    /// <exception cref="ArgumentNullException">The resource is null.</exception>
    /// <exception cref="ArgumentException">The resource has a different graph owner or the port is blank.</exception>
    /// <exception cref="InvalidOperationException">The port is already bound.</exception>
    private void AddBinding<T>(
        string port,
        GraphResource<T> resource,
        GraphResourceAccess access)
    {
        ArgumentNullException.ThrowIfNull(resource);
        if (resource.Resource.Owner != owner)
        {
            throw new ArgumentException(
                $"Operation '{id}' cannot directly reference resource '{resource.Id}' owned by graph " +
                $"'{resource.Resource.Owner}'. Bind graph instances at their containing call instead.",
                nameof(resource));
        }

        var name = LogicalGraphDefinition.RequireId(port, nameof(port));
        if (bindings.Any(binding => binding.Port == name))
        {
            throw new InvalidOperationException(
                $"Operation '{id}' already binds port '{name}'.");
        }

        bindings.Add(new PendingBinding(
            name, resource.Resource, access, GraphBindingOwner.Parent));
        resources.TryAdd(resource.Id, resource.Resource);
    }
}
