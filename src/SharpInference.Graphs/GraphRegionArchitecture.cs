namespace SharpInference.Graphs;

/// <summary>Selects the default architecture or computation presentation of a region.</summary>
public enum GraphArchitectureView
{
    /// <summary>Displays the architecture-oriented region view.</summary>
    Architecture,
    /// <summary>Displays the operation-oriented computation view.</summary>
    Computation,
}

/// <summary>Describes the execution step represented by an architecture region.</summary>
public enum GraphArchitectureStep
{
    /// <summary>The region represents one token-processing step.</summary>
    Token,
}

/// <summary>Specifies whether an architecture port is an input or output.</summary>
public enum GraphArchitecturePortDirection
{
    /// <summary>An architecture input port.</summary>
    In,
    /// <summary>An architecture output port.</summary>
    Out,
}

/// <summary>Associates a named architecture port with a resource and direction.</summary>
/// <param name="Resource">The referenced graph resource identifier.</param>
/// <param name="Direction">The input or output direction of the architecture port.</param>
/// <param name="Name">The port name.</param>
public sealed record GraphArchitecturePort(
    ResourceId Resource,
    GraphArchitecturePortDirection Direction,
    string Name);

/// <summary>Non-executable documentation and presentation for a region's actual contents.</summary>
/// <param name="Role">The optional semantic role of the region.</param>
/// <param name="Description">The human-readable architecture description.</param>
/// <param name="Formula">The optional mathematical description of the region.</param>
/// <param name="DefaultCollapsed">Whether the region is initially collapsed, or null to use the viewer default.</param>
/// <param name="RepeatGroup">Whether the region represents a repeated group, or null to use the viewer default.</param>
/// <param name="DefaultView">The preferred region presentation, or null to use the viewer default.</param>
/// <param name="Step">The optional execution step represented by the architecture region.</param>
public sealed record GraphRegionArchitecture(
    string? Role = null,
    string? Description = null,
    string? Formula = null,
    bool? DefaultCollapsed = null,
    bool? RepeatGroup = null,
    GraphArchitectureView? DefaultView = null,
    GraphArchitectureStep? Step = null)
{
    /// <summary>Gets the architecture region&apos;s named resource ports.</summary>
    public IReadOnlyList<GraphArchitecturePort> Ports { get; init; } = [];

    /// <summary>Creates architecture metadata with each resource port mapped to a replacement identifier.</summary>
    /// <param name="map">Maps each referenced resource identifier to a replacement identifier.</param>
    public GraphRegionArchitecture MapResources(Func<ResourceId, ResourceId> map)
    {
        ArgumentNullException.ThrowIfNull(map);
        if (Ports is null || Ports.Any(port => port is null))
            throw new InvalidDataException("Architecture ports cannot contain null entries.");
        return Ports.Count == 0 ? this : this with
        {
            Ports = Ports.Select(port => port with { Resource = map(port.Resource) }).ToArray(),
        };
    }

    internal void Validate(string regionId, IReadOnlySet<ResourceId> resources)
    {
        if (Role is not null && string.IsNullOrWhiteSpace(Role) ||
            DefaultView is { } view && !Enum.IsDefined(view) ||
            Step is { } step && !Enum.IsDefined(step) || Ports is null)
            throw new InvalidDataException($"Invalid architecture metadata in region '{regionId}'.");
        var seen = new HashSet<(ResourceId, GraphArchitecturePortDirection)>();
        foreach (var port in Ports)
            if (port is null || !resources.Contains(port.Resource) || !Enum.IsDefined(port.Direction) ||
                string.IsNullOrWhiteSpace(port.Name) || !seen.Add((port.Resource, port.Direction)))
                throw new InvalidDataException($"Invalid or duplicate architecture port in region '{regionId}'.");
    }

    internal void ValidateBindings(string regionId, IEnumerable<NodeResourceBinding> bindings)
    {
        if (Ports.Count == 0) return;
        var reads = new HashSet<ResourceId>();
        var writes = new HashSet<ResourceId>();
        foreach (var binding in bindings)
        {
            if (binding.Access != GraphResourceAccess.Write) reads.Add(binding.Resource);
            if (binding.Access != GraphResourceAccess.Read) writes.Add(binding.Resource);
        }
        foreach (var port in Ports)
            if (!(port.Direction == GraphArchitecturePortDirection.In ? reads : writes).Contains(port.Resource))
                throw new InvalidDataException($"Architecture port '{port.Resource}' does not match region '{regionId}' bindings.");
    }
}
