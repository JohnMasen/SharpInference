namespace SharpInference.Graphs;

public enum GraphArchitectureView
{
    Architecture,
    Computation,
}

public enum GraphArchitectureStep
{
    Token,
}

public enum GraphArchitecturePortDirection
{
    In,
    Out,
}

public sealed record GraphArchitecturePort(
    ResourceId Resource,
    GraphArchitecturePortDirection Direction,
    string Name);

/// <summary>Non-executable documentation and presentation for a region's actual contents.</summary>
public sealed record GraphRegionArchitecture(
    string? Role = null,
    string? Description = null,
    string? Formula = null,
    bool? DefaultCollapsed = null,
    bool? RepeatGroup = null,
    GraphArchitectureView? DefaultView = null,
    GraphArchitectureStep? Step = null)
{
    public IReadOnlyList<GraphArchitecturePort> Ports { get; init; } = [];

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
