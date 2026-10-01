using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpInference.Graphs;

public static class GraphJson
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter() },
    };

    public static string Serialize(LogicalGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        return JsonSerializer.Serialize(ToDto(graph), Options);
    }

    public static string Serialize(ExecutionGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        return JsonSerializer.Serialize(ToDto(graph), Options);
    }

    private static object ToDto(LogicalGraph graph) => new
    {
        Kind = "LogicalGraph",
        graph.Identity,
        graph.Model,
        Resources = graph.Resources.Select(ResourceDto),
        Regions = graph.Regions.Select(RegionDto),
        Nodes = graph.Nodes.Select(node => new
        {
            Id = node.Id.Value,
            Operation = node.Operation.ToString(),
            Region = node.Region.Value,
            Resources = node.Resources.Select(BindingDto),
            Dependencies = node.Dependencies.Select(value => value.Value),
            node.Attributes,
            node.Requirements,
        }),
        Inputs = graph.Inputs.Select(value => value.Value),
        Outputs = graph.Outputs.Select(value => value.Value),
        GraphState = StateDto(graph.GraphState),
    };

    private static object ToDto(ExecutionGraph graph) => new
    {
        Kind = "ExecutionGraph",
        graph.Identity,
        graph.Model,
        Resources = graph.Resources.Select(ResourceDto),
        Regions = graph.Regions.Select(RegionDto),
        Nodes = graph.Nodes.Select(node => new
        {
            Id = node.Id.Value,
            Operation = node.Operation.ToString(),
            Region = node.Region.Value,
            Resources = node.Resources.Select(BindingDto),
            Dependencies = node.Dependencies.Select(value => value.Value),
            node.Attributes,
            node.Requirements,
            InternalResources = node.InternalResources.Select(BindingDto),
            FirstWriteResources = node.FirstWriteResources.Select(value => value.Value),
            Source = new
            {
                LogicalNodes = node.Source.LogicalNodes.Select(value => value.Value),
                node.Source.AppliedRuleId,
            },
        }),
        Inputs = graph.Inputs.Select(value => value.Value),
        Outputs = graph.Outputs.Select(value => value.Value),
        GraphState = StateDto(graph.GraphState),
    };

    private static object? StateDto(GraphState? state) => state is null || state.Entries.Count == 0 ? null : new
    {
        Schema = new { state.Schema.Name },
        Slots = state.Slots.Select(slot => new { slot.Name, Resource = slot.Resource.Value }),
    };

    private static object ResourceDto(GraphResource resource) => new
    {
        Id = resource.Id.Value,
        resource.Name,
        resource.Kind,
        resource.Lifetime,
        Tensor = new
        {
            resource.Tensor.ElementType,
            resource.Tensor.Dimensions,
            resource.Tensor.Layout,
        },
        resource.BindingKey,
        resource.DeviceId,
        Scope = resource.Scope,
    };

    private static object RegionDto(GraphRegion region) => new
    {
        Id = region.Id.Value,
        ParentId = region.ParentId?.Value,
        region.Type,
        region.Role,
        region.Name,
        region.Attributes,
    };

    private static object BindingDto(NodeResourceBinding binding) => new
    {
        binding.Port,
        Resource = binding.Resource.Value,
        binding.Access,
        binding.InitializedBeforeRead,
    };

    public static LogicalGraph DeserializeLogical(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        RequireKind(root, "LogicalGraph");
        var parts = ReadParts(root);
        var nodes = root.GetProperty("Nodes").EnumerateArray().Select(node =>
            new LogicalNode(new LogicalNodeId(Text(node, "Id")), Operation(node),
                new RegionId(Text(node, "Region")), Bindings(node, "Resources"),
                Strings(node, "Dependencies").Select(value => new LogicalNodeId(value)).ToArray(),
                Attributes(node), Requirements(node))).ToArray();
        return new LogicalGraph(parts.Identity, parts.Model, parts.Resources, parts.Regions,
            nodes, parts.Inputs, parts.Outputs, parts.GraphState);
    }

    public static ExecutionGraph DeserializeExecution(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        RequireKind(root, "ExecutionGraph");
        var parts = ReadParts(root);
        var nodes = root.GetProperty("Nodes").EnumerateArray().Select(node =>
        {
            var source = node.GetProperty("Source");
            return new ExecutionNode(new ExecutionNodeId(Text(node, "Id")), Operation(node),
                new RegionId(Text(node, "Region")), Bindings(node, "Resources"),
                Strings(node, "Dependencies").Select(value => new ExecutionNodeId(value)).ToArray(),
                Attributes(node), Requirements(node),
                new ExecutionSourceMap(Strings(source, "LogicalNodes")
                    .Select(value => new LogicalNodeId(value)).ToArray(), NullableText(source, "AppliedRuleId")))
            {
                InternalResources = Bindings(node, "InternalResources"),
                FirstWriteResources = Strings(node, "FirstWriteResources")
                    .Select(value => new ResourceId(value)).ToArray(),
            };
        }).ToArray();
        return new ExecutionGraph(parts.Identity, parts.Model, parts.Resources, parts.Regions,
            nodes, parts.Inputs, parts.Outputs, parts.GraphState);
    }

    private static void RequireKind(JsonElement root, string kind)
    {
        if (Text(root, "Kind") != kind) throw new InvalidDataException($"Expected {kind} JSON.");
    }

    private static GraphParts ReadParts(JsonElement root)
    {
        var identity = root.GetProperty("Identity").Deserialize<GraphIdentity>(Options)!;
        var model = root.GetProperty("Model").Deserialize<GraphModelSignature>(Options)!;
        var resources = root.GetProperty("Resources").EnumerateArray().Select(item =>
        {
            var tensor = item.GetProperty("Tensor");
            return new GraphResource(new ResourceId(Text(item, "Id")), Text(item, "Name"),
                EnumValue<GraphResourceKind>(item, "Kind"), EnumValue<GraphResourceLifetime>(item, "Lifetime"),
                new TensorDescriptor(EnumValue<GraphElementType>(tensor, "ElementType"),
                    tensor.GetProperty("Dimensions").EnumerateArray().Select(value => value.GetInt32()),
                    Text(tensor, "Layout")), NullableText(item, "BindingKey"),
                NullableText(item, "DeviceId"),
                item.TryGetProperty("Scope", out var scope) ? EnumValue<GraphResourceScope>(item, "Scope") : null);
        }).ToArray();
        var regions = root.GetProperty("Regions").EnumerateArray().Select(item =>
            new GraphRegion(new RegionId(Text(item, "Id")),
                NullableText(item, "ParentId") is string parent ? new RegionId(parent) : null,
                Text(item, "Type"), NullableText(item, "Role"), Text(item, "Name"),
                Attributes(item))).ToArray();
        GraphState? graphState = null;
        if (root.TryGetProperty("GraphState", out var state) && state.ValueKind != JsonValueKind.Null)
        {
            graphState = new GraphState(new StateSchema(Text(state.GetProperty("Schema"), "Name")),
                state.GetProperty("Slots").EnumerateArray().Select(slot =>
                    new GraphStateSlot(Text(slot, "Name"), new ResourceId(Text(slot, "Resource")))));
        }
        return new GraphParts(identity, model, resources, regions,
            Strings(root, "Inputs").Select(value => new ResourceId(value)).ToArray(),
            Strings(root, "Outputs").Select(value => new ResourceId(value)).ToArray(), graphState);
    }

    private static string Text(JsonElement item, string property) => item.GetProperty(property).GetString()!;
    private static string? NullableText(JsonElement item, string property) =>
        item.TryGetProperty(property, out var value) && value.ValueKind != JsonValueKind.Null ? value.GetString() : null;
    private static string[] Strings(JsonElement item, string property) =>
        item.TryGetProperty(property, out var values)
            ? values.EnumerateArray().Select(value => value.GetString()!).ToArray() : [];
    private static T EnumValue<T>(JsonElement item, string property) where T : struct, Enum
    {
        var name = Text(item, property);
        if (!Enum.TryParse<T>(name, false, out var value) || !Enum.IsDefined(value) || value.ToString() != name)
            throw new InvalidDataException($"Invalid {property} value '{name}'.");
        return value;
    }
    private static GraphOperationId Operation(JsonElement node)
    {
        var value = Text(node, "Operation");
        var separator = value.LastIndexOf('@');
        if (separator < 1 || !int.TryParse(value[(separator + 1)..], out var version))
            throw new InvalidDataException($"Invalid graph operation '{value}'.");
        return new GraphOperationId(value[..separator], version);
    }
    private static NodeResourceBinding[] Bindings(JsonElement node, string property) =>
        node.TryGetProperty(property, out var bindings)
            ? bindings.EnumerateArray().Select(binding =>
                new NodeResourceBinding(Text(binding, "Port"), new ResourceId(Text(binding, "Resource")),
                    EnumValue<GraphResourceAccess>(binding, "Access"),
                    binding.TryGetProperty("InitializedBeforeRead", out var initialized) && initialized.GetBoolean())).ToArray()
            : [];
    private static Dictionary<string, string> Attributes(JsonElement item) =>
        item.GetProperty("Attributes").EnumerateObject().ToDictionary(property => property.Name,
            property => property.Value.GetString()!, StringComparer.Ordinal);
    private static PrecisionRequirement Requirements(JsonElement node)
    {
        var value = node.GetProperty("Requirements");
        return new PrecisionRequirement(EnumValue<GraphElementType>(value, "MinimumArithmeticType"),
            EnumValue<GraphElementType>(value, "MinimumAccumulatorType"));
    }
    private sealed record GraphParts(GraphIdentity Identity, GraphModelSignature Model,
        GraphResource[] Resources, GraphRegion[] Regions, ResourceId[] Inputs, ResourceId[] Outputs,
        GraphState? GraphState);
}
