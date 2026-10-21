using SharpInference.Instructions;

namespace SharpInference.Graphs;

public enum GraphIO { Internal, In, Out, InOut }

/// <summary>The runtime context owning a resource instance and its logical lifetime.</summary>
public enum ResourceScope { Call, Session, Global }

public enum ResourceUsage { Temporary, Weight, Constant }
public enum ResourcePermission { ReadOnly, ReadWrite }

public sealed class Resource(Graph graph, string id, TensorDescriptor tensor)
{
    public Graph Graph { get; } = graph ?? throw new ArgumentNullException(nameof(graph));
    public string Id { get; set; } = id;
    public TensorDescriptor Tensor { get; set; } = tensor ?? throw new ArgumentNullException(nameof(tensor));
    public GraphIO GraphIO { get; set; } = GraphIO.Internal;
    public ResourceScope Scope { get; set; } = ResourceScope.Call;
    public ResourceUsage? Usage { get; set; }
    public bool IsState { get; set; }
    public ResourcePermission Permission { get; set; } = ResourcePermission.ReadWrite;
    public string? BindingKey { get; set; }
    public string Category { get; set; } = "resource";
}

public enum ResourceTarget { Graph, Call }

/// <summary>Locates a declaration or a resource exposed through a direct call instance.</summary>
public sealed record ResourceRef(
    ResourceTarget TargetKind, string? Target, string TargetProperty)
{
    public ResourceRef(Resource resource) : this(
        ResourceTarget.Graph, resource.Graph.Id, resource.Id) { }

    public static ResourceRef Local(string id) => new(ResourceTarget.Graph, null, id);
    public static ResourceRef Local(Resource resource) => new(resource);
    public static ResourceRef InGraph(string graph, string resource) => new(ResourceTarget.Graph, graph, resource);
    public static ResourceRef InCall(string call, string resource) => new(ResourceTarget.Call, call, resource);
    public static ResourceRef InCall(CallAction call, string resource) => InCall(call.Id, resource);
}

public readonly record struct ActionRef(string Graph, string Action)
{
    public ActionRef(GraphAction action) : this(action.Graph.Id, action.Id) { }
}

public sealed record GraphFlow(ActionRef Source, ActionRef Target);
public sealed record Binding(string SourceProperty, ResourceRef Target, ResourceAccess Access);
