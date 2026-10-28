namespace SharpInference.Graphs;

/// <summary>An editable computation definition containing resources, actions and explicit flow.</summary>
public class Graph
{
    public string Id { get; set; } = "";
    public string? Name
    {
        get; set;
    }
    public string? Description
    {
        get; set;
    }
    public Dictionary<string, string> Metadata { get; } = new(StringComparer.Ordinal);
    public List<Resource> Resources { get; } = [];
    public List<GraphAction> Actions { get; } = [];
    public List<GraphFlow> GraphFlow { get; } = [];

    public IEnumerable<Resource> GraphIOs => Resources.Where(resource => resource.GraphIO != GraphIO.Internal);
    public IEnumerable<Resource> Inputs => Resources.Where(resource => resource.GraphIO is GraphIO.In or GraphIO.InOut);
    public IEnumerable<Resource> Outputs => Resources.Where(resource => resource.GraphIO is GraphIO.Out or GraphIO.InOut);

    public Resource Resource(
        string id, TensorDescriptor tensor, GraphIO graphIO = GraphIO.Internal,
        ResourceScope scope = ResourceScope.Call, ResourceUsage? usage = null,
        bool isState = false, ResourcePermission permission = ResourcePermission.ReadWrite,
        string? bindingKey = null)
    {
        var resource = new Resource(this, id, tensor)
        {
            GraphIO = graphIO,
            Scope = scope,
            Usage = usage,
            IsState = isState,
            Permission = permission,
            BindingKey = bindingKey,
        };
        Resources.Add(resource);
        return resource;
    }

    public virtual OperationAction Operation(string id, OperationId operation)
    {
        var action = new OperationAction(this, id, operation);
        Actions.Add(action);
        return action;
    }

    public virtual OperationAction Operation(string id, string operation) => Operation(id, new OperationId(operation));

    public virtual CallAction Call(string id, string target)
    {
        var action = new CallAction(this, id, target);
        Actions.Add(action);
        return action;
    }

    public virtual CallAction Call(string id, Graph target)
    {
        ArgumentNullException.ThrowIfNull(target);
        return Call(id, target.Id);
    }

    public virtual IReadOnlyList<GraphDiagnostic> Validate() => GraphValidation.Validate(this);

    public Graph Snapshot()
    {
        var copy = CreateSnapshot();
        copy.Id = Id;
        copy.Name = Name;
        copy.Description = Description;
        foreach (var pair in Metadata)
        {
            copy.Metadata.Add(pair.Key, pair.Value);
        }

        foreach (var resource in Resources)
        {
            copy.Resource(resource.Id, resource.Tensor, resource.GraphIO, resource.Scope, resource.Usage,
                resource.IsState, resource.Permission, resource.BindingKey).Category = resource.Category;
        }

        foreach (var action in Actions)
        {
            action.CopyTo(copy);
        }

        copy.GraphFlow.AddRange(GraphFlow);
        return copy;
    }

    protected virtual Graph CreateSnapshot() => new();
}

public static class InferenceGraphBuilder
{
    public static TGraph Builder<TGraph>(string id, Action<TGraph> configure) where TGraph : Graph, new()
    {
        ArgumentNullException.ThrowIfNull(configure);
        var graph = new TGraph { Id = id };
        configure(graph);
        return graph;
    }
}
