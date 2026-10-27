
namespace SharpInference.Graphs;

public abstract class GraphAction(Graph graph, string id)
{
    public Graph Graph { get; } = graph ?? throw new ArgumentNullException(nameof(graph));
    public string Id { get; set; } = id;
    public abstract string Category { get; set; }
    public List<Binding> Bindings { get; } = [];
    public Dictionary<string, string> Metadata { get; } = new(StringComparer.Ordinal);

    public GraphAction Bind(string property, ResourceRef target, ResourceAccess access)
    {
        ArgumentNullException.ThrowIfNull(target);
        Bindings.Add(new(property, target, access));
        return this;
    }

    public GraphAction Bind(string property, Resource target, ResourceAccess access) =>
        Bind(property, new ResourceRef(target), access);

    public GraphAction Bind(string property, string resource, ResourceAccess access) =>
        Bind(property, ResourceRef.Local(resource), access);

    public GraphAction Then(GraphAction next) => Then(new ActionRef(next));
    public GraphAction Then(string next) => Then(new ActionRef(Graph.Id, next));

    public GraphAction Then(ActionRef next)
    {
        Graph.GraphFlow.Add(new(new ActionRef(this), next));
        return this;
    }

    internal void CopyTo(Graph target)
    {
        var copy = CopyAction(target);
        copy.Category = Category;
        copy.Bindings.AddRange(Bindings);
        foreach (var pair in Metadata)
        {
            copy.Metadata.Add(pair.Key, pair.Value);
        }
    }

    protected abstract GraphAction CopyAction(Graph target);
}

public class OperationAction(Graph graph, string id, OperationId operation) : GraphAction(graph, id)
{
    public override string Category { get; set; } = "operation";
    public OperationId Operation { get; set; } = operation;
    public Dictionary<string, string> Attributes { get; } = new(StringComparer.Ordinal);
    public PrecisionRequirement? Precision { get; set; }

    public OperationAction RequirePrecision(ElementType arithmetic, ElementType accumulator)
    {
        Precision = new(arithmetic, accumulator);
        return this;
    }

    protected override GraphAction CopyAction(Graph target)
    {
        var copy = target.Operation(Id, Operation);
        copy.Precision = Precision;
        foreach (var pair in Attributes)
        {
            copy.Attributes.Add(pair.Key, pair.Value);
        }

        return copy;
    }
}

public class CallAction(Graph graph, string id, string target) : GraphAction(graph, id)
{
    public override string Category { get; set; } = "call";
    public string Target { get; set; } = target;

    protected override GraphAction CopyAction(Graph target) => target.Call(Id, Target);
}
