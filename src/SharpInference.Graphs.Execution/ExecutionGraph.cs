
namespace SharpInference.Graphs.Execution;

/// <summary>An editable computation source carrying selected target implementations and storage.</summary>
public sealed class ExecutionGraph : Graph
{
    public Dictionary<string, ExecutionStorage> Storage { get; } = new(StringComparer.Ordinal);

    public override IReadOnlyList<GraphDiagnostic> Validate()
    {
        var diagnostics = base.Validate().ToList();
        foreach (var action in Actions)
        {
            if (action is not (ExecutionOperation or ExecutionCall))
            {
                diagnostics.Add(new(GraphDiagnosticSeverity.Error, "G0017",
                    "Execution graphs require execution actions.", action.Id, Graph: Id));
            }
            var sources = action switch
            {
                ExecutionOperation operation => operation.Sources,
                ExecutionCall call => call.Sources,
                _ => [],
            };
            if (sources.Any(source => string.IsNullOrWhiteSpace(source.Document) ||
                string.IsNullOrWhiteSpace(source.Graph) || string.IsNullOrWhiteSpace(source.Node)))
            {
                diagnostics.Add(new(GraphDiagnosticSeverity.Error, "G0018",
                    "A source origin requires document, graph and node identities.", action.Id, Graph: Id));
            }
            if (action is ExecutionOperation { SelectedImplementation: { } implementation } &&
                (string.IsNullOrWhiteSpace(implementation.InstructionName) ||
                 string.IsNullOrWhiteSpace(implementation.Target.Architecture) ||
                 implementation.Precision is null ||
                 implementation.Dispatch is { } dispatch &&
                 (dispatch.ThreadsX == 0 || dispatch.ThreadsY == 0 || dispatch.ThreadsZ == 0 ||
                  dispatch.GroupsX == 0 || dispatch.GroupsY == 0 || dispatch.GroupsZ == 0)))
            {
                diagnostics.Add(new(GraphDiagnosticSeverity.Error, "G0021",
                    "An implementation requires instruction, target, precision and positive dispatch dimensions.", action.Id, Graph: Id));
            }
        }
        foreach (var (id, storage) in Storage)
        {
            if (!Resources.Any(resource => resource.Id == id) || !Enum.IsDefined(storage.Kind) ||
                storage.Slot is { } slot && (string.IsNullOrWhiteSpace(slot) || !Resources.Any(resource => resource.Id == slot)))
            {
                diagnostics.Add(new(GraphDiagnosticSeverity.Error, "G0019", $"Invalid storage selection for '{id}'.", Graph: Id));
            }
        }
        return diagnostics;
    }

    public override ExecutionOperation Operation(string id, OperationId operation)
    {
        var action = new ExecutionOperation(this, id, operation);
        Actions.Add(action);
        return action;
    }

    public override ExecutionOperation Operation(string id, string operation) => Operation(id, new OperationId(operation));

    public override ExecutionCall Call(string id, string target)
    {
        var action = new ExecutionCall(this, id, target);
        Actions.Add(action);
        return action;
    }

    public override ExecutionCall Call(string id, Graph target) => Call(id, target.Id);

    public Resource Resource(string id, TensorDescriptor tensor, ExecutionStorage storage,
        GraphIO graphIO = GraphIO.Internal, ResourceScope scope = ResourceScope.Call,
        ResourceUsage? usage = null, bool isState = false,
        ResourcePermission permission = ResourcePermission.ReadWrite, string? bindingKey = null)
    {
        var resource = base.Resource(id, tensor, graphIO, scope, usage, isState, permission, bindingKey);
        Storage.Add(id, storage);
        return resource;
    }

    protected override Graph CreateSnapshot()
    {
        var copy = new ExecutionGraph();
        foreach (var pair in Storage)
        {
            copy.Storage.Add(pair.Key, pair.Value);
        }

        return copy;
    }
}

public sealed record SourceOrigin(string Document, string Graph, string Node);

public sealed record ExecutionImplementation(
    Guid CollectionId, string InstructionName, ExecutionTarget Target,
    ExecutionPrecision Precision, ExecutionDispatch? Dispatch = null);

public enum ResourceStorageKind
{
    ExternalArgument, SharedBinding, LocalBuffer
}
public sealed record ExecutionStorage(ResourceStorageKind Kind, string? Slot = null);

public sealed class ExecutionOperation(Graph graph, string id, OperationId operation)
    : OperationAction(graph, id, operation)
{
    public ExecutionImplementation? SelectedImplementation
    {
        get; set;
    }
    public List<SourceOrigin> Sources { get; } = [];

    public ExecutionOperation Implementation(ExecutionImplementation implementation)
    {
        SelectedImplementation = implementation;
        return this;
    }

    public ExecutionOperation Source(string document, string graph, string node)
    {
        Sources.Add(new(document, graph, node));
        return this;
    }

    protected override GraphAction CopyAction(Graph target)
    {
        if (target is not ExecutionGraph execution)
        {
            throw new InvalidOperationException("Execution actions require an execution graph.");
        }

        var copy = execution.Operation(Id, Operation);
        copy.Precision = Precision;
        copy.SelectedImplementation = SelectedImplementation;
        copy.Sources.AddRange(Sources);
        foreach (var pair in Attributes)
        {
            copy.Attributes.Add(pair.Key, pair.Value);
        }

        return copy;
    }
}

public sealed class ExecutionCall(Graph graph, string id, string target) : CallAction(graph, id, target)
{
    public List<SourceOrigin> Sources { get; } = [];

    public ExecutionCall Source(string document, string graph, string node)
    {
        Sources.Add(new(document, graph, node));
        return this;
    }

    protected override GraphAction CopyAction(Graph target)
    {
        if (target is not ExecutionGraph execution)
        {
            throw new InvalidOperationException("Execution actions require an execution graph.");
        }

        var copy = execution.Call(Id, Target);
        copy.Sources.AddRange(Sources);
        return copy;
    }
}
