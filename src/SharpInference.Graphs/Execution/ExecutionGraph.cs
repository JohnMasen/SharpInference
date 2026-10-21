using SharpInference.Instructions;

namespace SharpInference.Graphs.Execution;

/// <summary>An editable computation source carrying selected target implementations and storage.</summary>
public sealed class ExecutionGraph : Graph
{
    public Dictionary<string, ExecutionStorage> Storage { get; } = new(StringComparer.Ordinal);

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
            copy.Storage.Add(pair.Key, pair.Value);
        return copy;
    }
}

public static class ExecutionPackage
{
    public const string BackendMetadataKey = "execution.backend";

    public static void SetTarget(this GraphPackage<ExecutionGraph> package, string backend)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(backend);
        package.Metadata[BackendMetadataKey] = backend;
    }

    public static string? GetTarget(this GraphPackage<ExecutionGraph> package) =>
        package.Metadata.GetValueOrDefault(BackendMetadataKey);
}

public sealed record SourceOrigin(string Document, string Graph, string Node);

public sealed record ExecutionImplementation(
    Guid CollectionId, string InstructionName, InstructionTarget Target,
    KernelPrecisionProfile Precision, GraphInstructionDispatch? Dispatch = null);

public enum ResourceStorageKind { ExternalArgument, SharedBinding, LocalBuffer }
public sealed record ExecutionStorage(ResourceStorageKind Kind, string? Slot = null);

public sealed class ExecutionOperation(Graph graph, string id, OperationId operation)
    : OperationAction(graph, id, operation)
{
    public ExecutionImplementation? SelectedImplementation { get; set; }
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
            throw new InvalidOperationException("Execution actions require an execution graph.");
        var copy = execution.Operation(Id, Operation);
        copy.Precision = Precision;
        copy.SelectedImplementation = SelectedImplementation;
        copy.Sources.AddRange(Sources);
        foreach (var pair in Attributes)
            copy.Attributes.Add(pair.Key, pair.Value);
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
            throw new InvalidOperationException("Execution actions require an execution graph.");
        var copy = execution.Call(Id, Target);
        copy.Sources.AddRange(Sources);
        return copy;
    }
}
