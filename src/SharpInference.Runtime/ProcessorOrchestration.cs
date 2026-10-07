using System.Runtime.ExceptionServices;
using SharpInference.Graphs;

namespace SharpInference.Runtime;

public sealed record ProcessorResourceAccess
{
    public ProcessorResourceAccess(string resource, GraphResourceAccess access)
    {
        Resource = string.IsNullOrWhiteSpace(resource)
            ? throw new ArgumentException("A resource name is required.", nameof(resource))
            : resource;
        if (!Enum.IsDefined(access)) throw new ArgumentOutOfRangeException(nameof(access));
        Access = access;
    }

    public string Resource { get; }
    public GraphResourceAccess Access { get; }
}

public sealed class ProcessorOrchestrationNode
{
    private static readonly Func<ProcessorOrchestrationContext, CancellationToken, ValueTask> NoOperation =
        static (_, _) => ValueTask.CompletedTask;

    public ProcessorOrchestrationNode(
        string name,
        IEnumerable<string>? dependencies,
        IEnumerable<ProcessorResourceAccess>? resources,
        Func<ProcessorOrchestrationContext, CancellationToken, ValueTask> execute)
    {
        Name = string.IsNullOrWhiteSpace(name)
            ? throw new ArgumentException("A node name is required.", nameof(name))
            : name;
        Dependencies = Array.AsReadOnly(dependencies?.ToArray() ?? []);
        Resources = Array.AsReadOnly(resources?.ToArray() ?? []);
        Execute = execute ?? throw new ArgumentNullException(nameof(execute));
        if (Dependencies.Any(string.IsNullOrWhiteSpace) ||
            Dependencies.Distinct(StringComparer.Ordinal).Count() != Dependencies.Count)
            throw new ArgumentException("Node dependencies must be unique names.", nameof(dependencies));
        if (Resources.GroupBy(value => value.Resource, StringComparer.Ordinal).Any(group => group.Count() != 1))
            throw new ArgumentException("A node may bind each orchestration resource only once.", nameof(resources));
    }

    public string Name { get; }
    public IReadOnlyList<string> Dependencies { get; }
    public IReadOnlyList<ProcessorResourceAccess> Resources { get; }
    public Func<ProcessorOrchestrationContext, CancellationToken, ValueTask> Execute { get; }

    public static ProcessorOrchestrationNode Join(string name, IEnumerable<string> dependencies) =>
        new(name, dependencies, [], NoOperation);
}

public sealed class ProcessorOrchestrationGraph
{
    public ProcessorOrchestrationGraph(IEnumerable<ProcessorOrchestrationNode> nodes)
    {
        ArgumentNullException.ThrowIfNull(nodes);
        var values = nodes.ToArray();
        if (values.Length == 0 || values.Any(static node => node is null))
            throw new ArgumentException("An orchestration graph requires nodes.", nameof(nodes));
        var byName = values.ToDictionary(node => node.Name, StringComparer.Ordinal);
        foreach (var node in values)
        {
            foreach (var dependency in node.Dependencies)
            {
                if (!byName.ContainsKey(dependency))
                    throw new InvalidDataException(
                        $"Orchestration node '{node.Name}' depends on unknown node '{dependency}'.");
                if (string.Equals(node.Name, dependency, StringComparison.Ordinal))
                    throw new InvalidDataException($"Orchestration node '{node.Name}' depends on itself.");
            }
        }

        ValidateAcyclic(values, byName);
        ValidateUnorderedConflicts(values, byName);
        Nodes = Array.AsReadOnly(values);
    }

    public IReadOnlyList<ProcessorOrchestrationNode> Nodes { get; }

    private static void ValidateAcyclic(
        IReadOnlyList<ProcessorOrchestrationNode> nodes,
        IReadOnlyDictionary<string, ProcessorOrchestrationNode> byName)
    {
        var active = new HashSet<string>(StringComparer.Ordinal);
        var complete = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in nodes) Visit(node);
        return;

        void Visit(ProcessorOrchestrationNode node)
        {
            if (complete.Contains(node.Name)) return;
            if (!active.Add(node.Name))
                throw new InvalidDataException($"Orchestration graph contains a cycle at '{node.Name}'.");
            foreach (var dependency in node.Dependencies) Visit(byName[dependency]);
            active.Remove(node.Name);
            complete.Add(node.Name);
        }
    }

    private static void ValidateUnorderedConflicts(
        IReadOnlyList<ProcessorOrchestrationNode> nodes,
        IReadOnlyDictionary<string, ProcessorOrchestrationNode> byName)
    {
        var ancestors = nodes.ToDictionary(
            node => node.Name,
            node => GetAncestors(node, byName),
            StringComparer.Ordinal);
        for (var leftIndex = 0; leftIndex < nodes.Count; leftIndex++)
        {
            var left = nodes[leftIndex];
            for (var rightIndex = leftIndex + 1; rightIndex < nodes.Count; rightIndex++)
            {
                var right = nodes[rightIndex];
                if (ancestors[left.Name].Contains(right.Name) ||
                    ancestors[right.Name].Contains(left.Name))
                    continue;
                foreach (var leftAccess in left.Resources)
                {
                    var rightAccess = right.Resources.FirstOrDefault(value =>
                        string.Equals(value.Resource, leftAccess.Resource, StringComparison.Ordinal));
                    if (rightAccess is not null &&
                        (leftAccess.Access != GraphResourceAccess.Read ||
                         rightAccess.Access != GraphResourceAccess.Read))
                    {
                        throw new InvalidDataException(
                            $"Unordered orchestration nodes '{left.Name}' and '{right.Name}' conflict on " +
                            $"resource '{leftAccess.Resource}'. Add an explicit dependency.");
                    }
                }
            }
        }
    }

    private static HashSet<string> GetAncestors(
        ProcessorOrchestrationNode node,
        IReadOnlyDictionary<string, ProcessorOrchestrationNode> byName)
    {
        var result = new HashSet<string>(StringComparer.Ordinal);
        Add(node);
        return result;

        void Add(ProcessorOrchestrationNode current)
        {
            foreach (var dependency in current.Dependencies)
            {
                if (result.Add(dependency)) Add(byName[dependency]);
            }
        }
    }
}

public sealed class ProcessorOrchestrationContext : IAsyncDisposable
{
    private readonly object gate = new();
    private readonly Dictionary<string, IStorageLease> storage = new(StringComparer.Ordinal);
    private bool disposed;

    public void Publish(string resource, IStorageLease lease)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(resource);
        ArgumentNullException.ThrowIfNull(lease);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (!storage.TryAdd(resource, lease))
                throw new InvalidOperationException($"Orchestration resource '{resource}' is already published.");
        }
    }

    public IStorageLease Acquire(string resource)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(resource);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return storage.TryGetValue(resource, out var lease)
                ? lease.Retain()
                : throw new KeyNotFoundException($"Orchestration resource '{resource}' has not been published.");
        }
    }

    public async ValueTask DisposeAsync()
    {
        IStorageLease[] leases;
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            var unique = new HashSet<IStorageLease>(ReferenceEqualityComparer.Instance);
            unique.UnionWith(storage.Values);
            leases = unique.ToArray();
            storage.Clear();
        }

        List<Exception>? errors = null;
        foreach (var lease in leases)
        {
            try { await lease.DisposeAsync().ConfigureAwait(false); }
            catch (Exception error) { (errors ??= []).Add(error); }
        }
        if (errors is not null)
            throw new AggregateException("One or more orchestration storage leases failed to release.", errors);
    }
}

public sealed class ProcessorOrchestrator
{
    private readonly int maximumConcurrency;

    public ProcessorOrchestrator(int maximumConcurrency = int.MaxValue)
    {
        if (maximumConcurrency <= 0) throw new ArgumentOutOfRangeException(nameof(maximumConcurrency));
        this.maximumConcurrency = maximumConcurrency;
    }

    public async ValueTask<ProcessorOrchestrationContext> ExecuteAsync(
        ProcessorOrchestrationGraph graph,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var context = new ProcessorOrchestrationContext();
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        using var concurrency = new SemaphoreSlim(maximumConcurrency, maximumConcurrency);
        var nodes = graph.Nodes.ToDictionary(node => node.Name, StringComparer.Ordinal);
        var tasks = new Dictionary<string, Task>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes) _ = GetTask(node);

        var all = Task.WhenAll(tasks.Values);
        try
        {
            await all.ConfigureAwait(false);
            return context;
        }
        catch
        {
            cancellation.Cancel();
            try { await Task.WhenAll(tasks.Values).ConfigureAwait(false); }
            catch { }

            var uniqueErrors = new HashSet<Exception>(ReferenceEqualityComparer.Instance);
            uniqueErrors.UnionWith(tasks.Values
                .Where(task => task.IsFaulted)
                .SelectMany(task => task.Exception!.Flatten().InnerExceptions));
            var errors = uniqueErrors.ToList();
            try { await context.DisposeAsync().ConfigureAwait(false); }
            catch (Exception cleanupError) { errors.Add(cleanupError); }

            if (errors.Count > 1)
                throw new AggregateException("Processor orchestration failed.", errors);
            if (errors.Count == 1)
                ExceptionDispatchInfo.Capture(errors[0]).Throw();
            cancellationToken.ThrowIfCancellationRequested();
            throw new OperationCanceledException(cancellation.Token);
        }

        Task GetTask(ProcessorOrchestrationNode node)
        {
            if (tasks.TryGetValue(node.Name, out var existing)) return existing;
            var dependencies = node.Dependencies.Select(name => GetTask(nodes[name])).ToArray();
            var created = RunNodeAsync(node, dependencies);
            tasks.Add(node.Name, created);
            return created;
        }

        async Task RunNodeAsync(ProcessorOrchestrationNode node, Task[] dependencies)
        {
            await Task.WhenAll(dependencies).ConfigureAwait(false);
            cancellation.Token.ThrowIfCancellationRequested();
            await concurrency.WaitAsync(cancellation.Token).ConfigureAwait(false);
            try
            {
                await node.Execute(context, cancellation.Token).ConfigureAwait(false);
            }
            catch
            {
                cancellation.Cancel();
                throw;
            }
            finally
            {
                concurrency.Release();
            }
        }
    }
}
