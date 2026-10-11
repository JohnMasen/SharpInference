using System.Runtime.ExceptionServices;
using SharpInference.Graphs;

namespace SharpInference.Runtime;

/// <summary>Declares a named orchestration resource and the access required by a node.</summary>
public sealed record ProcessorResourceAccess
{
    /// <summary>Creates and validates a resource name and its orchestration access mode.</summary>
    /// <param name="resource">The name of the orchestration resource.</param>
    /// <param name="access">The required read or write access to the resource.</param>
    public ProcessorResourceAccess(string resource, GraphResourceAccess access)
    {
        Resource = string.IsNullOrWhiteSpace(resource)
            ? throw new ArgumentException("A resource name is required.", nameof(resource))
            : resource;
        if (!Enum.IsDefined(access)) throw new ArgumentOutOfRangeException(nameof(access));
        Access = access;
    }

    /// <summary>Gets the name of the orchestration resource.</summary>
    public string Resource { get; }
    /// <summary>Gets the node&apos;s read or write access to the resource.</summary>
    public GraphResourceAccess Access { get; }
}

/// <summary>Defines a unit of orchestration work with explicit dependencies and resource access declarations.</summary>
public sealed class ProcessorOrchestrationNode
{
    private static readonly Func<ProcessorOrchestrationContext, CancellationToken, ValueTask> NoOperation =
        static (_, _) => ValueTask.CompletedTask;

    /// <summary>Creates a named work node with unique dependencies, resource bindings, and an execution callback.</summary>
    /// <param name="name">The unique orchestration node name used by dependency edges.</param>
    /// <param name="dependencies">The names of nodes that must complete before this node runs.</param>
    /// <param name="resources">The unique resource access declarations for the orchestration node.</param>
    /// <param name="execute">The asynchronous callback that executes the node in the shared resource context.</param>
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

    /// <summary>Gets the unique node name used by dependency edges.</summary>
    public string Name { get; }
    /// <summary>Gets the names of nodes that must finish before this node starts.</summary>
    public IReadOnlyList<string> Dependencies { get; }
    /// <summary>Gets the resource access declarations used to detect unordered conflicts.</summary>
    public IReadOnlyList<ProcessorResourceAccess> Resources { get; }
    /// <summary>Gets the callback that performs this node&apos;s work within the shared orchestration context.</summary>
    public Func<ProcessorOrchestrationContext, CancellationToken, ValueTask> Execute { get; }

    /// <summary>Creates a no-operation node that completes after all named dependencies finish.</summary>
    /// <param name="name">The unique name of the dependency-join node.</param>
    /// <param name="dependencies">The names of nodes that must complete before this node runs.</param>
    public static ProcessorOrchestrationNode Join(string name, IEnumerable<string> dependencies) =>
        new(name, dependencies, [], NoOperation);
}

/// <summary>Contains an acyclic orchestration graph whose conflicting resource accesses are explicitly ordered.</summary>
public sealed class ProcessorOrchestrationGraph
{
    /// <summary>Creates a graph and validates dependency names, cycles, and unordered resource conflicts.</summary>
    /// <param name="nodes">The orchestration nodes to validate and execute.</param>
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

    /// <summary>Gets the validated orchestration nodes.</summary>
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

/// <summary>Owns published storage leases and provides retained references to concurrent orchestration nodes.</summary>
public sealed class ProcessorOrchestrationContext : IAsyncDisposable
{
    private readonly object gate = new();
    private readonly Dictionary<string, IStorageLease> storage = new(StringComparer.Ordinal);
    private bool disposed;

    /// <summary>Publishes a storage lease under a unique resource name and transfers its ownership to this context.</summary>
    /// <param name="resource">The name of the orchestration resource.</param>
    /// <param name="lease">The storage lease whose ownership is transferred to the orchestration context.</param>
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

    /// <summary>Returns a retained lease for a previously published orchestration resource.</summary>
    /// <param name="resource">The name of the orchestration resource.</param>
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

    /// <summary>Releases all distinct published leases and aggregates cleanup failures.</summary>
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

/// <summary>Executes dependency-ordered processor nodes with bounded concurrency and shared leased resources.</summary>
public sealed class ProcessorOrchestrator
{
    private readonly int maximumConcurrency;

    /// <summary>Creates an orchestrator with a positive limit on simultaneously executing nodes.</summary>
    /// <param name="maximumConcurrency">The positive maximum number of nodes allowed to execute simultaneously.</param>
    public ProcessorOrchestrator(int maximumConcurrency = int.MaxValue)
    {
        if (maximumConcurrency <= 0) throw new ArgumentOutOfRangeException(nameof(maximumConcurrency));
        this.maximumConcurrency = maximumConcurrency;
    }

    /// <summary>Executes all graph nodes and returns their resource context, or cancels remaining work and cleans up on failure.</summary>
    /// <param name="graph">The validated dependency graph of orchestration work.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
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
