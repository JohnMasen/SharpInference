using System.Collections.ObjectModel;

namespace SharpInference.Graphs;

/// <summary>A physical scratch slot description. Allocate its backing separately for each active invocation.</summary>
public sealed record GraphResourceSlot(int Id, GraphResourceLifetime Lifetime, TensorDescriptor Tensor, string? DeviceId = null);

/// <summary>
/// Plans private local scratch by default, or all local resources via <see cref="PlanLocal"/>.
/// The plan contains no backing storage, and a buffer's contents are never assumed to be zero.
/// </summary>
public static class GraphResourceAllocator
{
    /// <remarks>
    /// Private ReadWrite bindings require a durable first-write proof from the graph optimizer,
    /// or an explicit backend guarantee supplied via <paramref name="initializedBeforeRead"/>.
    /// A Read binding cannot establish this; scratch is never implicitly initialized.
    /// </remarks>
    /// <param name="initializedBeforeRead">
    /// Backend-specific proof for ReadWrite scratch without graph metadata. Return true only when
    /// the concrete kernel writes the entire resource before its first read on every invocation.
    /// Do not use this to trust arbitrary serialized or custom operations.
    /// </param>
    public static GraphResourceAllocationPlan Plan(
        ExecutionGraph graph,
        Func<ResourceId, bool>? initializedBeforeRead = null) =>
        PlanCore(graph, initializedBeforeRead is null ? null : (_, id) => initializedBeforeRead(id),
            includePublic: false);

    /// <summary>
    /// Includes public local intermediates as well as private fused scratch. Only use this with a
    /// backend that binds public resources to the resulting slots; legacy backends allocate their
    /// public intermediates independently and should continue using <see cref="Plan"/>. Public
    /// ReadWrite resources require an optimizer first-write proof or an explicit backend guarantee;
    /// public read-only bindings cannot establish that proof themselves. Backend preparation must
    /// validate serialized first-write claims against the supported operation before executing it.
    /// </summary>
    public static GraphResourceAllocationPlan PlanLocal(
        ExecutionGraph graph,
        Func<ResourceId, bool>? initializedBeforeRead = null) =>
        PlanCore(graph, initializedBeforeRead is null ? null : (_, id) => initializedBeforeRead(id),
            includePublic: true);

    /// <summary>
    /// Per-operation form of the backend first-write proof. Use this when the same resource is
    /// accessed by different kernels and not every kernel has the same initialization guarantee.
    /// </summary>
    public static GraphResourceAllocationPlan PlanLocal(
        ExecutionGraph graph,
        Func<ExecutionNode, ResourceId, bool> initializedBeforeRead)
    {
        ArgumentNullException.ThrowIfNull(initializedBeforeRead);
        return PlanCore(graph, initializedBeforeRead, includePublic: true);
    }

    private static GraphResourceAllocationPlan PlanCore(
        ExecutionGraph graph,
        Func<ExecutionNode, ResourceId, bool>? initializedBeforeRead,
        bool includePublic)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        var slots = new List<GraphResourceSlot>();
        var occupants = new List<List<ResourceId>>();
        var assignments = new Dictionary<ResourceId, int>();
        var byNode = graph.Nodes.ToDictionary(node => node.Id);
        var predecessors = graph.Nodes.ToDictionary(node => node.Id, node =>
        {
            var result = new HashSet<ExecutionNodeId>();
            var pending = new Stack<ExecutionNodeId>(node.Dependencies);
            while (pending.TryPop(out var id))
            {
                if (result.Add(id))
                    foreach (var dependency in byNode[id].Dependencies) pending.Push(dependency);
            }
            return result;
        });
        var uses = graph.Nodes.SelectMany(node => node.Resources.Select(binding => (node.Id, binding)))
            .GroupBy(item => item.binding.Resource)
            .ToDictionary(group => group.Key, group => group.ToArray());
        var endpoints = graph.Inputs.Concat(graph.Outputs).ToHashSet();
        var candidates = new List<ResourceId>();

        foreach (var node in graph.Nodes)
        {
            foreach (var binding in node.InternalResources)
            {
                if (binding.Access == GraphResourceAccess.Read ||
                    (binding.Access == GraphResourceAccess.ReadWrite &&
                     !binding.InitializedBeforeRead &&
                     initializedBeforeRead?.Invoke(node, binding.Resource) != true))
                {
                    throw new InvalidDataException(
                        $"Private resource '{binding.Resource}' must be initialized before its first read; scratch is not zero-initialized.");
                }

                candidates.Add(binding.Resource);
            }
        }
        foreach (var resource in includePublic ? graph.Resources : [])
        {
            if (resource.Scope != GraphResourceScope.Local || endpoints.Contains(resource.Id) ||
                graph.InternalResourceOwners.ContainsKey(resource.Id) ||
                !uses.TryGetValue(resource.Id, out var accesses))
                continue;
            foreach (var (writer, binding) in accesses)
            {
                if (binding.Access == GraphResourceAccess.Read) continue;
                foreach (var (other, _) in accesses)
                {
                    if (writer != other && !predecessors[writer].Contains(other) &&
                        !predecessors[other].Contains(writer))
                        throw new InvalidDataException(
                            $"Local resource '{resource.Id}' has unordered accesses at '{writer}' and '{other}'.");
                }
            }
            foreach (var (reader, binding) in accesses)
            {
                if (binding.Access == GraphResourceAccess.Write) continue;
                if (!accesses.Any(access =>
                    (access.binding.Access == GraphResourceAccess.Write ||
                     access.binding.Access == GraphResourceAccess.ReadWrite &&
                     (byNode[access.Id].FirstWriteResources.Contains(resource.Id) ||
                      initializedBeforeRead?.Invoke(byNode[access.Id], resource.Id) == true)) &&
                    predecessors[reader].Contains(access.Id)) &&
                    !(binding.Access == GraphResourceAccess.ReadWrite &&
                        (byNode[reader].FirstWriteResources.Contains(resource.Id) ||
                         initializedBeforeRead?.Invoke(byNode[reader], resource.Id) == true)))
                {
                    throw new InvalidDataException(
                        $"Local resource '{resource.Id}' must be written before its first read.");
                }
            }
            candidates.Add(resource.Id);
        }

        var intervals = candidates.ToDictionary(id => id, LiveIntervals);
        foreach (var id in candidates)
        {
                var resource = resources[id];
                var slot = -1;
                for (var index = 0; index < slots.Count; index++)
                {
                    if (occupants[index].All(other => Compatible(resources[other], resource) &&
                        OrderedUses(other, id)))
                    {
                        slot = index;
                        break;
                    }
                }

                if (slot < 0)
                {
                    slot = slots.Count;
                    slots.Add(new GraphResourceSlot(slot, resource.Lifetime,
                        new TensorDescriptor(resource.Tensor.ElementType, resource.Tensor.Dimensions, resource.Tensor.Layout),
                        resource.DeviceId));
                    occupants.Add([]);
                }

                occupants[slot].Add(id);
                assignments.Add(id, slot);
        }

        if (assignments.Count != candidates.Count)
        {
            throw new InvalidDataException("Scratch assignments must cover exactly the local resources.");
        }

        return new GraphResourceAllocationPlan(slots, assignments);

        bool OrderedUses(ResourceId first, ResourceId second)
        {
            return intervals[first].All(left => intervals[second].All(right =>
                left.All(a => right.All(b => predecessors[b].Contains(a))) ||
                right.All(b => left.All(a => predecessors[a].Contains(b)))));
        }

        List<ExecutionNodeId[]> LiveIntervals(ResourceId id)
        {
            if (graph.InternalResourceOwners.TryGetValue(id, out var owner))
                return [[owner]];

            var accesses = uses[id].GroupBy(access => access.Id).ToArray();
            var overwrites = accesses.Where(group =>
            {
                var bindings = group.Select(access => access.binding).ToArray();
                // A write alongside a read on the same node is not an overwrite boundary without proof.
                return bindings.Any(binding => binding.Access != GraphResourceAccess.Read) &&
                    (bindings.All(binding => binding.Access == GraphResourceAccess.Write) ||
                     (byNode[group.Key].FirstWriteResources.Contains(id) ||
                      initializedBeforeRead?.Invoke(byNode[group.Key], id) == true));
            }).Select(group => group.Key).ToArray();
            var epochs = overwrites.ToDictionary(writer => writer, _ => new List<ExecutionNodeId>());
            foreach (var access in accesses)
            {
                var writer = overwrites
                    .Where(candidate => candidate == access.Key ||
                        predecessors[access.Key].Contains(candidate))
                    .OrderByDescending(candidate => predecessors[candidate].Count)
                    .FirstOrDefault();
                if (!epochs.TryGetValue(writer, out var epoch))
                    throw new InvalidDataException($"Local resource '{id}' has no ordered full write at '{access.Key}'.");
                epoch.Add(access.Key);
            }
            return epochs.Values.Where(epoch => epoch.Count != 0)
                .Select(epoch => epoch.ToArray()).ToList();
        }

        static bool Compatible(GraphResource left, GraphResource right) =>
            left.Lifetime == right.Lifetime &&
            string.Equals(left.DeviceId, right.DeviceId, StringComparison.Ordinal) &&
            left.Tensor.ElementType == right.Tensor.ElementType &&
            string.Equals(left.Tensor.Layout, right.Tensor.Layout, StringComparison.Ordinal) &&
            left.Tensor.Dimensions.SequenceEqual(right.Tensor.Dimensions);
    }
}

/// <summary>An immutable recipe; it never stores or shares backend buffers between invocations.</summary>
public sealed class GraphResourceAllocationPlan
{
    private static readonly HashSet<object> ActiveBuffers = new(ReferenceEqualityComparer.Instance);
    private static readonly object Gate = new();

    internal GraphResourceAllocationPlan(
        IReadOnlyList<GraphResourceSlot> slots,
        IDictionary<ResourceId, int> assignments)
    {
        Slots = Array.AsReadOnly(slots.ToArray());
        SlotByResource = new ReadOnlyDictionary<ResourceId, int>(new Dictionary<ResourceId, int>(assignments));
    }

    public IReadOnlyList<GraphResourceSlot> Slots { get; }
    public IReadOnlyDictionary<ResourceId, int> SlotByResource { get; }

    /// <summary>
    /// Calls the backend once per slot to allocate invocation-local storage. The factory must
    /// return fresh storage; reuse of the same object while another invocation is active is rejected.
    /// Dispose the returned lease after the invocation has finished (including asynchronous work).
    /// The lease disposes each successfully allocated buffer exactly once. The backend remains
    /// responsible for fully initializing each resource before its first read. Sequential token
    /// replays may retain one lease for the entire session and reuse its slots without allocating
    /// either the slots or the lease per token. Do not share that lease between concurrent replays.
    /// </summary>
    public GraphInvocationScratch<T> AllocateInvocation<T>(Func<GraphResourceSlot, T> allocate)
        where T : class, IDisposable =>
        AllocateInvocation(allocate, buffer => buffer.Dispose());

    /// <summary>
    /// Allocates invocation-local storage and transfers ownership of every accepted buffer to the
    /// lease. The release callback runs on lease disposal or if a later allocation fails. Keep the
    /// lease alive until all asynchronous users (including GPU work) have finished. A buffer returned
    /// again while already in use is rejected, not released, because another lease owns it.
    /// </summary>
    public GraphInvocationScratch<T> AllocateInvocation<T>(
        Func<GraphResourceSlot, T> allocate, Action<T> release)
        where T : class
    {
        ArgumentNullException.ThrowIfNull(allocate);
        ArgumentNullException.ThrowIfNull(release);
        lock (Gate)
        {
            var buffers = new T[Slots.Count];
            var added = new List<T>();
            try
            {
                foreach (var slot in Slots)
                {
                    var buffer = allocate(slot) ??
                        throw new InvalidOperationException($"Allocator returned null for scratch slot {slot.Id}.");
                    if (!ActiveBuffers.Add(buffer))
                    {
                        throw new InvalidOperationException("A scratch buffer is already in use by an active invocation or slot.");
                    }
                    added.Add(buffer);
                    buffers[slot.Id] = buffer;
                }

                return new GraphInvocationScratch<T>(this, buffers, release);
            }
            catch (Exception error)
            {
                var cleanupErrors = ReleaseCore(added, release);
                if (cleanupErrors.Count != 0)
                    throw new AggregateException("Scratch allocation and cleanup both failed.",
                        new[] { error }.Concat(cleanupErrors));
                throw;
            }
        }
    }

    internal void Release<T>(IReadOnlyList<T> buffers, Action<T> release) where T : class
    {
        lock (Gate)
        {
            var errors = ReleaseCore(buffers, release);
            if (errors.Count != 0)
                throw new AggregateException("Scratch buffer release failed.", errors);
        }
    }

    private static List<Exception> ReleaseCore<T>(IEnumerable<T> buffers, Action<T> release) where T : class
    {
        var errors = new List<Exception>();
        foreach (var buffer in buffers)
        {
            try
            {
                release(buffer);
            }
            catch (Exception error)
            {
                errors.Add(error);
            }
            finally
            {
                ActiveBuffers.Remove(buffer);
            }
        }
        return errors;
    }
}

/// <summary>Invocation-scoped slot buffers; dispose only after all users of the buffers finish.</summary>
public sealed class GraphInvocationScratch<T> : IDisposable where T : class
{
    private readonly GraphResourceAllocationPlan plan;
    private readonly IReadOnlyList<T> buffers;
    private readonly Action<T> release;
    private int disposed;

    internal GraphInvocationScratch(GraphResourceAllocationPlan plan, IReadOnlyList<T> buffers, Action<T> release)
    {
        this.plan = plan;
        this.buffers = buffers;
        this.release = release;
    }

    public T GetBuffer(ResourceId resource)
    {
        if (Volatile.Read(ref disposed) != 0) throw new ObjectDisposedException(nameof(GraphInvocationScratch<T>));
        return buffers[plan.SlotByResource[resource]];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0) return;
        plan.Release(buffers, release);
    }
}
