#:project ..\src\SharpInference.Graphs.Core\SharpInference.Graphs.Core.csproj
#:project ..\src\SharpInference.Graphs\SharpInference.Graphs.csproj

using System.Diagnostics;
using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using static SharpInference.Graphs.Core.PatternElements;

// Run with: dotnet run --file scripts\GraphQueryBenchmark.cs -c Release
Console.WriteLine("Median of 7 warmed runs; allocations are per invocation on this thread.");
Console.WriteLine("Scenario | ms | allocated bytes | checksum");
foreach (var size in new[] { 128, 1024 })
{
    var graph = CreateExecutionGraph(size);
    var view = graph.Structure.DependencyView();
    var nodes = view.Nodes.ToArray();
    var pairs = Enumerable.Range(0, 2000)
        .Select(i => (From: nodes[(i * 17) % size], To: nodes[(i * 37 + 11) % size])).ToArray();
    var idPairs = pairs.Select(pair => (From: pair.From.GraphId, To: pair.To.GraphId)).ToArray();
    var reference = new LegacyDagIndex(view);
    var index = view.CreateReachabilityIndex(nodes, nodes);
    foreach (var pair in pairs)
        if (reference.Before(pair.From.GraphId, pair.To.GraphId) !=
            (pair.From.GraphId != pair.To.GraphId && index.IsReachable(pair.From, pair.To)))
            throw new InvalidDataException("Reachability mismatch");
    Measure($"{size}: legacy DAG build", () => new LegacyDagIndex(view).Count);
    Measure($"{size}: new DAG build", () => view.CreateReachabilityIndex(nodes, nodes).SourceCount);
    Measure($"{size}: legacy 2000 queries", () => idPairs.Count(pair => reference.Before(pair.From, pair.To)));
    Measure($"{size}: new 2000 queries", () => idPairs.Count(pair =>
        pair.From != pair.To && index.IsReachable(pair.From, pair.To)));
    Measure($"{size}: BFS 2000 queries", () => idPairs.Count(pair =>
        pair.From != pair.To && view.IsReachable(pair.From, pair.To)));
    Measure($"{size}: new build + 2000 queries", () =>
    {
        var built = view.CreateReachabilityIndex(nodes, nodes);
        return idPairs.Count(pair => pair.From != pair.To && built.IsReachable(pair.From, pair.To));
    });
    var anchor = graph.Structure.GetNode(new("operation:n0"));
    var target = graph.Structure.GetNode(new("operation:n1"));
    var pattern = Pattern.Create(Node(anchor.GraphId))
        - Edge().Where<ExecutionDependencyEdge>(_ => true) - Node();
    var expected = graph.Structure.EdgesFromNode(anchor.GraphId)
        .OfType<ExecutionDependencyEdge>().Count();
    if (graph.Structure.Match(pattern).Count() != expected)
        throw new InvalidDataException("Pattern count mismatch");
    Measure($"{size}: 100 new patterns", () =>
        Enumerable.Range(0, 100).Sum(_ => graph.Structure.Match(pattern).Count()));
    var resource = graph.Structure.NodesOfType<ExecutionResourceNode>().First();
    var readPattern = Pattern.Create(Node(resource.GraphId))
        - Edge().Where<ExecutionResourceReadEdge>(_ => true) - Node();
    Measure($"{size}: 1000 direct readers", () =>
        Enumerable.Range(0, 1000).Sum(_ => graph.Structure.EdgesFromNode(resource.GraphId)
            .OfType<ExecutionResourceReadEdge>().Select(edge => graph.Structure.GetNode(edge.Target))
            .DistinctBy(node => node.GraphId).Count()));
    Measure($"{size}: 1000 production readers", () =>
        Enumerable.Range(0, 1000).Sum(_ => graph.Structure.Readers(resource).Count));
    Measure($"{size}: 1000 matched readers", () =>
        Enumerable.Range(0, 1000).Sum(_ => graph.Structure.Match(readPattern)
            .Select(match => match.Nodes.Last()).DistinctBy(n => n.GraphId).Count()));
    var writePattern = Pattern.Create(Node())
        - Edge().Where<ExecutionResourceWriteEdge>(_ => true) - Node(resource.GraphId);
    var expectedWriters = graph.Structure.EdgesToNode(resource.GraphId)
        .OfType<ExecutionResourceWriteEdge>().Select(edge => edge.Source).Distinct()
        .OrderBy(id => id.Value, StringComparer.Ordinal).ToArray();
    var actualWriters = graph.Structure.Match(writePattern)
        .Select(match => match.Nodes.First().GraphId).Distinct()
        .OrderBy(id => id.Value, StringComparer.Ordinal).ToArray();
    if (!expectedWriters.SequenceEqual(actualWriters) ||
        !expectedWriters.SequenceEqual(graph.Structure.Writers(resource).Select(node => node.GraphId)
            .OrderBy(id => id.Value, StringComparer.Ordinal)))
        throw new InvalidDataException("Writer identities mismatch");
    Measure($"{size}: 1000 indexed writers", () =>
        Enumerable.Range(0, 1000).Sum(_ => graph.Structure.EdgesToNode(resource.GraphId)
            .OfType<ExecutionResourceWriteEdge>().Select(edge => edge.Source).Distinct().Count()));
    Measure($"{size}: 1000 production writers", () =>
        Enumerable.Range(0, 1000).Sum(_ => graph.Structure.Writers(resource).Count));
    Measure($"{size}: 1000 terminal-anchored matched writers", () =>
        Enumerable.Range(0, 1000).Sum(_ => graph.Structure.Match(writePattern)
            .Select(match => match.Nodes.First()).DistinctBy(n => n.GraphId).Count()));
    var planningPattern = Pattern.Create(Node().Where<ExecutionGraphStructureNode>(_ => true))
        - Edge() - Node().Where<ExecutionGraphStructureNode>(node => node.GraphId == resource.GraphId);
    var noEdges = graph.Structure.Filter(edge: _ => false);
    Measure($"{size}: general candidate planning without path expansion", () =>
        noEdges.Match(planningPattern).Count());
    Measure($"{size}: match + ToGraph", () => graph.Structure.Match(pattern).ToGraph().Edges.Count);
    Measure($"{size}: one-off BFS", () => view.IsReachable(anchor.GraphId, target.GraphId) ? 1 : 0);
    Measure($"{size}: one-off index build", () =>
        view.CreateReachabilityIndex([anchor], [target]).IsReachable(anchor, target) ? 1 : 0);
    Console.WriteLine($"{size}: raw full relation masks = {(long)size * ((size + 63) / 64) * 8} bytes");
}

static void Measure(string name, Func<int> action)
{
    for (var i = 0; i < 3; i++) _ = action();
    var samples = new List<(double Time, long Allocated, int Value)>();
    for (var i = 0; i < 7; i++)
    {
        var allocated = GC.GetAllocatedBytesForCurrentThread();
        var started = Stopwatch.GetTimestamp();
        var value = action();
        var elapsed = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
        samples.Add((elapsed, GC.GetAllocatedBytesForCurrentThread() - allocated, value));
    }
    if (samples.Any(sample => sample.Value != samples[0].Value))
        throw new InvalidDataException($"Non-deterministic benchmark: {name}");
    var median = samples.OrderBy(sample => sample.Time).ElementAt(3);
    Console.WriteLine($"{name} | {median.Time:F3} | {median.Allocated} | {median.Value}");
}

static ExecutionGraph CreateExecutionGraph(int size)
{
    var builder = new ExecutionGraphBuilder(new("benchmark"));
    var tensor = new TensorDescriptor(GraphElementType.Float32, [4]);
    for (var i = 0; i < size; i++)
        builder.AddResource(new(new($"r{i}"), $"r{i}", GraphDefinitionResourceKind.Temporary, tensor));
    for (var i = 0; i < size; i++)
    {
        var bindings = new List<ModelResourceBinding>
        {
            new("output", new(GraphBindingOwner.Parent, new($"r{i}")), GraphResourceAccess.Write, $"n{i}:output"),
        };
        if (i > 0)
            bindings.Add(new("input", new(GraphBindingOwner.Parent, new($"r{i - 1}")),
                GraphResourceAccess.Read, $"n{i}:input"));
        var dependencies = new List<ExecutionNodeId>();
        if (i > 0) dependencies.Add(new($"n{i - 1}"));
        if (i > 5) dependencies.Add(new($"n{i - 5}"));
        builder.AddOperation(new(new($"n{i}"), new("core.copy", 1), new Dictionary<string, string>(),
            new(GraphElementType.Float32, GraphElementType.Float32), new([new($"n{i}")])),
            bindings, dependencies);
    }
    return builder.Build();
}

/// <summary>A benchmark-only reference implementation of the pre-migration DAG reachability algorithm.</summary>
sealed class LegacyDagIndex
{
    private readonly Dictionary<GraphNodeId, int> positions;
    private readonly ulong[][] reachable;
    /// <summary>Gets the number of indexed operation nodes.</summary>
    public int Count => positions.Count;

    /// <summary>Builds full strict reachability bit rows in reverse topological order.</summary>
    /// <param name="view">The dependency-only DAG whose operations are indexed.</param>
    /// <exception cref="InvalidDataException">The view contains a directed cycle.</exception>
    public LegacyDagIndex(IGraphView<ExecutionGraphStructureNode, ExecutionGraphEdge> view)
    {
        var order = view.TopologicalOrder();
        positions = order.Select((node, index) => (node.GraphId, index))
            .ToDictionary(item => item.GraphId, item => item.index);
        var successors = order.Select(node => view.EdgesFromNode(node.GraphId)
            .Select(edge => positions[edge.Target]).Distinct().ToArray()).ToArray();
        reachable = order.Select(_ => new ulong[(order.Count + 63) / 64]).ToArray();
        for (var i = order.Count - 1; i >= 0; i--)
            foreach (var successor in successors[i])
            {
                reachable[i][successor / 64] |= 1UL << (successor % 64);
                for (var word = 0; word < reachable[i].Length; word++)
                    reachable[i][word] |= reachable[successor][word];
            }
    }

    /// <summary>Tests strict reachability using the benchmark reference bit rows.</summary>
    /// <param name="from">The indexed source operation identity.</param>
    /// <param name="to">The indexed target operation identity.</param>
    /// <returns>Whether a nonempty directed path connects the two operations.</returns>
    /// <exception cref="KeyNotFoundException">Either identity is not indexed.</exception>
    public bool Before(GraphNodeId from, GraphNodeId to)
    {
        var target = positions[to];
        return (reachable[positions[from]][target / 64] & (1UL << (target % 64))) != 0;
    }
}
