#:project ..\src\SharpInference.Graphs.Core\SharpInference.Graphs.Core.csproj
#:project ..\src\SharpInference.Graphs\SharpInference.Graphs.csproj
#:project ..\src\SharpInference.Instructions\SharpInference.Instructions.csproj

using System.Diagnostics;
using SharpInference.Instructions;
using SharpInference.Graphs;
using SharpInference.Graphs.Core;
using SharpInference.Graphs.Execution;
using static SharpInference.Graphs.Core.PatternElements;

// Run with: dotnet run --file scripts\GraphQueryBenchmark.cs -c Release
Console.WriteLine("Median of 7 warmed runs; allocations are per invocation on this thread.");
Console.WriteLine("Scenario | ms | allocated bytes | checksum");
foreach (var size in new[] { 128, 1024 })
{
    var package = CreatePackage(size);
    var graph = package.MainGraph!;
    var view = graph.DependencyGraph();
    var nodes = view.Nodes.ToArray();
    var pairs = Enumerable.Range(0, 2000)
        .Select(i => (From: nodes[(i * 17) % size], To: nodes[(i * 37 + 11) % size])).ToArray();
    var reference = new ReferenceDagIndex(view);
    var index = view.CreateReachabilityIndex(nodes, nodes);
    foreach (var pair in pairs)
        if (reference.Before(pair.From.Id, pair.To.Id) !=
            (pair.From.Id != pair.To.Id && index.IsReachable(pair.From, pair.To)))
            throw new InvalidDataException("Reachability mismatch");
    Measure($"{size}: reference DAG build", () => new ReferenceDagIndex(view).Count);
    Measure($"{size}: indexed DAG build", () => view.CreateReachabilityIndex(nodes, nodes).SourceCount);
    Measure($"{size}: reference 2000 queries", () => pairs.Count(pair => reference.Before(pair.From.Id, pair.To.Id)));
    Measure($"{size}: indexed 2000 queries", () => pairs.Count(pair =>
        pair.From.Id != pair.To.Id && index.IsReachable(pair.From, pair.To)));
    Measure($"{size}: BFS 2000 queries", () => pairs.Count(pair =>
        pair.From.Id != pair.To.Id && view.IsReachable(pair.From.Id, pair.To.Id)));
    Measure($"{size}: index build + 2000 queries", () =>
    {
        var built = view.CreateReachabilityIndex(nodes, nodes);
        return pairs.Count(pair => pair.From.Id != pair.To.Id && built.IsReachable(pair.From, pair.To));
    });
    var anchor = view.GetNode(new("n0"));
    var target = view.GetNode(new("n1"));
    var pattern = Pattern.Create(Node(anchor.Id)) - Edge().Where<FlowEdge>(_ => true) - Node();
    var expected = view.EnumerateEdgesFromNode(anchor.Id).Count();
    if (view.Match(pattern).Count() != expected)
        throw new InvalidDataException("Pattern count mismatch");
    Measure($"{size}: 100 patterns", () => Enumerable.Range(0, 100).Sum(_ => view.Match(pattern).Count()));
    var resource = graph.Resources[0];
    var readers = graph.Actions.Where(action => action.Bindings.Any(binding =>
        binding.Target.TargetProperty == resource.Id && binding.Access != ResourceAccess.Write)).ToArray();
    var writers = graph.Actions.Where(action => action.Bindings.Any(binding =>
        binding.Target.TargetProperty == resource.Id && binding.Access != ResourceAccess.Read)).ToArray();
    if (!readers.SequenceEqual(graph.Readers(resource)) || !writers.SequenceEqual(graph.Writers(resource)))
        throw new InvalidDataException("Reader/writer identities mismatch");
    Measure($"{size}: 1000 indexed readers", () => Enumerable.Range(0, 1000).Sum(_ => readers.Length));
    Measure($"{size}: 1000 production readers", () => Enumerable.Range(0, 1000).Sum(_ => graph.Readers(resource).Count()));
    Measure($"{size}: 1000 indexed writers", () => Enumerable.Range(0, 1000).Sum(_ => writers.Length));
    Measure($"{size}: 1000 production writers", () => Enumerable.Range(0, 1000).Sum(_ => graph.Writers(resource).Count()));
    var planningPattern = Pattern.Create(Node().Where<ActionNode>(_ => true))
        - Edge() - Node().Where<ActionNode>(node => node.Id == anchor.Id);
    var noEdges = view.Filter(edge: _ => false);
    Measure($"{size}: candidate planning without path expansion", () => noEdges.Match(planningPattern).Count());
    Measure($"{size}: match + ToGraph", () => view.Match(pattern).ToGraph().Edges.Count);
    Measure($"{size}: one-off BFS", () => view.IsReachable(anchor.Id, target.Id) ? 1 : 0);
    Measure($"{size}: one-off index build", () => view.CreateReachabilityIndex([anchor], [target]).IsReachable(anchor, target) ? 1 : 0);
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
        samples.Add((Stopwatch.GetElapsedTime(started).TotalMilliseconds,
            GC.GetAllocatedBytesForCurrentThread() - allocated, value));
    }
    if (samples.Any(sample => sample.Value != samples[0].Value))
        throw new InvalidDataException($"Non-deterministic benchmark: {name}");
    var median = samples.OrderBy(sample => sample.Time).ElementAt(3);
    Console.WriteLine($"{name} | {median.Time:F3} | {median.Allocated} | {median.Value}");
}

static GraphPackage<ExecutionGraph> CreatePackage(int size)
{
    var graph = new ExecutionGraph { Id = "benchmark" };
    var tensor = new TensorDescriptor(ElementType.Float32, [4]);
    for (var i = 0; i < size; i++)
        graph.Resource($"r{i}", tensor);
    for (var i = 0; i < size; i++)
    {
        var operation = graph.Operation($"n{i}", Operations.Copy);
        operation.Bind("output", $"r{i}", ResourceAccess.Write);
        if (i > 0)
        {
            operation.Bind("input", $"r{i - 1}", ResourceAccess.Read);
            graph.Actions[i - 1].Then(operation);
        }
        if (i > 5)
            graph.Actions[i - 5].Then(operation);
    }
    return new(graph);
}

sealed class ReferenceDagIndex
{
    private readonly Dictionary<GraphNodeId, int> positions;
    private readonly ulong[][] reachable;
    public int Count => positions.Count;

    public ReferenceDagIndex(IGraphView<ActionNode, FlowEdge> view)
    {
        var order = view.TopologicalOrder();
        positions = order.Select((node, index) => (node.Id, index)).ToDictionary(item => item.Id, item => item.index);
        var successors = order.Select(node => view.EnumerateEdgesFromNode(node.Id)
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

    public bool Before(GraphNodeId from, GraphNodeId to)
    {
        var target = positions[to];
        return (reachable[positions[from]][target / 64] & (1UL << (target % 64))) != 0;
    }
}
