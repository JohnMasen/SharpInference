using SharpInference.Graphs.Core;
using SharpInference.Graphs;
using static SharpInference.Graphs.Core.PatternElements;

namespace SharpInference.Tests;

/// <summary>Verifies composable graph queries, exact match views, and scoped reachability indexes.</summary>
public sealed class GraphViewQueryTests
{
    /// <summary>A named test node with queryable classification and enabled state.</summary>
    /// <param name="Name">The shared test name and structural identifier.</param>
    /// <param name="Kind">The classification used to distinguish operations from resources.</param>
    /// <param name="Enabled">Whether enabled-node queries should retain the node.</param>
    private sealed record TestNode(string Name, string Kind = "op", bool Enabled = true) : IGraphNode
    {
        /// <inheritdoc/>
        public GraphNodeId Id => new(Name);
    }

    /// <summary>A directed test edge with a queryable relationship classification.</summary>
    /// <param name="Name">The unique edge identifier.</param>
    /// <param name="From">The structural identifier of the source node.</param>
    /// <param name="To">The structural identifier of the target node.</param>
    /// <param name="Kind">The dependency, read, or write classification.</param>
    private sealed record TestEdge(string Name, string From, string To, string Kind = "dep") : IGraphEdge
    {
        /// <inheritdoc/>
        public GraphEdgeId Id => new(Name);
        /// <inheritdoc/>
        public GraphNodeId Source => new(From);
        /// <inheritdoc/>
        public GraphNodeId Target => new(To);
    }

    /// <summary>Creates an edge constraint for one test relationship classification.</summary>
    /// <param name="kind">The classification to retain.</param>
    /// <returns>A typed edge constraint matching that classification.</returns>
    private static EdgePattern Kind(string kind) => Edge().Where<TestEdge>(edge => edge.Kind == kind);
    /// <summary>Formats node names without changing their enumeration order.</summary>
    /// <param name="nodes">The nodes to format.</param>
    /// <returns>Comma-separated node names.</returns>
    private static string Names(IEnumerable<TestNode> nodes) => string.Join(",", nodes.Select(node => node.Name));

    /// <summary>Builds the shared twenty-node, thirty-five-edge query fixture.</summary>
    /// <returns>A graph containing operations, resources, parallel dependencies, and resource accesses.</returns>
    private static Graph<TestNode, TestEdge> Example()
    {
        var nodes = "ABCDEFGHIJKL".Select(id => new TestNode(id.ToString(), Enabled: id != 'H'))
            .Concat(Enumerable.Range(1, 8).Select(i => new TestNode($"R{i}", "resource")));
        (string, string)[] dependencies =
        [
            ("A", "C"), ("B", "C"), ("C", "D"), ("C", "E"), ("D", "F"), ("E", "F"),
            ("F", "G"), ("F", "H"), ("G", "I"), ("H", "I"), ("I", "J"), ("J", "K"),
            ("K", "L"), ("A", "F"), ("C", "D"),
        ];
        var edges = dependencies.Select((pair, i) => new TestEdge($"d{i + 1}", pair.Item1, pair.Item2))
            .Concat(new TestEdge[]
            {
                new("w1", "A", "R1", "write"),
                new("r1", "R1", "C", "read"), new("r2", "R1", "C", "read"), new("r3", "R1", "D", "read"),
                new("w2", "B", "R2", "write"), new("r4", "R2", "C", "read"),
                new("w3", "C", "R3", "write"), new("r5", "R3", "D", "read"), new("r6", "R3", "E", "read"),
                new("w4", "D", "R4", "write"), new("r7", "R4", "F", "read"),
                new("w5", "E", "R5", "write"), new("r8", "R5", "F", "read"),
                new("w6", "F", "R6", "write"), new("r9", "R6", "G", "read"), new("r10", "R6", "H", "read"),
                new("w7", "I", "R7", "write"), new("r11", "R7", "J", "read"),
                new("w8", "K", "R8", "write"), new("r12", "R8", "L", "read"),
            });
        return new(nodes, edges);
    }

    /// <summary>Checks extension-only query contracts and composition across graph, filter, and match views.</summary>
    [Fact]
    public void QueryOperationsAreExtensionsAndComposeAcrossViewKinds()
    {
        var contract = typeof(IGraphView<TestNode, TestEdge>);
        foreach (var name in new[] { "Filter", "Match", "ToGraph", "TopologicalOrder", "CreateReachabilityIndex" })
            Assert.DoesNotContain(contract.GetMethods(), method => method.Name == name);

        IGraphView<TestNode, TestEdge> graph = Example();
        IGraphView<TestNode, TestEdge> filtered = graph.Filter(node => node.Kind == "op", edge => edge.Kind == "dep");
        var matches = filtered.Match(
            Pattern.Create(Node(new GraphNodeId("A"))) - Edge() - Node("F"));
        var match = Assert.Single(matches);
        var merged = matches.ToGraph();
        Assert.Equal("A,F", Names(merged.Nodes));
        Assert.Single(merged.Edges);
        Assert.Single(match.Filter().Edges);
        Assert.Single(match.Match(Pattern.Create(Node("A")) - Edge() - Node("F")));
        Assert.Equal("A,F", Names(match.TopologicalOrder()));
        Assert.Single(match.ToGraph().Edges);
        Assert.True(match.CreateReachabilityIndex(match.Nodes, match.Nodes)
            .IsReachable(new GraphNodeId("A"), new GraphNodeId("F")));
    }

    /// <summary>Checks implicit string identities in direct lookups, adjacency queries, and indexed reachability.</summary>
    [Fact]
    public void StringsConvertToNodeIdsInGraphQueries()
    {
        GraphNodeId id = "A";
        Assert.Equal(new GraphNodeId("A"), id);
        Assert.Equal("A", id.ToString());
        var graph = Example();
        Assert.Equal(id, graph.GetNode("A").Id);
        Assert.True(graph.ContainsNode("A"));
        Assert.Equal(3, graph.EdgesFromNode("A").Count);
        Assert.Equal(5, graph.EdgesToNode("F").Count);
        Assert.Single(graph.Match(Pattern.Create(Node(id))));
        var view = graph.Filter(edge: edge => edge.Kind == "dep");
        var index = view.CreateReachabilityIndex([graph.GetNode("A")], [graph.GetNode("F")]);
        Assert.True(index.IsReachable("A", "F"));
        Assert.Throws<KeyNotFoundException>(() => graph.GetNode("missing"));
    }

    /// <summary>Checks production model queries against adjacency order, duplicates, hidden nodes, and self-loops.</summary>
    [Fact]
    public void ProductionModelQueriesPreserveAdjacencySemantics()
    {
        var first = new ExecutionOperationNode(new("first"), new("core.copy", 1),
            new Dictionary<string, string>(), new(GraphElementType.Float32, GraphElementType.Float32), new([]));
        var second = first with { Id = new("second") };
        var resource = new ExecutionResourceNode(new(GraphBindingOwner.Parent, new("state")),
            new(new("state"), "display-state", GraphDefinitionResourceKind.InOut,
                new(GraphElementType.Float32, [4])));
        var graph = new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(
            [first, second, resource],
            [
                new ExecutionResourceReadEdge(new("r2"), resource.GraphId, second.GraphId, "input", "r2"),
                new ExecutionResourceReadEdge(new("r1"), resource.GraphId, first.GraphId, "input", "r1"),
                new ExecutionResourceReadEdge(new("r3"), resource.GraphId, second.GraphId, "other", "r3"),
                new ExecutionDependencyEdge(new("d2"), first.GraphId, second.GraphId),
                new ExecutionDependencyEdge(new("d1"), first.GraphId, second.GraphId),
                new ExecutionResourceWriteEdge(new("w1"), first.GraphId, resource.GraphId, "output", "w1"),
            ]);
        Assert.Equal([second.GraphId, first.GraphId], graph.Readers(resource).Select(node => node.GraphId));
        Assert.Equal([second.GraphId, second.GraphId], graph.Successors(first).Select(node => node.GraphId));
        Assert.Equal([first.GraphId], graph.Writers(resource).Select(node => node.GraphId));
        Assert.Equal(["r1", "w1"], graph.Bindings(first).Select(binding => binding.BindingId));
        var filtered = graph.Filter(node => node.GraphId != second.GraphId);
        Assert.Equal([first.GraphId], filtered.Readers(resource).Select(node => node.GraphId));
        Assert.Empty(filtered.Successors(first));
        Assert.Throws<KeyNotFoundException>(() =>
            graph.Filter(node => node.GraphId != resource.GraphId).Readers(resource));
        Assert.Single(graph.Match(Pattern.Create(Node("resource:parent:state"))));
        Assert.Empty(graph.Match(Pattern.Create(Node("display-state"))));
        var cyclic = new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(graph.Nodes,
            graph.Edges.Append(new ExecutionDependencyEdge(new("self"), first.GraphId, first.GraphId)));
        Assert.Equal([second.GraphId, second.GraphId, first.GraphId],
            cyclic.Successors(first).Select(node => node.GraphId));
        var otherResource = new ExecutionResourceNode(new(GraphBindingOwner.Parent, new("other")),
            resource.Resource with { Id = new("other") });
        var boundaryGraph = new Graph<ExecutionGraphStructureNode, ExecutionGraphEdge>(
            graph.Nodes.Append(otherResource),
            new ExecutionGraphEdge[]
            {
                new ExecutionResourceReadEdge(new("other-read"), otherResource.GraphId, first.GraphId,
                    "other-input", "other-read"),
                new ExecutionResourceWriteEdge(new("other-write"), first.GraphId, otherResource.GraphId,
                    "other-output", "other-write"),
            }.Concat(graph.Edges));
        var boundary = boundaryGraph.BoundaryResources([first.GraphId, second.GraphId]);
        Assert.Equal([otherResource.GraphId, resource.GraphId], boundary.Inputs.Select(node => node.GraphId));
        Assert.Equal([otherResource.GraphId, resource.GraphId], boundary.Outputs.Select(node => node.GraphId));
    }

    /// <summary>Validates the eighteen progressively more complex queries from the design examples.</summary>
    /// <param name="query">The one-based example number to execute.</param>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(10)]
    [InlineData(11)]
    [InlineData(12)]
    [InlineData(13)]
    [InlineData(14)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(18)]
    public void TwentyNodeQueries(int query)
    {
        var graph = Example();
        Assert.Equal(20, graph.Nodes.Count);
        Assert.Equal(35, graph.Edges.Count);
        var dep = graph.Filter(node => node.Kind == "op", edge => edge.Kind == "dep");
        IEnumerable<TestNode> Readers(string name) => graph.Match(
            Pattern.Create(Node(name)) - Kind("read") - Node())
            .Select(match => match.Nodes.Last()).DistinctBy(node => node.Id);
        IEnumerable<TestNode> Writers(string name) => graph.Match(
            Pattern.Create(Node()) - Kind("write") - Node(name))
            .Select(match => match.Nodes.First()).DistinctBy(node => node.Id);
        var index = dep.CreateReachabilityIndex(dep.Nodes, dep.Nodes);
        TestNode[] Terminal(string name)
        {
            var accesses = Readers(name).Concat(Writers(name)).DistinctBy(node => node.Id).ToArray();
            return accesses.Where(first => !accesses.Any(second =>
                first.Id != second.Id && index.IsReachable(first, second))).ToArray();
        }

        switch (query)
        {
            case 1:
                Assert.Equal("A,B,C,D,E,F,G,I,J,K,L", Names(graph.Match(
                    Pattern.Create(Node().Where<TestNode>(node => node.Kind == "op" && node.Enabled)))
                    .Select(match => match.Nodes.Single())));
                break;
            case 2:
                Assert.Equal("A,B", Names(graph.Match(Pattern.Create(
                    Node().Where<TestNode>(node => node.Name == "A" || node.Name == "B")))
                    .Select(match => match.Nodes.Single())));
                break;
            case 3:
                Assert.Equal("D,E,D", Names(dep.Match(Pattern.Create(Node("C")) - Edge() - Node())
                    .Select(match => match.Nodes.Last())));
                break;
            case 4:
                Assert.Equal(["A", "D", "E"], dep.Match(Pattern.Create(Node()) - Edge() - Node("F"))
                    .Select(match => match.Nodes.First().Name).Order());
                Assert.Equal("D,E,A", Names(dep.EdgesToNode(new("F"))
                    .Select(edge => dep.GetNode(edge.Source))));
                break;
            case 5:
                Assert.Equal("C,D", Names(Readers("R1")));
                break;
            case 6:
                Assert.Equal("C", Names(Writers("R3")));
                break;
            case 7:
                Assert.Equal(12, graph.Match(Pattern.Create(Node())
                    - Kind("write") - Node() - Kind("read") - Node()).Count());
                break;
            case 8:
                Assert.Equal(["D:d3", "E:d4", "D:d15"], dep.Match(Pattern.Create(Node("C"))
                    - Edge() - Node() - Edge() - Node("F"))
                    .Select(match => $"{match.Nodes.ElementAt(1).Name}:{match.Edges.First().Name}"));
                break;
            case 9:
                var merged = graph.Match(Pattern.Create(Node("R1")) - Kind("read") - Node()).ToGraph();
                Assert.Equal("R1,C,D", Names(merged.Nodes));
                Assert.Equal(["r1", "r2", "r3"], merged.Edges.Select(edge => edge.Name));
                break;
            case 10:
                var reads = graph.Match(Pattern.Create(Node("R1")) - Kind("read") - Node()).ToGraph();
                Assert.False(reads.Match(Pattern.Create(Node("C")) - Kind("dep") - Node("D")).Any());
                break;
            case 11:
                var withoutH = dep.Filter(node => node.Name != "H");
                Assert.Equal(11, withoutH.Nodes.Count());
                Assert.Equal(13, withoutH.Edges.Count());
                break;
            case 12:
                Assert.True(dep.Match(Pattern.Create(Node("A")) - Edge() - Node("F")).Any());
                Assert.False(dep.Match(Pattern.Create(Node("C")) - Edge() - Node("L")).Any());
                break;
            case 13:
                Assert.Equal("A,B,C,D,E,F,G,H,I,J,K,L", Names(dep.TopologicalOrder()));
                break;
            case 14:
                var scoped = dep.CreateReachabilityIndex(
                    dep.Nodes.Where(node => new[] { "A", "B", "D", "E", "G", "H" }.Contains(node.Name)),
                    dep.Nodes.Where(node => new[] { "F", "I", "L" }.Contains(node.Name)));
                Assert.Equal(6, scoped.SourceCount);
                Assert.Equal(3, scoped.TargetCount);
                foreach (var source in new[] { "A", "B", "D", "E", "G", "H" })
                    foreach (var target in new[] { "F", "I", "L" })
                        Assert.Equal(target != "F" || source is not ("G" or "H"),
                            scoped.IsReachable(new GraphNodeId(source), new GraphNodeId(target)));
                break;
            case 15:
                Assert.Equal("D", Names(Terminal("R1")));
                Assert.Equal("G,H", Names(Terminal("R6")));
                break;
            case 16:
                Assert.All(Terminal("R2"), last => Assert.True(
                    last.Id != new GraphNodeId("D") && index.IsReachable(last.Id, new GraphNodeId("D"))));
                Assert.False(Terminal("R4").All(last => index.IsReachable(last.Id, new GraphNodeId("E"))));
                break;
            case 17:
                var seen = new HashSet<(GraphNodeId, GraphNodeId)>();
                var redundant = dep.Edges.Where(edge => !seen.Add((edge.Source, edge.Target)) ||
                    dep.EdgesFromNode(edge.Source).Any(other => other.Target != edge.Target &&
                        index.IsReachable(other.Target, edge.Target)));
                Assert.Equal(["d14", "d15"], redundant.Select(edge => edge.Name));
                break;
            case 18:
                Assert.Equal("L", Names(dep.Nodes.Where(node => !dep.EdgesFromNode(node.Id).Any())));
                break;
        }
    }

    /// <summary>Checks the strict fifty-percent direction threshold and single evaluation of endpoint predicates.</summary>
    /// <param name="targetCount">The number of nodes satisfying the terminal condition.</param>
    /// <param name="reverse">Whether the strict threshold requires incoming-edge traversal.</param>
    [Theory]
    [InlineData(0, false)]
    [InlineData(49, true)]
    [InlineData(50, false)]
    [InlineData(51, false)]
    public void EndpointCountsSelectSearchDirection(int targetCount, bool reverse)
    {
        var nodes = Enumerable.Range(0, 100).Select(i => new TestNode($"N{i}")).ToArray();
        var graph = new Graph<TestNode, TestEdge>(nodes, [new("edge", "N99", "N0")]);
        var view = new CountingView(graph);
        var sourceCalls = 0;
        var targetCalls = 0;
        var pattern = Pattern.Create(Node().Where<TestNode>(_ => { sourceCalls++; return true; }))
            - Edge() - Node().Where<TestNode>(node =>
            {
                targetCalls++;
                return int.Parse(node.Name.AsSpan(1)) < targetCount;
            });
        var results = view.Match(pattern);
        Assert.Equal(0, sourceCalls);
        Assert.Equal(targetCount == 0 ? 0 : 1, results.Count());
        Assert.Equal(100, sourceCalls);
        Assert.Equal(100, targetCalls);
        Assert.Equal(reverse ? targetCount : 0, view.IncomingCalls);
        Assert.Equal(targetCount == 0 || reverse ? 0 : 100, view.OutgoingCalls);
        Assert.Equal(100, view.EnumeratedNodes);
    }

    /// <summary>Checks terminal identity planning, short-circuit traversal, and bounded candidate counting.</summary>
    [Fact]
    public void TerminalIdentityUsesIncomingIndexAndShortCircuits()
    {
        var graph = Example();
        var view = new CountingView(graph);
        var edgeCalls = 0;
        var pattern = Pattern.Create(Node())
            - Edge().Where<TestEdge>(_ => { edgeCalls++; return true; }) - Node("F");
        var results = view.Match(pattern);
        Assert.Equal(0, view.EnumeratedNodes);
        var first = results.First();
        Assert.Equal("F", first.Nodes.Last().Name);
        Assert.Single(first.Edges);
        Assert.Equal(3, view.EnumeratedNodes);
        Assert.Equal(1, view.IncomingCalls);
        Assert.Equal(0, view.OutgoingCalls);
        Assert.Equal(1, edgeCalls);
        var missing = new CountingView(graph);
        Assert.Empty(missing.Match(Pattern.Create(Node()) - Edge() - Node("missing")));
        Assert.Equal(0, missing.EnumeratedNodes);
    }

    /// <summary>Checks that a terminal identity predicate and filtered source candidates are evaluated once.</summary>
    [Fact]
    public void PlannedIdentityAndSourcePredicatesAreNotRepeated()
    {
        var graph = Example();
        var view = new CountingView(graph);
        var sourceCalls = 0;
        var targetCalls = 0;
        var pattern = Pattern.Create(Node().Where<TestNode>(node =>
            { sourceCalls++; return node.Kind == "op"; }))
            - Edge() - Node("F").Where<TestNode>(_ => { targetCalls++; return true; });
        Assert.Equal(3, view.Match(pattern).Count());
        Assert.Equal(20, sourceCalls);
        Assert.Equal(1, targetCalls);
        Assert.Equal(1, view.IncomingCalls);
        Assert.Equal(0, view.OutgoingCalls);
    }

    /// <summary>Checks reverse multi-edge matching against forward searches, including parallel edges and cycles.</summary>
    [Fact]
    public void ReverseMatchesHaveExactForwardPathsAndStableResults()
    {
        var original = Example();
        var graph = new Graph<TestNode, TestEdge>(original.Nodes,
            original.Edges.Append(new("cycle", "F", "C")));
        var view = graph.Filter(edge: edge => edge.Kind == "dep");
        var expected = view.Nodes.SelectMany(node => view.Match(
                Pattern.Create(Node(node.Id)) - Edge() - Node() - Edge() - Node("F")))
            .Select(match => string.Join(",", match.Edges.Select(edge => edge.Id.Value))).Order().ToArray();
        var results = view.Match(Pattern.Create(Node()) - Edge() - Node() - Edge() - Node("F")).ToArray();
        Assert.Equal(expected, results
            .Select(match => string.Join(",", match.Edges.Select(edge => edge.Id.Value))).Order());
        Assert.All(results, match =>
        {
            var nodes = match.Nodes.ToArray();
            var edges = match.Edges.ToArray();
            Assert.Equal("F", nodes[^1].Name);
            Assert.Equal(nodes.Length, nodes.Select(node => node.Id).Distinct().Count());
            for (var i = 0; i < edges.Length; i++)
            {
                Assert.Equal(nodes[i].Id, edges[i].Source);
                Assert.Equal(nodes[i + 1].Id, edges[i].Target);
            }
        });
        var retained = results[0].Edges.ToArray();
        Assert.Equal(retained, results[0].Edges);
        Assert.Empty(view.Filter(node => node.Name != "F").Match(
            Pattern.Create(Node()) - Edge() - Node("F")));
    }

    /// <summary>Checks small views at and below the strict identity-anchor threshold.</summary>
    /// <param name="count">The total visible node count.</param>
    /// <param name="reverse">Whether three or more source candidates select reverse search.</param>
    [Theory]
    [InlineData(1, false)]
    [InlineData(2, false)]
    [InlineData(3, true)]
    public void TerminalIdentityThresholdHandlesSmallViews(int count, bool reverse)
    {
        var nodes = Enumerable.Range(0, count).Select(i => new TestNode($"N{i}")).ToArray();
        var view = new CountingView(new(nodes,
            count == 1 ? [] : [new TestEdge("edge", "N0", $"N{count - 1}")]));
        Assert.Equal(count == 1 ? 0 : 1, view.Match(
            Pattern.Create(Node()) - Edge() - Node((GraphNodeId)$"N{count - 1}")).Count());
        Assert.Equal(reverse ? 1 : 0, view.IncomingCalls);
        Assert.Equal(reverse ? 0 : count, view.OutgoingCalls);
    }

    /// <summary>A view instrumenting candidate enumeration and the direction of adjacency access.</summary>
    /// <param name="graph">The backing graph whose content and order are preserved.</param>
    private sealed class CountingView(Graph<TestNode, TestEdge> graph) : IGraphView<TestNode, TestEdge>
    {
        /// <summary>Gets the number of nodes consumed from the visible-node sequence.</summary>
        public int EnumeratedNodes { get; private set; }
        /// <summary>Gets the number of incoming adjacency lookups.</summary>
        public int IncomingCalls { get; private set; }
        /// <summary>Gets the number of outgoing adjacency lookups.</summary>
        public int OutgoingCalls { get; private set; }
        /// <inheritdoc/>
        public IEnumerable<TestNode> Nodes => Enumerate();
        /// <inheritdoc/>
        public IEnumerable<TestEdge> Edges => graph.Edges;
        /// <summary>Retrieves a backing node by identity.</summary>
        /// <param name="id">The node identity.</param>
        /// <returns>The backing node.</returns>
        public TestNode GetNode(GraphNodeId id) => graph.GetNode(id);
        /// <summary>Checks whether the backing graph contains a node.</summary>
        /// <param name="id">The identity to check.</param>
        /// <returns>Whether the identity exists.</returns>
        public bool ContainsNode(GraphNodeId id) => graph.ContainsNode(id);
        /// <summary>Counts an incoming lookup and returns its edges.</summary>
        /// <param name="id">The target node identity.</param>
        /// <returns>The backing incoming edges.</returns>
        public IEnumerable<TestEdge> EdgesToNode(GraphNodeId id)
        {
            IncomingCalls++;
            return graph.EdgesToNode(id);
        }
        /// <summary>Counts an outgoing lookup and returns its edges.</summary>
        /// <param name="id">The source node identity.</param>
        /// <returns>The backing outgoing edges.</returns>
        public IEnumerable<TestEdge> EdgesFromNode(GraphNodeId id)
        {
            OutgoingCalls++;
            return graph.EdgesFromNode(id);
        }
        /// <summary>Counts visible nodes as they are consumed.</summary>
        /// <returns>The backing nodes in declaration order.</returns>
        private IEnumerable<TestNode> Enumerate()
        {
            foreach (var node in graph.Nodes)
            {
                EnumeratedNodes++;
                yield return node;
            }
        }
    }

    /// <summary>Checks pattern branching, deferred searches, short-circuiting, and stable retained results.</summary>
    [Fact]
    public void PatternsAreImmutableAndMatchesAreStableLazyAndRepeatable()
    {
        var graph = Example();
        var checks = 0;
        var start = Pattern.Create(Node().Where<TestNode>(node => { checks++; return true; }));
        var sequence = graph.Match(start);
        Assert.Equal(0, checks);
        Assert.True(sequence.Any());
        Assert.Equal(1, checks);
        Assert.Equal(20, sequence.Count());
        Assert.Equal(21, checks);

        var fromC = Pattern.Create(Node("C"));
        var toD = fromC - Kind("dep") - Node("D");
        var toE = fromC - Kind("dep") - Node("E");
        Assert.Single(graph.Match(fromC));
        Assert.Equal(2, graph.Match(toD).Count());
        Assert.Single(graph.Match(toE));
        var results = graph.Match(toD).ToArray();
        Assert.Equal(["d3"], results[0].Edges.Select(edge => edge.Name));
        Assert.Equal(["d15"], results[1].Edges.Select(edge => edge.Name));
        Assert.Equal(["d3"], results[0].Edges.Select(edge => edge.Name));
    }

    /// <summary>Checks that an identity-anchored match uses lookup instead of scanning all nodes.</summary>
    [Fact]
    public void ExactIdentityUsesLookupWithoutEnumeratingNodes()
    {
        var graph = Example();
        var view = new LookupOnlyView(graph);
        var match = Assert.Single(view.Match(
            Pattern.Create(Node(new GraphNodeId("A"))) - Kind("write") - Node()));
        Assert.Equal("A,R1", Names(match.Nodes));
        Assert.Empty(view.Match(Pattern.Create(Node(new GraphNodeId("unknown")))));
    }

    /// <summary>A view that rejects node enumeration to detect unintended full scans.</summary>
    /// <param name="graph">The backing graph providing identity and adjacency lookups.</param>
    private sealed class LookupOnlyView(Graph<TestNode, TestEdge> graph) : IGraphView<TestNode, TestEdge>
    {
        /// <summary>Throws if a query attempts to enumerate all nodes.</summary>
        public IEnumerable<TestNode> Nodes => throw new InvalidOperationException("Unexpected full scan");
        /// <inheritdoc/>
        public IEnumerable<TestEdge> Edges => graph.Edges;
        /// <summary>Delegates node lookup to the backing graph.</summary>
        /// <param name="id">The structural identity to retrieve.</param>
        /// <returns>The matching backing graph node.</returns>
        public TestNode GetNode(GraphNodeId id) => graph.GetNode(id);
        /// <summary>Delegates node existence checks to the backing graph.</summary>
        /// <param name="id">The structural identity to test.</param>
        /// <returns>Whether the backing graph contains the identity.</returns>
        public bool ContainsNode(GraphNodeId id) => graph.ContainsNode(id);
        /// <summary>Delegates incoming-edge lookup to the backing graph.</summary>
        /// <param name="id">The target node identity.</param>
        /// <returns>Incoming edges in backing graph order.</returns>
        public IEnumerable<TestEdge> EdgesToNode(GraphNodeId id) => graph.EdgesToNode(id);
        /// <summary>Delegates outgoing-edge lookup to the backing graph.</summary>
        /// <param name="id">The source node identity.</param>
        /// <returns>Outgoing edges in backing graph order.</returns>
        public IEnumerable<TestEdge> EdgesFromNode(GraphNodeId id) => graph.EdgesFromNode(id);
    }

    /// <summary>Checks that nested queries and materialization cannot restore edges outside a matched view.</summary>
    [Fact]
    public void MatchAndFilterNeverRestoreHiddenEdges()
    {
        var graph = Example();
        var first = graph.Match(Pattern.Create(Node("C")) - Kind("dep") - Node("D")).First();
        Assert.Single(first.Edges);
        Assert.Single(first.Match(Pattern.Create(Node()) - Edge() - Node()));
        Assert.Single(first.Filter().Edges);
        Assert.Single(first.ToGraph().Edges);
        var hiddenD = first.Filter(node => node.Name != "D");
        Assert.Empty(hiddenD.Edges);
        Assert.Empty(hiddenD.EdgesFromNode(new("C")));
        Assert.Throws<KeyNotFoundException>(() => hiddenD.EdgesToNode(new("D")));
        Assert.Throws<KeyNotFoundException>(() => hiddenD.GetNode(new("D")));
    }

    /// <summary>Checks nested filter scope, isolated nodes, and declaration order of parallel edges.</summary>
    [Fact]
    public void FiltersComposeAndPreserveIsolatedNodesAndParallelEdges()
    {
        var graph = Example();
        var dep = graph.Filter(edge: edge => edge.Kind == "dep");
        Assert.Equal(20, dep.Nodes.Count());
        Assert.Empty(dep.EdgesFromNode(new("R1")));
        Assert.Equal(["d3", "d4", "d15"], dep.EdgesFromNode(new("C")).Select(edge => edge.Name));
        var onlyD = dep.Filter(node => node.Name == "D");
        Assert.Single(onlyD.Nodes);
        Assert.Empty(onlyD.Edges);
        Assert.Empty(onlyD.Match(Pattern.Create(Node("C"))));
    }

    /// <summary>Checks that strings anchor structural identities independently of display names.</summary>
    [Fact]
    public void NamedPatternsDoNotConfuseNamesWithIds()
    {
        var graph = new Graph<NamedNode, TestEdge>([new(new("operation:1"), "display")], []);
        Assert.Empty(graph.Match(Pattern.Create(Node("display"))));
        Assert.Single(graph.Match(Pattern.Create(Node("operation:1"))));
        Assert.Single(graph.Match(Pattern.Create(Node().Where<NamedNode>(node => node.Name == "display"))));
        Assert.Single(graph.Match(Pattern.Create(Node(new GraphNodeId("operation:1")))));
    }

    /// <summary>A test node whose display name is independent of its identity.</summary>
    /// <param name="Id">The structural node identifier.</param>
    /// <param name="Name">The display name available to explicit typed predicates.</param>
    private sealed record NamedNode(GraphNodeId Id, string Name) : IGraphNode;
    /// <summary>A test node that deliberately has no name contract.</summary>
    /// <param name="Id">The structural node identifier.</param>
    private sealed record UnnamedNode(GraphNodeId Id) : IGraphNode;

    /// <summary>Checks that type guards prevent predicate execution on incompatible nodes.</summary>
    [Fact]
    public void TypedPredicatesDoNotExecuteForOtherTypes()
    {
        var graph = new Graph<IGraphNode, IGraphEdge>(
            [new TestNode("A"), new UnnamedNode(new("B"))],
            [new TestEdge("ab", "A", "B")]);
        var calls = 0;
        var matches = graph.Match(Pattern.Create(Node().Where<TestNode>(node =>
            { calls++; return true; })));
        Assert.Single(matches);
        Assert.Equal(1, calls);
        Assert.Single(graph.Match(Pattern.Create(Node("B"))));
    }

    /// <summary>Checks explicit failures for invalid pattern alternation, empty identities, and null inputs.</summary>
    [Fact]
    public void InvalidPatternsAndNullInputsFailExplicitly()
    {
        var graph = Example();
        var start = Pattern.Create(Node());
        Assert.Throws<ArgumentException>(() => start - Node());
        Assert.Throws<ArgumentException>(() => start - Edge() - Edge());
        Assert.Throws<ArgumentException>(() => graph.Match(start - Edge()));
        Assert.Throws<ArgumentNullException>(() => Pattern.Create(null!));
        Assert.Throws<ArgumentNullException>(() => Node().Where<TestNode>(null!));
        Assert.Throws<ArgumentNullException>(() => Edge().Where<TestEdge>(null!));
        Assert.Throws<ArgumentException>(() => Node(""));
        Assert.Throws<ArgumentException>(() => Node(default(GraphNodeId)));
    }

    /// <summary>Checks that linear matches exclude repeated nodes, including self-loop paths.</summary>
    [Fact]
    public void SimpleLinearMatchesDoNotRevisitNodesOnCycles()
    {
        var graph = new Graph<TestNode, TestEdge>(
            [new("A"), new("B")], [new("ab", "A", "B"), new("ba", "B", "A"), new("aa", "A", "A")]);
        Assert.Equal(2, graph.Match(Pattern.Create(Node()) - Edge() - Node()).Count());
        Assert.Empty(graph.Match(Pattern.Create(Node()) - Edge() - Node() - Edge() - Node()));
    }

    /// <summary>Checks exact-content union, empty unions, and rejection of conflicting node or edge identities.</summary>
    [Fact]
    public void MaterializationMergesExactContentAndRejectsConflictingIds()
    {
        var graph = Example();
        var merged = graph.Match(Pattern.Create(Node("C")) - Kind("dep") - Node("D")).ToGraph();
        Assert.Equal("C,D", Names(merged.Nodes));
        Assert.Equal(["d3", "d15"], merged.Edges.Select(edge => edge.Name));
        Assert.Empty(Array.Empty<IGraphView<TestNode, TestEdge>>().ToGraph().Nodes);
        var conflictingNodes = new Graph<TestNode, TestEdge>([new("C", "resource")], []);
        Assert.Throws<InvalidDataException>(() =>
            new IGraphView<TestNode, TestEdge>[] { merged, conflictingNodes }.ToGraph());
        var conflictingEdges = new Graph<TestNode, TestEdge>(merged.Nodes, [new("d3", "D", "C")]);
        Assert.Throws<InvalidDataException>(() =>
            new IGraphView<TestNode, TestEdge>[] { merged, conflictingEdges }.ToGraph());
    }

    /// <summary>Compares all indexed endpoint pairs with independent breadth-first reachability.</summary>
    /// <param name="cyclic">Whether the fixture includes a cycle-closing edge.</param>
    /// <param name="sparse">Whether only a sparse subset of source nodes is indexed.</param>
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ReachabilityMatchesIndependentBfsForDenseSparseDagAndCycles(bool cyclic, bool sparse)
    {
        var nodes = Enumerable.Range(0, 130).Select(i => new TestNode($"N{i}")).ToArray();
        var edges = Enumerable.Range(0, 129).Select(i => new TestEdge($"chain{i}", $"N{i}", $"N{i + 1}"))
            .Concat(Enumerable.Range(0, 125).Select(i => new TestEdge($"skip{i}", $"N{i}", $"N{i + 5}")))
            .ToList();
        if (cyclic)
            edges.Add(new("back", "N129", "N0"));
        var graph = new Graph<TestNode, TestEdge>(nodes, edges);
        var sources = sparse ? nodes.Where((_, i) => i % 50 == 0).ToArray() : nodes;
        var targets = nodes.Reverse().ToArray();
        var index = graph.CreateReachabilityIndex(sources, targets);
        foreach (var source in sources)
            foreach (var target in targets)
                Assert.Equal(graph.IsReachable(source.Id, target.Id), index.IsReachable(source, target));
        Assert.True(index.IsReachable(nodes[0], nodes[64]));
        Assert.True(index.IsReachable(nodes[0], nodes[129]));
    }

    /// <summary>Checks single enumeration, identity deduplication, and strict endpoint-scope validation.</summary>
    [Fact]
    public void EndpointsAreEnumeratedOnceDeduplicatedAndRangeChecked()
    {
        var graph = Example();
        var from = graph.GetNode(new("A"));
        var to = graph.GetNode(new("L"));
        var sourceEnumerations = 0;
        var targetEnumerations = 0;
        IEnumerable<TestNode> Sources()
        {
            if (++sourceEnumerations > 1) throw new InvalidOperationException("Re-enumeration");
            yield return from;
            yield return from;
        }
        IEnumerable<TestNode> Targets()
        {
            if (++targetEnumerations > 1) throw new InvalidOperationException("Re-enumeration");
            yield return to;
            yield return to;
        }
        var index = graph.Filter(edge: edge => edge.Kind == "dep").CreateReachabilityIndex(Sources(), Targets());
        Assert.Equal(1, index.SourceCount);
        Assert.Equal(1, index.TargetCount);
        Assert.True(index.IsReachable(from, to));
        Assert.True(index.IsReachable(from.Id, to.Id));
        Assert.Throws<KeyNotFoundException>(() => index.IsReachable(new GraphNodeId("B"), to.Id));
        Assert.Throws<KeyNotFoundException>(() => index.IsReachable(from.Id, new GraphNodeId("C")));
        Assert.Throws<KeyNotFoundException>(() => graph.CreateReachabilityIndex([new TestNode("missing")], [to]));
        Assert.Equal(1, sourceEnumerations);
        Assert.Equal(1, targetEnumerations);
    }

    /// <summary>Checks index snapshots against mutable filters and paths through unindexed intermediate nodes.</summary>
    [Fact]
    public void IndexFreezesViewScopeAndTraversesNonEndpointNodes()
    {
        var graph = Example();
        var hidden = "";
        var view = graph.Filter(node => node.Name != hidden, edge => edge.Kind == "dep");
        var a = graph.GetNode(new("A"));
        var l = graph.GetNode(new("L"));
        var index = view.CreateReachabilityIndex([a], [l]);
        Assert.True(index.IsReachable(a, l));
        hidden = "I";
        Assert.True(index.IsReachable(a, l));
        Assert.False(view.CreateReachabilityIndex([a], [l]).IsReachable(a, l));
        var withoutL = graph.Filter(node => node.Name != "L");
        Assert.Throws<KeyNotFoundException>(() => withoutL.CreateReachabilityIndex([a], [l]));
    }

    /// <summary>Checks empty scopes, zero-step reachability, and null endpoint failures.</summary>
    [Fact]
    public void EmptyAndSelfIndexesHaveDefinedBehavior()
    {
        var graph = Example();
        var a = graph.GetNode(new("A"));
        var self = graph.CreateReachabilityIndex([a], [a]);
        Assert.True(self.IsReachable(a, a));
        Assert.Equal(0, graph.CreateReachabilityIndex([], []).SourceCount);
        var noTargets = graph.CreateReachabilityIndex([a], []);
        Assert.Equal(0, noTargets.TargetCount);
        Assert.Throws<KeyNotFoundException>(() => noTargets.IsReachable(a, a));
        Assert.Throws<ArgumentNullException>(() => graph.CreateReachabilityIndex(null!, [a]));
        Assert.Throws<ArgumentNullException>(() => graph.CreateReachabilityIndex([null!], [a]));
    }

    /// <summary>Checks declared adjacency order, parallel edges, and errors for hidden or unknown nodes.</summary>
    [Fact]
    public void AdjacencyMethodsPreserveOrderingAndRejectHiddenOrUnknownNodes()
    {
        var graph = Example();
        Assert.Equal(["d5", "d6", "d14", "r7", "r8"],
            graph.EdgesToNode(new("F")).Select(edge => edge.Id.Value));
        Assert.Equal(["d3", "d4", "d15", "w3"],
            graph.EdgesFromNode(new("C")).Select(edge => edge.Id.Value));
        Assert.Throws<KeyNotFoundException>(() => graph.EdgesToNode(new("missing")));
        Assert.Throws<KeyNotFoundException>(() => graph.EdgesFromNode(new("missing")));
        var view = graph.Filter(node => node.Kind == "op");
        Assert.Equal(["d5", "d6", "d14"],
            view.EdgesToNode(new("F")).Select(edge => edge.Id.Value));
        Assert.Equal(["d3", "d4", "d15"],
            view.EdgesFromNode(new("C")).Select(edge => edge.Id.Value));
        Assert.Throws<KeyNotFoundException>(() => view.EdgesToNode(new("R3")));
        Assert.Throws<KeyNotFoundException>(() => view.EdgesFromNode(new("R3")));
    }

    /// <summary>Checks that removed compatibility aliases and legacy matcher types are absent.</summary>
    [Fact]
    public void LegacyGraphApisAreNotExposed()
    {
        var graphType = typeof(Graph<TestNode, TestEdge>);
        foreach (var type in new[] { graphType, typeof(GraphView<TestNode, TestEdge>) })
            Assert.DoesNotContain(type.GetMethods(), method => method.Name is "View" or "InEdges" or "OutEdges");
        Assert.Null(graphType.Assembly.GetType("SharpInference.Graphs.Core.GraphPattern`2"));
        Assert.Null(graphType.Assembly.GetType("SharpInference.Graphs.Core.GraphMatch`2"));
    }
}
