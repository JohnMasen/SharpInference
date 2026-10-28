namespace SharpInference.Graphs.Logical;

/// <summary>A logical source package with read-only value and dependency analysis.</summary>
public sealed class LogicalGraphPackage : GraphPackage<LogicalGraph>
{
    public LogicalGraphPackage()
    {
    }
    public LogicalGraphPackage(LogicalGraph main, params LogicalGraph[] definitions) : base(main, definitions) { }

    public override IReadOnlyList<GraphDiagnostic> Validate()
    {
        var diagnostics = base.Validate();
        if (diagnostics.Any(item => item.Severity == GraphDiagnosticSeverity.Error))
        {
            return diagnostics;
        }
        return diagnostics.Concat(AnalyzeDefinitions(diagnostics).Values.SelectMany(result => result.Diagnostics))
            .Distinct().ToArray();
    }

    public IReadOnlyDictionary<string, GraphAnalysis> Analyze() => AnalyzeDefinitions(base.Validate());

    public GraphAnalysis Analyze(LogicalGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        if (!Contains(graph))
        {
            throw new ArgumentException("The graph is not a definition in this package.", nameof(graph));
        }
        return Analyze()[graph.Id];
    }

    public override LogicalGraphPackage Snapshot() => (LogicalGraphPackage)base.Snapshot();
    protected override GraphPackage<LogicalGraph> CreateSnapshot() => new LogicalGraphPackage();
}
